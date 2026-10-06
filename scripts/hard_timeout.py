"""Run an experiment callable in a killable child process."""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from typing import Any, Callable


class ExperimentTimeout(TimeoutError):
    """Raised after the child process is killed at its wall-clock limit."""


def failure_reason(error: BaseException) -> str:
    if isinstance(error, ExperimentTimeout):
        return f"timeout: {error}"
    return f"{type(error).__name__}: {error}"


def _worker(output: Any, function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    try:
        output.put((True, function(*args, **kwargs)))
    except BaseException as error:  # noqa: BLE001
        output.put((False, (type(error).__name__, str(error), traceback.format_exc())))


def run_with_hard_timeout(
    function: Callable[..., Any], *args: Any, timeout_seconds: int = 600, **kwargs: Any
) -> Any:
    """Run work in a forked process so native code can be forcibly stopped."""
    context = mp.get_context("fork")
    output = context.Queue(maxsize=1)
    process = context.Process(target=_worker, args=(output, function, args, kwargs))
    process.start()
    try:
        # Read concurrently with the child so large NumPy results cannot fill
        # the pipe and block the child's feeder thread while the parent joins.
        succeeded, result = output.get(timeout=timeout_seconds)
    except queue.Empty:
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            output.close()
            process.close()
            raise ExperimentTimeout(f"experiment exceeded {timeout_seconds} seconds; child process killed")
        output.close()
        process.close()
        raise RuntimeError(f"experiment child exited with code {process.exitcode} without a result")
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        output.close()
        process.close()
        raise RuntimeError("experiment child sent a result but failed to exit cleanly")
    output.close()
    process.close()
    if not succeeded:
        name, message, child_traceback = result
        raise RuntimeError(f"child {name}: {message}\n{child_traceback}")
    return result
