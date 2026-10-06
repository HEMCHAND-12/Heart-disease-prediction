import time

import pytest

from scripts.hard_timeout import ExperimentTimeout, failure_reason, run_with_hard_timeout


def _sleep(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


def _large_result(size: int) -> bytes:
    return b"x" * size


def test_child_result_returns_before_deadline() -> None:
    assert run_with_hard_timeout(_sleep, 0.01, timeout_seconds=2) == "done"


def test_child_is_killed_at_hard_deadline() -> None:
    with pytest.raises(ExperimentTimeout, match="child process killed"):
        run_with_hard_timeout(_sleep, 1.0, timeout_seconds=0.1)


def test_large_child_result_does_not_deadlock_parent() -> None:
    result = run_with_hard_timeout(_large_result, 1_000_000, timeout_seconds=2)
    assert len(result) == 1_000_000


def test_timeout_failure_has_machine_readable_reason() -> None:
    assert failure_reason(ExperimentTimeout("killed")).startswith("timeout:")
