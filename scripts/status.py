"""Read-only progress dashboard for background experiments."""

from __future__ import annotations

import csv
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXTENSION_RESULTS = RESULTS / "extension_results.csv"
ALL_RUNS = RESULTS / "all_runs.csv"


def _process_rows() -> list[dict[str, str]]:
    output = subprocess.run(
        ["ps", "-eo", "pid=,etime=,pcpu=,cmd="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) == 4 and any(
            marker in parts[3]
            for marker in ("scripts.proposed_model", "scripts.run_experiments", "scripts.run_e4")
        ):
            rows.append({"pid": parts[0], "elapsed": parts[1], "cpu": parts[2], "cmd": parts[3]})
    return rows


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _e1_expected_rows(rows: list[dict[str, str]]) -> int:
    runtime = next((row for row in rows if row.get("record_type") == "runtime_plan"), None)
    if runtime and runtime.get("expected_rows"):
        return int(runtime["expected_rows"])
    # One plan row + 3*5 PaRSEL folds + 5*5*5 standalone folds + 6*8 summaries + 5 held-out tests.
    return 1 + 15 + 125 + 48 + 5


def _eta_seconds(rows: list[dict[str, str]]) -> float | None:
    fold_rows = [row for row in rows if row.get("record_type") == "cv_fold" and row.get("status") == "ok"]
    if not fold_rows:
        return None
    elapsed = [float(row["train_seconds"]) + float(row["predict_seconds"]) for row in fold_rows if row.get("train_seconds")]
    if not elapsed:
        return None
    expected = _e1_expected_rows(rows)
    remaining = max(expected - len(rows), 0)
    return remaining * (sum(elapsed) / len(elapsed))


def main() -> None:
    extension_rows = _read_rows(EXTENSION_RESULTS)
    all_rows = _read_rows(ALL_RUNS)
    processes = _process_rows()
    print("Processes:")
    if processes:
        for process in processes:
            print(f"  alive pid={process['pid']} elapsed={process['elapsed']} cpu={process['cpu']}% cmd={process['cmd']}")
    else:
        print("  none")
    print(f"Rows: extension_results.csv={len(extension_rows)}; all_runs.csv={len(all_rows)}")
    print("Rows by extension/model:")
    counts = Counter((row.get("extension", "phase1"), row.get("model", "unknown")) for row in extension_rows)
    for (extension, model), count in sorted(counts.items()):
        print(f"  {extension}/{model}: {count}")
    expected = _e1_expected_rows(extension_rows)
    percent = 100 * len(extension_rows) / expected if expected else 0.0
    eta = _eta_seconds(extension_rows)
    print(f"E1 estimate: {len(extension_rows)}/{expected} rows ({percent:.1f}%)")
    print(f"E1 rough ETA: {eta / 60:.1f} minutes" if eta is not None else "E1 rough ETA: unavailable")
    print("Last 3 extension rows:")
    for row in extension_rows[-3:]:
        print("  " + ",".join(row.get(key, "") for key in ("run_key", "record_type", "model", "seed", "fold", "status")))


if __name__ == "__main__":
    main()
