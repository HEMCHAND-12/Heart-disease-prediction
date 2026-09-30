import csv

from scripts.status import _e1_expected_rows, _eta_seconds, _read_rows


def test_status_reads_fixture_and_estimates_progress(tmp_path):
    path = tmp_path / "extension_results.csv"
    fields = ["record_type", "run_key", "model", "seed", "fold", "status", "train_seconds", "predict_seconds"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"record_type": "runtime_plan", "run_key": "plan", "model": "", "seed": "42", "fold": "", "status": "planned", "train_seconds": "", "predict_seconds": ""})
        writer.writerow({"record_type": "cv_fold", "run_key": "fold0", "model": "PAC", "seed": "0", "fold": "0", "status": "ok", "train_seconds": "10", "predict_seconds": "1"})
        writer.writerow({"record_type": "cv_fold", "run_key": "fold1", "model": "Ridge", "seed": "0", "fold": "1", "status": "ok", "train_seconds": "20", "predict_seconds": "1"})
    rows = _read_rows(path)
    assert len(rows) == 3
    assert _e1_expected_rows(rows) == 194
    assert _eta_seconds(rows) > 0
