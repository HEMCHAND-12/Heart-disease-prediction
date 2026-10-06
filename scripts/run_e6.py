"""Calibrate the E2 PaRSEL score stream using its train-derived validation scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve

from scripts.hard_timeout import failure_reason, run_with_hard_timeout
from scripts.proposed_model import _append_extension, _default_record, extension_completed_run_keys
from scripts.run_experiments import RESULTS, PAPER_PATHS, TARGET, metrics
from scripts.extension_extra import save_reliability_diagram


def _fit_calibrator(method: str, scores: np.ndarray, targets: np.ndarray):
    if method == "platt":
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(scores.reshape(-1, 1), targets)
        return model
    model = IsotonicRegression(out_of_bounds="clip").fit(scores, targets)
    return model


def _probabilities(model, method: str, scores: np.ndarray) -> np.ndarray:
    if method == "platt":
        return model.predict_proba(scores.reshape(-1, 1))[:, 1]
    return model.predict(scores)


def _max_f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[int(np.nanargmax(f1))])


def _recall_threshold(targets: np.ndarray, probabilities: np.ndarray, target_recall: float = 0.90) -> float:
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    valid = np.flatnonzero(recall[:-1] >= target_recall)
    return float(thresholds[valid[-1]]) if len(valid) else 0.0


def run() -> None:
    validation = pd.read_csv(RESULTS / "e2_validation_scores.csv")
    test_scores = pd.read_csv(RESULTS / "e2_test_scores.csv")
    test_targets = pd.read_csv(PAPER_PATHS["test"])[TARGET].to_numpy()
    old_rows = pd.read_csv(RESULTS / "extension_results.csv")
    old_thresholds = {
        str(row.model_variant): float(row.threshold)
        for row in old_rows.loc[old_rows.run_key.str.startswith("E2:threshold:")].itertuples()
    }
    done = extension_completed_run_keys()
    for method in ("platt", "isotonic"):
        key = f"E6:calibration:{method}:ROS:PaRSEL"
        if key in done:
            continue
        record = _default_record("E6", key, record_type="calibration", model="PaRSEL",
                                 model_variant=method, sampler="ROS", reducer="none")
        try:
            calibrator = run_with_hard_timeout(
                _fit_calibrator, method, validation.score.to_numpy(), validation.target.to_numpy(),
                timeout_seconds=600,
            )
            calibrated_validation = _probabilities(calibrator, method, validation.score.to_numpy())
            calibrated_test = _probabilities(calibrator, method, test_scores.score.to_numpy())
            thresholds = {
                "max_f1": _max_f1_threshold(validation.target.to_numpy(), calibrated_validation),
                "recall_at_least_0.90": _recall_threshold(validation.target.to_numpy(), calibrated_validation),
            }
            threshold_rows = []
            for variant, threshold in thresholds.items():
                predictions = (calibrated_test >= threshold).astype(int)
                values = metrics(pd.Series(test_targets), predictions, calibrated_test, 0.0, 0.0)
                threshold_rows.append({
                    "method": method, "operating_point": variant, "threshold": threshold,
                    "brier_score": float(brier_score_loss(test_targets, calibrated_test)), **values,
                    "e2_threshold": old_thresholds.get(variant),
                })
            record.update({
                "threshold": thresholds["max_f1"],
                "brier_score": float(brier_score_loss(test_targets, calibrated_test)),
                "status": "ok", "error": "",
                "mean": float(np.mean(calibrated_test)),
                "std": float(np.std(calibrated_test)),
            })
            RESULTS.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(threshold_rows).to_csv(RESULTS / f"e6_{method}_thresholds.csv", index=False)
            from sklearn.calibration import calibration_curve
            fraction, mean = calibration_curve(test_targets, calibrated_test, n_bins=10, strategy="quantile")
            save_reliability_diagram(
                {"fraction_positive": fraction.tolist(), "mean_predicted": mean.tolist()},
                RESULTS / f"e6_{method}_reliability.png",
            )
        except Exception as error:  # noqa: BLE001
            record.update(status="failed", error=failure_reason(error))
        _append_extension(record)
        print(f"{key}: {record['status']}", flush=True)


if __name__ == "__main__":
    run()
