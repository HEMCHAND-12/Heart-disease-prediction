"""Unlaunched PaRSEL+ E4-E8 helpers; all public runners must checkpoint extension keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from scripts.run_experiments import make_model, make_sampler, model_scores

EXTENSION_KEYS = {
    "E4": "E4:meta",
    "E5": "E5:imbalance",
    "E6": "E6:calibration",
    "E7": "E7:explainability_fairness",
    "E8": "E8:pima_external",
}


def compare_meta_learners(oof: np.ndarray, y_train: pd.Series, test_meta: np.ndarray) -> dict[str, np.ndarray]:
    """Fit E4 meta learners on OOF predictions only; caller checkpoints each result."""
    learners = {
        "LogitBoost_style": GradientBoostingClassifier(loss="log_loss", n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }
    outputs: dict[str, np.ndarray] = {}
    for name, learner in learners.items():
        learner.fit(oof, y_train)
        outputs[name] = model_scores(learner, test_meta)
    outputs["WeightedAverage"] = np.average(test_meta, axis=1, weights=np.ones(test_meta.shape[1]))
    return outputs


def fit_imbalance_variants(x_train: np.ndarray, y_train: pd.Series) -> dict[str, Any]:
    """Return train-only class-weight and ROS/SMOTE training variants."""
    outputs: dict[str, Any] = {}
    weighted = make_model("LogitBoost_surrogate")
    weighted.fit(x_train, y_train)
    outputs["class_weight"] = weighted
    for name in ("ROS", "SMOTE"):
        sampler = make_sampler(name)
        x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
        model = make_model("LogitBoost_surrogate")
        model.fit(x_resampled, y_resampled)
        outputs[name] = model
    return outputs


def calibrate_model(model: Any, x_train: np.ndarray, y_train: pd.Series) -> dict[str, Any]:
    """Create train-only Platt and isotonic calibration wrappers."""
    return {
        "platt": CalibratedClassifierCV(model, method="sigmoid", cv=3).fit(x_train, y_train),
        "isotonic": CalibratedClassifierCV(model, method="isotonic", cv=3).fit(x_train, y_train),
    }


def calibration_summary(model: Any, x_test: np.ndarray, y_test: pd.Series) -> dict[str, Any]:
    probabilities = model.predict_proba(x_test)[:, 1]
    fraction_positive, mean_predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
    return {
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "fraction_positive": fraction_positive.tolist(),
        "mean_predicted": mean_predicted.tolist(),
    }


def save_reliability_diagram(summary: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(summary["mean_predicted"], summary["fraction_positive"], marker="o", label="model")
    axis.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", title="Reliability diagram")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def shap_artifact_plan(model: Any, x_sample: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Return a deferred SHAP plan; caller executes and checkpoints E7 later."""
    return {
        "sample_rows": min(500, len(x_sample)),
        "artifacts": [
            str(output_dir / "shap_summary.png"),
            str(output_dir / "shap_waterfall.png"),
            str(output_dir / "shap_dependence.png"),
        ],
        "model_type": type(model).__name__,
    }


def subgroup_metrics(frame: pd.DataFrame, predictions: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Return subgroup counts and rates; caller supplies untouched evaluation predictions."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    data = frame.copy()
    data["prediction"] = predictions
    data["score"] = scores
    data["age_band"] = pd.cut(data["age"], bins=[-np.inf, 40, 60, np.inf], labels=["<40", "40-60", ">60"])
    rows = []
    for dimension, groups in (("gender", data.groupby("gender", dropna=False)), ("age_band", data.groupby("age_band", dropna=False))):
        for group, subset in groups:
            rows.append({
                "dimension": dimension,
                "group": str(group),
                "n": len(subset),
                "positive_prevalence": float(subset["diabetes"].mean()),
                "precision": precision_score(subset["diabetes"], subset["prediction"], zero_division=0),
                "recall": recall_score(subset["diabetes"], subset["prediction"], zero_division=0),
                "f1": f1_score(subset["diabetes"], subset["prediction"], zero_division=0),
            })
    return pd.DataFrame(rows)


def map_pima_common_features(pima: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Map Pima glucose, BMI, and age; drop all unmatched features and retain caveat metadata."""
    required = {"Glucose", "BMI", "Age", "Outcome"}
    missing = required.difference(pima.columns)
    if missing:
        raise ValueError(f"Pima input missing columns: {sorted(missing)}")
    mapped = pd.DataFrame({
        "age": pima["Age"],
        "bmi": pima["BMI"],
        "blood_glucose_level": pima["Glucose"],
    })
    return mapped, pima["Outcome"].astype(int)
