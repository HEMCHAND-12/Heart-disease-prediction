import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.extension_extra import (
    calibrate_model,
    calibration_summary,
    compare_meta_learners,
    fit_imbalance_variants,
    map_pima_common_features,
    subgroup_metrics,
)


def synthetic_data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 4))
    y = pd.Series((x[:, 0] + x[:, 1] > 0).astype(int), name="diabetes")
    return x, y


def test_e4_meta_learners_return_scores():
    x, y = synthetic_data()
    outputs = compare_meta_learners(x, y, x[:10])
    assert set(outputs) == {"LogitBoost_style", "LogisticRegression", "WeightedAverage"}
    assert all(values.shape == (10,) for values in outputs.values())


def test_e5_returns_train_only_variants():
    x, y = synthetic_data()
    variants = fit_imbalance_variants(x, y)
    assert {"class_weight", "ROS", "SMOTE"} <= set(variants)


def test_e6_calibration_reports_brier_score():
    x, y = synthetic_data()
    model = LogisticRegression(max_iter=1000).fit(x, y)
    calibrated = calibrate_model(model, x, y)
    summary = calibration_summary(calibrated["platt"], x, y)
    assert summary["brier_score"] >= 0


def test_e7_subgroups_and_e8_mapping():
    _, y = synthetic_data()
    frame = pd.DataFrame({
        "gender": ["Female", "Male"] * 40,
        "age": np.linspace(20, 80, 80),
        "diabetes": y,
    })
    groups = subgroup_metrics(frame, y.to_numpy(), y.to_numpy(dtype=float))
    assert {"gender", "age_band"} <= set(groups.dimension)
    pima = pd.DataFrame({"Glucose": [100, 150], "BMI": [25, 30], "Age": [30, 50], "Outcome": [0, 1]})
    mapped, labels = map_pima_common_features(pima)
    assert list(mapped.columns) == ["age", "bmi", "blood_glucose_level"]
    assert labels.tolist() == [0, 1]
