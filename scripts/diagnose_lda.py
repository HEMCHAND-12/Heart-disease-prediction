"""Diagnose LDA shape and class-prior effects on the fixed Phase 1 split."""

from __future__ import annotations

import json
import time
import uuid

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from scripts.run_experiments import (
    PAPER_PATHS,
    SEED,
    TARGET,
    append_record,
    completed_run_keys,
    make_model,
    make_sampler,
    metrics,
    model_scores,
    now_utc,
    transform_pair,
)
from scripts.hard_timeout import failure_reason, run_with_hard_timeout

MODELS = ["PAC", "Ridge", "SGD", "XGBoost", "LogitBoost_surrogate"]


def _fit_diagnostic_model(x_train, y_train, x_test, model_name):
    started = time.perf_counter()
    x_fit, y_fit = make_sampler("ROS").fit_resample(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    model = make_model(model_name)
    started = time.perf_counter()
    model.fit(x_fit, y_fit)
    fit_seconds += time.perf_counter() - started
    started = time.perf_counter()
    predictions = model.predict(x_test)
    scores = model_scores(model, x_test)
    return predictions, scores, fit_seconds, time.perf_counter() - started


def main() -> None:
    train = pd.read_csv(PAPER_PATHS["train"])
    test = pd.read_csv(PAPER_PATHS["test"])
    x_train_raw, x_test_raw, _, preprocessing_metadata = transform_pair(
        train, test, "none", scale=True
    )
    if x_train_raw.ndim != 2 or x_test_raw.ndim != 2:
        raise AssertionError("Pre-LDA features must be two-dimensional matrices")
    if np.isnan(x_train_raw).any() or np.isnan(x_test_raw).any():
        raise AssertionError("Pre-LDA train/test inputs contain NaN values")

    prior_variants = {
        "lda_fit_imbalanced_train": (x_train_raw, train[TARGET]),
    }
    x_ros, y_ros = make_sampler("ROS").fit_resample(x_train_raw, train[TARGET])
    prior_variants["lda_fit_ros_balanced_train"] = (x_ros, y_ros)

    completed = completed_run_keys()
    for prior_name, (lda_fit_x, lda_fit_y) in prior_variants.items():
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(lda_fit_x, lda_fit_y)
        x_train_lda = lda.transform(x_train_raw)
        x_test_lda = lda.transform(x_test_raw)
        if x_train_lda.ndim != 2 or x_test_lda.ndim != 2 or x_train_lda.shape[1] != 1 or x_test_lda.shape[1] != 1:
            raise AssertionError(f"Expected one-dimensional binary LDA output, got {x_train_lda.shape}, {x_test_lda.shape}")
        if np.isnan(x_train_lda).any() or np.isnan(x_test_lda).any():
            raise AssertionError("LDA transform returned NaN values")

        for model_name in MODELS:
            run_key = f"phase1_diagnostic:paper_faithful:ROS:LDA:{prior_name}:{model_name}:{SEED}"
            if run_key in completed:
                continue
            metadata = {
                "reducer": "LDA",
                "component_count": 1,
                "input_shape": list(lda_fit_x.shape),
                "transformed_train_shape": list(x_train_lda.shape),
                "transformed_test_shape": list(x_test_lda.shape),
                "scaling": True,
                "train_has_nan": False,
                "test_has_nan": False,
                "lda_fit_variant": prior_name,
                "lda_fit_class_counts": {str(key): int(value) for key, value in pd.Series(lda_fit_y).value_counts().sort_index().items()},
                "classifier_train_counts_after_ros": {
                    "0": int(train[TARGET].value_counts().max()),
                    "1": int(train[TARGET].value_counts().max()),
                },
                "preprocessing": preprocessing_metadata,
            }
            record = {
                "run_id": uuid.uuid4().hex[:12],
                "run_key": run_key,
                "timestamp_utc": now_utc(),
                "phase": "phase1_lda_diagnostic",
                "dataset_variant": "paper_faithful",
                "split": "paper_faithful",
                "balancer": "ROS",
                "sampler": "ROS",
                "reducer": "LDA",
                "model": model_name,
                "model_variant": prior_name,
                "seed": SEED,
                "train_rows": len(train),
                "test_rows": len(test),
                "status": "ok",
                "error": "",
                "rfe_features": "",
                "tuned_parameters": "{}",
                "reducer_metadata": json.dumps(metadata, sort_keys=True),
                "decision_threshold": 0.5,
            }
            try:
                predictions, scores, fit_seconds, predict_seconds = run_with_hard_timeout(
                    _fit_diagnostic_model, x_train_lda, train[TARGET], x_test_lda,
                    model_name, timeout_seconds=600,
                )
                record.update(metrics(test[TARGET], predictions, scores, fit_seconds, predict_seconds))
                record["brier_score"] = ""
            except Exception as error:  # noqa: BLE001
                record.update({"status": "failed", "error": failure_reason(error)})
            append_record(record)
            print(f"completed {prior_name} / {model_name}")


if __name__ == "__main__":
    main()
