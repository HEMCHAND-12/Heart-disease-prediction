"""Time-boxed standalone ROS + Factor Analysis Phase 1 comparison."""

from __future__ import annotations

import csv
import time
import uuid

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

from scripts.hard_timeout import ExperimentTimeout, run_with_hard_timeout
from scripts.run_experiments import (
    FEATURES, PAPER_PATHS, RESULTS, SEED, TARGET, append_record,
    completed_run_keys, make_model, make_sampler, metrics, model_scores,
    now_utc, preprocessor, TrainWinsorizer,
)

MODELS = ("PAC", "Ridge", "SGD", "XGBoost", "LogitBoost_surrogate")
COMPONENTS = (2, 4, 6)


def _prepare_fa_data(train: pd.DataFrame, test: pd.DataFrame, components: int, cache_path: str):
    cache = RESULTS / "fa_cache" / cache_path
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return
    y = train[TARGET]
    winsor = TrainWinsorizer().fit(train[FEATURES])
    prep = preprocessor(scale=True)
    x_train = prep.fit_transform(winsor.transform(train[FEATURES]), y)
    x_test = prep.transform(winsor.transform(test[FEATURES]))
    reducer = FactorAnalysis(n_components=components, random_state=SEED).fit(x_train)
    x_train, x_test = reducer.transform(x_train), reducer.transform(x_test)
    temporary = cache.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, x_train=x_train, x_test=x_test, y_train=y.to_numpy())
    temporary.replace(cache)


def _fit_fa_model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, model_name: str):
    x_balanced, y_balanced = make_sampler("ROS").fit_resample(x_train, y_train)
    model = make_model(model_name)
    start = time.perf_counter()
    model.fit(x_balanced, y_balanced)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    scores = model_scores(model, x_test)
    predictions = model.predict(x_test)
    predict_seconds = time.perf_counter() - start
    return predictions, scores, fit_seconds, predict_seconds


def run() -> None:
    train = pd.read_csv(PAPER_PATHS["train"])
    test = pd.read_csv(PAPER_PATHS["test"])
    finished = completed_run_keys()
    for components in COMPONENTS:
        cache_name = f"fa_scaled_{components}.npz"
        try:
            run_with_hard_timeout(
                _prepare_fa_data, train, test, components, cache_name, timeout_seconds=600
            )
        except ExperimentTimeout as error:
            prep_error = f"timeout: {error}"
        except Exception as error:  # noqa: BLE001
            prep_error = f"{type(error).__name__}: {error}"
        else:
            prep_error = ""
        if prep_error:
            for model_name in MODELS:
                key = f"phase1:paper_faithful:ROS:FA{components}:{model_name}:42:standalone_hardtimeout_v1"
                if key in finished:
                    continue
                append_record({
                    "run_id": uuid.uuid4().hex[:12], "run_key": key,
                    "timestamp_utc": now_utc(), "phase": "phase1_baseline",
                    "dataset_variant": "paper_faithful", "split": "paper_faithful",
                    "balancer": "ROS", "sampler": "ROS", "reducer": f"FA{components}",
                    "model": model_name, "model_variant": "paper_parameters",
                    "seed": SEED, "train_rows": len(train), "test_rows": len(test),
                    "status": "failed", "error": prep_error,
                })
                finished.add(key)
            print(f"FA{components}: {prep_error}", flush=True)
            continue
        data = np.load(RESULTS / "fa_cache" / cache_name)
        x_train, x_test, y_train = data["x_train"], data["x_test"], data["y_train"]
        for model_name in MODELS:
            key = f"phase1:paper_faithful:ROS:FA{components}:{model_name}:42:standalone_hardtimeout_v1"
            if key in finished:
                continue
            start = now_utc()
            row = {
                "run_id": uuid.uuid4().hex[:12], "run_key": key,
                "timestamp_utc": start, "phase": "phase1_baseline",
                "dataset_variant": "paper_faithful", "split": "paper_faithful",
                "balancer": "ROS", "sampler": "ROS", "reducer": f"FA{components}",
                "model": model_name, "model_variant": "paper_parameters",
                "seed": SEED, "train_rows": len(train), "test_rows": len(test),
                "status": "ok", "error": "", "decision_threshold": 0.5,
            }
            try:
                predicted, scores, fit_s, pred_s = run_with_hard_timeout(
                    _fit_fa_model, x_train, y_train, x_test, model_name, timeout_seconds=600
                )
                row.update(metrics(test[TARGET], predicted, scores, fit_s, pred_s))
                curve_path = RESULTS / "phase1_fa_predictions.csv"
                write_header = not curve_path.exists()
                with curve_path.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["run_key", "target", "score", "prediction"])
                    if write_header:
                        writer.writeheader()
                    writer.writerows({"run_key": key, "target": int(y), "score": float(s), "prediction": int(p)}
                                     for y, s, p in zip(test[TARGET], scores, predicted))
            except ExperimentTimeout as error:
                row.update(status="failed", error=f"timeout: {error}")
            except Exception as error:  # noqa: BLE001
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            append_record(row)
            finished.add(key)
            print(f"{key}: {row['status']}", flush=True)


if __name__ == "__main__":
    run()
