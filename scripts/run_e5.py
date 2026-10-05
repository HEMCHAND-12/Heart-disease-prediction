"""Run E5 class-weight/ROS/SMOTE comparisons with train-fold resampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from scripts.proposed_model import (
    _append_extension,
    _default_record,
    _delong_pvalue,
    _mcnemar_pvalue,
    extension_completed_run_keys,
)
from scripts.run_experiments import (
    FEATURES,
    PAPER_PATHS,
    RESULTS,
    SEED,
    TARGET,
    experiment_timeout,
    make_model,
    make_sampler,
    metrics,
    model_scores,
    transform_pair,
)

CACHE_DIR = RESULTS / "e5_cache"
BASE_MODELS = ("PAC", "Ridge", "SGD", "XGBoost")
STRATEGIES = ("class_weight", "ROS", "SMOTE")
META_LEARNERS = ("LogitBoost_style", "LogisticRegression")
SEEDS = (0, 1, 2)


def _fit_model(name: str, x: np.ndarray, y: pd.Series):
    model = make_model(name)
    if name != "XGBoost" and hasattr(model, "set_params") and name in ("PAC", "Ridge", "SGD"):
        model.set_params(class_weight="balanced")
    if name == "XGBoost":
        model.set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
    model.fit(x, y)
    return model


def _prepare_training(strategy: str, fold_train: pd.DataFrame, x_train: np.ndarray, y_train: pd.Series):
    if strategy == "class_weight":
        return x_train, y_train
    sampler = make_sampler(strategy)
    return sampler.fit_resample(x_train, y_train)


def _meta_scores(name: str, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    if name == "LogisticRegression":
        learner = LogisticRegression(max_iter=1000, random_state=SEED)
    else:
        learner = GradientBoostingClassifier(loss="log_loss", n_estimators=100, random_state=SEED)
    learner.fit(x_train, y_train)
    return model_scores(learner, x_eval)


def _cache_path(strategy: str, seed: int) -> Path:
    return CACHE_DIR / f"{strategy}_seed{seed}.npz"


def build_cache(strategy: str, seed: int, train: pd.DataFrame, test: pd.DataFrame):
    path = _cache_path(strategy, seed)
    if path.exists():
        data = np.load(path)
        return data["oof"], data["target"], data["test"]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros((len(train), len(BASE_MODELS)))
    for fit_index, validation_index in splitter.split(train[FEATURES], train[TARGET]):
        fold_train, fold_valid = train.iloc[fit_index], train.iloc[validation_index]
        x_fit, x_valid, _, _ = transform_pair(fold_train, fold_valid, "none", scale=True)
        x_fit_xgb, x_valid_xgb, _, _ = transform_pair(fold_train, fold_valid, "none", scale=False)
        balanced_x, balanced_y = _prepare_training(strategy, fold_train, x_fit, fold_train[TARGET])
        balanced_xgb, balanced_y_xgb = _prepare_training(strategy, fold_train, x_fit_xgb, fold_train[TARGET])
        for column, name in enumerate(BASE_MODELS):
            if strategy == "class_weight":
                model = _fit_model(name, balanced_xgb if name == "XGBoost" else balanced_x, balanced_y_xgb if name == "XGBoost" else balanced_y)
            else:
                model = make_model(name)
                model.fit(balanced_xgb if name == "XGBoost" else balanced_x, balanced_y_xgb if name == "XGBoost" else balanced_y)
            oof[validation_index, column] = model_scores(model, x_valid_xgb if name == "XGBoost" else x_valid)
    x_train, x_test, _, _ = transform_pair(train, test, "none", scale=True)
    x_train_xgb, x_test_xgb, _, _ = transform_pair(train, test, "none", scale=False)
    balanced_x, balanced_y = _prepare_training(strategy, train, x_train, train[TARGET])
    balanced_xgb, balanced_y_xgb = _prepare_training(strategy, train, x_train_xgb, train[TARGET])
    test_scores = []
    for name in BASE_MODELS:
        if strategy == "class_weight":
            model = _fit_model(name, balanced_xgb if name == "XGBoost" else balanced_x, balanced_y_xgb if name == "XGBoost" else balanced_y)
        else:
            model = make_model(name)
            model.fit(balanced_xgb if name == "XGBoost" else balanced_x, balanced_y_xgb if name == "XGBoost" else balanced_y)
        test_scores.append(model_scores(model, x_test_xgb if name == "XGBoost" else x_test))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, oof=oof, target=train[TARGET].to_numpy(), test=np.column_stack(test_scores))
    return oof, train[TARGET].to_numpy(), np.column_stack(test_scores)


def run() -> None:
    train = pd.read_csv(PAPER_PATHS["train"])
    test = pd.read_csv(PAPER_PATHS["test"])
    phase1 = pd.read_csv(RESULTS / "all_runs.csv")
    baseline = phase1.loc[phase1.run_key.astype(str).str.contains("phase1:paper_faithful:ROS:none:PaRSEL:42", regex=False)].iloc[-1]
    baseline_curve = pd.read_csv(RESULTS / f"curve_{baseline.run_id}.csv")
    baseline_predictions, baseline_scores = baseline_curve.prediction.to_numpy(), baseline_curve.score.to_numpy()
    finished = extension_completed_run_keys()
    caches = {}
    for strategy in STRATEGIES:
        for seed in SEEDS:
            cache_key = f"E5:cache:{strategy}:seed{seed}"
            with experiment_timeout():
                cache = build_cache(strategy, seed, train, test)
            caches[(strategy, seed)] = cache
            if cache_key not in finished:
                _append_extension(_default_record("E5", cache_key, record_type="base_oof_cache", seed=seed, model="PaRSEL_base_layer", model_variant=strategy, sampler=strategy, status="ok"))
                finished.add(cache_key)

    for strategy in STRATEGIES:
        for meta_name in META_LEARNERS:
            for seed in SEEDS:
                run_key = f"E5:cv:{strategy}:{meta_name}:seed{seed}"
                if run_key in finished:
                    continue
                oof, target, _ = caches[(strategy, seed)]
                record = _default_record("E5", run_key, record_type="cv_summary", seed=seed, model=meta_name, model_variant=strategy, sampler=strategy)
                try:
                    with experiment_timeout():
                        scores = _meta_scores(meta_name, oof, target, oof)
                        predictions = (scores >= 0.5).astype(int)
                    record.update(metrics(pd.Series(target), predictions, scores, 0.0, 0.0))
                except Exception as error:  # noqa: BLE001
                    record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                _append_extension(record)
                finished.add(run_key)

    for strategy in STRATEGIES:
        for meta_name in META_LEARNERS:
            run_key = f"E5:paired_test:{strategy}:{meta_name}"
            if run_key in finished:
                continue
            oof, target, test_meta = caches[(strategy, SEEDS[0])]
            record = _default_record("E5", run_key, record_type="paired_test", model=meta_name, model_variant=strategy, sampler=strategy)
            try:
                with experiment_timeout():
                    scores = _meta_scores(meta_name, oof, target, test_meta)
                    predictions = (scores >= 0.5).astype(int)
                record.update(metrics(test[TARGET], predictions, scores, 0.0, 0.0))
                record["mcnemar_p"] = _mcnemar_pvalue(test[TARGET].to_numpy(), baseline_predictions, predictions)
                record["delong_p"] = _delong_pvalue(test[TARGET].to_numpy(), baseline_scores, scores)
            except Exception as error:  # noqa: BLE001
                record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            _append_extension(record)
            finished.add(run_key)


if __name__ == "__main__":
    run()
