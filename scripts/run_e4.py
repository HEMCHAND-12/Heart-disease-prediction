"""Run E4 meta-learner comparisons from a shared OOF base-score cache."""

from __future__ import annotations

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

CACHE_DIR = RESULTS / "e4_cache"
BASE_MODELS = ("PAC", "Ridge", "SGD", "XGBoost")
META_LEARNERS = ("LogitBoost_style", "LogisticRegression", "WeightedAverage")
SEEDS = (0, 1, 2)


def _meta_scores(name: str, train_scores: np.ndarray, y_train: np.ndarray, evaluation_scores: np.ndarray) -> np.ndarray:
    if name == "WeightedAverage":
        return np.mean(evaluation_scores, axis=1)
    learner = (
        LogisticRegression(max_iter=1000, random_state=SEED)
        if name == "LogisticRegression"
        else GradientBoostingClassifier(loss="log_loss", n_estimators=100, random_state=SEED)
    )
    learner.fit(train_scores, y_train)
    return model_scores(learner, evaluation_scores)


def _build_cache(seed: int, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oof_path = CACHE_DIR / f"seed_{seed}_oof.npz"
    test_path = CACHE_DIR / f"seed_{seed}_test.npz"
    if oof_path.exists() and test_path.exists():
        oof = np.load(oof_path)
        test_scores = np.load(test_path)
        return oof["scores"], oof["target"], test_scores["scores"]

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros((len(train), len(BASE_MODELS)), dtype=float)
    for fit_index, validation_index in splitter.split(train[FEATURES], train[TARGET]):
        fold_train = train.iloc[fit_index]
        fold_valid = train.iloc[validation_index]
        x_fit, x_valid, _, _ = transform_pair(fold_train, fold_valid, "none", scale=True)
        x_fit_xgb, x_valid_xgb, _, _ = transform_pair(fold_train, fold_valid, "none", scale=False)
        x_balanced, y_balanced = make_sampler("ROS").fit_resample(x_fit, fold_train[TARGET])
        x_balanced_xgb, y_balanced_xgb = make_sampler("ROS").fit_resample(x_fit_xgb, fold_train[TARGET])
        for column, model_name in enumerate(BASE_MODELS):
            model = make_model(model_name)
            fit_x = x_balanced_xgb if model_name == "XGBoost" else x_balanced
            fit_y = y_balanced_xgb if model_name == "XGBoost" else y_balanced
            model.fit(fit_x, fit_y)
            score_x = x_valid_xgb if model_name == "XGBoost" else x_valid
            oof[validation_index, column] = model_scores(model, score_x)

    x_train, x_test, _, _ = transform_pair(train, test, "none", scale=True)
    x_train_xgb, x_test_xgb, _, _ = transform_pair(train, test, "none", scale=False)
    x_balanced, y_balanced = make_sampler("ROS").fit_resample(x_train, train[TARGET])
    x_balanced_xgb, y_balanced_xgb = make_sampler("ROS").fit_resample(x_train_xgb, train[TARGET])
    test_scores = []
    for model_name in BASE_MODELS:
        model = make_model(model_name)
        fit_x = x_balanced_xgb if model_name == "XGBoost" else x_balanced
        fit_y = y_balanced_xgb if model_name == "XGBoost" else y_balanced
        model.fit(fit_x, fit_y)
        score_x = x_test_xgb if model_name == "XGBoost" else x_test
        test_scores.append(model_scores(model, score_x))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(oof_path, scores=oof, target=train[TARGET].to_numpy())
    np.savez_compressed(test_path, scores=np.column_stack(test_scores))
    return oof, train[TARGET].to_numpy(), np.column_stack(test_scores)


def run() -> None:
    train = pd.read_csv(PAPER_PATHS["train"])
    test = pd.read_csv(PAPER_PATHS["test"])
    finished = extension_completed_run_keys()
    caches = {}
    for seed in SEEDS:
        cache_key = f"E4:cache:seed{seed}"
        with experiment_timeout():
            cache = _build_cache(seed, train, test)
        caches[seed] = cache
        if cache_key not in finished:
            _append_extension(_default_record("E4", cache_key, record_type="base_oof_cache", seed=seed, model="PaRSEL_base_layer", model_variant="shared_cache", sampler="ROS", status="ok"))
            finished.add(cache_key)

    for seed, (oof, target, _) in caches.items():
        for meta_name in META_LEARNERS:
            run_key = f"E4:cv:{meta_name}:seed{seed}"
            if run_key in finished:
                continue
            record = _default_record("E4", run_key, record_type="cv_summary", seed=seed, model=meta_name, model_variant="shared_oof_base_scores", sampler="ROS")
            try:
                with experiment_timeout():
                    # OOF base scores are cached; no base model is refit per meta learner.
                    scores = _meta_scores(meta_name, oof, target, oof)
                    predictions = (scores >= 0.5).astype(int)
                record.update(metrics(pd.Series(target), predictions, scores, 0.0, 0.0))
            except Exception as error:  # noqa: BLE001
                record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            _append_extension(record)
            finished.add(run_key)

    runs = pd.read_csv(RESULTS / "all_runs.csv")
    baseline_key = "phase1:paper_faithful:ROS:none:PaRSEL:42"
    baseline_rows = runs.loc[
        runs["run_key"].astype(str).str.contains("phase1:paper_faithful:ROS:none:PaRSEL:42", regex=False)
        & runs["status"].astype(str).str.strip().eq("ok")
    ]
    if baseline_rows.empty:
        raise RuntimeError(f"Missing completed baseline row: {baseline_key}")
    baseline = baseline_rows.iloc[-1]
    baseline_curve = pd.read_csv(RESULTS / f"curve_{baseline.run_id}.csv")
    baseline_predictions = baseline_curve.prediction.to_numpy()
    baseline_scores = baseline_curve.score.to_numpy()
    oof, target, test_meta = caches[SEEDS[0]]
    for meta_name in META_LEARNERS:
        run_key = f"E4:paired_test:{meta_name}"
        if run_key in finished:
            continue
        record = _default_record("E4", run_key, record_type="paired_test", model=meta_name, model_variant="shared_final_base_scores", sampler="ROS")
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
