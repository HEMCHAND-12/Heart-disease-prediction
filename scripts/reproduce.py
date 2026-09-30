"""Run the PaRSEL reproduction experiments from the raw diabetes data."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import platform
import time
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "diabetes_prediction_dataset.csv"
SPLITS = ROOT / "splits"
RESULTS = ROOT / "results"
TARGET = "diabetes"
SEED = 42
CATEGORICAL = ["gender", "smoking_history"]
NUMERIC = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
CONTINUOUS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
FEATURES = [*CATEGORICAL, *NUMERIC]
SPLIT_PATHS = {name: SPLITS / f"{name}.csv" for name in ("stratified_train", "stratified_test", "grouped_train", "grouped_test")}

PAPER_PARAMS: dict[str, dict[str, Any]] = {
    "PAC": {"C": 1.6063676e-05, "max_iter": 2000, "random_state": 30},
    "Ridge": {"alpha": 7.3, "solver": "saga"},
    "SGD": {"alpha": 0.000745, "loss": "perceptron", "penalty": "l2", "max_iter": 3000, "random_state": 30},
    "XGBoost": {"learning_rate": 0.0582, "n_estimators": 57, "max_depth": 7, "subsample": 0.534, "colsample_bytree": 0.731, "random_state": 30, "n_jobs": 4, "tree_method": "hist", "eval_metric": "logloss"},
    "LogitBoost_surrogate": {"learning_rate": 0.132, "n_estimators": 400, "max_leaf_nodes": 127, "min_samples_leaf": 4, "max_depth": 5, "random_state": 30},
}


class TrainWinsorizer:
    """Fit 1.5-IQR bounds on training data and reuse them unchanged."""

    def fit(self, frame: pd.DataFrame) -> TrainWinsorizer:
        self.bounds_ = {}
        for name in CONTINUOUS:
            low, high = frame[name].quantile([0.25, 0.75])
            iqr = high - low
            self.bounds_[name] = (float(low - 1.5 * iqr), float(high + 1.5 * iqr))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for name, (low, high) in self.bounds_.items():
            result[name] = result[name].clip(low, high)
        return result


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def load_source() -> tuple[pd.DataFrame, int, int]:
    frame = pd.read_csv(SOURCE)
    null_rows = int(frame.isna().any(axis=1).sum())
    frame = frame.dropna().reset_index(drop=True)
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates(keep="first").reset_index(drop=True)
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Expected diabetes labels 0 and 1")
    return frame, duplicate_rows, null_rows


def group_ids(frame: pd.DataFrame) -> np.ndarray:
    return pd.factorize(pd.MultiIndex.from_frame(frame[FEATURES]))[0]


def prepare_splits(force: bool = False) -> dict[str, Any]:
    data, duplicates, null_rows = load_source()
    if force or not all(path.exists() for path in SPLIT_PATHS.values()):
        train, test = train_test_split(data, test_size=0.2, random_state=SEED, stratify=data[TARGET])
        group_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        group_train_idx, group_test_idx = next(group_splitter.split(data[FEATURES], data[TARGET], group_ids(data)))
        grouped_train, grouped_test = data.iloc[group_train_idx], data.iloc[group_test_idx]
        SPLITS.mkdir(parents=True, exist_ok=True)
        train.to_csv(SPLIT_PATHS["stratified_train"], index=False)
        test.to_csv(SPLIT_PATHS["stratified_test"], index=False)
        grouped_train.to_csv(SPLIT_PATHS["grouped_train"], index=False)
        grouped_test.to_csv(SPLIT_PATHS["grouped_test"], index=False)
    strat_train, strat_test, group_train, group_test = (pd.read_csv(path) for path in SPLIT_PATHS.values())
    for train, test in ((strat_train, strat_test), (group_train, group_test)):
        overlap = len(set(map(tuple, train.to_numpy())) & set(map(tuple, test.to_numpy())))
        if overlap:
            raise AssertionError(f"Exact train/test row overlap: {overlap}")
    shared_predictors = len(
        set(map(tuple, group_train[FEATURES].to_numpy()))
        & set(map(tuple, group_test[FEATURES].to_numpy()))
    )
    if shared_predictors:
        raise AssertionError(f"Grouped split predictor overlap: {shared_predictors}")
    counts = lambda frame: {int(k): int(v) for k, v in frame[TARGET].value_counts().sort_index().items()}
    config = {
        "created_utc": now_utc(), "seed": SEED, "target": TARGET, "features": FEATURES,
        "source_rows": len(data) + duplicates, "null_rows_removed_before_split": null_rows,
        "exact_duplicates_removed_before_split": duplicates,
        "null_policy": "Remove rows with nulls before split; source currently has none.",
        "outlier_policy": "Train-fitted 1.5-IQR winsorization for continuous columns; bounds reused for validation/test.",
        "encoding": "OrdinalEncoder fitted on train; unknown categories map to -1.",
        "scaling": "StandardScaler fitted on training data for linear models and reducers; XGBoost uses unscaled data.",
        "reduction_settings": "RFE uses RandomForest and train-only 3-fold AP selection over 4/6/8 features; LDA uses one component; FA uses min(4, input_features) components.",
        "stratified_split": {"type": "80/20 stratified split after exact deduplication", "train_rows": len(strat_train), "test_rows": len(strat_test), "train_counts": counts(strat_train), "test_counts": counts(strat_test), "test_ratio_percent": {str(k): round(v / len(strat_test) * 100, 4) for k, v in counts(strat_test).items()}},
        "grouped_split": {"type": "first shuffled StratifiedGroupKFold(5) fold grouped by identical predictors", "train_rows": len(group_train), "test_rows": len(group_test), "train_counts": counts(group_train), "test_counts": counts(group_test)},
        "paper_parameters": PAPER_PARAMS,
        "meta_model_deviation": "GradientBoostingClassifier(loss='log_loss') used as a documented LogitBoost-style surrogate, not canonical LogitBoost.",
        "python_version": platform.python_version(),
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "smote-variants", "matplotlib", "shap")},
        "specialized_sampler_status": {},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2, default=int) + "\n")
    return config


def preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categories = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    numeric = Pipeline(numeric_steps)
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categories, CATEGORICAL)], verbose_feature_names_out=False)


def select_rfe_count(x: np.ndarray, y: pd.Series) -> int:
    candidates = [n for n in (4, 6, 8) if n <= x.shape[1]]
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for count in candidates:
        fold_scores = []
        for fit_idx, valid_idx in folds.split(x, y):
            selector = RFE(
                RandomForestClassifier(n_estimators=80, max_depth=8, random_state=SEED, n_jobs=4),
                n_features_to_select=count,
            ).fit(x[fit_idx], y.iloc[fit_idx])
            x_fit, x_valid = selector.transform(x[fit_idx]), selector.transform(x[valid_idx])
            classifier = SGDClassifier(loss="log_loss", alpha=0.001, random_state=SEED).fit(x_fit, y.iloc[fit_idx])
            fold_scores.append(average_precision_score(y.iloc[valid_idx], model_scores(classifier, x_valid)))
        scores.append((float(np.mean(fold_scores)), count))
    return max(scores)[1]


def transform_pair(train: pd.DataFrame, test: pd.DataFrame, reducer_name: str, scale: bool = True, rfe_count: int | None = None) -> tuple[np.ndarray, np.ndarray, int | None]:
    winsor = TrainWinsorizer().fit(train[FEATURES])
    pre = preprocessor(scale)
    x_train = pre.fit_transform(winsor.transform(train[FEATURES]), train[TARGET])
    x_test = pre.transform(winsor.transform(test[FEATURES]))
    reducer: Any = None
    if reducer_name == "RFE":
        rfe_count = rfe_count or select_rfe_count(x_train, train[TARGET])
        reducer = RFE(RandomForestClassifier(n_estimators=100, max_depth=8, random_state=SEED, n_jobs=4), n_features_to_select=rfe_count).fit(x_train, train[TARGET])
    elif reducer_name == "LDA":
        reducer = LinearDiscriminantAnalysis(n_components=1).fit(x_train, train[TARGET])
    elif reducer_name == "FA":
        reducer = FactorAnalysis(n_components=min(4, x_train.shape[1]), random_state=SEED).fit(x_train)
    elif reducer_name != "none":
        raise ValueError(f"Unknown reducer: {reducer_name}")
    if reducer is not None:
        x_train, x_test = reducer.transform(x_train), reducer.transform(x_test)
    return x_train, x_test, rfe_count


def make_model(name: str, overrides: dict[str, Any] | None = None) -> Any:
    params = PAPER_PARAMS[name].copy()
    params.update(overrides or {})
    if name == "PAC": return PassiveAggressiveClassifier(**params)
    if name == "Ridge": return RidgeClassifier(**params)
    if name == "SGD": return SGDClassifier(**params)
    if name == "XGBoost": return XGBClassifier(**params)
    if name == "LogitBoost_surrogate": return GradientBoostingClassifier(loss="log_loss", **params)
    raise ValueError(name)


def make_sampler(name: str) -> Any:
    if name == "ROS": return RandomOverSampler(random_state=SEED)
    if name == "ADASYN": return ADASYN(random_state=SEED, n_neighbors=5)
    if name == "SMOTE": return SMOTE(random_state=SEED, k_neighbors=5)
    if name == "Borderline-SMOTE": return BorderlineSMOTE(random_state=SEED, k_neighbors=5)
    try:
        sv = importlib.import_module("smote_variants")
    except ImportError as error:
        raise RuntimeError("smote-variants missing; no substitute was applied") from error
    options = {"ProWRAS": ("ProWRAS",), "LoRAS": ("LoRAS",), "MWMOTE": ("MWMOTE",), "RWOS": ("RWO_sampling", "RWOS", "RandomWalkOversampling")}
    for candidate in options[name]:
        cls = getattr(sv, candidate, None)
        if cls:
            return cls(random_state=SEED)
    raise RuntimeError(f"Sampler {name} not provided by installed smote-variants")


def model_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x))[:, 1]
    return np.asarray(model.decision_function(x)).reshape(-1)


def metrics(y: pd.Series, predicted: np.ndarray, scores: np.ndarray, fit_s: float, predict_s: float) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, predicted), "precision": precision_score(y, predicted, zero_division=0), "recall": recall_score(y, predicted, zero_division=0), "f1": f1_score(y, predicted, zero_division=0), "specificity": tn / (tn + fp), "roc_auc": roc_auc_score(y, scores), "pr_auc": average_precision_score(y, scores), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "train_seconds": fit_s, "predict_seconds": predict_s}


def fit_parsel(train: pd.DataFrame, test: pd.DataFrame, reducer_name: str, sampler_name: str, tune: bool = False) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any]]:
    y = train[TARGET]
    base_names = ["PAC", "Ridge", "SGD", "XGBoost"]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), len(base_names)))
    fit_time = 0.0
    for fit_idx, valid_idx in folds.split(train[FEATURES], y):
        fold_start = time.perf_counter()
        fold_train = train.iloc[fit_idx]
        fold_valid = train.iloc[valid_idx]
        x_fold, x_valid, _ = transform_pair(fold_train, fold_valid, reducer_name, scale=True)
        if reducer_name == "none":
            x_fold_xgb, x_valid_xgb, _ = transform_pair(fold_train, fold_valid, reducer_name, scale=False)
        else:
            x_fold_xgb, x_valid_xgb = x_fold, x_valid
        x_fold, y_fold = make_sampler(sampler_name).fit_resample(x_fold, fold_train[TARGET])
        if reducer_name == "none":
            x_fold_xgb, y_fold_xgb = make_sampler(sampler_name).fit_resample(x_fold_xgb, fold_train[TARGET])
        else:
            x_fold_xgb, y_fold_xgb = x_fold, y_fold
        for col, name in enumerate(base_names):
            model = make_model(name)
            model.fit(x_fold_xgb, y_fold_xgb) if name == "XGBoost" else model.fit(x_fold, y_fold)
            if name == "XGBoost":
                oof[valid_idx, col] = model_scores(model, x_valid_xgb)
            else:
                oof[valid_idx, col] = model_scores(model, x_valid)
        fit_time += time.perf_counter() - fold_start
    meta = make_model("LogitBoost_surrogate")
    best_params: dict[str, Any] = {}
    meta_start = time.perf_counter()
    if tune:
        search = RandomizedSearchCV(GradientBoostingClassifier(loss="log_loss", random_state=SEED), {
            "learning_rate": [0.05, 0.1, 0.132, 0.2], "n_estimators": [100, 200, 300, 400],
            "max_depth": [1, 2, 3, 5], "max_leaf_nodes": [7, 15, 31, 63, 127], "min_samples_leaf": [2, 4, 8],
        }, n_iter=12, scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=SEED), random_state=SEED, n_jobs=1)
        search.fit(oof, y); meta = search.best_estimator_; best_params = search.best_params_
    else:
        meta.fit(oof, y)
    fit_time += time.perf_counter() - meta_start
    final_start = time.perf_counter()
    x_train, x_test, _ = transform_pair(train, test, reducer_name, scale=True)
    x_balanced, y_balanced = make_sampler(sampler_name).fit_resample(x_train, y)
    if reducer_name == "none":
        x_xgb_train, x_xgb_test, _ = transform_pair(train, test, reducer_name, scale=False)
        x_xgb_balanced, _ = make_sampler(sampler_name).fit_resample(x_xgb_train, y)
    else:
        x_xgb_train, x_xgb_test, x_xgb_balanced = x_train, x_test, x_balanced
    final_models = []
    for name in base_names:
        model = make_model(name)
        model.fit(x_xgb_balanced if name == "XGBoost" else x_balanced, y_balanced)
        final_models.append(model)
    fit_time += time.perf_counter() - final_start
    meta_test = np.column_stack([model_scores(model, x_xgb_test if name == "XGBoost" else x_test) for name, model in zip(base_names, final_models)])
    start = time.perf_counter()
    predicted = meta.predict(meta_test)
    scores = model_scores(meta, meta_test)
    predict_time = time.perf_counter() - start
    return predicted, scores, fit_time, predict_time, best_params


def append_record(record: dict[str, Any]) -> None:
    path = RESULTS / "all_runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id", "timestamp_utc", "split", "sampler", "reducer", "model", "seed",
        "train_rows", "test_rows", "status", "error", "rfe_features", "tuned_parameters",
        "accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc",
        "tp", "fp", "tn", "fn", "train_seconds", "predict_seconds",
    ]
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if path.stat().st_size == 0: writer.writeheader()
        writer.writerow(record)


def run_combination(train: pd.DataFrame, test: pd.DataFrame, split: str, sampler: str, reducer: str, models: list[str], tune: bool = False) -> None:
    base = {"timestamp_utc": now_utc(), "split": split, "sampler": sampler, "reducer": reducer, "seed": SEED, "train_rows": len(train), "test_rows": len(test)}
    try:
        make_sampler(sampler)
    except Exception as error:  # noqa: BLE001
        for name in models:
            append_record({
                **base,
                "run_id": uuid.uuid4().hex[:12],
                "model": name,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            })
        return
    try:
        for name in models:
            run_id = uuid.uuid4().hex[:12]
            record = {**base, "run_id": run_id, "model": name, "status": "ok", "error": "", "rfe_features": "", "tuned_parameters": ""}
            try:
                if name == "PaRSEL":
                    predicted, scores, fit_s, pred_s, params = fit_parsel(train, test, reducer, sampler, tune)
                else:
                    scale = name != "XGBoost" or reducer != "none"
                    x_train, x_test, rfe_count = transform_pair(train, test, reducer, scale=scale)
                    record["rfe_features"] = rfe_count or ""
                    start = time.perf_counter()
                    x_bal, y_bal = make_sampler(sampler).fit_resample(x_train, train[TARGET])
                    model = make_model(name); model.fit(x_bal, y_bal); fit_s = time.perf_counter() - start
                    start = time.perf_counter(); scores = model_scores(model, x_test); predicted = model.predict(x_test); pred_s = time.perf_counter() - start; params = {}
                record.update(metrics(test[TARGET], predicted, scores, fit_s, pred_s)); record["tuned_parameters"] = json.dumps(params, sort_keys=True)
                if name == "PaRSEL" and split == "stratified" and sampler == "ROS" and reducer == "none":
                    curve = pd.DataFrame({"target": test[TARGET].to_numpy(), "score": scores, "prediction": predicted})
                    curve.to_csv(RESULTS / f"curve_{run_id}.csv", index=False)
            except Exception as error:  # noqa: BLE001
                record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            append_record(record)
    except Exception as error:  # noqa: BLE001
        append_record({**base, "run_id": uuid.uuid4().hex[:12], "model": "ALL", "status": "failed", "error": f"{type(error).__name__}: {error}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--priority-only", action="store_true")
    parser.add_argument("--include-tuned", action="store_true")
    parser.add_argument("--split", choices=("stratified", "grouped", "both"), default="stratified")
    args = parser.parse_args()
    config = prepare_splits(args.force_splits)
    specialized = {}
    for name in ("ProWRAS", "LoRAS", "MWMOTE", "RWOS"):
        try: make_sampler(name); specialized[name] = "available"
        except Exception as error:  # noqa: BLE001
            specialized[name] = f"unavailable: {error}"
    config["specialized_sampler_status"] = specialized
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2, default=int) + "\n")
    print(json.dumps({"stratified": config["stratified_split"], "grouped": config["grouped_split"], "specialized_samplers": specialized}, indent=2))
    if args.prepare_only: return
    models = ["PAC", "Ridge", "SGD", "XGBoost", "LogitBoost_surrogate", "PaRSEL"]
    combos = [("ROS", "none"), ("ProWRAS", "LDA"), ("ProWRAS", "RFE")] if args.priority_only else [(s, r) for s in ("ROS", "ProWRAS", "LoRAS", "ADASYN", "SMOTE", "Borderline-SMOTE", "MWMOTE", "RWOS") for r in ("none", "RFE", "LDA", "FA")]
    splits = ("stratified", "grouped") if args.split == "both" else (args.split,)
    for split in splits:
        train = pd.read_csv(SPLIT_PATHS[f"{split}_train"]); test = pd.read_csv(SPLIT_PATHS[f"{split}_test"])
        for sampler, reducer in combos:
            run_combination(train, test, split, sampler, reducer, models, tune=args.include_tuned and sampler == "ROS" and reducer == "none")


if __name__ == "__main__":
    main()