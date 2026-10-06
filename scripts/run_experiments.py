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
from scripts.hard_timeout import ExperimentTimeout, failure_reason, run_with_hard_timeout
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from scipy.stats import randint
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
    balanced_accuracy_score,
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
PAPER_PATHS = {name: SPLITS / f"paper_faithful_{name}.csv" for name in ("train", "test")}
RFE_SELECTION_CACHE: dict[str, int] = {}

PAPER_PARAMS: dict[str, dict[str, Any]] = {
    "PAC": {"C": 1.6063676259174505e-05, "max_iter": 2000, "random_state": 30},
    "Ridge": {"alpha": 7.3, "solver": "saga", "random_state": SEED},
    "SGD": {"alpha": 0.000745, "loss": "perceptron", "penalty": "l2", "max_iter": 3000, "random_state": SEED},
    "XGBoost": {"learning_rate": 0.05820509320520235, "n_estimators": 57, "max_depth": 7, "subsample": 0.5343885211152184, "colsample_bytree": 0.730893825622149, "random_state": SEED, "n_jobs": 4, "tree_method": "hist", "eval_metric": "logloss"},
    "LogitBoost_surrogate": {"learning_rate": 0.1323705789444759, "n_estimators": 400, "max_leaf_nodes": 127, "min_samples_leaf": 4, "max_depth": 5, "random_state": SEED},
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
    raw = pd.read_csv(SOURCE)
    null_rows = int(raw.isna().any(axis=1).sum())
    paper_data = raw.dropna().reset_index(drop=True)
    data, duplicates, _ = load_source()
    if force or not all(path.exists() for path in (*SPLIT_PATHS.values(), *PAPER_PATHS.values())):
        paper_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        paper_train_idx, paper_test_idx = next(
            paper_splitter.split(paper_data[FEATURES], paper_data[TARGET], group_ids(paper_data))
        )
        paper_train = paper_data.iloc[paper_train_idx].copy()
        paper_test = paper_data.iloc[paper_test_idx].copy()
        train, test = train_test_split(data, test_size=0.2, random_state=SEED, stratify=data[TARGET])
        group_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        group_train_idx, group_test_idx = next(group_splitter.split(data[FEATURES], data[TARGET], group_ids(data)))
        grouped_train, grouped_test = data.iloc[group_train_idx], data.iloc[group_test_idx]
        SPLITS.mkdir(parents=True, exist_ok=True)
        paper_train.to_csv(PAPER_PATHS["train"], index=False)
        paper_test.to_csv(PAPER_PATHS["test"], index=False)
        train.to_csv(SPLIT_PATHS["stratified_train"], index=False)
        test.to_csv(SPLIT_PATHS["stratified_test"], index=False)
        grouped_train.to_csv(SPLIT_PATHS["grouped_train"], index=False)
        grouped_test.to_csv(SPLIT_PATHS["grouped_test"], index=False)
    paper_train, paper_test = (pd.read_csv(path) for path in PAPER_PATHS.values())
    strat_train, strat_test, group_train, group_test = (pd.read_csv(path) for path in SPLIT_PATHS.values())
    for train, test in ((paper_train, paper_test), (strat_train, strat_test), (group_train, group_test)):
        overlap = len(set(map(tuple, train.to_numpy())) & set(map(tuple, test.to_numpy())))
        if overlap:
            raise AssertionError(f"Exact train/test row overlap: {overlap}")
    shared_predictors = len(
        set(map(tuple, group_train[FEATURES].to_numpy()))
        & set(map(tuple, group_test[FEATURES].to_numpy()))
    )
    if shared_predictors:
        raise AssertionError(f"Grouped split predictor overlap: {shared_predictors}")
    paper_shared_predictors = len(
        set(map(tuple, paper_train[FEATURES].to_numpy()))
        & set(map(tuple, paper_test[FEATURES].to_numpy()))
    )
    if paper_shared_predictors:
        raise AssertionError(f"Paper-faithful split predictor overlap: {paper_shared_predictors}")
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
        "meta_model_deviation": "GradientBoostingClassifier(loss='log_loss') used as additive logistic boosting surrogate; not canonical LogitBoost.",
        "python_version": platform.python_version(),
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "smote-variants", "matplotlib", "shap")},
        "specialized_sampler_status": {
            name: "unavailable (smote-variants import failed: metric-learn missing)"
            for name in ("ProWRAS", "LoRAS", "MWMOTE", "RWOS")
        },
        "sampler_install_attempt": {
            "package": "smote-variants==1.0.1",
            "environment": ".venv/sampler-install",
            "wheel_install": "completed with --no-deps",
            "import_validation": "failed: metric_learn is absent; no retry or main-environment install performed",
            "available_methods": [],
        },
        "dataset_variants": {
            "paper_faithful": {
                "deduplicated": False,
                "split_rule": "first StratifiedGroupKFold(5) fold grouped by identical predictor tuples; all 100k source rows retained",
                "train_rows": len(paper_train), "test_rows": len(paper_test),
                "train_counts": counts(paper_train), "test_counts": counts(paper_test),
                "ros_train_counts": {str(label): max(counts(paper_train).values()) for label in (0, 1)},
                "ros_train_rows": 2 * max(counts(paper_train).values()),
                "test_prevalence": float(paper_test[TARGET].mean()),
                "files": {key: str(path.relative_to(ROOT)) for key, path in PAPER_PATHS.items()},
            },
            "deduplicated": {
                "deduplicated": True,
                "split_rule": "stratified 80/20 row split after exact duplicate removal",
                "train_rows": len(strat_train), "test_rows": len(strat_test),
                "train_counts": counts(strat_train), "test_counts": counts(strat_test),
                "test_prevalence": float(strat_test[TARGET].mean()),
                "files": {"train": str(SPLIT_PATHS["stratified_train"].relative_to(ROOT)), "test": str(SPLIT_PATHS["stratified_test"].relative_to(ROOT))},
            },
            "predictor_grouped": {
                "deduplicated": True,
                "split_rule": "first StratifiedGroupKFold(5) fold grouped by identical predictors",
                "train_rows": len(group_train), "test_rows": len(group_test),
                "train_counts": counts(group_train), "test_counts": counts(group_test),
                "test_prevalence": float(group_test[TARGET].mean()),
                "files": {"train": str(SPLIT_PATHS["grouped_train"].relative_to(ROOT)), "test": str(SPLIT_PATHS["grouped_test"].relative_to(ROOT))},
            },
        },
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


def transform_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    reducer_name: str,
    scale: bool = True,
    rfe_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int | None, dict[str, Any]]:
    winsor = TrainWinsorizer().fit(train[FEATURES])
    pre = preprocessor(scale)
    x_train = pre.fit_transform(winsor.transform(train[FEATURES]), train[TARGET])
    x_test = pre.transform(winsor.transform(test[FEATURES]))
    reducer: Any = None
    feature_names = list(pre.get_feature_names_out())
    if reducer_name == "RFE":
        rfe_count = rfe_count or select_rfe_count(x_train, train[TARGET])
        reducer = RFE(RandomForestClassifier(n_estimators=100, max_depth=8, random_state=SEED, n_jobs=4), n_features_to_select=rfe_count).fit(x_train, train[TARGET])
        selected_features = [name for name, selected in zip(feature_names, reducer.support_) if selected]
        metadata = {
            "reducer": "RFE",
            "n_features": int(rfe_count),
            "selected_features": selected_features,
            "estimator": "RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)",
            "scaling": scale,
        }
    elif reducer_name == "LDA":
        reducer = LinearDiscriminantAnalysis(n_components=1).fit(x_train, train[TARGET])
        metadata = {
            "reducer": "LDA",
            "component_count": 1,
            "input_shape": list(x_train.shape),
            "scaling": scale,
        }
    elif reducer_name == "FA":
        component_count = min(4, x_train.shape[1])
        reducer = FactorAnalysis(n_components=component_count, random_state=SEED).fit(x_train)
        metadata = {
            "reducer": "FA",
            "component_count": int(component_count),
            "input_shape": list(x_train.shape),
            "scaling": scale,
        }
    elif reducer_name != "none":
        raise ValueError(f"Unknown reducer: {reducer_name}")
    else:
        metadata = {"reducer": "none", "input_features": feature_names, "scaling": scale}
    if reducer is not None:
        x_train, x_test = reducer.transform(x_train), reducer.transform(x_test)
    metadata.update({
        "train_shape_after_reduction": list(x_train.shape),
        "test_shape_after_reduction": list(x_test.shape),
        "train_has_nan": bool(np.isnan(x_train).any()),
        "test_has_nan": bool(np.isnan(x_test).any()),
    })
    return x_train, x_test, rfe_count, metadata


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
    return {"accuracy": accuracy_score(y, predicted), "precision": precision_score(y, predicted, zero_division=0), "recall": recall_score(y, predicted, zero_division=0), "f1": f1_score(y, predicted, zero_division=0), "specificity": tn / (tn + fp), "balanced_accuracy": balanced_accuracy_score(y, predicted), "roc_auc": roc_auc_score(y, scores), "pr_auc": average_precision_score(y, scores), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "train_seconds": fit_s, "predict_seconds": predict_s}


def fit_parsel(
    train: pd.DataFrame,
    test: pd.DataFrame,
    reducer_name: str,
    sampler_name: str,
    tune: bool = False,
    rfe_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any], dict[str, Any]]:
    y = train[TARGET]
    base_names = ["PAC", "Ridge", "SGD", "XGBoost"]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), len(base_names)))
    fit_time = 0.0
    selected_rfe_count = rfe_count
    if reducer_name == "RFE" and selected_rfe_count is None:
        x_for_selection, _, _, _ = transform_pair(train, train.iloc[[0]], "none", scale=True)
        selected_rfe_count = select_rfe_count(x_for_selection, y)
    for fit_idx, valid_idx in folds.split(train[FEATURES], y):
        fold_start = time.perf_counter()
        fold_train = train.iloc[fit_idx]
        fold_valid = train.iloc[valid_idx]
        x_fold, x_valid, _, _ = transform_pair(
            fold_train, fold_valid, reducer_name, scale=True, rfe_count=selected_rfe_count
        )
        if reducer_name == "none":
            x_fold_xgb, x_valid_xgb, _, _ = transform_pair(fold_train, fold_valid, reducer_name, scale=False)
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
        meta_search_base = make_model("LogitBoost_surrogate")
        search = RandomizedSearchCV(meta_search_base, {
            "learning_rate": [0.1, 0.01, 0.001], "n_estimators": randint(50, 200), "max_depth": randint(3, 10),
        }, n_iter=3, scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=SEED), random_state=SEED, n_jobs=1)
        search.fit(oof, y); meta = search.best_estimator_; best_params = search.best_params_
    else:
        meta.fit(oof, y)
    fit_time += time.perf_counter() - meta_start
    final_start = time.perf_counter()
    x_train, x_test, _, reducer_metadata = transform_pair(
        train, test, reducer_name, scale=True, rfe_count=selected_rfe_count
    )
    x_balanced, y_balanced = make_sampler(sampler_name).fit_resample(x_train, y)
    if reducer_name == "none":
        x_xgb_train, x_xgb_test, _, _ = transform_pair(train, test, reducer_name, scale=False)
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
    return predicted, scores, fit_time, predict_time, best_params, reducer_metadata


def _fit_phase1_model(
    train: pd.DataFrame, test: pd.DataFrame, name: str, reducer: str,
    sampler: str, tuned: bool, selected_rfe_count: int | None,
):
    """Fit and score one Phase 1 configuration inside a killable child."""
    if name == "PaRSEL":
        predicted, scores, fit_s, pred_s, params, metadata = fit_parsel(
            train, test, reducer, sampler, tuned, selected_rfe_count
        )
        metadata["tuned_hyperparameters"] = params
        rfe_features = metadata.get("n_features", "")
    else:
        scale = name != "XGBoost" or reducer != "none"
        x_train, x_test, rfe_count, _ = transform_pair(
            train, test, reducer, scale=scale, rfe_count=selected_rfe_count
        )
        rfe_features = rfe_count or ""
        started = time.perf_counter()
        x_balanced, y_balanced = make_sampler(sampler).fit_resample(x_train, train[TARGET])
        model = make_model(name)
        model.fit(x_balanced, y_balanced)
        fit_s = time.perf_counter() - started
        started = time.perf_counter()
        scores = model_scores(model, x_test)
        predicted = model.predict(x_test)
        pred_s = time.perf_counter() - started
        params, metadata = {}, {}
    return predicted, scores, fit_s, pred_s, params, metadata, rfe_features


def append_record(record: dict[str, Any]) -> None:
    if record.get("model") == "PaRSEL" and not record.get("reducer_metadata"):
        raise ValueError("PaRSEL result records require reducer_metadata")
    path = RESULTS / "all_runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id", "run_key", "timestamp_utc", "phase", "dataset_variant", "split",
        "balancer", "sampler", "reducer", "model", "model_variant", "seed",
        "train_rows", "test_rows", "status", "error", "rfe_features", "tuned_parameters",
        "reducer_metadata",
        "decision_threshold", "accuracy", "precision", "recall", "f1", "specificity",
        "balanced_accuracy", "roc_auc", "pr_auc", "brier_score", "tp", "fp", "tn", "fn",
        "train_seconds", "predict_seconds",
    ]
    ensure_result_schema(path, fields)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(record)


def ensure_result_schema(path: Path, fields: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        old_fields = reader.fieldnames or []
        if old_fields == fields:
            return
        old_rows = list(reader)
    for row in old_rows:
        row.setdefault("phase", "historical_reference")
        row.setdefault("dataset_variant", "deduplicated")
        row.setdefault("balancer", row.get("sampler", "ROS"))
        row.setdefault("model_variant", "previous_runner")
        row.setdefault("run_key", "")
        row.setdefault("decision_threshold", 0.5)
        row.setdefault("balanced_accuracy", "")
        row.setdefault("brier_score", "")
        if row.get("model") == "PaRSEL":
            row.setdefault("reducer_metadata", json.dumps({"reducer": row.get("reducer", "unknown"), "legacy_metadata_unavailable": True}))
        else:
            row.setdefault("reducer_metadata", "")
    temporary = path.with_suffix(".csv.migrating")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(old_rows)
    temporary.replace(path)


def completed_run_keys() -> set[str]:
    path = RESULTS / "all_runs.csv"
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["run_key"] for row in csv.DictReader(stream) if row.get("run_key")}


def run_combination(train: pd.DataFrame, test: pd.DataFrame, split: str, dataset_variant: str, sampler: str, reducer: str, models: list[str]) -> None:
    base = {"timestamp_utc": now_utc(), "phase": "phase1_baseline", "dataset_variant": dataset_variant, "split": split, "balancer": sampler, "sampler": sampler, "reducer": reducer, "seed": SEED, "train_rows": len(train), "test_rows": len(test)}
    finished = completed_run_keys()
    try:
        make_sampler(sampler)
    except Exception as error:  # noqa: BLE001
        for name in models:
            model_variant = "tuned" if name == "PaRSEL_tuned" else "paper_parameters"
            rfe_version = ":rfe_fixedn_v3" if reducer == "RFE" else ""
            run_key = f"phase1:{dataset_variant}:{sampler}:{reducer}:{name}:{SEED}{rfe_version}"
            if run_key in finished:
                continue
            append_record({
                **base,
                "run_id": uuid.uuid4().hex[:12],
                "run_key": run_key,
                "model": name,
                "model_variant": model_variant,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "reducer_metadata": json.dumps({"reducer": reducer, "fit_status": "unavailable_sampler"}) if name.startswith("PaRSEL") else "",
            })
        return
    selected_rfe_count: int | None = None
    if reducer == "RFE":
        try:
            if dataset_variant in RFE_SELECTION_CACHE:
                selected_rfe_count = RFE_SELECTION_CACHE[dataset_variant]
            else:
                def _select_count():
                    x_for_selection, _, _, _ = transform_pair(train, train.iloc[[0]], "none", scale=True)
                    return select_rfe_count(x_for_selection, train[TARGET])
                selected_rfe_count = run_with_hard_timeout(_select_count, timeout_seconds=600)
                RFE_SELECTION_CACHE[dataset_variant] = selected_rfe_count
        except Exception as error:  # noqa: BLE001
            for name in models:
                is_parsel = name.startswith("PaRSEL")
                append_record({
                    **base,
                    "run_id": uuid.uuid4().hex[:12],
                    "run_key": f"phase1:{dataset_variant}:{sampler}:{reducer}:{name}:{SEED}:rfe_fixedn_v3",
                    "model": "PaRSEL" if name == "PaRSEL_tuned" else name,
                    "model_variant": "tuned" if name == "PaRSEL_tuned" else "paper_parameters",
                    "status": "failed",
                    "error": f"RFE count selection failed: {failure_reason(error)}",
                    "rfe_features": "",
                    "tuned_parameters": "",
                    "reducer_metadata": json.dumps({"reducer": "RFE", "fit_status": "selection_failed"}) if is_parsel else "",
                })
            return
    try:
        for name in models:
            rfe_version = ":rfe_fixedn_v3" if reducer == "RFE" else ""
            run_key = f"phase1:{dataset_variant}:{sampler}:{reducer}:{name}:{SEED}{rfe_version}"
            if run_key in finished:
                continue
            run_id = uuid.uuid4().hex[:12]
            tuned = name == "PaRSEL_tuned"
            model_name = "PaRSEL" if tuned else name
            planned_metadata = {"reducer": reducer, "fit_status": "not_completed"}
            record = {**base, "run_id": run_id, "run_key": run_key, "model": model_name, "model_variant": "tuned" if tuned else "paper_parameters", "status": "ok", "error": "", "rfe_features": "", "tuned_parameters": "", "reducer_metadata": json.dumps(planned_metadata) if model_name == "PaRSEL" else ""}
            try:
                predicted, scores, fit_s, pred_s, params, reducer_metadata, rfe_features = run_with_hard_timeout(
                    _fit_phase1_model, train, test, model_name, reducer, sampler, tuned,
                    selected_rfe_count, timeout_seconds=600,
                )
                if model_name == "PaRSEL":
                    record["reducer_metadata"] = json.dumps(reducer_metadata, sort_keys=True)
                record["rfe_features"] = rfe_features
                record.update(metrics(test[TARGET], predicted, scores, fit_s, pred_s)); record["tuned_parameters"] = json.dumps(params, sort_keys=True)
                if model_name == "PaRSEL" and not tuned:
                    curve = pd.DataFrame({"target": test[TARGET].to_numpy(), "score": scores, "prediction": predicted})
                    curve.to_csv(RESULTS / f"curve_{run_id}.csv", index=False)
            except ExperimentTimeout as error:
                record.update({"status": "failed", "error": f"timeout: {error}"})
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
    parser.add_argument("--split", choices=("paper_faithful", "deduplicated", "predictor_grouped", "both"), default="paper_faithful")
    parser.add_argument("--phase1-priority", action="store_true")
    parser.add_argument("--primary-only", action="store_true", help="Run only the Phase 1 PaRSEL ROS/no-reduction baseline")
    args = parser.parse_args()
    config = prepare_splits(args.force_splits)
    specialized = {}
    for name in ("ProWRAS", "LoRAS", "MWMOTE", "RWOS"):
        try:
            make_sampler(name)
            specialized[name] = "available"
        except Exception:  # noqa: BLE001
            specialized[name] = "unavailable (smote-variants import failed: metric-learn missing)"
    config["specialized_sampler_status"] = specialized
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2, default=int) + "\n")
    print(json.dumps({"stratified": config["stratified_split"], "grouped": config["grouped_split"], "specialized_samplers": specialized}, indent=2))
    if args.prepare_only: return
    if args.primary_only:
        combos = [("ROS", "none")]
        models_by_combo = {("ROS", "none"): ["PaRSEL"]}
    elif args.phase1_priority:
        combos = [
            ("ROS", "none"),
            ("ROS", "RFE"),
            ("ROS", "LDA"),
            ("SMOTE", "LDA"),
            ("SMOTE", "RFE"),
            ("ADASYN", "LDA"),
            ("ADASYN", "RFE"),
            ("Borderline-SMOTE", "LDA"),
            ("ROS", "FA"),
            ("SMOTE", "FA"),
            ("ProWRAS", "LDA"),
            ("ProWRAS", "RFE"),
            ("LoRAS", "LDA"),
            ("MWMOTE", "LDA"),
            ("RWOS", "LDA"),
        ]
        reference_models = ["PaRSEL", "PAC", "Ridge", "SGD", "XGBoost", "LogitBoost_surrogate"]
        if args.include_tuned:
            reference_models.append("PaRSEL_tuned")
        models_by_combo = {
            combo: reference_models if combo == ("ROS", "none") else ["PaRSEL", "PAC", "Ridge", "SGD", "XGBoost", "LogitBoost_surrogate"]
            for combo in combos
        }
    else:
        models_by_combo = {}
        combos = [(sampler, reducer) for sampler in ("ROS", "ProWRAS", "LoRAS", "ADASYN", "SMOTE", "Borderline-SMOTE", "MWMOTE", "RWOS") for reducer in ("none", "RFE", "LDA", "FA")]
    for combo in combos:
        models_by_combo.setdefault(combo, ["PAC", "Ridge", "SGD", "XGBoost", "LogitBoost_surrogate", "PaRSEL"])
    variants = [args.split] if args.split != "both" else ["paper_faithful", "deduplicated", "predictor_grouped"]
    variant_paths = {
        "paper_faithful": PAPER_PATHS,
        "deduplicated": {"train": SPLIT_PATHS["stratified_train"], "test": SPLIT_PATHS["stratified_test"]},
        "predictor_grouped": {"train": SPLIT_PATHS["grouped_train"], "test": SPLIT_PATHS["grouped_test"]},
    }
    for variant in variants:
        files = variant_paths[variant]
        train, test = pd.read_csv(files["train"]), pd.read_csv(files["test"])
        variant_combos = combos
        if args.phase1_priority and variant != "paper_faithful":
            variant_combos = [("ROS", "none"), ("ROS", "RFE"), ("ROS", "LDA")]
        for sampler, reducer in variant_combos:
            models = models_by_combo.get((sampler, reducer), ["PaRSEL", "PAC", "Ridge", "SGD", "XGBoost", "LogitBoost_surrogate"])
            run_combination(train, test, variant, variant, sampler, reducer, models)


if __name__ == "__main__":
    main()
