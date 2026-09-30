"""Run PaRSEL+ E1/E2 extensions without fitting on the held-out test data."""

from __future__ import annotations

import argparse
import csv
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from scripts.run_experiments import (
    FEATURES,
    PAPER_PATHS,
    RESULTS,
    SEED,
    TARGET,
    experiment_timeout,
    fit_parsel,
    group_ids,
    make_model,
    make_sampler,
    metrics,
    model_scores,
    now_utc,
    transform_pair,
)

EXTENSION_RESULTS = RESULTS / "extension_results.csv"
ALT_MODELS = ["PaRSEL", "PAC", "Ridge", "SGD", "XGBoost", "LogisticRegression"]


def extension_completed_run_keys() -> set[str]:
    if not EXTENSION_RESULTS.exists():
        return set()
    with EXTENSION_RESULTS.open(newline="", encoding="utf-8") as stream:
        return {row["run_key"] for row in csv.DictReader(stream) if row.get("run_key")}


def _append_extension(record: dict[str, Any]) -> None:
    fields = [
        "run_id", "run_key", "timestamp_utc", "extension", "record_type", "seed", "fold",
        "dataset_variant", "model", "model_variant", "sampler", "reducer", "threshold",
        "accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy",
        "roc_auc", "pr_auc", "brier_score", "mcnemar_p", "delong_p", "mean", "std",
        "ci_low", "ci_high", "metric_name", "tp", "fp", "tn", "fn", "train_seconds", "predict_seconds", "status", "error",
    ]
    EXTENSION_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    exists = EXTENSION_RESULTS.exists() and EXTENSION_RESULTS.stat().st_size > 0
    with EXTENSION_RESULTS.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def _default_record(extension: str, run_key: str, **fields: Any) -> dict[str, Any]:
    return {
        "run_id": uuid.uuid4().hex[:12],
        "run_key": run_key,
        "timestamp_utc": now_utc(),
        "extension": extension,
        "record_type": "run",
        "seed": SEED,
        "fold": "",
        "dataset_variant": "paper_faithful",
        "model": "",
        "model_variant": "",
        "sampler": "",
        "reducer": "none",
        "threshold": 0.5,
        "status": "ok",
        "error": "",
        **fields,
    }


def _make_alternative(name: str) -> Any:
    if name == "LogisticRegression":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    return make_model(name)


def _delong_pvalue(y_true: np.ndarray, first_scores: np.ndarray, second_scores: np.ndarray) -> float:
    """Two-sided paired DeLong test for correlated ROC-AUCs."""

    def midrank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranks[start:stop] = 0.5 * (start + stop - 1) + 1
            start = stop
        result = np.empty(len(values), dtype=float)
        result[order] = ranks
        return result

    positives = y_true == 1
    m = int(positives.sum())
    n = int(len(y_true) - m)
    if m < 2 or n < 2:
        return float("nan")
    predictions = np.vstack([first_scores, second_scores])
    v01 = []
    v10 = []
    aucs = []
    for scores in predictions:
        positive_scores = scores[positives]
        negative_scores = scores[~positives]
        all_scores = np.concatenate([positive_scores, negative_scores])
        rank_pos = midrank(positive_scores)
        rank_neg = midrank(negative_scores)
        rank_all = midrank(all_scores)
        auc = (rank_all[:m].sum() / m - (m + 1) / 2) / n
        aucs.append(auc)
        v01.append((rank_all[:m] - rank_pos) / n)
        v10.append(1 - (rank_all[m:] - rank_neg) / m)
    covariance = np.cov(np.asarray(v01)) / m + np.cov(np.asarray(v10)) / n
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    if variance <= 0:
        return 1.0 if np.isclose(aucs[0], aucs[1]) else 0.0
    z_score = (aucs[0] - aucs[1]) / np.sqrt(variance)
    return float(2 * norm.sf(abs(z_score)))


def _mcnemar_pvalue(y_true: np.ndarray, baseline: np.ndarray, alternative: np.ndarray) -> float:
    baseline_correct = baseline == y_true
    alternative_correct = alternative == y_true
    baseline_only = int(np.sum(baseline_correct & ~alternative_correct))
    alternative_only = int(np.sum(~baseline_correct & alternative_correct))
    discordant = baseline_only + alternative_only
    return 1.0 if discordant == 0 else float(binomtest(min(baseline_only, alternative_only), discordant, 0.5).pvalue)


def _ci(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    if len(array) < 2:
        return mean, std, mean, mean
    half = float(student_t.ppf(0.975, len(array) - 1) * std / np.sqrt(len(array)))
    return mean, std, mean - half, mean + half


def run_e1() -> None:
    train = pd.read_csv(PAPER_PATHS["train"])
    test = pd.read_csv(PAPER_PATHS["test"])
    y_test = test[TARGET].to_numpy()
    baseline_path = None
    runs = pd.read_csv(RESULTS / "all_runs.csv")
    baseline_row = runs.loc[
        (runs.run_key == f"phase1:paper_faithful:ROS:none:PaRSEL:{SEED}")
        & (runs.status == "ok")
    ]
    if baseline_row.empty:
        raise RuntimeError("E1 requires the completed Phase 1 untuned PaRSEL baseline")
    baseline_run_id = str(baseline_row.iloc[-1].run_id)
    baseline_path = RESULTS / f"curve_{baseline_run_id}.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline held-out predictions: {baseline_path}")
    baseline_curve = pd.read_csv(baseline_path)
    baseline_predictions = baseline_curve.prediction.to_numpy()
    baseline_scores = baseline_curve.score.to_numpy()

    finished = extension_completed_run_keys()
    baseline_fit_seconds = float(baseline_row.iloc[-1].train_seconds)
    stack_fits = 3 * 5
    standalone_fits = 5 * 5 * 5
    estimated_seconds = baseline_fit_seconds * stack_fits + standalone_fits * 3.0
    plan_key = "E1:runtime_estimate:v1"
    if plan_key not in finished:
        _append_extension(_default_record(
            "E1", plan_key, record_type="runtime_plan", model="PaRSEL_vs_standalones",
            model_variant="3 seeds x 5 folds for PaRSEL; 5 seeds x 5 folds for standalone learners",
            mean=estimated_seconds, metric_name="estimated_total_seconds",
            status="planned",
        ))
        finished.add(plan_key)
    for seed in range(5):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (fit_index, validation_index) in enumerate(splitter.split(train[FEATURES], train[TARGET])):
            fold_train = train.iloc[fit_index]
            fold_validation = train.iloc[validation_index]
            for model_name in ALT_MODELS:
                if model_name == "PaRSEL" and seed >= 3:
                    continue
                run_key = f"E1:cv:{model_name}:seed{seed}:fold{fold}"
                if run_key in finished:
                    continue
                record = _default_record(
                    "E1", run_key, record_type="cv_fold", seed=seed, fold=fold,
                    model=model_name, model_variant="paper_parameters", sampler="ROS",
                )
                try:
                    with experiment_timeout():
                        if model_name == "PaRSEL":
                            predictions, scores, fit_seconds, predict_seconds, _, _ = fit_parsel(
                                fold_train, fold_validation, "none", "ROS", False
                            )
                        else:
                            scale = model_name != "XGBoost"
                            x_fit, x_valid, _, _ = transform_pair(
                                fold_train, fold_validation, "none", scale=scale
                            )
                            x_balanced, y_balanced = make_sampler("ROS").fit_resample(x_fit, fold_train[TARGET])
                            model = _make_alternative(model_name)
                            started = time.perf_counter()
                            model.fit(x_balanced, y_balanced)
                            fit_seconds = time.perf_counter() - started
                            started = time.perf_counter()
                            scores = model_scores(model, x_valid)
                            predictions = model.predict(x_valid)
                            predict_seconds = time.perf_counter() - started
                    values = metrics(fold_validation[TARGET], predictions, scores, fit_seconds, predict_seconds)
                    record.update(values)
                    record["train_seconds"] = fit_seconds
                    record["predict_seconds"] = predict_seconds
                except Exception as error:  # noqa: BLE001
                    record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                _append_extension(record)
                finished.add(run_key)

    existing = pd.read_csv(EXTENSION_RESULTS) if EXTENSION_RESULTS.exists() else pd.DataFrame()
    summary_metrics = ["accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "roc_auc", "pr_auc"]
    for model_name in ALT_MODELS:
        for metric_name in summary_metrics:
            values = []
            for seed in range(5):
                if not existing.empty:
                    rows = existing.loc[
                        (existing.extension == "E1")
                        & (existing.record_type == "cv_fold")
                        & (existing.model == model_name)
                        & (existing.seed == seed)
                        & (existing.status == "ok")
                    ]
                    seed_values = rows[metric_name].dropna().astype(float).tolist()
                else:
                    seed_values = []
                if seed_values:
                    values.append(float(np.mean(seed_values)))
            if not values:
                continue
            summary_key = f"E1:cv_summary:{model_name}:{metric_name}"
            if summary_key in finished:
                continue
            mean, std, low, high = _ci(values)
            summary_record = _default_record(
                "E1", summary_key, record_type="cv_summary",
                model=model_name, model_variant="mean_of_fold_metric_by_seed", metric_name=metric_name,
                mean=mean, std=std, ci_low=low, ci_high=high, status="ok",
            )
            summary_record[metric_name] = mean
            _append_extension(summary_record)
            finished.add(summary_key)

    for model_name in ALT_MODELS:
        if model_name == "PaRSEL":
            continue
        run_key = f"E1:holdout:{model_name}"
        if run_key in finished:
            continue
        record = _default_record("E1", run_key, record_type="paired_test", model=model_name, model_variant="paper_parameters", sampler="ROS")
        try:
            with experiment_timeout():
                scale = model_name != "XGBoost"
                x_fit, x_test, _, _ = transform_pair(train, test, "none", scale=scale)
                x_balanced, y_balanced = make_sampler("ROS").fit_resample(x_fit, train[TARGET])
                model = _make_alternative(model_name)
                start = time.perf_counter(); model.fit(x_balanced, y_balanced); fit_seconds = time.perf_counter() - start
                start = time.perf_counter(); alt_scores = model_scores(model, x_test); alt_predictions = model.predict(x_test); predict_seconds = time.perf_counter() - start
            record.update(metrics(test[TARGET], alt_predictions, alt_scores, fit_seconds, predict_seconds))
            record["mcnemar_p"] = _mcnemar_pvalue(y_test, baseline_predictions, alt_predictions)
            record["delong_p"] = _delong_pvalue(y_test, baseline_scores, alt_scores)
            record["baseline_roc_auc"] = float(roc_auc_score(y_test, baseline_scores))
            record["baseline_pr_auc"] = float(average_precision_score(y_test, baseline_scores))
            record["train_seconds"] = fit_seconds; record["predict_seconds"] = predict_seconds
        except Exception as error:  # noqa: BLE001
            record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        _append_extension(record)
        finished.add(run_key)


def _threshold_for_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[int(np.nanargmax(f1_values))])


def _threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, minimum_recall: float = 0.90) -> float:
    _, recall, thresholds = precision_recall_curve(y_true, scores)
    eligible = thresholds[recall[:-1] >= minimum_recall]
    return float(np.max(eligible)) if len(eligible) else float(np.min(scores))


def run_e2() -> None:
    train = pd.read_csv(PAPER_PATHS["train"])
    test = pd.read_csv(PAPER_PATHS["test"])
    validation_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    fit_indices, validation_indices = next(
        validation_splitter.split(train[FEATURES], train[TARGET], group_ids(train))
    )
    fit_train, validation = train.iloc[fit_indices], train.iloc[validation_indices]
    validation_key = "E2:validation_scores:PaRSEL:ROS:none"
    cached_path = RESULTS / "e2_validation_scores.csv"
    finished = extension_completed_run_keys()
    if validation_key not in finished:
        with experiment_timeout():
            _, validation_scores, _, _, _, _ = fit_parsel(fit_train, validation, "none", "ROS", False)
        pd.DataFrame({"target": validation[TARGET].to_numpy(), "score": validation_scores}).to_csv(cached_path, index=False)
        _append_extension(_default_record("E2", validation_key, record_type="validation_scores", model="PaRSEL", sampler="ROS", status="ok"))
        finished.add(validation_key)
    validation_scores = pd.read_csv(cached_path).score.to_numpy()
    validation_targets = pd.read_csv(cached_path).target.to_numpy()
    thresholds = {
        "max_f1": _threshold_for_f1(validation_targets, validation_scores),
        "recall_at_least_0.90": _threshold_for_recall(validation_targets, validation_scores),
    }
    test_key = "E2:final_test_scores:PaRSEL:ROS:none"
    cached_test = RESULTS / "e2_test_scores.csv"
    if test_key not in finished:
        with experiment_timeout():
            _, test_scores, _, _, _, _ = fit_parsel(train, test, "none", "ROS", False)
        pd.DataFrame({"target": test[TARGET].to_numpy(), "score": test_scores}).to_csv(cached_test, index=False)
        _append_extension(_default_record("E2", test_key, record_type="fixed_test_scores", model="PaRSEL", sampler="ROS", status="ok"))
        finished.add(test_key)
    test_frame = pd.read_csv(cached_test)
    for variant, threshold in thresholds.items():
        run_key = f"E2:threshold:{variant}:PaRSEL"
        if run_key in finished:
            continue
        scores = test_frame.score.to_numpy()
        predictions = (scores >= threshold).astype(int)
        record = _default_record("E2", run_key, model="PaRSEL", model_variant=variant, sampler="ROS", threshold=threshold)
        record.update(metrics(test_frame.target, predictions, scores, 0.0, 0.0))
        record["brier_score"] = float(np.mean((scores - test_frame.target.to_numpy()) ** 2))
        _append_extension(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", choices=("E1", "E2"))
    args = parser.parse_args()
    if args.extension == "E1":
        run_e1()
    else:
        run_e2()


if __name__ == "__main__":
    main()
