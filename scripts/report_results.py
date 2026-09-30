"""Generate paper comparison tables and charts from append-only experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER_VALUES = {
    "ProWRAS": (0.98, 0.95, 0.94, 0.97),
    "LoRAS": (0.97, 0.80, 0.99, 0.67),
    "ROS": (0.97, 0.80, 0.99, 0.67),
    "ADASYN": (0.97, 0.81, 0.99, 0.67),
    "SMOTE": (0.97, 0.80, 0.99, 0.77),
    "Borderline-SMOTE": (0.97, 0.80, 0.99, 0.68),
    "MWMOTE": (0.97, 0.80, 0.93, 0.68),
    "RWOS": (0.97, 0.81, 0.99, 0.67),
}


def verdict(value: float | None, target: float, claim_type: str = "approx", tolerance: float = 0.015) -> str:
    if value is None or pd.isna(value):
        return "not run"
    if claim_type == "minimum":
        return "reproduced" if value >= target else "not reproduced"
    if claim_type == "range":
        return "reproduced" if 0.67 <= value <= 0.70 else "partially / not reproduced"
    return "reproduced" if abs(value - target) <= tolerance else "partially / not reproduced"


def metric_charts(runs: pd.DataFrame) -> None:
    successful_ros = runs.loc[(runs.status == "ok") & (runs.split == "stratified") & (runs.sampler == "ROS")]
    if successful_ros.empty:
        return
    for metric in ("accuracy", "precision", "recall", "f1"):
        pivot = successful_ros.pivot_table(index="model", columns="reducer", values=metric, aggfunc="mean")
        axis = pivot.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=0)
        axis.set_title(f"ROS: {metric.title()} by model and reduction")
        axis.set_ylabel(metric.title())
        axis.figure.tight_layout()
        axis.figure.savefig(RESULTS / f"figure_{metric}.png", dpi=160)
        plt.close(axis.figure)


def curve_charts(runs: pd.DataFrame) -> None:
    primary = runs.loc[
        (runs.status == "ok")
        & (runs.split == "stratified")
        & (runs.sampler == "ROS")
        & (runs.reducer == "none")
        & (runs.model == "PaRSEL")
    ].sort_values("timestamp_utc")
    for row in primary.itertuples():
        path = RESULTS / f"curve_{row.run_id}.csv"
        if not path.exists():
            continue
        curve = pd.read_csv(path)
        fpr, tpr, _ = roc_curve(curve.target, curve.score)
        precision, recall, _ = precision_recall_curve(curve.target, curve.score)
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.plot(fpr, tpr, label=f"ROC-AUC {roc_auc_score(curve.target, curve.score):.3f}")
        axis.plot([0, 1], [0, 1], "--", color="grey")
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title="PaRSEL ROC: ROS + no reduction")
        axis.legend()
        fig.tight_layout(); fig.savefig(RESULTS / "roc_curve.png", dpi=160); plt.close(fig)
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.plot(recall, precision, label=f"PR-AUC {average_precision_score(curve.target, curve.score):.3f}")
        axis.axhline(curve.target.mean(), linestyle="--", color="grey", label="test prevalence")
        axis.set(xlabel="Recall", ylabel="Precision", title="PaRSEL precision-recall: ROS + no reduction")
        axis.legend()
        fig.tight_layout(); fig.savefig(RESULTS / "pr_curve.png", dpi=160); plt.close(fig)
        break


def comparison_report(runs: pd.DataFrame) -> None:
    successful = runs.loc[runs.status == "ok"]
    primary = successful.loc[
        (successful.split == "stratified")
        & (successful.sampler == "ROS")
        & (successful.reducer == "none")
        & (successful.model == "PaRSEL")
    ]
    result = primary.iloc[-1] if not primary.empty else None
    paper_claims = [
        ("Accuracy", "about 97%", "accuracy", 0.97, "approx"),
        ("F1", "about 80%", "f1", 0.80, "approx"),
        ("Precision", ">90% (often about 99%)", "precision", 0.90, "minimum"),
        ("Recall", "67-70% (one passage says 77% for SMOTE)", "recall", 0.685, "range"),
        ("ROC-AUC", "about 98%", "roc_auc", 0.98, "approx"),
    ]
    lines = [
        "# Paper vs. Reproduction Results",
        "",
        "Primary comparison is fixed in advance as the stratified seed-42 split, ROS, no dimensionality reduction, PaRSEL. It is not selected by test performance. Values in the paper column are rounded claims from its discussion; several result tables are embedded as raster figures, so cell-level values cannot be reliably extracted from the supplied PDF.",
        "",
        "Current status: five standalone ROS/no-reduction models completed before the PaRSEL run was cancelled. PaRSEL, tuning, the grouped model run, the remaining sampler/reducer grid, PaRSEL ROC/PR curves, and SHAP analysis are not complete. These results are partial and are not a completed paper reproduction.",
        "",
        "## Headline Claims",
        "",
        "| Metric | Paper | Ours (primary ROS/no reduction) | Delta | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for label, paper, key, target, claim_type in paper_claims:
        value = float(result[key]) if result is not None and pd.notna(result[key]) else None
        ours = f"{value:.4f}" if value is not None else "not run"
        delta = f"{value - target:+.4f}" if value is not None else "n/a"
        lines.append(f"| {label} | {paper} | {ours} | {delta} | {verdict(value, target, claim_type)} |")
    lines += [
        "",
        "## Completed Standalone Reference Runs",
        "",
        "These are individual classifiers on the same fixed ROS/no-reduction training and test split. They are not the PaRSEL stack and are included only as partial progress.",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Fit seconds | Predict seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    standalone = successful.loc[
        (successful.split == "stratified")
        & (successful.sampler == "ROS")
        & (successful.reducer == "none")
        & (successful.model != "PaRSEL")
    ]
    if standalone.empty:
        lines.append("| No completed standalone rows | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        for row in standalone.itertuples():
            lines.append(f"| {row.model} | {row.accuracy:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.roc_auc:.4f} | {row.pr_auc:.4f} | {row.train_seconds:.2f} | {row.predict_seconds:.3f} |")
    lines += [
        "",
        "## Balancing Comparisons",
        "",
        "The paper's discussion gives rounded PaRSEL results after reduction for these samplers. The supplied PDF does not expose the per-classifier table cells as text. Our rows are reported separately for every reducer to avoid choosing the best-looking test configuration.",
        "",
        "| Balancer | Paper accuracy / F1 / precision / recall | Reducer | Ours accuracy / F1 / precision / recall | Status |",
        "|---|---|---|---|---|",
    ]
    for sampler, (acc, f1, precision, recall) in PAPER_VALUES.items():
        ours_rows = successful.loc[
            (successful.split == "stratified")
            & (successful.sampler == sampler)
            & (successful.model == "PaRSEL")
        ]
        if ours_rows.empty:
            lines.append(f"| {sampler} | {acc:.0%} / {f1:.0%} / {precision:.0%} / {recall:.0%} | n/a | not run | not run |")
            continue
        for row in ours_rows.itertuples():
            ours_text = f"{row.accuracy:.1%} / {row.f1:.1%} / {row.precision:.1%} / {row.recall:.1%}"
            matches = abs(row.accuracy - acc) <= 0.015 and abs(row.f1 - f1) <= 0.02
            status = "partially reproduced" if matches else "not reproduced"
            lines.append(f"| {sampler} | {acc:.0%} / {f1:.0%} / {precision:.0%} / {recall:.0%} | {row.reducer} | {ours_text} | {status} |")
    lines += [
        "",
        "## Robustness Split",
        "",
    ]
    grouped = successful.loc[
        (successful.split == "grouped")
        & (successful.sampler == "ROS")
        & (successful.reducer == "none")
        & (successful.model == "PaRSEL")
    ]
    if grouped.empty or result is None:
        lines.append("Grouped-split PaRSEL result is not available yet. Run `make robustness` after the main run.")
    else:
        group_result = grouped.iloc[-1]
        lines.append(f"Grouped PaRSEL ROC-AUC: {group_result.roc_auc:.4f}; PR-AUC: {group_result.pr_auc:.4f}; F1: {group_result.f1:.4f}. Stratified row-split ROC-AUC: {result.roc_auc:.4f}; PR-AUC: {result.pr_auc:.4f}; F1: {result.f1:.4f}.")
    lines += [
        "",
        "## Tables 3-11",
        "",
        "The narrative allows approximate sampler-level PaRSEL comparisons above. The individual classifier cells, exact reduction-specific values, and some table entries are rasterized in the included PDF and were not machine-readable. They are not reconstructed or guessed here. The full local model-by-sampler-by-reducer metrics are in `all_runs.csv`.",
        "",
        "## Threats to Validity and Deviations",
        "",
        "- The raw diabetes file contains 3,854 exact duplicate rows. Exact duplicates are removed before the fixed split to enforce zero exact-row overlap. This differs from the paper, which does not document duplicate handling, and changes the class ratio slightly.",
        "- The stratified holdout remains imbalanced; accuracy can look strong while missing minority cases. Recall, precision, PR-AUC, specificity, and the confusion matrix must be read together.",
        "- Predictor combinations can repeat with conflicting labels. The grouped split keeps identical predictor tuples together and is reported separately as a stricter robustness test.",
        "- The paper says split before balancing, but does not specify seed or split ratio. Seed 42 and 80/20 are reproducibility choices, not verified author settings.",
        "- The paper reports recall as both 67% and 70%; its Table 4/tuned result is described as roughly 98% accuracy and 97% recall, while other reported recall values are lower. These claims are internally inconsistent.",
        "- The paper claims reduced execution time but does not provide a sufficiently detailed, hardware-matched timing protocol. Our times are local and include model fitting and sampling, but are not directly comparable across machines.",
        "- `GradientBoostingClassifier(loss='log_loss')` is a documented LogitBoost-style substitute, not a canonical LogitBoost implementation. It is not labeled as exact reproduction.",
        "- `smote-variants==1.0.1` installation was cancelled. ProWRAS, LoRAS, MWMOTE, and RWOS are therefore recorded as unavailable, not approximated. They must be run after that dependency is installed and the matching implementation names are verified.",
        "- Outliers are capped at training-fitted 1.5-IQR bounds because the paper's outlier-removal rule is unspecified. Test rows are not dropped.",
        "- SHAP summary, waterfall, and dependence plots are deferred until the PaRSEL and comparison runs finish; no explanation is attributed to an uncompleted stack.",
        "",
        "## Artifacts",
        "",
        "- `config.json`: seeds, split counts, preprocessing decisions, hyperparameters, and library versions.",
        "- `all_runs.csv`: append-only metrics, timings, statuses, and errors.",
        "- `figure_accuracy.png`, `figure_precision.png`, `figure_recall.png`, `figure_f1.png`: ROS metric comparisons.",
        "- `roc_curve.png`, `pr_curve.png`: primary PaRSEL ROS/no-reduction curves, when that run completed.",
        "",
    ]
    (RESULTS / "comparison_to_paper.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    path = RESULTS / "all_runs.csv"
    if not path.exists():
        RESULTS.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=["status", "split", "sampler", "reducer", "model"])
        comparison_report(empty)
        print("No experiment rows found; wrote report with not-run statuses.")
        return
    runs = pd.read_csv(path)
    metric_charts(runs)
    curve_charts(runs)
    comparison_report(runs)
    print(f"Wrote paper comparison and charts under {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()