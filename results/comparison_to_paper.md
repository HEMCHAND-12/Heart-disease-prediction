# Paper vs. Reproduction Results

Primary comparison is fixed in advance as the no-dedup, seed-42, predictor-group-isolated split, ROS, no dimensionality reduction, PaRSEL with paper parameters. This retains all source rows while preventing identical predictor tuples from crossing train/test; the paper does not specify a grouped split, so this is a leakage-safe faithful-data variant, not an exact split reproduction. Values in the paper column are rounded claims from its discussion; several result tables are embedded as raster figures, so cell-level values cannot be reliably extracted from the supplied PDF.

Current status is calculated from append-only `all_runs.csv`. Historical standalone rows from the earlier runner are excluded from the paper-faithful PaRSEL verdict. Specialized samplers that fail to import are explicitly recorded as unavailable rather than replaced.

## Headline Claims

| Metric | Paper | Ours (primary ROS/no reduction) | Delta | Verdict |
|---|---:|---:|---:|---|
| Accuracy | about 97% | 0.9544 | -0.0156 | partially / not reproduced |
| F1 | about 80% | 0.7231 | -0.0769 | partially / not reproduced |
| Precision | >90% (often about 99%) | 0.7472 | -0.1528 | not reproduced |
| Recall | 67-70% (one passage says 77% for SMOTE) | 0.7006 | +0.0156 | partially / not reproduced |
| ROC-AUC | about 98% | 0.9607 | -0.0193 | partially / not reproduced |

## Tuned PaRSEL vs. Paper Table 4 Claim

Our tuned variant uses ROS with no reduction; the paper's tuned narrative is for ProWRAS+LDA. Values are shown for context, but this is not a configuration-matched comparison.

| Metric | Paper narrative | Our train-only tuned variant | Difference | Verdict |
|---|---:|---:|---:|---|
| Accuracy | 98% | 0.9636 | -0.0164 | not comparable |
| F1 | 95% | 0.7640 | -0.1860 | not comparable |
| Precision | 94% | 0.8495 | -0.0905 | not comparable |
| Recall | 97% | 0.6941 | -0.2759 | not comparable |

## Completed Standalone Reference Runs

These are individual classifiers on the same fixed ROS/no-reduction training and test split. They are not the PaRSEL stack and are included only as partial progress.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Fit seconds | Predict seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PAC | 0.8803 | 0.4060 | 0.8824 | 0.5561 | 0.9563 | 0.7885 | 0.60 | 0.004 |
| Ridge | 0.8840 | 0.4135 | 0.8735 | 0.5613 | 0.9564 | 0.7872 | 1.21 | 0.001 |
| SGD | 0.8522 | 0.3529 | 0.8871 | 0.5049 | 0.9415 | 0.6378 | 0.28 | 0.001 |
| XGBoost | 0.9075 | 0.4766 | 0.9041 | 0.6242 | 0.9743 | 0.8734 | 0.62 | 0.038 |
| LogitBoost_surrogate | 0.9309 | 0.5610 | 0.8576 | 0.6783 | 0.9746 | 0.8686 | 136.00 | 0.380 |

## Balancing Comparisons

The paper's discussion gives rounded PaRSEL results after reduction for these samplers. The supplied PDF does not expose the per-classifier table cells as text. Our rows are reported separately for every reducer to avoid choosing the best-looking test configuration.

| Balancer | Paper accuracy / F1 / precision / recall | Reducer | Ours accuracy / F1 / precision / recall | Status |
|---|---|---|---|---|
| ProWRAS | 98% / 95% / 94% / 97% | n/a | not run | not run |
| LoRAS | 97% / 80% / 99% / 67% | n/a | not run | not run |
| ROS | 97% / 80% / 99% / 67% | none | 95.4% / 72.3% / 74.7% / 70.1% | not reproduced |
| ROS | 97% / 80% / 99% / 67% | RFE | 95.4% / 72.3% / 74.7% / 70.1% | not reproduced |
| ROS | 97% / 80% / 99% / 67% | LDA | 90.2% / 49.2% / 43.8% / 56.0% | not reproduced |
| ADASYN | 97% / 81% / 99% / 67% | n/a | not run | not run |
| SMOTE | 97% / 80% / 99% / 77% | n/a | not run | not run |
| Borderline-SMOTE | 97% / 80% / 99% / 68% | n/a | not run | not run |
| MWMOTE | 97% / 80% / 93% / 68% | n/a | not run | not run |
| RWOS | 97% / 81% / 99% / 67% | n/a | not run | not run |

## Robustness Split

Grouped-split PaRSEL result is not available yet. Run `make robustness` after the main run.

## Tables 3-11

The narrative allows approximate sampler-level PaRSEL comparisons above. The individual classifier cells, exact reduction-specific values, and some table entries are rasterized in the included PDF and were not machine-readable. They are not reconstructed or guessed here. The full local model-by-sampler-by-reducer metrics are in `all_runs.csv`.

## Threats to Validity and Deviations

- The raw diabetes file contains 3,854 exact duplicate rows. The paper-faithful variant keeps all source rows but groups identical predictor tuples into one partition to prevent train/test duplication; deduplication is reported as a separate robustness variant.
- The holdouts remain about 91/9 imbalanced; accuracy can look strong while missing minority cases. Recall, precision, PR-AUC, specificity, balanced accuracy, and the confusion matrix must be read together.
- Predictor combinations can repeat with conflicting labels. Grouped variants keep identical predictor tuples together, so their metrics may be more conservative than an ordinary row-wise split.
- The paper says split before balancing, but does not specify seed or split ratio. Seed 42 and 80/20 are reproducibility choices, not verified author settings.
- The paper reports recall as both 67% and 70%; its Table 4/tuned result is described as roughly 98% accuracy and 97% recall, while other reported recall values are lower. These claims are internally inconsistent.
- The paper claims reduced execution time but does not provide a sufficiently detailed, hardware-matched timing protocol. Our times are local and include model fitting and sampling, but are not directly comparable across machines.
- `GradientBoostingClassifier(loss='log_loss')` is a documented LogitBoost-style substitute, not a canonical LogitBoost implementation. It is not labeled as exact reproduction.
- `smote-variants==1.0.1` installed as a no-deps wheel in an isolated venv, but import validation failed because `metric-learn` is absent. Per the one-attempt rule no retry or main-environment install was made; ProWRAS, LoRAS, MWMOTE, and RWOS are unavailable, not approximated.
- Outliers are capped at training-fitted 1.5-IQR bounds because the paper's outlier-removal rule is unspecified. Test rows are not dropped.
- SHAP summary, waterfall, and dependence plots are deferred until the PaRSEL and comparison runs finish; no explanation is attributed to an uncompleted stack.

## Artifacts

- `config.json`: seeds, split counts, preprocessing decisions, hyperparameters, and library versions.
- `all_runs.csv`: append-only metrics, timings, statuses, and errors.
- `figure_accuracy.png`, `figure_precision.png`, `figure_recall.png`, `figure_f1.png`: ROS metric comparisons.
- `roc_curve.png`, `pr_curve.png`: primary PaRSEL ROS/no-reduction curves, when that run completed.
