# Paper vs. Reproduction Results

Primary comparison is fixed in advance as the stratified seed-42 split, ROS, no dimensionality reduction, PaRSEL. It is not selected by test performance. Values in the paper column are rounded claims from its discussion; several result tables are embedded as raster figures, so cell-level values cannot be reliably extracted from the supplied PDF.

Current status: five standalone ROS/no-reduction models completed before the PaRSEL run was cancelled. PaRSEL, tuning, the grouped model run, the remaining sampler/reducer grid, PaRSEL ROC/PR curves, and SHAP analysis are not complete. These results are partial and are not a completed paper reproduction.

## Headline Claims

| Metric | Paper | Ours (primary ROS/no reduction) | Delta | Verdict |
|---|---:|---:|---:|---|
| Accuracy | about 97% | not run | n/a | not run |
| F1 | about 80% | not run | n/a | not run |
| Precision | >90% (often about 99%) | not run | n/a | not run |
| Recall | 67-70% (one passage says 77% for SMOTE) | not run | n/a | not run |
| ROC-AUC | about 98% | not run | n/a | not run |

## Completed Standalone Reference Runs

These are individual classifiers on the same fixed ROS/no-reduction training and test split. They are not the PaRSEL stack and are included only as partial progress.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Fit seconds | Predict seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PAC | 0.8728 | 0.4005 | 0.8903 | 0.5525 | 0.9581 | 0.8016 | 0.50 | 0.005 |
| Ridge | 0.8790 | 0.4127 | 0.8791 | 0.5617 | 0.9574 | 0.7984 | 1.21 | 0.001 |
| SGD | 0.7506 | 0.2526 | 0.9334 | 0.3976 | 0.9372 | 0.7024 | 0.30 | 0.002 |
| XGBoost | 0.8951 | 0.4533 | 0.9204 | 0.6074 | 0.9763 | 0.8798 | 0.47 | 0.028 |
| LogitBoost_surrogate | 0.9236 | 0.5421 | 0.8614 | 0.6655 | 0.9748 | 0.8733 | 106.20 | 0.370 |

## Balancing Comparisons

The paper's discussion gives rounded PaRSEL results after reduction for these samplers. The supplied PDF does not expose the per-classifier table cells as text. Our rows are reported separately for every reducer to avoid choosing the best-looking test configuration.

| Balancer | Paper accuracy / F1 / precision / recall | Reducer | Ours accuracy / F1 / precision / recall | Status |
|---|---|---|---|---|
| ProWRAS | 98% / 95% / 94% / 97% | n/a | not run | not run |
| LoRAS | 97% / 80% / 99% / 67% | n/a | not run | not run |
| ROS | 97% / 80% / 99% / 67% | n/a | not run | not run |
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

- The raw diabetes file contains 3,854 exact duplicate rows. Exact duplicates are removed before the fixed split to enforce zero exact-row overlap. This differs from the paper, which does not document duplicate handling, and changes the class ratio slightly.
- The stratified holdout remains imbalanced; accuracy can look strong while missing minority cases. Recall, precision, PR-AUC, specificity, and the confusion matrix must be read together.
- Predictor combinations can repeat with conflicting labels. The grouped split keeps identical predictor tuples together and is reported separately as a stricter robustness test.
- The paper says split before balancing, but does not specify seed or split ratio. Seed 42 and 80/20 are reproducibility choices, not verified author settings.
- The paper reports recall as both 67% and 70%; its Table 4/tuned result is described as roughly 98% accuracy and 97% recall, while other reported recall values are lower. These claims are internally inconsistent.
- The paper claims reduced execution time but does not provide a sufficiently detailed, hardware-matched timing protocol. Our times are local and include model fitting and sampling, but are not directly comparable across machines.
- `GradientBoostingClassifier(loss='log_loss')` is a documented LogitBoost-style substitute, not a canonical LogitBoost implementation. It is not labeled as exact reproduction.
- `smote-variants==1.0.1` installation was cancelled. ProWRAS, LoRAS, MWMOTE, and RWOS are therefore recorded as unavailable, not approximated. They must be run after that dependency is installed and the matching implementation names are verified.
- Outliers are capped at training-fitted 1.5-IQR bounds because the paper's outlier-removal rule is unspecified. Test rows are not dropped.
- SHAP summary, waterfall, and dependence plots are deferred until the PaRSEL and comparison runs finish; no explanation is attributed to an uncompleted stack.

## Artifacts

- `config.json`: seeds, split counts, preprocessing decisions, hyperparameters, and library versions.
- `all_runs.csv`: append-only metrics, timings, statuses, and errors.
- `figure_accuracy.png`, `figure_precision.png`, `figure_recall.png`, `figure_f1.png`: ROS metric comparisons.
- `roc_curve.png`, `pr_curve.png`: primary PaRSEL ROS/no-reduction curves, when that run completed.
