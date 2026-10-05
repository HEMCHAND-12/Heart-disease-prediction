# Reproduction Decisions

Updated: 2026-09-30

## Data and Split

- The primary target is the `diabetes` column in the 100,000-row diabetes source. `cardio_train.csv` is not merged or used in this experiment.
- Remove exact duplicate full rows once, before splitting. This is required because the current source has 3,854 exact duplicates and its existing row split puts 1,091 exact rows in both train and test. Keep the first copy in source order and log the number removed.
- Create one stratified 80/20 split with `random_state=42`; persist the raw train/test rows and reuse them across all experiments. The paper says split before balancing but does not publish a split ratio or seed, so this is a documented reproduction choice, not a claim that the split matches the authors'.
- Run a secondary `StratifiedGroupKFold`-based holdout grouped on the identical predictor tuple. This is the strict generalization check; preserve any conflicting-label groups as groups rather than splitting them apart.
- Do not overwrite the existing tracked ROS files. They remain audit artifacts because of their train/test overlap.

## Preprocessing and Model Inputs

- The raw source has no missing cells. Assert this; if missing cells appear in a later input, reject that input rather than silently dropping different rows from train and test.
- Fit category encoders on training rows only, with an explicit unknown-category value for test-time categories.
- The paper says outliers were removed but gives no feature bounds or rule. Use train-fitted 1.5-IQR winsorization for continuous columns (`age`, `bmi`, `HbA1c_level`, `blood_glucose_level`): learn bounds from train only, cap train and test with those same bounds, and report this as a deviation from unspecified paper preprocessing. Do not drop test rows.
- Standardize numeric inputs for PAC, Ridge, SGD, LDA, and Factor Analysis where scale matters. Keep XGBoost unscaled. Fit each scaler on the relevant training fold only.
- Fit RFE, LDA, and Factor Analysis on training data only; apply the fitted transform unchanged to validation/test rows. For RFE, use Random Forest and select feature count using training-only CV.
- Apply samplers only after the reducer and only to training data. In stacking OOF generation, resampling must happen separately inside each OOF training fold so copied/synthetic points do not enter the OOF validation fold.

## Classifiers and Search

- Use the user-specified Table 2 starting parameters for the paper-configured variant. Record any unsupported parameter combinations explicitly instead of silently substituting them.
- Use stratified 5-fold out-of-fold decision scores from PAC, Ridge, and SGD, and probabilities/scores from XGBoost, to train the stack's meta layer.
- The paper names LogitBoost but does not give a sufficiently reproducible implementation. Use scikit-learn `GradientBoostingClassifier(loss="log_loss")` with the supplied learning rate/tree settings as a documented LogitBoost-style surrogate if no maintained canonical implementation is available. Do not label this as exact LogitBoost.
- Run `RandomizedSearchCV` only within the fixed training set, with stratified folds and average precision as the search objective because the untouched evaluation set is strongly imbalanced. Store the search space, seed, CV settings, and chosen parameters. The reported-paper-parameter run remains separate from the tuned run.

## Reporting and Runtime

- Append experiment rows to `results/all_runs.csv`; assign each a unique run ID and UTC timestamp. Never overwrite earlier experiment results.
- Evaluate every run on the same untouched test split and report accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, confusion counts, fit time, and prediction time.
- Treat library/runtime limitations as deviations. Any deterministic subsampling must be training-only, seeded, and recorded with its original and sampled sizes.
- Compare measured results to paper claims without tuning against the test set. Explain the paper's internal metric inconsistencies and any failure to reproduce rather than adjusting the test protocol to match.
- Generate SHAP plots only after core experiments and comparisons are complete.

## Phase Framing Update (2026-09-30)

- Present the project as an extension of Noor et al., not a reproduction-only project: Phase 1 is the paper-method baseline; Phase 2 is the separately evaluated PaRSEL+ proposal.
- Keep a paper-faithful no-dedup dataset variant, but assign all identical predictor tuples to one partition. A plain row-stratified split would put duplicate records in both train and test, conflicting with the no-leakage requirement. This grouped-stratified split is the primary compromise: it retains all 100,000 rows and approximate 80/20 class-stratified proportions but is not the paper's undocumented split.
- Preserve the earlier 76,916/19,230 deduplicated stratified split and predictor-grouped split as separate robustness variants. Do not overwrite existing result rows; add the new schema fields by a one-time lossless CSV schema migration before appending further runs.
- The single permitted install attempt placed `smote-variants==1.0.1` in `.venv/sampler-install` with `--no-deps` and a 10-minute timeout. Wheel installation completed, but import failed because `metric-learn` is absent. No dependency install was attempted in the main environment and no retry will be made. Mark ProWRAS, LoRAS, MWMOTE, and RWO_sampling unavailable; the wheel remains isolated.
- The standalone ROS/no-reduction rows in `results/all_runs.csv` predate the Phase 1 exact-parameter runner. Preserve them as historical standalone reference rows, but do not treat them as PaRSEL or as a completed Phase 1 comparison.
- The Phase 1 runner uses a 600-second alarm per model/configuration; timeout and sampler/import failures are checkpointed and skipped on resume.
- The requested RandomizedSearchCV ranges are retained, but the search is time-boxed to three seeded parameter draws with 3-fold CV (rather than a broad search) to fit the 600-second per-run budget. Report this as a limited tuned variant, not exhaustive hyperparameter optimization.
- First successful Phase 1 primary run (paper-faithful, ROS, no reducer, untuned PaRSEL): accuracy 0.9544, precision 0.7472, recall 0.7006, F1 0.7231, specificity 0.97798, balanced accuracy 0.8393, ROC-AUC 0.9607, PR-AUC 0.6470. This does not reproduce the paper's approximate 97% accuracy / 80% F1 or 98% ROC-AUC claims; do not tune against the test set to close the gap.
- Direct ROS verification on paper-faithful training labels: pre-ROS counts {0: 73,199, 1: 6,800}; post-ROS counts {0: 73,199, 1: 73,199}; 146,398 rows. This is nine examples per class below the paper's 73,208 due to the group-safe split allocation.
- Train-only randomized-search Phase 1 variant completed after three candidate draws (3-fold CV; same untouched test): learning_rate=0.1, max_depth=7, n_estimators=152; accuracy 0.9636, precision 0.8495, recall 0.6941, F1 0.7640, ROC-AUC 0.9713, PR-AUC 0.8458. Its ROS/no-reducer configuration differs from the paper's ProWRAS+LDA Table 4 tuned narrative, so this comparison is not configuration-matched. Keep the untuned model as the primary baseline.

## LDA Diagnostic Hypothesis (before run)

Hypothesis: binary LDA is producing the expected one-dimensional, finite projection; the observed ROS+LDA loss is due to information compression and/or the fitted class prior, not a malformed input shape.
Discriminator: inspect train/test shape and NaNs, then compare the five standalone learners using LDA fitted on imbalanced training data versus LDA fitted on ROS-balanced training data, with the same untouched test rows.

LDA diagnostic outcome: scaled, train-fitted preprocessing produced 79,999 x 8 training and 20,001 x 8 test inputs; both contained no NaNs. Binary LDA produced exactly 79,999 x 1 and 20,001 x 1 finite arrays. Standalone mean F1 across PAC/Ridge/SGD/XGBoost/LogitBoost surrogate was 0.530 with LDA fitted on imbalanced train and 0.564 when LDA was fitted on ROS-balanced train; mean ROC-AUC was 0.949 and 0.955 respectively. Individual LDA standalone F1 ranged 0.413-0.579 (imbalanced prior) and 0.545-0.577 (balanced prior), versus no-reduction standalone F1 0.505-0.678. Shape/scaling/NaN handling is correct. The LDA collapse is consistent with genuine one-component information compression, with class prior contributing; the larger PaRSEL drop to F1 0.492 may additionally reflect stack score/meta-threshold behavior and needs E2 calibration/threshold analysis.

## RFE Runtime Decision (before rerun)

Hypothesis: one training-only CV selection of RFE feature count, followed by fold-local Random Forest RFE using that fixed count, will finish within the run budget without changing test evaluation.
Discriminator: rerun ROS+RFE once with the selected count and selected feature names saved in reducer_metadata; compare only on the fixed untouched test and label this count-selection compromise.

- Compromise: select `n_features` once by 3-fold average-precision CV on the training split, then reuse that fixed count inside OOF folds and the final train fit. Random Forest RFE remains fit only on each relevant training partition, with at most 100 trees. This is faster than independently selecting feature count inside every OOF fold, but the count selection sees the full training labels; the untouched test remains isolated.

## PaRSEL+ E1 Hypothesis (before run)

Hypothesis: repeated five-fold CV over five fixed seeds will show that the primary PaRSEL baseline's apparent high accuracy coexists with materially weaker minority-class precision/PR-AUC than ranking metrics alone suggest.
Discriminator: report mean, standard deviation, and 95% intervals across the five-seed CV for fixed standalone alternatives, and compare paired test-set predictions/scores to the saved PaRSEL baseline with McNemar and DeLong tests.

## PaRSEL+ E2 Hypothesis (before run)

Hypothesis: selecting a threshold on a training-only validation fold can raise PaRSEL recall to at least 0.90, while lowering precision relative to its default-threshold held-out result.
Discriminator: choose max-F1 and recall-constrained thresholds on the same validation predictions, then evaluate each exactly once on the untouched test set and record the precision/recall trade-off.

## Handoff While E1 Runs (2026-09-30)

- Added read-only `scripts/status.py` and `make status`; it reads process state and CSVs but never writes experiment results. Its E1 row estimate is one runtime-plan row plus 15 PaRSEL folds, 125 standalone folds, 48 metric summaries, and 5 paired held-out rows: 194 rows.
- Added ignore patterns for future logs, per-run figures, prepared split CSVs, caches, and virtual environments. Existing tracked generated artifacts were not removed or rewritten.
- Drafted the report problem statement, paper limitations, leakage-free methodology, and threats-to-validity sections without inventing PaRSEL+ numbers.
- Added unlaunched E4-E8 helper code and synthetic unit tests. These helpers do not write `extension_results.csv` until a future explicit runner invokes them with extension run keys.
- Added an untrained Streamlit shell in `app.py`. It refuses prediction until a separately validated pipeline and frozen threshold are saved; it displays an educational-use-only/non-diagnosis notice.
- E1 remains the only active experiment writer. E2 and Phase 1 queue resumption remain intentionally blocked until E1 finishes and is checked with `make status`.

## E2 Outcome (2026-10-05)

- E2 selected thresholds on a group-safe validation partition from training only and evaluated once on the untouched test. Default 0.5: precision 0.7472, recall 0.7006, F1 0.7231, specificity 0.9780, balanced accuracy 0.8393, [TN, FP, FN, TP] = [17,898, 403, 509, 1,191]. Max-F1 threshold 0.613937: precision 0.7646, recall 0.6876, F1 0.7241, specificity 0.9803, balanced accuracy 0.8340, [17,941, 360, 531, 1,169]. Recall-target threshold 0.081557: precision 0.4649, recall 0.8888, F1 0.6105, specificity 0.9050, balanced accuracy 0.8969, [16,562, 1,739, 189, 1,511].
- Max-F1 gave no meaningful gain. The recall-target variant is a precision/recall trade-off, with about 4.2 additional false alarms per additional true positive; its validation recall target was not fully preserved on test. ROS-inflated, uncalibrated scores explain why the max-F1 threshold is 0.61 rather than a calibrated probability.