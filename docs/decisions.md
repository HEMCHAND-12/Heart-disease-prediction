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