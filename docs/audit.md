# Reproduction Audit

Audit date: 2026-09-30

## Scope and Branches

Remote branches inspected:

| Branch | Tip | Relevant tracked files |
|---|---|---|
| `main` | `1c10ad42f0b548de517ee91a3110d414c7c8d781` | `README.md`, `pyproject.toml`, `requirements.txt`; no data or experiment code |
| `DataCleaning&DimensionalityReduction` | `d8ada5df8e9a2fa51cfdce1d9901060ddf9c1e80` | `cldr.py`, raw diabetes/Cardio data, merged cleaned data, full-data LDA output, reduction-results CSV |
| `data/paper-balanced-training` (current) | `969b78ae9bb998a4d094e4ed50b69d80e08decf1` | raw diabetes data, ROS train/test files, `scripts/balance_dataset.py`, `scripts/baseline.py`, paper PDF |

The root-level `diabetes_prediction_dataset.csv` is untracked and has the same SHA-256 as `data/diabetes_prediction_dataset.csv`; it was left untouched. The friend branch also contains a committed `venv/` tree; it is not needed for this reproduction and was not changed.

## Current-Branch Data Inventory

| File | Rows | Target counts | Finding |
|---|---:|---:|---|
| `data/diabetes_prediction_dataset.csv` | 100,000 | 0: 91,500; 1: 8,500 | Raw source; 3,854 exact duplicate rows; no missing cells |
| `data/diabetes_prediction_dataset_balanced_train.csv` | 146,416 | 0: 73,208; 1: 73,208 | ROS output; oversampling follows a random 80/20 row split in `balance_dataset.py` |
| `data/diabetes_prediction_dataset_test.csv` | 20,000 | 0: 18,292; 1: 1,708 | Original-source rows, not synthetically generated; class ratio 91.46% / 8.54% |
| `data/cardio_train.csv` | 70,000 | cardio 0: 35,021; 1: 34,979 | Separate cardiovascular dataset; not the paper-matching diabetes target |

The existing split is not stratified. A stratified 20,000-row holdout at the source proportions would have approximately 18,300 negatives and 1,700 positives. The existing test counts differ slightly.

### Existing ROS Pair Leakage Check

The script splits before ROS, so no synthetic observations are intentionally made in the test set. However, the original source itself contains duplicates that crossed the row-wise split:

- Exact full-row intersection between ROS training data and test data: **1,091 rows**.
- Predictor-only combinations shared across training and test: **1,122**.
- The test contains 193 exact duplicate rows within itself; those are present in the source and are not synthetic test examples.

Therefore, this ROS/test pair fails the requested zero train/test exact-row-overlap rule and must not be used for reported reproduction results. Rebuild a fixed split from the raw data after resolving exact source duplicates, then apply each sampler to training data only. Keep the current files as provenance, not as the clean evaluation split.

## Friend-Branch Data Audit

No friend-uploaded diabetes file on `DataCleaning&DimensionalityReduction` is a valid, train-only balanced dataset with a separately untouched imbalanced test set.

| File | Provenance / method | Test status and overlap | Decision |
|---|---|---|---|
| `data/diabetes_prediction_dataset.csv` | Raw source; branch history associates the file with commit `bccc52203e7a5749c77774ff10dd06687cbaa456` | No train/test split in the file | Use as source only |
| `data/cardio_train.csv` | Separate 70,000-row cardiovascular dataset, naturally close to balanced; no ROS/SMOTE method or split recorded | No separate test file; not the same target or cohort | Do not use for PaRSEL reproduction |
| `data/diabetes_cleaned_all_features.csv` | Commit `d8ada5df8e9a2fa51cfdce1d9901060ddf9c1e80` (`update2`); `cldr.py` appends 70,000 mapped cardiovascular rows to the diabetes rows; it assigns a diabetes label from `cardio_train.gluc == 3` | 170,000 rows, no independent test split; not a balancing output. A cardiovascular glucose category is converted into a fabricated diabetes label, and mapped glucose remains an input feature | Reject: mixed populations and target leakage |
| `data/diabetes_reduced_lda.csv` | Same `update2` commit; one LDA component plus target, computed from the merged 170,000-row dataset | LDA is refit on all rows when the file is exported, before any held-out evaluation | Reject: invalid merged labels and full-data transformation leakage |
| `objective2_reduction_results.csv` | Same `update2` commit; results from `cldr.py` on the merged data | Reported metrics inherit the invalid labels/data construction; not a paper result | Do not report as reproduction evidence |

The friend branch has no train/test-balanced diabetes pair to reuse. It has no valid file for which a balancing algorithm, split-before/after status, untouched test status, and zero train/test overlap can all be established.

## Paper Protocol Check

The paper describes 100,000 CDC/Kaggle records and reports an imbalanced class distribution matching the diabetes source. Its Algorithm 16 places the train/test split before class balancing. Thus, **split-before-balancing is supported**. The paper does not specify an 80/20 ratio, random seed, or enough split details to establish that the existing seed-42 split matches the paper. The 73,208-per-class training size in the current ROS artifact is a result of this repository's chosen 80/20 split and ROS, not a fully specified paper split.

The paper compares eight balancing approaches (ProWRAS, LoRAS, ADASYN, SMOTE, ROS, MWMOTE, Borderline-SMOTE, and RWOS), three reductions (RFE, LDA, and Factor Analysis), and a stack of Passive Aggressive, Ridge, SGD, and XGBoost base learners with LogitBoost as the meta learner. The paper's claimed headline values include about 97% accuracy, 80% F1, precision above 90%, recall reported as both 67% and 70% in different passages, and about 98% ROC-AUC. It also reports a tuned result around 98% accuracy, 95% F1, 94% precision, and 97% recall. These discrepancies and the paper's incomplete split/hyperparameter details must be preserved in the comparison report.

## Audit Decision

1. Use only the original diabetes source for the primary reproduction. Do not merge `cardio_train.csv` or use the friend branch's merged/LDA artifacts.
2. Do not report results from the current ROS/test pair because 1,091 exact rows overlap.
3. Create one saved seed-42, stratified split after removing exact duplicate source rows; also run the requested predictor-grouped holdout separately. Document that deduplicating before splitting is a deliberate deviation needed to satisfy the no-exact-row-overlap requirement.
4. Fit all category mappings, outlier rules, scaling, dimensionality reduction, resampling, hyperparameter search, and stacking meta-features using training data only. Preserve the untouched test set for final evaluation.