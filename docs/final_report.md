# PaRSEL Baseline and PaRSEL+ Extension

**Project status:** Phase 1 priority experiments are running. Phase 2 has not started. The PaRSEL+ comparison, statistical tests, and improvement claims are intentionally pending.

## Introduction and Gaps in the Base Paper

Noor et al. (2023) introduce PaRSEL, a stacking approach with Passive Aggressive, Ridge, SGD, and XGBoost base learners and a LogitBoost meta-learner, alongside balancing and dimensionality-reduction comparisons. Despite the paper title's reference to heart disease, its principal 100,000-row dataset and reported 91,500/8,500 class counts match the diabetes prediction dataset and its `diabetes` target. This project therefore treats the work as a diabetes-risk classification study and keeps the separate cardiovascular dataset independent.

Several details needed for strict replication are unspecified, including the exact split ratio/seed, duplicate policy, outlier rules, and fully detailed LogitBoost implementation. The paper also reports recall inconsistently (about 67% versus 70%, with another method-specific value near 77%) and describes a tuned result near 98% accuracy together with 97% recall. These discrepancies are retained as limitations rather than resolved by test-set tuning.

## Methodology

### Phase 1: PaRSEL Baseline

The paper-faithful data variant retains all 100,000 rows. Its fixed seed-42 split assigns identical predictor tuples to one partition, preventing repeated records from crossing train/test. This is a deliberate deviation from a simple row-wise stratified split to meet the no-leakage requirement. It contains 79,999 training rows (73,199 negatives, 6,800 positives) and 20,001 original, imbalanced test rows (18,301 negatives, 1,700 positives; 8.50% positive). Random Over Sampling is applied to training data only.

Two additional variants are reported separately: the 76,916/19,230 exact-deduplicated stratified split and the exact-deduplicated predictor-grouped split. The historical ROS pair is excluded because exact rows crossed its train/test boundary.

All encoders, train-fitted 1.5-IQR caps for age/BMI/HbA1c/glucose, scalers, reducers, samplers, OOF models, and tuning are learned inside training partitions only. PaRSEL meta-features are 5-fold stratified out-of-fold base-model scores. The meta-learner is scikit-learn `GradientBoostingClassifier(loss="log_loss")` configured as additive logistic boosting; it is a transparent surrogate, not canonical LogitBoost. Each run is checkpointed to `results/all_runs.csv` and has a 600-second limit.

The single isolated `smote-variants==1.0.1` wheel install completed with `--no-deps`, but import validation failed because `metric-learn` is absent. No retry or installation in the primary environment was made. ProWRAS, LoRAS, MWMOTE, and RWO are recorded unavailable; no alternative is mislabeled under those names.

### Phase 2: PaRSEL+ Proposal

PaRSEL+ is a planned extension, not yet an evaluated model. The ordered ablations are: E1 repeated-seed evaluation and paired significance; E2 validation-only threshold selection; E3 additional base learners and drop-one ablations; E4 meta-learner comparison; E5 class/cost weighting versus available oversampling; E6 probability calibration; E7 SHAP and subgroup metrics; E8 independent Pima validation. E9 is optional and will not be built before analytical validation.

## Baseline Reproduction vs. Paper

The primary held-out run uses the same 20,001 test rows for all Phase 1 configurations. At the default 0.5 decision threshold, the first untuned PaRSEL + ROS + no-reduction run obtained:

| Metric | Paper's rounded claim | Our Phase 1 result | Difference from point reference | Verdict |
|---|---:|---:|---:|---|
| Accuracy | about 97% | 95.44% | -1.56 percentage points | Not reproduced |
| F1 | about 80% | 72.31% | -7.69 percentage points | Not reproduced |
| Precision | above 90% (often about 99%) | 74.72% | below lower bound | Not reproduced |
| Recall | about 67-70% | 70.06% | within reported range | Partially reproduced |
| ROC-AUC | about 98% | 97.80% | -0.20 percentage points | Partially reproduced |
| PR-AUC | Not clearly reported | 83.93% | n/a | Not comparable |

Specificity was 97.80%; balanced accuracy was 83.93%. The test positive prevalence is 8.50%, so accuracy alone masks the precision/recall trade-off. The paper's confidence and metric inconsistencies prevent a claim of exact reproduction. Remaining prioritized Phase 1 results will be added from the append-only log; the final baseline table must not select a run by test score.

## Proposed Improvements

PaRSEL+ will only be described as an improvement if it outperforms the prespecified Phase 1 baseline on the same test split and the E1 paired tests support that conclusion. Candidate mechanisms are threshold selection for screening recall, added base-model diversity, meta-learner alternatives, cost-sensitive learning, and calibration. Their hypotheses and outcomes will be added to `docs/decisions.md` one extension at a time.

## Ablation Study

**Pending Phase 2 execution.** The required table will compare paper's four base learners against the expanded layer, each drop-one model, meta-learning approaches, weighting/sampling choices, and calibration. Every run will retain its split, seed, model variant, metrics, timings, and status in `results/extension_results.csv`. Unavailable learners will remain unavailable rather than being replaced without disclosure.

## Statistical Significance

**Pending E1.** Planned analysis uses repeated stratified cross-validation within development data over five fixed seeds, reporting mean, standard deviation, and 95% confidence intervals. The untouched test set is reserved for one final paired comparison. McNemar's test will compare paired classifications; DeLong's test will compare paired ROC-AUC. No improvement claim is made before those results exist.

## Limitations

- The source contains 3,854 exact duplicate rows and repeated predictor combinations with conflicting labels. The primary variant retains all rows but groups identical predictors to prevent train/test contamination; deduplicated and predictor-grouped results remain separate.
- The positive class is approximately 8.5% of the primary test set. Accuracy can be misleading; PR-AUC, recall, precision, specificity, balanced accuracy, and confusion counts are reported alongside it.
- The group-safe primary split and 1.5-IQR train-fitted winsorization are leakage controls but are not specified by Noor et al.
- The LogitBoost-style additive logistic boosting model is a substitute, not canonical LogitBoost.
- The specialized sampler wheel could not import without `metric-learn`; those comparisons are unavailable under the no-retry rule.
- The paper's title/target mismatch, inconsistent recall values, and incompletely specified split and runtime protocol limit numerical comparability.
- The dataset is observational and its source/population/labeling process may introduce subgroup and geographic biases. It is not a clinical dataset for diagnosis.

## Future Work

Complete the Phase 1 priority queue, then perform E1-E8 in order. Assess external validation only on documented common features: map Pima plasma glucose to the project's glucose feature, Pima mass to BMI, and age to age; drop features without counterparts (and document that the glucose measurement and population/label criteria differ). Do not merge the Pima rows into the development data. Consider a small demo only after calibration, fairness checks, and statistical comparisons, with an explicit not-for-clinical-use notice.

## Citation

Noor, A., Javaid, N., Alrajeh, N., Mansoor, B., Khaqan, A., & Bouk, S. H. (2023). *Heart Disease Prediction Using Stacking Model With Balancing Techniques and Dimensionality Reduction*. IEEE Access, 11. https://doi.org/10.1109/ACCESS.2023.3325681
