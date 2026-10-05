# PaRSEL Baseline and PaRSEL+ Extension

**Project status:** Phase 1 priority experiments are running. Phase 2 has not started. The PaRSEL+ comparison, statistical tests, and improvement claims are intentionally pending.

## Introduction and Gaps in the Base Paper

Noor et al. (2023) introduce PaRSEL, a stacking approach with Passive Aggressive, Ridge, SGD, and XGBoost base learners and a LogitBoost meta-learner, alongside balancing and dimensionality-reduction comparisons. Despite the paper title's reference to heart disease, its principal 100,000-row dataset and reported 91,500/8,500 class counts match the diabetes prediction dataset and its `diabetes` target. This project therefore treats the work as a diabetes-risk classification study and keeps the separate cardiovascular dataset independent.

Several details needed for strict replication are unspecified, including the exact split ratio/seed, duplicate policy, outlier rules, and fully detailed LogitBoost implementation. The paper also reports recall inconsistently (about 67% versus 70%, with another method-specific value near 77%) and describes a tuned result near 98% accuracy together with 97% recall. These discrepancies are retained as limitations rather than resolved by test-set tuning.

### Problem Statement

The project addresses binary diabetes-risk prediction under a strongly
imbalanced class distribution. The objective is not simply high accuracy: the
minority class must be detected with an explicit precision/recall trade-off,
while repeated records, preprocessing, balancing, and threshold selection stay
isolated from the final test estimate. PaRSEL+ extends the base workflow by
testing stricter evaluation and decision-making procedures.

### Base-Paper Limitations

- The title says heart disease, while the primary data and target are diabetes.
- Algorithm 16 presents dimensionality reduction before the split. This is
	ambiguous and can allow test-derived transformation information; this project
	fits reducers on training data only.
- The approximately 91/9 test imbalance makes accuracy misleading and exposes
	the paper's relatively weak reported recall.
- Reported values are internally inconsistent, including recall around 67%
	versus 70% and the Table 4 narrative claiming roughly 98% accuracy with 97%
	recall.
- The paper does not describe threshold optimization, probability calibration,
	paired statistical testing, or confidence intervals.

## Methodology

### Phase 1: PaRSEL Baseline

The paper-faithful data variant retains all 100,000 rows. Its fixed seed-42 split assigns identical predictor tuples to one partition, preventing repeated records from crossing train/test. This is a deliberate deviation from a simple row-wise stratified split to meet the no-leakage requirement. It contains 79,999 training rows (73,199 negatives, 6,800 positives) and 20,001 original, imbalanced test rows (18,301 negatives, 1,700 positives; 8.50% positive). Random Over Sampling is applied to training data only and produces 73,199 examples per class (146,398 total), nine per class below the paper's 73,208 figure.

Two additional variants are reported separately: the 76,916/19,230 exact-deduplicated stratified split and the exact-deduplicated predictor-grouped split. The historical ROS pair is excluded because exact rows crossed its train/test boundary.

All encoders, train-fitted 1.5-IQR caps for age/BMI/HbA1c/glucose, scalers, reducers, samplers, OOF models, and tuning are learned inside training partitions only. PaRSEL meta-features are 5-fold stratified out-of-fold base-model scores. The meta-learner is scikit-learn `GradientBoostingClassifier(loss="log_loss")` configured as additive logistic boosting; it is a transparent surrogate, not canonical LogitBoost. Each run is checkpointed to `results/all_runs.csv` and has a 600-second limit.

The single isolated `smote-variants==1.0.1` wheel install completed with `--no-deps`, but import validation failed because `metric-learn` is absent. No retry or installation in the primary environment was made. ProWRAS, LoRAS, MWMOTE, and RWO are recorded unavailable; no alternative is mislabeled under those names.

### Phase 2: PaRSEL+ Proposal

PaRSEL+ is a planned extension, not yet an evaluated model. The ordered ablations are: E1 repeated-seed evaluation and paired significance; E2 validation-only threshold selection; E3 additional base learners and drop-one ablations; E4 meta-learner comparison; E5 class/cost weighting versus available oversampling; E6 probability calibration; E7 SHAP and subgroup metrics; E8 independent Pima validation. E9 is optional and will not be built before analytical validation.

### Evaluation Methodology

The pipeline removes nulls before splitting, caps continuous outliers using
training-fitted 1.5-IQR bounds, encodes categories and scales numeric inputs
using training data only, and keeps identical predictor tuples together in the
primary split. Reducers and samplers are applied only after the split; samplers
operate on training partitions, and PaRSEL meta-features are out-of-fold base
predictions. The untouched test set is imbalanced and is used only for final
evaluation. Reported metrics include precision, recall, F1, specificity,
balanced accuracy, ROC-AUC, PR-AUC, and confusion counts.

## Baseline Reproduction vs. Paper

The primary held-out run uses the same 20,001 test rows for all Phase 1 configurations. At the default 0.5 decision threshold, the first untuned PaRSEL + ROS + no-reduction run obtained:

| Metric | Paper's rounded claim | Phase 1 baseline | PaRSEL+ | Baseline minus paper | Verdict |
|---|---:|---:|---:|---:|---|
| Accuracy | about 97% | 95.44% | pending E1-E8 | -1.56 percentage points | Not reproduced |
| F1 | about 80% | 72.31% | pending E1-E8 | -7.69 percentage points | Not reproduced |
| Precision | above 90% (often about 99%) | 74.72% | pending E1-E8 | below lower bound | Not reproduced |
| Recall | about 67-70% | 70.06% | pending E1-E8 | within reported range | Partially reproduced |
| ROC-AUC | about 98% | 96.07% | pending E1-E8 | -1.93 percentage points | Not reproduced |
| PR-AUC | Not clearly reported | 64.70% | pending E1-E8 | n/a | Not comparable |

Specificity was 97.80%; balanced accuracy was 83.93%. The test positive prevalence is 8.50%, so accuracy alone masks the precision/recall trade-off. The paper's confidence and metric inconsistencies prevent a claim of exact reproduction. Remaining prioritized Phase 1 results will be added from the append-only log; the final baseline table must not select a run by test score.

The separate three-draw, training-only randomized-search variant selected learning rate 0.1, depth 7, and 152 estimators. It scored 96.36% accuracy, 84.95% precision, 69.41% recall, 76.40% F1, 0.9713 ROC-AUC, and 0.8458 PR-AUC on the same test set. The paper's Table 4 narrative is for ProWRAS+LDA, whereas this tuned run is ROS/no reduction, so the numbers are not configuration-comparable. The untuned paper-parameter result remains the prespecified primary comparator.

## Proposed Improvements

PaRSEL+ will only be described as an improvement if it outperforms the prespecified Phase 1 baseline on the same test split and the E1 paired tests support that conclusion. Candidate mechanisms are threshold selection for screening recall, added base-model diversity, meta-learner alternatives, cost-sensitive learning, and calibration. Their hypotheses and outcomes will be added to `docs/decisions.md` one extension at a time.

## Ablation Study

**Pending Phase 2 execution.** The required table will compare paper's four base learners against the expanded layer, each drop-one model, meta-learning approaches, weighting/sampling choices, and calibration. Every run will retain its split, seed, model variant, metrics, timings, and status in `results/extension_results.csv`. Unavailable learners will remain unavailable rather than being replaced without disclosure.

## Statistical Significance

### E1 Cross-Validated Evaluation

E1 is complete. It used five-fold stratified CV over five fixed seeds for the
standalone models and three fixed seeds for PaRSEL because its nested OOF stack
is substantially more expensive. The reported means are first averaged within
each seed, then summarized across seeds; the intervals are 95% t-based CIs. The
held-out test set was not used for CV fitting or model selection and was used
separately for the paired McNemar and DeLong tests.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| PaRSEL | 0.9672 +/- 0.0008 [0.9652, 0.9693] | 0.8935 +/- 0.0067 [0.8768, 0.9103] | 0.6981 +/- 0.0041 [0.6881, 0.7082] | 0.7836 +/- 0.0052 [0.7707, 0.7965] | 0.9741 +/- 0.0004 [0.9730, 0.9752] | 0.8480 +/- 0.0012 [0.8450, 0.8509] |
| PAC | 0.8774 +/- 0.0004 [0.8769, 0.8779] | 0.4010 +/- 0.0008 [0.4000, 0.4020] | 0.8949 +/- 0.0009 [0.8937, 0.8960] | 0.5538 +/- 0.0007 [0.5529, 0.5547] | 0.9611 +/- 0.0001 [0.9610, 0.9611] | 0.8038 +/- 0.0001 [0.8037, 0.8040] |
| Ridge | 0.8829 +/- 0.0003 [0.8825, 0.8832] | 0.4122 +/- 0.0006 [0.4115, 0.4129] | 0.8867 +/- 0.0008 [0.8858, 0.8877] | 0.5628 +/- 0.0005 [0.5621, 0.5635] | 0.9611 +/- 0.0000 [0.9611, 0.9612] | 0.8039 +/- 0.0001 [0.8038, 0.8041] |
| SGD | 0.8495 +/- 0.0370 [0.8036, 0.8955] | 0.3892 +/- 0.0789 [0.2913, 0.4871] | 0.8569 +/- 0.0739 [0.7651, 0.9486] | 0.5147 +/- 0.0482 [0.4548, 0.5746] | 0.9479 +/- 0.0066 [0.9398, 0.9560] | 0.7438 +/- 0.0185 [0.7208, 0.7668] |
| XGBoost | 0.9082 +/- 0.0009 [0.9071, 0.9094] | 0.4794 +/- 0.0027 [0.4761, 0.4828] | 0.9203 +/- 0.0006 [0.9196, 0.9211] | 0.6304 +/- 0.0024 [0.6274, 0.6334] | 0.9784 +/- 0.0001 [0.9783, 0.9785] | 0.8850 +/- 0.0003 [0.8846, 0.8853] |
| Logistic regression | 0.8851 +/- 0.0004 [0.8846, 0.8857] | 0.4171 +/- 0.0009 [0.4160, 0.4183] | 0.8834 +/- 0.0009 [0.8822, 0.8845] | 0.5667 +/- 0.0007 [0.5658, 0.5676] | 0.9619 +/- 0.0001 [0.9618, 0.9620] | 0.8113 +/- 0.0001 [0.8111, 0.8114] |

Held-out paired comparisons against PaRSEL:

| Alternative | McNemar p | DeLong p | Verdict |
|---|---:|---:|---|
| PAC | 1.27e-200 | 0.00666 | Mixed: PaRSEL has better accuracy/F1/ranking, PAC has higher recall; no universal improvement claim |
| Ridge | 6.31e-188 | 0.00864 | Mixed: PaRSEL has better accuracy/F1/ranking, Ridge has higher recall; no universal improvement claim |
| SGD | 5.27e-321 | 2.42e-27 | Mixed: PaRSEL has better accuracy/F1/ranking, SGD has higher recall; no universal improvement claim |
| XGBoost | 1.44e-99 | 3.03e-44 | Mixed: PaRSEL has better accuracy/F1/PR-AUC, XGBoost has higher recall and ROC-AUC; no universal improvement claim |
| Logistic regression | 9.49e-181 | 0.02598 | Mixed: PaRSEL has better accuracy/F1/PR-AUC, logistic regression has higher recall; no universal improvement claim |

All paired p-values are below 0.05, so these differences are statistically
detectable on the held-out set, but significance does not decide which clinical
operating point is preferable. PaRSEL+ is therefore not claimed to improve all
metrics; the principal trade-off is stronger precision/F1/ranking versus lower
recall than the standalone screening-oriented models.

## E1 Verification

The append-only E1 file contains 194 rows: one runtime-plan row, 140 CV-fold
rows, 48 metric summaries, and five paired held-out rows. There are no duplicate
run keys and all executable rows completed successfully.

### E4 Meta-Learner Comparison

E4 reused shared out-of-fold base predictions, so the three meta-learners did
not refit the base layer separately. It used three seeds with five folds per
seed. The intervals below are 95% t-based CIs across the three seed-level
results and are the primary E4 evidence. The held-out comparison uses one
fixed untouched test split, exactly the same ROS/no-reduction PaRSEL baseline
run used in E1.

| Meta-learner | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| LogitBoost-style | 0.9718 +/- 0.0002 [0.9712, 0.9723] | 0.9641 +/- 0.0028 [0.9570, 0.9712] | 0.6935 +/- 0.0023 [0.6877, 0.6993] | 0.8067 +/- 0.0017 [0.8025, 0.8110] | 0.9813 +/- 0.0004 [0.9802, 0.9824] | 0.8931 +/- 0.0033 [0.8849, 0.9012] |
| Logistic regression | 0.9665 +/- 0.0001 [0.9662, 0.9669] | 0.8442 +/- 0.0010 [0.8417, 0.8467] | 0.7437 +/- 0.0006 [0.7422, 0.7451] | 0.7908 +/- 0.0007 [0.7889, 0.7926] | 0.9775 +/- 0.0001 [0.9773, 0.9776] | 0.8799 +/- 0.0002 [0.8795, 0.8803] |
| Weighted average | 0.9548 +/- 0.0001 [0.9545, 0.9551] | 0.7391 +/- 0.0019 [0.7345, 0.7437] | 0.7240 +/- 0.0018 [0.7195, 0.7285] | 0.7314 +/- 0.0002 [0.7310, 0.7319] | 0.9665 +/- 0.0000 [0.9664, 0.9666] | 0.8254 +/- 0.0000 [0.8253, 0.8255] |

| Meta-learner | McNemar p | DeLong p | Verdict vs. PaRSEL baseline |
|---|---:|---:|---|
| LogitBoost-style | 9.60e-40 | 6.49e-37 | Mixed: better accuracy, precision, F1, ROC-AUC and PR-AUC, but worse recall; no universal improvement claim |
| Logistic regression | 1.13e-19 | 7.45e-35 | Better on the measured held-out accuracy, precision, recall, F1, ROC-AUC and PR-AUC; both paired tests are significant |
| Weighted average | 0.330 | 0.457 | No significant difference by either paired test; no improvement claim |

The held-out test comparison is a single fixed split; the cross-validated
intervals above are the primary evidence for E4 stability. No E4 improvement is
claimed where a paired p-value is at least 0.05.

### E5 Imbalance Handling

E5 compared class weighting, ROS, and SMOTE with both the LogitBoost-style and
logistic-regression meta-learners. Resampling occurred inside each training CV
fold only. Three seeds and five folds were used per configuration; intervals
are 95% t-based CIs across seed-level metric means. Paired tests use the same
single untouched test split and Phase 1 PaRSEL baseline predictions.

| Strategy / meta-learner | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Class weight / LogitBoost-style | 0.9719 +/- 0.0002 [0.9714, 0.9725] | 0.9653 +/- 0.0016 [0.9613, 0.9694] | 0.6948 +/- 0.0028 [0.6879, 0.7016] | 0.8080 +/- 0.0018 [0.8036, 0.8124] | 0.9812 +/- 0.0001 [0.9809, 0.9816] | 0.8950 +/- 0.0010 [0.8925, 0.8975] |
| Class weight / Logistic regression | 0.9666 +/- 0.0001 [0.9663, 0.9668] | 0.8443 +/- 0.0011 [0.8416, 0.8470] | 0.7437 +/- 0.0006 [0.7421, 0.7453] | 0.7908 +/- 0.0007 [0.7892, 0.7925] | 0.9776 +/- 0.0001 [0.9775, 0.9777] | 0.8804 +/- 0.0003 [0.8797, 0.8811] |
| ROS / LogitBoost-style | 0.9718 +/- 0.0002 [0.9712, 0.9723] | 0.9641 +/- 0.0028 [0.9570, 0.9712] | 0.6935 +/- 0.0023 [0.6877, 0.6993] | 0.8067 +/- 0.0017 [0.8025, 0.8110] | 0.9813 +/- 0.0004 [0.9802, 0.9824] | 0.8931 +/- 0.0033 [0.8849, 0.9012] |
| ROS / Logistic regression | 0.9665 +/- 0.0001 [0.9662, 0.9669] | 0.8442 +/- 0.0010 [0.8417, 0.8467] | 0.7437 +/- 0.0006 [0.7422, 0.7451] | 0.7908 +/- 0.0007 [0.7889, 0.7926] | 0.9775 +/- 0.0001 [0.9773, 0.9776] | 0.8799 +/- 0.0002 [0.8795, 0.8803] |
| SMOTE / LogitBoost-style | 0.9729 +/- 0.0001 [0.9727, 0.9731] | 0.9881 +/- 0.0014 [0.9846, 0.9917] | 0.6895 +/- 0.0021 [0.6842, 0.6948] | 0.8122 +/- 0.0010 [0.8097, 0.8147] | 0.9790 +/- 0.0001 [0.9787, 0.9793] | 0.8892 +/- 0.0009 [0.8871, 0.8913] |
| SMOTE / Logistic regression | 0.9702 +/- 0.0003 [0.9694, 0.9710] | 0.9227 +/- 0.0038 [0.9134, 0.9321] | 0.7088 +/- 0.0006 [0.7074, 0.7102] | 0.8017 +/- 0.0017 [0.7974, 0.8060] | 0.9754 +/- 0.0001 [0.9751, 0.9757] | 0.8770 +/- 0.0002 [0.8765, 0.8774] |

| Strategy / meta-learner | Test precision | Test recall | Test F1 | McNemar p | DeLong p | Verdict vs. baseline |
|---|---:|---:|---:|---:|---:|---|
| Class weight / LogitBoost-style | 94.42% | 66.65% | 78.14% | 1.57e-38 | 3.29e-35 | Mixed: higher precision/F1 and ranking, lower recall |
| Class weight / Logistic regression | 83.84% | 72.94% | 78.01% | 1.10e-19 | 4.32e-35 | Better on measured accuracy, precision, recall, F1 and ranking; significant paired differences |
| ROS / LogitBoost-style | 94.70% | 66.24% | 77.95% | 9.60e-40 | 6.49e-37 | Mixed: higher precision/F1 and ranking, lower recall |
| ROS / Logistic regression | 83.48% | 72.82% | 77.79% | 1.13e-19 | 7.45e-35 | Better on measured accuracy, precision, recall, F1 and ranking; significant paired differences |
| SMOTE / LogitBoost-style | 97.66% | 66.35% | 79.02% | 1.37e-45 | 3.06e-17 | Mixed: higher precision/F1, lower recall; ROC-AUC/PR-AUC lower than baseline |
| SMOTE / Logistic regression | 90.55% | 69.35% | 78.55% | 9.71e-33 | 2.31e-18 | Mixed: higher F1 and PR-AUC, lower recall and ROC-AUC |

All paired tests here have p < 0.05, but conclusions remain metric-specific;
statistical significance does not make a strategy universally superior. The
CV intervals are the primary stability evidence, while the held-out comparison
uses a single fixed test split.

### E2 Threshold Operating Points

E2 selected thresholds on a group-safe validation partition cut from the
training split only. The selected thresholds were then frozen and evaluated
once on the untouched 20,001-row test set. The default row below was recomputed
from the cached test scores without refitting.

| Operating point | Frozen threshold | Precision | Recall | F1 | Specificity | Balanced accuracy | Confusion matrix [TN, FP, FN, TP] |
|---|---:|---:|---:|---:|---:|---:|---|
| Default baseline | 0.500000 | 74.72% | 70.06% | 72.31% | 97.80% | 83.93% | [17,898, 403, 509, 1,191] |
| Validation max-F1 | 0.613937 | 76.46% | 68.76% | 72.41% | 98.03% | 83.40% | [17,941, 360, 531, 1,169] |
| Validation recall >= 0.90 | 0.081557 | 46.49% | 88.88% | 61.05% | 90.50% | 89.69% | [16,562, 1,739, 189, 1,511] |

Max-F1 thresholding produced no meaningful gain over the 72.31% baseline F1:
it reached 72.41% while recall fell from 70.06% to 68.76%. The recall-target
variant improved recall by 18.82 percentage points but reduced precision by
28.23 points and F1 by 11.26 points. It generated 1,336 additional false
alarms for 320 additional true positives, about 4.2 false alarms per additional
case caught. The validation target of 0.90 recall was not fully preserved on the
test set (88.88%). These are operating-point trade-offs, not an overall model
improvement. Because ROS changes the training class prior and the probabilities
are uncalibrated, the best-F1 threshold being 0.61 should not be interpreted as
a calibrated risk probability; calibration is a later PaRSEL+ stage.

## Limitations

- The source contains 3,854 exact duplicate rows and repeated predictor combinations with conflicting labels. The primary variant retains all rows but groups identical predictors to prevent train/test contamination; deduplicated and predictor-grouped results remain separate.
- The positive class is approximately 8.5% of the primary test set. Accuracy can be misleading; PR-AUC, recall, precision, specificity, balanced accuracy, and confusion counts are reported alongside it.
- The group-safe primary split and 1.5-IQR train-fitted winsorization are leakage controls but are not specified by Noor et al.
- The LogitBoost-style additive logistic boosting model is a substitute, not canonical LogitBoost.
- The specialized sampler wheel could not import without `metric-learn`; those comparisons are unavailable under the no-retry rule.
- The paper's title/target mismatch, inconsistent recall values, and incompletely specified split and runtime protocol limit numerical comparability.
- The dataset is observational and its source/population/labeling process may introduce subgroup and geographic biases. It is not a clinical dataset for diagnosis.

## Threats to Validity

- Duplicate rows and repeated predictor combinations can make a row-wise split
	optimistic; group-safe splitting is used as a control, but it differs from the
	paper's unspecified protocol.
- The test prevalence is about 8.5%, so accuracy and ROC-AUC can hide poor
	minority precision; PR-AUC and class-specific metrics are necessary.
- The paper's reducer ordering is ambiguous, and the project's train-only fit is
	intentionally more conservative. This can lower measured numbers while
	reducing leakage risk.
- The LogitBoost-style surrogate is not canonical LogitBoost, and unavailable
	specialized samplers cannot support configuration-matched claims.
- A single held-out split is not enough to establish general superiority;
	PaRSEL+ claims remain pending E1 confidence intervals and paired tests.

## Future Work

Complete the Phase 1 priority queue, then perform E1-E8 in order. Assess external validation only on documented common features: map Pima plasma glucose to the project's glucose feature, Pima mass to BMI, and age to age; drop features without counterparts (and document that the glucose measurement and population/label criteria differ). Do not merge the Pima rows into the development data. Consider a small demo only after calibration, fairness checks, and statistical comparisons, with an explicit not-for-clinical-use notice.

## Citation

Noor, A., Javaid, N., Alrajeh, N., Mansoor, B., Khaqan, A., & Bouk, S. H. (2023). *Heart Disease Prediction Using Stacking Model With Balancing Techniques and Dimensionality Reduction*. IEEE Access, 11. https://doi.org/10.1109/ACCESS.2023.3325681
