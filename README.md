# Heart Disease PaRSEL

Starter workspace for reproducing the paper's heart-disease stacking model with
feature reduction, class balancing, XGBoost, and SHAP explanations.

## Environment

The project is configured for a CPU-only Python virtual environment in `.venv`.
The laptop has 8 logical CPUs, approximately 7 GiB RAM, and Python 3.14.
Large oversampling experiments should be run one technique at a time to keep
memory use predictable.

Create or refresh the environment with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Activate it in a terminal with:

```bash
source .venv/bin/activate
```

The paper's reported results are research benchmarks, not clinical advice.

## First Baseline

The paper's Table 1 reports 91,500 healthy and 8,500 unhealthy rows, matching
the `diabetes` label in `data/diabetes_prediction_dataset.csv`. The baseline
label. Run it from the repository root with:

```bash
.venv/bin/python scripts/baseline.py
```

This is a logistic-regression baseline with one-hot encoding, class weighting,
and a stratified grouped holdout split. Repeated predictor combinations stay in
one partition. It is a starting point, not a full PaRSEL reproduction. The
paper's `cardio_train.csv` dataset is independent data and should not be merged
into the main training labels.

## Paper-Matched ROS Split

Generate the balanced training set and untouched test split with:

```bash
.venv/bin/python scripts/balance_dataset.py
```

The script applies the paper's Random Over Sampling (ROS) method after an
80/20 split (`random_state=42`). It writes 146,416 training rows with 73,208
examples per class to `data/diabetes_prediction_dataset_balanced_train.csv`,
and preserves the original class distribution in
`data/diabetes_prediction_dataset_test.csv`. The original source CSV remains
unchanged. `data/cardio_train.csv` is a separate, already-balanced dataset.