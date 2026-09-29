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