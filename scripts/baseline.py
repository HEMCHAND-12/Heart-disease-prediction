"""Run a reproducible heart-disease baseline on the project CSV."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetes_prediction_dataset.csv"
TARGET = "diabetes"
RANDOM_STATE = 42


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column {TARGET!r} in {DATA_PATH}")

    features = data.drop(columns=TARGET)
    target = data[TARGET]
    categorical_columns = features.select_dtypes(include=["object", "str"]).columns.tolist()
    numeric_columns = features.select_dtypes(exclude=["object", "str"]).columns.tolist()

    repeated_predictors = data.assign(_target=target).groupby(
        list(features.columns), dropna=False
    )["_target"].nunique()
    conflicting_predictors = int(repeated_predictors.gt(1).sum())

    print(f"Rows: {len(data):,}; exact duplicate rows: {int(data.duplicated().sum()):,}")
    print(f"Target counts: {target.value_counts().sort_index().to_dict()}")
    print(f"Repeated feature combinations with conflicting labels: {conflicting_predictors:,}")
    print("Split: stratified 5-fold with identical predictor combinations grouped together.")

    groups = pd.factorize(pd.MultiIndex.from_frame(features))[0]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_indices, test_indices = next(splitter.split(features, target, groups=groups))
    x_train, x_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
    model = make_pipeline(
        preprocessing,
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    print(f"\nTrain rows: {len(x_train):,}; test rows: {len(x_test):,}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"Average precision: {average_precision_score(y_test, probabilities):.4f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("Classification report:")
    print(classification_report(y_test, predictions, labels=[0, 1], zero_division=0))


if __name__ == "__main__":
    main()