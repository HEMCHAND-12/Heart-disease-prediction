"""Create the paper-sized ROS training set and preserve an untouched test split."""

from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "diabetes_prediction_dataset.csv"
BALANCED_PATH = ROOT / "data" / "diabetes_prediction_dataset_balanced_train.csv"
TEST_PATH = ROOT / "data" / "diabetes_prediction_dataset_test.csv"
TARGET = "diabetes"
RANDOM_STATE = 42
EXPECTED_TRAIN_CLASS_SIZE = 73_208


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column {TARGET!r} in {DATA_PATH}")

    features = data.drop(columns=TARGET)
    target = data[TARGET]
    if set(target.unique()) != {0, 1}:
        raise ValueError(f"Expected binary 0/1 values in target {TARGET!r}")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    train_counts = y_train.value_counts().sort_index()
    if train_counts.to_dict() != {0: EXPECTED_TRAIN_CLASS_SIZE, 1: 6_792}:
        raise ValueError(
            "The source data or split no longer matches the paper's Table 1: "
            f"training counts are {train_counts.to_dict()}"
        )

    sampler = RandomOverSampler(random_state=RANDOM_STATE)
    x_balanced, y_balanced = sampler.fit_resample(x_train, y_train)
    balanced = x_balanced.copy()
    balanced[TARGET] = y_balanced

    balanced_counts = balanced[TARGET].value_counts().sort_index().to_dict()
    if balanced_counts != {0: EXPECTED_TRAIN_CLASS_SIZE, 1: EXPECTED_TRAIN_CLASS_SIZE}:
        raise RuntimeError(f"Unexpected balanced class counts: {balanced_counts}")

    test = x_test.copy()
    test[TARGET] = y_test
    balanced.to_csv(BALANCED_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Source rows: {len(data):,}")
    print(f"Source target counts: {target.value_counts().sort_index().to_dict()}")
    print(f"Training rows after ROS: {len(balanced):,}")
    print(f"Balanced training target counts: {balanced_counts}")
    print(f"Unchanged test rows: {len(test):,}")
    print(f"Unchanged test target counts: {test[TARGET].value_counts().sort_index().to_dict()}")
    print(f"Wrote: {BALANCED_PATH.relative_to(ROOT)}")
    print(f"Wrote: {TEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()