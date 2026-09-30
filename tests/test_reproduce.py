import pandas as pd

from scripts.reproduce import (
    FEATURES,
    NUMERIC,
    TARGET,
    TrainWinsorizer,
    group_ids,
    prepare_splits,
    transform_pair,
)


def test_saved_splits_have_no_exact_row_overlap() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "splits" / "stratified_train.csv")
    test = pd.read_csv(root / "splits" / "stratified_test.csv")
    assert not set(map(tuple, train.to_numpy())) & set(map(tuple, test.to_numpy()))


def test_grouped_split_has_no_shared_predictor_tuples() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "splits" / "grouped_train.csv")
    test = pd.read_csv(root / "splits" / "grouped_test.csv")
    assert not set(map(tuple, train[FEATURES].to_numpy())) & set(map(tuple, test[FEATURES].to_numpy()))


def test_paper_faithful_variant_retains_duplicates_without_group_leakage() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / "splits" / "paper_faithful_train.csv")
    test = pd.read_csv(root / "splits" / "paper_faithful_test.csv")
    assert len(train) + len(test) == 100_000
    assert train.duplicated().sum() + test.duplicated().sum() > 0
    assert not set(map(tuple, train[FEATURES].to_numpy())) & set(map(tuple, test[FEATURES].to_numpy()))
    assert test[TARGET].value_counts().to_dict() == {0: 18_301, 1: 1_700}


def test_group_ids_keep_equal_predictors_together() -> None:
    frame = pd.DataFrame(
        {
            "gender": ["Female", "Female", "Male"],
            "age": [20.0, 20.0, 30.0],
            "hypertension": [0, 0, 1],
            "heart_disease": [0, 0, 0],
            "smoking_history": ["never", "never", "former"],
            "bmi": [20.0, 20.0, 25.0],
            "HbA1c_level": [5.0, 5.0, 6.0],
            "blood_glucose_level": [100, 100, 130],
            "diabetes": [0, 1, 1],
        }
    )
    identifiers = group_ids(frame)
    assert identifiers[0] == identifiers[1]
    assert identifiers[0] != identifiers[2]


def test_winsorizer_bounds_are_learned_from_train_only() -> None:
    train = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0],
            "bmi": [20.0, 21.0, 22.0, 23.0],
            "HbA1c_level": [4.0, 5.0, 6.0, 7.0],
            "blood_glucose_level": [80, 90, 100, 110],
        }
    )
    test = train.iloc[[0]].copy()
    test.loc[test.index[0], "age"] = 10_000
    fitted = TrainWinsorizer().fit(train)
    upper_bound = fitted.bounds_["age"][1]
    assert fitted.transform(test)["age"].iloc[0] == upper_bound
    assert fitted.bounds_["age"][1] < test["age"].iloc[0]


def test_test_rows_cannot_change_training_preprocessing() -> None:
    from scripts.reproduce import CATEGORICAL

    train = pd.DataFrame({
        "gender": ["Female", "Male"] * 10,
        "smoking_history": ["never", "former"] * 10,
        "age": [30.0, 60.0] * 10,
        "hypertension": [0, 1] * 10,
        "heart_disease": [0, 1] * 10,
        "bmi": [22.0, 34.0] * 10,
        "HbA1c_level": [5.2, 7.0] * 10,
        "blood_glucose_level": [90, 180] * 10,
        "diabetes": [0, 1] * 10,
    })
    test = train.iloc[:4].copy()
    changed_test = test.copy()
    changed_test.loc[:, CATEGORICAL[0]] = "unseen-category"
    changed_test.loc[:, NUMERIC[0]] = 1_000_000
    original_train, _, _ = transform_pair(train, test, "none")
    changed_train, _, _ = transform_pair(train, changed_test, "none")
    assert (original_train == changed_train).all()


def test_split_builder_records_binary_classes() -> None:
    config = prepare_splits()
    assert config["target"] == TARGET
    assert config["features"] == FEATURES
    assert config["stratified_split"]["test_rows"] > 0
    assert config["stratified_split"]["test_counts"] == {0: 17534, 1: 1696}