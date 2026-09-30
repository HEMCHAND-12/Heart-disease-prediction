import pandas as pd

from scripts.reproduce import (
    FEATURES,
    TARGET,
    TrainWinsorizer,
    group_ids,
    prepare_splits,
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


def test_split_builder_records_binary_classes() -> None:
    config = prepare_splits()
    assert config["target"] == TARGET
    assert config["features"] == FEATURES
    assert config["stratified_split"]["test_rows"] > 0
    assert config["stratified_split"]["test_counts"] == {0: 17534, 1: 1696}