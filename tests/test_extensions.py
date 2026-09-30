import numpy as np

from scripts.proposed_model import _delong_pvalue, _mcnemar_pvalue


def test_paired_delong_is_one_for_identical_scores() -> None:
    targets = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
    assert _delong_pvalue(targets, scores, scores) == 1.0


def test_mcnemar_is_one_for_identical_predictions() -> None:
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 0])
    assert _mcnemar_pvalue(targets, predictions, predictions) == 1.0


def test_mcnemar_detects_asymmetric_discordant_errors() -> None:
    targets = np.array([0, 0, 0, 0, 1, 1])
    baseline = np.array([1, 1, 1, 0, 0, 0])
    alternative = np.array([0, 0, 0, 0, 1, 1])
    assert _mcnemar_pvalue(targets, baseline, alternative) < 0.1
