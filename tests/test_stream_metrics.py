"""Tests for numcompute_stream.metrics."""

import numpy as np
import pytest

from numcompute_stream.metrics import (
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)


def test_confusion_matrix_counts():
    cm = confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 1, 0])
    expected = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 0]])
    np.testing.assert_array_equal(cm, expected)


def test_f1_class_never_predicted_is_zero_not_nan():
    # 2.1's f1score returned NaN here (0/0 in precision).
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    assert f1_score(y_true, y_pred, average="binary", pos_label=1) == 0.0
    assert precision(y_true, y_pred, average="binary") == 0.0
    # macro: class 0 has P=0.5, R=1 -> F1=2/3; class 1 -> 0
    assert f1_score(y_true, y_pred) == pytest.approx((2 / 3 + 0) / 2)


def test_binary_scores_hand_computed():
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0])
    # TP=2, FP=1, FN=1
    assert precision(y_true, y_pred, average="binary") == pytest.approx(2 / 3)
    assert recall(y_true, y_pred, average="binary") == pytest.approx(2 / 3)
    assert f1_score(y_true, y_pred, average="binary") == pytest.approx(2 / 3)
    assert accuracy(y_true, y_pred) == pytest.approx(3 / 5)


def test_bad_average_and_length_mismatch():
    with pytest.raises(ValueError):
        f1_score([0, 1], [0, 1], average="weighted")
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])
