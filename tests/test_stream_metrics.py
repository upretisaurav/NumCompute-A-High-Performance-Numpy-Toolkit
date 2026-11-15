"""Tests for numcompute_stream.metrics."""

import numpy as np
import pytest

from numcompute_stream.metrics import (
    F1,
    Accuracy,
    ConfusionMatrix,
    Precision,
    Recall,
    RollingMetric,
    StreamingAUC,
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


# ---------------------------------------------------------------------------
# Streaming metrics
# ---------------------------------------------------------------------------


def stream(metric, y_true, y_pred, size=7):
    for start in range(0, len(y_true), size):
        metric.update(y_true[start : start + size], y_pred[start : start + size])
    return metric


@pytest.mark.parametrize(
    "metric, func",
    [
        (Accuracy(), accuracy),
        (Precision(), precision),
        (Recall(), recall),
        (F1(), f1_score),
        (F1(average="binary", pos_label=2), lambda t, p: f1_score(t, p, "binary", 2)),
    ],
)
def test_streaming_result_equals_all_data_at_once(metric, func):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=100)
    y_pred = np.where(rng.random(100) < 0.7, y_true, rng.integers(0, 3, size=100))
    assert stream(metric, y_true, y_pred).result() == pytest.approx(
        func(y_true, y_pred)
    )


def test_confusion_matrix_grows_when_new_label_appears():
    cm = ConfusionMatrix().update([0, 1], [0, 1])
    cm.update([2, 0], [2, 1])
    np.testing.assert_array_equal(cm.labels_, [0, 1, 2])
    np.testing.assert_array_equal(cm.matrix_, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])


def test_reset_really_resets():
    acc = Accuracy().update([0, 1], [1, 1])
    acc.reset()
    assert np.isnan(acc.result())
    acc.update([1], [1])
    assert acc.result() == 1.0


def test_rolling_window_forgets_old_chunks():
    rolling = RollingMetric(Accuracy(), window=2)
    rolling.update([0, 0], [1, 1])  # 0% correct, will drop out
    rolling.update([0, 0], [0, 0])
    rolling.update([1, 1], [1, 1])
    assert rolling.result() == 1.0


def test_auc_perfect_ranking_and_constant_scores():
    y = np.array([0, 0, 1, 1])
    assert StreamingAUC().update(y, [0.1, 0.2, 0.8, 0.9]).result() == 1.0
    assert StreamingAUC().update(y, [0.9, 0.8, 0.2, 0.1]).result() == 0.0
    assert StreamingAUC().update(y, [0.5, 0.5, 0.5, 0.5]).result() == 0.5


def test_auc_matches_pairwise_count_over_chunks():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=300)
    scores = np.clip(rng.normal(0.4 + 0.2 * y, 0.15), 0, 1)
    auc = stream(StreamingAUC(n_bins=100000), y, scores, size=50).result()
    pos, neg = scores[y == 1], scores[y == 0]
    exact = np.mean(pos[:, None] > neg[None, :])
    assert auc == pytest.approx(exact, abs=1e-3)


def test_auc_single_class_is_nan():
    assert np.isnan(StreamingAUC().update([1, 1], [0.2, 0.9]).result())
