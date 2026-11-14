"""Tests for numcompute_stream.tree."""

import numpy as np
import pytest

from numcompute_stream.tree import best_split, entropy, gini

# ---------------------------------------------------------------------------
# 1.1 Impurity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impurity", [gini, entropy])
def test_pure_group_has_zero_impurity(impurity):
    assert impurity([10, 0]) == 0.0
    assert impurity([0, 0, 7]) == 0.0


def test_balanced_binary_values():
    assert gini([5, 5]) == pytest.approx(0.5)
    assert entropy([5, 5]) == pytest.approx(1.0)


def test_three_equal_classes():
    assert gini([4, 4, 4]) == pytest.approx(2.0 / 3.0)
    assert entropy([4, 4, 4]) == pytest.approx(np.log2(3.0))


@pytest.mark.parametrize("impurity", [gini, entropy])
def test_empty_group_is_zero_not_nan(impurity):
    assert impurity([0, 0]) == 0.0


@pytest.mark.parametrize("impurity", [gini, entropy])
def test_impurity_on_2d_counts_matches_row_by_row(impurity):
    counts = np.array([[3, 1], [0, 4], [2, 2], [0, 0]])
    expected = [impurity(row) for row in counts]
    np.testing.assert_allclose(impurity(counts), expected)


# ---------------------------------------------------------------------------
# 1.2 Vectorised split search
# ---------------------------------------------------------------------------


def test_best_split_hand_computed_example():
    # Perfect split between 3 and 10 -> threshold 6.5, gain = parent gini 0.5
    x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    threshold, gain = best_split(x, y, n_classes=2)
    assert threshold == 6.5
    assert gain == pytest.approx(0.5)


def test_best_split_entropy_gain():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0, 0, 1, 1])
    threshold, gain = best_split(x, y, 2, criterion="entropy")
    assert threshold == 2.5
    assert gain == pytest.approx(1.0)


def test_constant_feature_has_no_split():
    threshold, gain = best_split(np.ones(5), np.array([0, 1, 0, 1, 0]), 2)
    assert threshold is None
    assert gain == 0.0


def test_never_splits_between_equal_values():
    # Best cut by position would be between the two 2.0s; that is not allowed.
    x = np.array([1.0, 2.0, 2.0, 3.0])
    y = np.array([0, 0, 1, 1])
    threshold, _ = best_split(x, y, 2)
    assert threshold in (1.5, 2.5)
    assert not np.any(x == threshold)


def test_tie_picks_first_threshold():
    # Cutting at 1.5 or 3.5 gives the same gain; the lower one is kept.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0, 1, 1, 0])
    threshold, _ = best_split(x, y, 2)
    assert threshold == 1.5
