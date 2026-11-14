"""Tests for numcompute_stream.tree."""

import numpy as np
import pytest

from numcompute_stream.tree import entropy, gini

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
