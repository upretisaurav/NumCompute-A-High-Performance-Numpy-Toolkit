"""Tests for numcompute_stream.tree."""

import numpy as np
import pytest

from numcompute_stream.exceptions import NotFittedError
from numcompute_stream.tree import (
    DecisionTreeClassifier,
    best_split,
    best_split_loop,
    entropy,
    gini,
)

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


# ---------------------------------------------------------------------------
# 1.3 Loop version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_loop_and_vectorised_give_same_split(criterion, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=200)
    y = (x + rng.normal(scale=0.8, size=200) > 0).astype(int)
    t_vec, g_vec = best_split(x, y, 2, criterion)
    t_loop, g_loop = best_split_loop(x, y, 2, criterion)
    assert t_vec == t_loop
    assert g_vec == pytest.approx(g_loop)


def test_loop_and_vectorised_agree_with_duplicates_and_3_classes():
    rng = np.random.default_rng(5)
    x = rng.integers(0, 10, size=300).astype(float)
    y = rng.integers(0, 3, size=300)
    assert best_split(x, y, 3)[0] == best_split_loop(x, y, 3)[0]


# ---------------------------------------------------------------------------
# 1.4 / 1.5 Building, fit, predict
# ---------------------------------------------------------------------------


def make_blobs(n=200, seed=0):
    """Two well separated clusters in 2D."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n // 2, 2)), rng.normal(6, 1, (n // 2, 2))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_separable_data_gives_full_train_accuracy():
    X, y = make_blobs()
    tree = DecisionTreeClassifier().fit(X, y)
    assert np.all(tree.predict(X) == y)


def test_max_depth_one_gives_a_stump():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = rng.integers(0, 2, size=100)
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert not tree.tree_.is_leaf()
    assert tree.tree_.left.is_leaf() and tree.tree_.right.is_leaf()
    assert tree.n_nodes_ == 3


def test_min_samples_split_stops_splitting():
    X, y = make_blobs(n=20)
    tree = DecisionTreeClassifier(min_samples_split=50).fit(X, y)
    assert tree.tree_.is_leaf()


def test_multiclass_and_string_labels():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [20.0], [21.0]])
    y = np.array(["a", "a", "a", "b", "b", "c", "c"])
    tree = DecisionTreeClassifier().fit(X, y)
    assert list(tree.classes_) == ["a", "b", "c"]
    assert np.all(tree.predict(X) == y)


def test_predict_proba_shape_and_rows_sum_to_one():
    X, y = make_blobs()
    proba = DecisionTreeClassifier(max_depth=2).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_majority_tie_goes_to_smallest_class():
    X = np.array([[1.0], [1.0]])
    tree = DecisionTreeClassifier().fit(X, np.array([1, 0]))
    assert tree.predict(X)[0] == 0


def test_feature_tie_picks_lowest_feature_index():
    # Both columns are identical, so both give the same gain.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.column_stack([x, x])
    tree = DecisionTreeClassifier(max_depth=1).fit(X, np.array([0, 0, 1, 1]))
    assert tree.tree_.feature == 0


def test_nan_values_are_filled_with_training_median():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [10.0, np.nan], [11.0, np.nan]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    np.testing.assert_array_equal(tree.fill_values_, [6.0, 0.0])  # all-NaN -> 0
    # NaN at predict time goes to the median (6.0 -> left of 6.0 is <= 6).
    assert tree.predict(np.array([[np.nan, np.nan]]))[0] == 0
    assert np.all(tree.predict(X) == y)


def test_max_features_sqrt_is_reproducible_with_seed():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(150, 9))
    y = (X[:, 4] > 0).astype(int)
    a = DecisionTreeClassifier(max_features="sqrt", random_state=7).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=7).fit(X, y)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def test_predict_loop_matches_vectorised_predict():
    X, y = make_blobs(n=300, seed=2)
    X = X + np.random.default_rng(9).normal(scale=2.0, size=X.shape)
    tree = DecisionTreeClassifier(max_depth=4).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), tree.predict_loop(X))


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        DecisionTreeClassifier().predict(np.zeros((2, 2)))


@pytest.mark.parametrize(
    "X, y",
    [
        (np.zeros(4), np.zeros(4)),  # X 1D
        (np.zeros((4, 2)), np.zeros((4, 1))),  # y 2D
        (np.zeros((4, 2)), np.zeros(3)),  # row mismatch
        (np.zeros((0, 2)), np.zeros(0)),  # empty
        (np.array([[1.0, np.inf]]), np.zeros(1)),  # Inf
    ],
)
def test_fit_shape_errors(X, y):
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(X, y)


def test_non_numeric_x_raises_type_error():
    with pytest.raises(TypeError):
        DecisionTreeClassifier().fit([["a"], ["b"]], [0, 1])


def test_predict_with_wrong_number_of_features_raises():
    X, y = make_blobs()
    tree = DecisionTreeClassifier().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        tree.predict(np.zeros((3, 5)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": 0},
        {"min_samples_split": 1},
        {"max_features": "log"},
        {"buffer_size": 0},
        {"criterion": "mse"},
    ],
)
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kwargs)
