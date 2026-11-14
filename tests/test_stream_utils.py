"""Boundary tests for numcompute_stream.utils."""

import numpy as np
import pytest

from numcompute_stream.exceptions import NotFittedError
from numcompute_stream.utils import (
    check_is_fitted,
    check_positive_int,
    check_X_y,
    chunk_moments,
    logsumexp,
    merge_moments,
    safe_divide,
    sigmoid,
    softmax,
)


def test_safe_divide_by_zero_gives_fill():
    out = safe_divide([1.0, 0.0, 6.0], [0.0, 0.0, 3.0])
    np.testing.assert_array_equal(out, [0.0, 0.0, 2.0])


def test_logsumexp_huge_and_tiny_values():
    assert logsumexp([1000.0, 1000.0]) == pytest.approx(1000 + np.log(2))
    assert logsumexp([-1000.0, -1000.0]) == pytest.approx(-1000 + np.log(2))
    assert logsumexp([-np.inf, -np.inf]) == -np.inf


def test_logsumexp_empty_raises():
    with pytest.raises(ValueError):
        logsumexp([])


def test_softmax_huge_values_sum_to_one():
    p = softmax(np.array([1e4, 1e4, -1e4]))
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(1.0)


def test_sigmoid_no_overflow():
    np.testing.assert_allclose(sigmoid([-1000.0, 0.0, 1000.0]), [0.0, 0.5, 1.0])


def test_merge_moments_matches_full_data_with_nans():
    rng = np.random.default_rng(1)
    X = rng.normal(loc=1e6, scale=1e-3, size=(300, 3))
    X[rng.random(X.shape) < 0.1] = np.nan

    count, mean, M2 = np.zeros(3), np.zeros(3), np.zeros(3)
    for start in range(0, 300, 37):
        chunk = chunk_moments(X[start : start + 37])
        count, mean, M2 = merge_moments(count, mean, M2, *chunk)

    np.testing.assert_allclose(mean, np.nanmean(X, axis=0))
    np.testing.assert_allclose(M2 / count, np.nanvar(X, axis=0), rtol=1e-6)


def test_chunk_moments_all_nan_column():
    count, mean, M2 = chunk_moments(np.array([[np.nan, 1.0], [np.nan, 3.0]]))
    np.testing.assert_array_equal(count, [0, 2])
    np.testing.assert_array_equal(mean, [0.0, 2.0])


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_check_positive_int_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        check_positive_int(bad, "chunk_size")


def test_check_x_y_errors():
    with pytest.raises(ValueError):
        check_X_y(np.zeros((3, 2)), [0, 1])
    with pytest.raises(ValueError):
        check_X_y([[1.0, np.inf]], [0])
    with pytest.raises(TypeError):
        check_X_y([["a", "b"]], [0])


def test_check_is_fitted_raises():
    with pytest.raises(NotFittedError):
        check_is_fitted(object(), "tree_")
