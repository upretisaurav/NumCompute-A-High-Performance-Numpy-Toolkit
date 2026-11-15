"""Tests for numcompute_stream.preprocessing."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler


# 2.1 fix: tiny-scale standardisation
def test_tiny_scale_column_is_standardised():
    # 2.1 treated this column as constant (np.isclose(std, 0)) and left it as is.
    X = np.array([[1e-12], [2e-12], [3e-12]])
    out = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(out.ravel(), [-1.2247449, 0.0, 1.2247449], rtol=1e-6)


@pytest.mark.parametrize("value", [0.0, 0.1, 5.0, 1e8])
def test_constant_column_gives_finite_zeros(value):
    X = np.full((50, 1), value)
    out = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_streaming_mean_var_match_full_data():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=50, scale=3, size=(500, 4))
    scaler = StandardScaler()
    for start in range(0, 500, 64):
        scaler.partial_fit(X[start : start + 64])
    np.testing.assert_allclose(scaler.mean_, X.mean(axis=0))
    np.testing.assert_allclose(scaler.var_, X.var(axis=0))
    np.testing.assert_allclose(
        scaler.transform(X), StandardScaler().fit_transform(X), atol=1e-10
    )


def test_scaler_ignores_nan_and_keeps_it():
    X = np.array([[1.0, np.nan], [3.0, 2.0], [np.nan, 4.0]])
    scaler = StandardScaler().fit(X)
    np.testing.assert_allclose(scaler.mean_, [2.0, 3.0])
    assert np.isnan(scaler.transform(X)[2, 0])


def test_zero_variance_chunk_then_varied_chunk():
    scaler = StandardScaler().partial_fit(np.full((10, 1), 7.0))
    assert scaler.scale_[0] == 1.0
    scaler.partial_fit(np.array([[5.0], [9.0]]))
    assert scaler.scale_[0] > 0 and scaler.scale_[0] != 1.0


def test_scaler_feature_mismatch_raises():
    scaler = StandardScaler().fit(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        scaler.partial_fit(np.zeros((3, 3)))


def test_imputer_fills_nan_with_running_mean():
    imputer = SimpleImputer()
    imputer.partial_fit(np.array([[1.0, np.nan], [3.0, np.nan]]))
    imputer.partial_fit(np.array([[5.0, 10.0]]))
    np.testing.assert_allclose(imputer.statistics_, [3.0, 10.0])
    out = imputer.transform(np.array([[np.nan, np.nan]]))
    np.testing.assert_allclose(out, [[3.0, 10.0]])


def test_imputer_column_with_no_values_uses_fill_value():
    imputer = SimpleImputer(fill_value=-1.0).fit(np.array([[np.nan], [np.nan]]))
    np.testing.assert_array_equal(imputer.transform([[np.nan]]), [[-1.0]])


def test_imputer_constant_strategy():
    out = SimpleImputer(strategy="constant", fill_value=9).fit_transform(
        [[np.nan, 1.0]]
    )
    np.testing.assert_array_equal(out, [[9.0, 1.0]])


def test_one_hot_new_category_gets_its_own_column():
    enc = OneHotEncoder()
    enc.partial_fit(np.array([["red"], ["blue"]]))
    enc.partial_fit(np.array([["green"], ["red"]]))
    np.testing.assert_array_equal(enc.categories_[0], ["blue", "red", "green"])
    out = enc.transform(np.array([["green"]]))
    np.testing.assert_array_equal(out, [[0.0, 0.0, 1.0]])


def test_one_hot_expanding_keeps_old_columns_unchanged():
    X1 = np.array([["b", "x"], ["a", "y"]])
    enc = OneHotEncoder().partial_fit(X1)
    before = enc.transform(X1)
    enc.partial_fit(np.array([["c", "x"]]))
    after = enc.transform(X1)
    # "c" is added at the end of feature 0's columns; a, b, x, y keep their order.
    np.testing.assert_array_equal(after[:, [0, 1, 3, 4]], before)
    np.testing.assert_array_equal(after[:, 2], [0.0, 0.0])


def test_one_hot_new_category_in_chunk_2_keeps_width_when_not_expanding():
    enc = OneHotEncoder(expand=False)
    enc.partial_fit(np.array([["red"], ["blue"]]))
    width = enc.transform(np.array([["red"]])).shape[1]

    chunk2 = np.array([["green"], ["red"]])
    enc.partial_fit(chunk2)
    out = enc.transform(chunk2)

    assert out.shape[1] == width == 2
    np.testing.assert_array_equal(out[0], [0.0, 0.0])  # unseen -> zeros
    assert enc.unseen_categories_[0] == {"green"}


def test_one_hot_handle_unknown_error():
    enc = OneHotEncoder(handle_unknown="error").fit(np.array([["a"], ["b"]]))
    with pytest.raises(ValueError):
        enc.transform(np.array([["c"]]))
