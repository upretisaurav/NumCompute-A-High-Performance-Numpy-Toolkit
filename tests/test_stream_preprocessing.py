"""Tests for numcompute_stream.preprocessing."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import StandardScaler


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
