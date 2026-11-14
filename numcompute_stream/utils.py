"""Shared helpers used across the package."""

import numpy as np

from numcompute_stream.exceptions import NotFittedError


def check_array(X, dtype=float, allow_nan=False):

    """
    Convert X to a non-empty 2D array of shape (n_samples, n_features).

    Raises TypeError if X can't be converted to dtype, ValueError if it is
    not 2D, empty, or has NaN/Inf while allow_nan is False.
    """

    try:
        X_arr = np.asarray(X, dtype=dtype)
    except (TypeError, ValueError) as e:
        if "inhomogeneous" in str(e):
            raise ValueError(
                "Input forms a ragged array with inconsistent row lengths."
            ) from e
        dtype_name = getattr(dtype, "__name__", dtype)
        raise TypeError(f"Input cannot be converted to {dtype_name}.") from e

    if X_arr.ndim == 0:
        raise ValueError("Expected a 2D array, got a scalar instead.")
    if X_arr.ndim == 1:
        raise ValueError(
            "Expected a 2D array, got a 1D array instead. Use "
            "X.reshape(-1, 1) for a single feature or X.reshape(1, -1) "
            "for a single sample."
        )
    if X_arr.ndim > 2:
        raise ValueError(
            f"Expected a 2D array, got a {X_arr.ndim}D array instead."
        )

    if X_arr.size == 0:
        raise ValueError("Input array is empty (zero rows or columns).")

    # Only numeric arrays can hold NaN/Inf, so string arrays skip this.
    if not allow_nan and np.issubdtype(X_arr.dtype, np.number):
        if not np.all(np.isfinite(X_arr)):
            raise ValueError(
                "Input contains NaN or Inf. Set allow_nan=True to allow them."
            )

    return X_arr


def check_is_fitted(estimator, attributes):

    """Raise NotFittedError if any of the given attributes is missing."""

    if isinstance(attributes, str):
        attributes = [attributes]

    if not all(hasattr(estimator, attr) for attr in attributes):
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' before using this estimator."
        )

def check_1d_array(X, dtype=float, allow_nan=False, allow_empty_arr=False):
    """
    Convert X to a 1D array of shape (n_samples,).

    Raises TypeError if it can't be converted, ValueError if it is not 1D,
    empty (unless allowed), or has NaN/Inf while allow_nan is False.
    """

    try:
        X_arr = np.asarray(X, dtype=dtype)
    except (TypeError, ValueError) as e:
        if "inhomogeneous" in str(e):
            raise ValueError(
                "Input forms a ragged array with inconsistent row lengths."
            ) from e

        dtype_name = getattr(dtype, "__name__", dtype)
        raise TypeError(
            f"Input cannot be converted to {dtype_name}."
        ) from e

    if X_arr.ndim != 1:
        raise ValueError(
            f"Expected a 1D array, got a {X_arr.ndim}D array instead."
        )

    if X_arr.size == 0 and not allow_empty_arr:
        raise ValueError("Input array is empty.")

    # Only numeric arrays can hold NaN/Inf, so string arrays skip this.
    if not allow_nan and np.issubdtype(X_arr.dtype, np.number):
        if not np.all(np.isfinite(X_arr)):
            raise ValueError(
                "Input contains NaN or Inf. Set allow_nan=True to allow them."
            )

    return X_arr