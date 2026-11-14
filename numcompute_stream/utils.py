"""
Shared helpers: input checks, numerically stable functions and the
running mean/variance helpers used by the scaler and StreamingStats.
"""

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

    if X_arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, got a {X_arr.ndim}D array instead. Use "
            "X.reshape(-1, 1) for a single feature."
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


def check_1d_array(y, dtype=None):
    """
    Convert labels to a non-empty 1D array of shape (n_samples,).

    Raises ValueError if y is not 1D, empty or contains NaN.
    """

    y_arr = np.asarray(y, dtype=dtype)

    if y_arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got a {y_arr.ndim}D array instead.")

    if y_arr.size == 0:
        raise ValueError("Input array is empty.")

    if np.issubdtype(y_arr.dtype, np.floating) and np.isnan(y_arr).any():
        raise ValueError("Labels contain NaN.")

    return y_arr


def check_X_y(X, y):  # noqa: N802 (X is the usual name for the data matrix)
    """
    Check X (NaN allowed, Inf not) and y together.

    Raises TypeError if X is not numeric, ValueError for bad shapes, Inf or
    a different number of rows.
    """

    X = check_array(X, allow_nan=True)
    y = check_1d_array(y)

    if np.isinf(X).any():
        raise ValueError("Input contains Inf.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have different numbers of rows: {X.shape[0]} vs {y.shape[0]}."
        )

    return X, y


def check_is_fitted(estimator, attributes, msg=None):
    """Raise NotFittedError if any of the given attributes is missing."""

    if isinstance(attributes, str):
        attributes = [attributes]

    if not all(hasattr(estimator, attr) for attr in attributes):
        if msg is None:
            msg = (
                f"This {type(estimator).__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )
        raise NotFittedError(msg)


def check_positive_int(value, name):
    """Return value as an int, or raise ValueError if it is not an int >= 1."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive int, got {value!r}.")
    if value < 1:
        raise ValueError(f"{name} must be a positive int, got {value}.")
    return int(value)


def safe_divide(a, b, fill=0.0):
    """Element-wise a / b, giving fill (default 0.0) where b == 0."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, fill, dtype=float)
    np.divide(a, b, out=out, where=b != 0)
    return out


def logsumexp(a, axis=None):
    """log(sum(exp(a))) without overflow (subtracts the max first)."""

    a = np.asarray(a, dtype=float)
    if a.size == 0:
        raise ValueError("logsumexp of an empty array is undefined.")

    a_max = np.max(a, axis=axis, keepdims=True)
    # If the max is inf or -inf, subtracting it would give nan.
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)

    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max

    if axis is None:
        return float(out.ravel()[0])
    return np.squeeze(out, axis=axis)


def softmax(a, axis=-1):
    """Numerically stable softmax; values along axis sum to 1."""

    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    e = np.exp(a - a_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(x):
    """Numerically stable 1 / (1 + exp(-x)); never overflows."""

    x = np.asarray(x, dtype=float)
    # exp(-|x|) is always <= 1, so it can't overflow.
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def chunk_moments(X):
    """
    Per-column (count, mean, M2) of one chunk, ignoring NaN.

    Each output has shape (n_features,). Empty columns give 0, 0, 0.
    """

    X = np.asarray(X, dtype=float)
    count = np.sum(~np.isnan(X), axis=0).astype(float)
    mean = safe_divide(np.nansum(X, axis=0), count)
    M2 = np.nansum((X - mean) ** 2, axis=0)
    return count, mean, M2


def merge_moments(count_a, mean_a, M2_a, count_b, mean_b, M2_b):
    """
    Merge the (count, mean, M2) of two groups (Chan et al.), so mean and
    variance can be updated chunk by chunk. Variance is M2 / count.
    """

    count = count_a + count_b
    delta = mean_b - mean_a
    mean = mean_a + delta * safe_divide(count_b, count)
    M2 = M2_a + M2_b + delta**2 * safe_divide(count_a * count_b, count)
    return count, mean, M2
