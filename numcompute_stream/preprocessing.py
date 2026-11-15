"""
Transformers for numeric and categorical data.

StandardScaler, SimpleImputer and OneHotEncoder also have partial_fit(X)
to update their state one chunk at a time. Every fit takes an optional y
(ignored) so the transformers work inside a Pipeline.
"""

import numpy as np

from numcompute_stream.utils import (
    check_array,
    check_is_fitted,
    chunk_moments,
    merge_moments,
    safe_divide,
)

# A std smaller than this fraction of |mean| is only rounding error, so
# the column is treated as constant.
_CONSTANT_TOL = 1e3 * np.finfo(float).eps


def _safe_scale(std, mean):
    """
    Return std, with 1.0 for constant columns.

    2.1 used np.isclose(std, 0), whose 1e-8 tolerance treated real tiny-scale
    columns (e.g. ~1e-12) as constant. This check is relative to the mean.
    """

    constant = std <= _CONSTANT_TOL * np.abs(mean)
    return np.where(constant, 1.0, std)


def _check_n_features(transformer, X):
    if X.shape[1] != transformer.n_features_in_:
        raise ValueError(
            f"X has {X.shape[1]} features, but {type(transformer).__name__} "
            f"was fitted with {transformer.n_features_in_}."
        )


class StandardScaler:
    """Standardise features to zero mean and unit variance. NaN is ignored."""

    def fit(self, X, y=None):
        """Learn mean_ and scale_ from scratch. X: (n_samples, n_features)."""

        if hasattr(self, "mean_"):
            del self.mean_
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """
        Update the running mean and variance with one chunk.

        Raises ValueError if the number of features changes.
        """

        X = check_array(X, allow_nan=True)

        if not hasattr(self, "mean_"):
            self.n_features_in_ = X.shape[1]
            self.n_samples_seen_ = np.zeros(X.shape[1])
            self.mean_ = np.zeros(X.shape[1])
            self._M2 = np.zeros(X.shape[1])
        else:
            _check_n_features(self, X)

        self.n_samples_seen_, self.mean_, self._M2 = merge_moments(
            self.n_samples_seen_, self.mean_, self._M2, *chunk_moments(X)
        )
        self.var_ = safe_divide(self._M2, self.n_samples_seen_)
        self.scale_ = _safe_scale(np.sqrt(self.var_), self.mean_)

        return self

    def transform(self, X):
        """
        Return (X - mean_) / scale_. NaN stays NaN.

        Raises NotFittedError before fit, ValueError on a feature mismatch.
        """

        check_is_fitted(self, ["mean_", "scale_"])
        X = check_array(X, allow_nan=True)
        _check_n_features(self, X)

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        """Same as fit(X).transform(X)."""

        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale each feature into feature_range using its min and max."""

    def __init__(self, feature_range=(0, 1)):
        """Raises ValueError if feature_range min is not below max."""

        low, high = feature_range

        if low >= high:
            raise ValueError("feature_range min must be less than max.")

        self.feature_range = (low, high)

    def fit(self, X, y=None):
        """Learn the min and max of each feature. X: (n_samples, n_features)."""

        X = check_array(X)

        self.n_features_in_ = X.shape[1]
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        return self

    def transform(self, X):
        """
        Scale X into feature_range (values are not clipped).

        Raises NotFittedError before fit, ValueError on a feature mismatch.
        """

        check_is_fitted(self, ["data_min_", "data_range_", "n_features_in_"])

        X = check_array(X)
        _check_n_features(self, X)

        safe_range = np.where(self.data_range_ == 0, 1.0, self.data_range_)

        low, high = self.feature_range
        X_std = (X - self.data_min_) / safe_range

        return X_std * (high - low) + low

    def fit_transform(self, X, y=None):
        """Same as fit(X).transform(X)."""

        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    Encode categorical columns as 0/1 indicator columns.

    Categories are fixed after the first chunk so the output width never
    changes (a growing width would break a model trained earlier). Later
    unseen categories become all zeros and are listed in unseen_categories_.
    """

    def __init__(self, handle_unknown="ignore"):
        """handle_unknown: "ignore" (default, unknown -> all zeros) or "error"."""

        if handle_unknown not in ("error", "ignore"):
            raise ValueError(
                f"handle_unknown must be 'error' or 'ignore', got {handle_unknown!r}."
            )

        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Learn the sorted categories of each column from scratch."""

        if hasattr(self, "categories_"):
            del self.categories_
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """Learn categories on the first call; afterwards only record unseen ones."""

        X = check_array(X, dtype=None)

        if not hasattr(self, "categories_"):
            self.n_features_in_ = X.shape[1]
            self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
            self.unseen_categories_ = [set() for _ in range(X.shape[1])]
            return self

        _check_n_features(self, X)
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            unknown = col[~np.isin(col, cats)]
            self.unseen_categories_[j].update(unknown.tolist())

        return self

    def transform(self, X):
        """
        Return a float array of shape (n_samples, total categories).

        Raises ValueError on a feature mismatch, or on an unknown category when
        handle_unknown="error".
        """

        check_is_fitted(self, ["categories_", "n_features_in_"])

        X = check_array(X, dtype=None)
        _check_n_features(self, X)

        blocks = []

        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            block = col[:, None] == cats

            unknown = ~block.any(axis=1)
            if self.handle_unknown == "error" and unknown.any():
                raise ValueError(
                    f"Unknown categories in feature {j}: {np.unique(col[unknown])}"
                )

            blocks.append(block)

        return np.hstack(blocks).astype(float)

    def fit_transform(self, X, y=None):
        """Same as fit(X).transform(X)."""

        return self.fit(X).transform(X)


class SimpleImputer:
    """Replace NaN with a per-column fill value."""

    def __init__(self, strategy="mean", fill_value=0.0):
        """
        strategy: "mean" (running column mean) or "constant" (fill_value).
        fill_value is also used for a column with no values seen yet.
        """

        if strategy not in ("mean", "constant"):
            raise ValueError(
                f"strategy must be 'mean' or 'constant', got {strategy!r}."
            )
        if not isinstance(fill_value, (int, float, np.number)):
            raise TypeError("fill_value must be a number.")

        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """Learn the fill values from scratch."""

        if hasattr(self, "statistics_"):
            del self.statistics_
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """
        Update the running per-column sum and count with one chunk.

        Raises ValueError if the number of features changes.
        """

        X = check_array(X, allow_nan=True)

        if not hasattr(self, "statistics_"):
            self.n_features_in_ = X.shape[1]
            self._sum = np.zeros(X.shape[1])
            self._count = np.zeros(X.shape[1])
        else:
            _check_n_features(self, X)

        self._sum += np.nansum(X, axis=0)
        self._count += np.sum(~np.isnan(X), axis=0)

        if self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1], float(self.fill_value))
        else:
            # Columns with no values yet fall back to fill_value.
            self.statistics_ = np.where(
                self._count > 0, safe_divide(self._sum, self._count), self.fill_value
            )

        return self

    def transform(self, X):
        """Return a copy of X with NaN replaced by statistics_."""

        check_is_fitted(self, ["statistics_"])

        X = check_array(X, allow_nan=True)
        _check_n_features(self, X)

        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X, y=None):
        """Same as fit(X).transform(X)."""

        return self.fit(X).transform(X)
