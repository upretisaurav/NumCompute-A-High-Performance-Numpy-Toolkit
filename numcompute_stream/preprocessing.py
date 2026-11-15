"""
Transformers for numeric and categorical data, in a fit-then-transform
style: StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer.
"""

import numpy as np

from numcompute_stream.utils import check_array, check_is_fitted

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


class StandardScaler:
    """Standardise features to zero mean and unit variance."""

    def fit(self, X):
        """
        Learn mean_ and scale_ (1 for constant columns). X: (n_samples, n_features).
        """

        X = check_array(X)

        self.n_features_in_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = _safe_scale(np.std(X, axis=0), self.mean_)

        return self

    def transform(self, X):
        """
        Return (X - mean_) / scale_.

        Raises NotFittedError before fit, ValueError on a feature mismatch.
        """

        check_is_fitted(self, ["mean_", "scale_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but StandardScaler was "
                f"fitted with {self.n_features_in_}."
            )

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
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

    def fit(self, X):
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

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but MinMaxScaler was "
                f"fitted with {self.n_features_in_}."
            )

        safe_range = np.where(self.data_range_ == 0, 1.0, self.data_range_)

        low, high = self.feature_range
        X_std = (X - self.data_min_) / safe_range

        return X_std * (high - low) + low

    def fit_transform(self, X):
        """Same as fit(X).transform(X)."""

        return self.fit(X).transform(X)


class OneHotEncoder:
    """Encode categorical columns as 0/1 indicator columns."""

    def __init__(self, handle_unknown="error"):
        """
        handle_unknown: "error" (default) raises on unknown categories; "ignore" gives zeros.
        """

        if handle_unknown not in ("error", "ignore"):
            raise ValueError(
                f"handle_unknown must be 'error' or 'ignore', got {handle_unknown!r}."
            )

        self.handle_unknown = handle_unknown

    def fit(self, X):
        """Learn the sorted categories of each column."""

        X = check_array(X, dtype=None)

        self.n_features_in_ = X.shape[1]

        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]

        return self

    def transform(self, X):
        """
        Return a float array of shape (n_samples, total categories).

        Raises ValueError on a feature mismatch, or on an unknown category when
        handle_unknown="error".
        """

        check_is_fitted(self, ["categories_", "n_features_in_"])

        X = check_array(X, dtype=None)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneHotEncoder was "
                f"fitted with {self.n_features_in_}."
            )

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

    def fit_transform(self, X):
        """Same as fit(X).transform(X)."""

        return self.fit(X).transform(X)


class SimpleImputer:
    """Replace NaN with a constant."""

    def __init__(self, fill_value=0.0):
        """
        fill_value: number used for NaN (default 0.0). Raises TypeError if not a number.
        """

        if not isinstance(fill_value, (int, float, np.number)):
            raise TypeError("fill_value must be a number.")

        self.fill_value = fill_value

    def fit(self, X):
        """Record the number of features."""

        X = check_array(X, allow_nan=True)

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        """Return a copy of X with NaN replaced by fill_value."""

        check_is_fitted(self, ["n_features_in_"])

        X = check_array(X, allow_nan=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SimpleImputer was "
                f"fitted with {self.n_features_in_}."
            )

        return np.where(np.isnan(X), self.fill_value, X)

    def fit_transform(self, X):
        """Same as fit(X).transform(X)."""

        return self.fit(X).transform(X)
