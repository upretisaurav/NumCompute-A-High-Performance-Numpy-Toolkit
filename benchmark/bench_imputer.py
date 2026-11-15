"""
SimpleImputer vs a Python loop (fixed 2.1 benchmark).

2.1 bugs: test_pyloop = test_arr aliased the arrays, so the check
compared the loop output with itself; the loop filled every cell, not
only NaN; timings went to a leftover index; the input was 5x5.
Run: python -m benchmark.bench_imputer
"""

import numpy as np

from benchmark.common import median_time, save_table
from numcompute_stream.preprocessing import SimpleImputer


def impute_loop(X, fill_values):
    """Replace NaN cell by cell with the column's fill value."""

    out = X.copy()
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if np.isnan(out[i, j]):
                out[i, j] = fill_values[j]
    return out


def column_means_loop(X):
    """Mean of the non-NaN values of each column, with loops."""

    means = []
    for j in range(X.shape[1]):
        total, count = 0.0, 0
        for i in range(X.shape[0]):
            if not np.isnan(X[i, j]):
                total += X[i, j]
                count += 1
        means.append(total / count if count > 0 else 0.0)
    return np.array(means)


def run_vectorised(X, strategy):
    return SimpleImputer(strategy=strategy).fit_transform(X.copy())


def run_loop(X, strategy):
    X_copy = X.copy()
    if strategy == "mean":
        fill = column_means_loop(X_copy)
    else:
        fill = np.zeros(X.shape[1])
    return impute_loop(X_copy, fill)


def main():
    rng = np.random.default_rng(0)
    rows = []

    for n in (1000, 10000, 100000):
        X = rng.normal(size=(n, 10))
        X[rng.random(X.shape) < 0.1] = np.nan

        for strategy in ("constant", "mean"):
            # Same input, independent copies: check before timing.
            np.testing.assert_allclose(
                run_vectorised(X, strategy), run_loop(X, strategy)
            )
            assert not np.isnan(run_vectorised(X, strategy)).any()

            t_vec = median_time(run_vectorised, X, strategy)
            t_loop = median_time(run_loop, X, strategy)
            rows.append(
                {
                    "n_samples": n,
                    "strategy": strategy,
                    "loop_ms": round(t_loop * 1000, 2),
                    "vectorised_ms": round(t_vec * 1000, 3),
                    "speedup": round(t_loop / t_vec, 1),
                }
            )

    save_table(rows, "imputer")


if __name__ == "__main__":
    main()
