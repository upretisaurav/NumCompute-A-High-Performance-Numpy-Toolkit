"""Loop vs vectorised split search. Run: python -m benchmark.bench_split"""

import matplotlib.pyplot as plt
import numpy as np

from benchmark.common import RESULTS, median_time, save_table
from numcompute_stream.tree import best_split, best_split_loop


def search_all_features(X, y, split_func):
    """Best (feature, threshold, gain) over all columns."""

    best = (None, None, 0.0)
    for f in range(X.shape[1]):
        threshold, gain = split_func(X[:, f], y, 2)
        if threshold is not None and gain > best[2]:
            best = (f, threshold, gain)
    return best


def main():
    rng = np.random.default_rng(0)
    rows = []

    for n_features in (1, 5):
        for n in (500, 2000, 10000):
            X = rng.normal(size=(n, n_features))
            y = (X.sum(axis=1) + rng.normal(scale=1.0, size=n) > 0).astype(int)

            vec = search_all_features(X, y, best_split)
            loop = search_all_features(X, y, best_split_loop)
            assert vec[0] == loop[0] and vec[1] == loop[1], "results differ"
            assert np.isclose(vec[2], loop[2])

            t_vec = median_time(search_all_features, X, y, best_split)
            t_loop = median_time(search_all_features, X, y, best_split_loop)
            rows.append(
                {
                    "n_samples": n,
                    "n_features": n_features,
                    "loop_ms": round(t_loop * 1000, 2),
                    "vectorised_ms": round(t_vec * 1000, 2),
                    "speedup": round(t_loop / t_vec, 1),
                }
            )

    save_table(rows, "split")

    fig, ax = plt.subplots(figsize=(7, 4))
    for n_features, marker in ((1, "o"), (5, "s")):
        sub = [r for r in rows if r["n_features"] == n_features]
        n = [r["n_samples"] for r in sub]
        ax.plot(
            n,
            [r["loop_ms"] for r in sub],
            marker=marker,
            label=f"loop, {n_features} feature(s)",
        )
        ax.plot(
            n,
            [r["vectorised_ms"] for r in sub],
            marker=marker,
            linestyle="--",
            label=f"vectorised, {n_features} feature(s)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Median time (ms)")
    ax.set_title("Best split search: loop vs vectorised")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "split.png", dpi=150)


if __name__ == "__main__":
    main()
