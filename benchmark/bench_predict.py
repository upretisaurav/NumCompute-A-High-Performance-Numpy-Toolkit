"""Per-row loop vs vectorised predict. Run: python -m benchmark.bench_predict"""

import numpy as np

from benchmark.common import median_time, save_table
from numcompute_stream.tree import DecisionTreeClassifier


def main():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(5000, 5))
    y_train = (X_train[:, 0] * X_train[:, 1] + X_train[:, 2] > 0).astype(int)
    tree = DecisionTreeClassifier(max_depth=8).fit(X_train, y_train)

    rows = []
    for n in (1000, 10000, 50000):
        X = rng.normal(size=(n, 5))
        np.testing.assert_array_equal(tree.predict(X), tree.predict_loop(X))

        t_vec = median_time(tree.predict, X)
        t_loop = median_time(tree.predict_loop, X)
        rows.append(
            {
                "n_samples": n,
                "tree_nodes": tree.n_nodes_,
                "loop_ms": round(t_loop * 1000, 2),
                "vectorised_ms": round(t_vec * 1000, 2),
                "speedup": round(t_loop / t_vec, 1),
            }
        )

    save_table(rows, "predict")


if __name__ == "__main__":
    main()
