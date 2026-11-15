"""
Final accuracy, retrains and retrain time for several buffer sizes on
the Electricity stream. Run: python -m benchmark.bench_buffer_size
"""

import matplotlib.pyplot as plt

from benchmark.common import RESULTS, save_table
from numcompute_stream.io import iter_chunks, load_csv
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier

BUFFER_SIZES = (100, 250, 500, 1000, 2000)
CHUNK_SIZE = 250


def main():
    X, y = load_csv("demo/data/electricity.csv")
    rows = []

    for buffer_size in BUFFER_SIZES:
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size)),
            ]
        )
        history = StreamTrainer(pipe).run(iter_chunks(X, y, CHUNK_SIZE))
        rows.append(
            {
                "buffer_size": buffer_size,
                "final_cumulative_accuracy": round(
                    history["cumulative_accuracy"][-1], 4
                ),
                "n_retrains": sum(history["retrained"]),
                "total_retrain_time_s": round(sum(history["retrain_time"]), 3),
                "max_buffer_KB": round(max(history["buffer_bytes"]) / 1024, 1),
            }
        )

    save_table(rows, "buffer_size")

    sizes = [r["buffer_size"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(sizes, [r["final_cumulative_accuracy"] for r in rows], marker="o")
    ax1.set_xscale("log")
    ax1.set_xlabel("buffer_size")
    ax1.set_ylabel("Final cumulative accuracy")
    ax1.set_title("Accuracy vs buffer size")
    ax1.grid(alpha=0.3)
    ax2.bar([str(s) for s in sizes], [r["total_retrain_time_s"] for r in rows])
    ax2.set_xlabel("buffer_size")
    ax2.set_ylabel("Total retrain time (s)")
    ax2.set_title("Retrain cost vs buffer size")
    fig.tight_layout()
    fig.savefig(RESULTS / "buffer_size.png", dpi=150)


if __name__ == "__main__":
    main()
