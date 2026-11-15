"""Helpers shared by the benchmark scripts."""

import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

RESULTS = Path(__file__).parent / "results"


def median_time(func, *args, repeats=5):
    """Median time in seconds of `repeats` calls to func(*args)."""

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


def save_table(rows, name):
    """Save rows (list of dicts) as results/<name>.csv and .md."""

    RESULTS.mkdir(exist_ok=True)
    keys = list(rows[0].keys())

    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    table = "\n".join(lines)
    (RESULTS / f"{name}.md").write_text(table + "\n")
    print(table)
