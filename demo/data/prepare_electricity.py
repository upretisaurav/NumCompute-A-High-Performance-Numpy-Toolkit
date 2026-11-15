"""
Download the Electricity (elec2) dataset from OpenML and save it as
electricity.csv: drop "date", map UP -> 1 / DOWN -> 0, and blank 2% of
feature cells (seed 0) to exercise the imputer.
Run from the repo root: python demo/data/prepare_electricity.py
"""

import csv
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://www.openml.org/data/get_csv/2419/electricity-normalized.arff"
OUT = Path(__file__).parent / "electricity.csv"


def main():
    text = urllib.request.urlopen(URL).read().decode().splitlines()
    rows = list(csv.reader(text))
    header, rows = rows[0], rows[1:]

    features = header[1:-1]  # drop "date" and "class"
    X = np.array([[float(v) for v in row[1:-1]] for row in rows])
    y = np.array([1 if row[-1] == "UP" else 0 for row in rows])

    rng = np.random.default_rng(0)
    X[rng.random(X.shape) < 0.02] = np.nan

    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(features + ["class"])
        for x_row, label in zip(X, y, strict=True):
            writer.writerow(
                ["" if np.isnan(v) else f"{v:.6g}" for v in x_row] + [label]
            )

    print(f"Saved {len(y)} rows to {OUT}")


if __name__ == "__main__":
    main()
