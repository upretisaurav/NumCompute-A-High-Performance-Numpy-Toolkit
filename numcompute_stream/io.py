"""CSV loading and chunking for streaming. Missing cells become NaN."""

from itertools import islice

import numpy as np

from numcompute_stream.utils import check_positive_int


def _parse_lines(lines, delimiter):
    """Parse a list of CSV lines into a 2D float array."""

    data = np.genfromtxt(
        lines, dtype=float, delimiter=delimiter, filling_values=np.nan, ndmin=2
    )
    return data


def _split_target(data, target_col):
    """Split a 2D array into X and y (column target_col)."""

    if target_col is None:
        return data

    n_cols = data.shape[1]
    if not -n_cols <= target_col < n_cols:
        raise ValueError(
            f"target_col {target_col} is out of range for {n_cols} columns."
        )

    y = data[:, target_col]
    X = np.delete(data, target_col, axis=1)
    return X, y


def load_csv(csv_path, target_col=-1, skip_header=True, delimiter=","):
    """
    Load a numeric CSV and split off the target column.

    Returns (X, y) with shapes (n_rows, n_cols - 1) and (n_rows,), or only
    X if target_col is None. Raises ValueError if there are no data rows or
    target_col is out of range.
    """

    with open(csv_path) as f:
        lines = f.readlines()

    if skip_header:
        lines = lines[1:]
    lines = [line for line in lines if line.strip()]

    if len(lines) == 0:
        raise ValueError("The CSV file has no data rows.")

    return _split_target(_parse_lines(lines, delimiter), target_col)


def iter_chunks(X, y=None, chunk_size=100):
    """
    Yield consecutive (X_chunk, y_chunk) pairs (or X_chunk if y is None).

    The last chunk may be smaller. Raises ValueError if chunk_size is not a
    positive int or X and y have different lengths.
    """

    chunk_size = check_positive_int(chunk_size, "chunk_size")
    X = np.asarray(X)

    if y is not None:
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError(
                f"X and y have different numbers of rows: {len(X)} vs {len(y)}."
            )

    for start in range(0, len(X), chunk_size):
        end = start + chunk_size
        if y is None:
            yield X[start:end]
        else:
            yield X[start:end], y[start:end]


def read_csv_chunks(
    csv_path, chunk_size=100, target_col=-1, skip_header=True, delimiter=","
):
    """
    Like load_csv, but reads the file chunk_size lines at a time so only one
    chunk is in memory. Yields (X_chunk, y_chunk).
    """

    chunk_size = check_positive_int(chunk_size, "chunk_size")

    with open(csv_path) as f:
        if skip_header:
            next(f, None)

        while True:
            lines = [line for line in islice(f, chunk_size) if line.strip()]
            if len(lines) == 0:
                break
            yield _split_target(_parse_lines(lines, delimiter), target_col)
