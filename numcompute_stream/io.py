"""Loads numeric CSV files into NumPy arrays. Missing cells become NaN."""

import numpy as np


def load_csv(csv_path, file_delimiter=","):

    """
    Load a CSV file as a 2D float array of shape (n_rows, n_columns).

    Raises TypeError/ValueError for a bad delimiter, ValueError for an empty
    file, FileNotFoundError if the file does not exist.
    """

    if not isinstance(file_delimiter, str):
        raise TypeError(f"Delimiter: {file_delimiter} is not string.")

    if len(file_delimiter) != 1:
        raise ValueError("Delimiter must contain exactly one character.")

    csv_data = np.genfromtxt(
        csv_path,
        dtype=float,
        delimiter=file_delimiter,
        filling_values=np.nan,
        ndmin=2
    )

    if csv_data.size == 0:
        raise ValueError("The CSV file is empty.")

    return csv_data
