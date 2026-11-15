"""Tests for numcompute_stream.io (2.1 fix: chunked input)."""

import numpy as np
import pytest

from numcompute_stream.io import iter_chunks, load_csv, read_csv_chunks


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["a,b,label"] + [f"{i},{i * 2},{i % 2}" for i in range(10)]
    rows[3] = "2,,0"  # missing cell
    path.write_text("\n".join(rows) + "\n")
    return path


def test_load_csv_splits_target_and_fills_missing(csv_file):
    X, y = load_csv(csv_file)
    assert X.shape == (10, 2)
    assert y.shape == (10,)
    assert np.isnan(X[2, 1])
    np.testing.assert_array_equal(y, np.arange(10) % 2)


def test_load_csv_target_col_none_and_out_of_range(csv_file):
    assert load_csv(csv_file, target_col=None).shape == (10, 3)
    with pytest.raises(ValueError):
        load_csv(csv_file, target_col=5)


def test_load_csv_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("a,b\n")
    with pytest.raises(ValueError):
        load_csv(path)


def test_iter_chunks_cover_all_rows_once_in_order():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    chunks = list(iter_chunks(X, y, chunk_size=4))
    assert [len(c[1]) for c in chunks] == [4, 4, 2]  # last chunk smaller
    np.testing.assert_array_equal(np.concatenate([c[1] for c in chunks]), y)
    np.testing.assert_array_equal(np.vstack([c[0] for c in chunks]), X)


def test_chunk_size_larger_than_data_gives_one_chunk():
    chunks = list(iter_chunks(np.zeros((3, 2)), chunk_size=10))
    assert len(chunks) == 1
    assert chunks[0].shape == (3, 2)


@pytest.mark.parametrize("bad", [0, -2, 1.5])
def test_bad_chunk_size_raises(bad):
    with pytest.raises(ValueError):
        list(iter_chunks(np.zeros((3, 2)), chunk_size=bad))


def test_iter_chunks_length_mismatch_raises():
    with pytest.raises(ValueError):
        list(iter_chunks(np.zeros((3, 2)), np.zeros(4), chunk_size=2))


def test_read_csv_chunks_matches_load_csv(csv_file):
    X, y = load_csv(csv_file)
    chunks = list(read_csv_chunks(csv_file, chunk_size=3))
    assert [len(c[1]) for c in chunks] == [3, 3, 3, 1]
    np.testing.assert_array_equal(np.vstack([c[0] for c in chunks]), X)
    np.testing.assert_array_equal(np.concatenate([c[1] for c in chunks]), y)
