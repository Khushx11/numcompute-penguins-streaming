"""
tests for numcompute_stream/io.py
"""

import numpy as np
import pytest
import tempfile
import os
from numcompute_stream.io import read_csv, read_csv_chunked, write_csv, make_stream_chunks


def _write_temp_csv(content):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_read_csv_basic():
    path = _write_temp_csv("a,b,c\n1,2,3\n4,5,6\n")
    headers, data = read_csv(path)
    assert headers == ["a", "b", "c"]
    assert data.shape == (2, 3)
    os.unlink(path)


def test_read_csv_missing_values():
    path = _write_temp_csv("a,b\n1,\n,3\n")
    headers, data = read_csv(path)
    assert np.isnan(data[0, 1])
    assert np.isnan(data[1, 0])
    os.unlink(path)


def test_read_csv_chunked_yields_correct_shape():
    path = _write_temp_csv("a,b\n" + "\n".join(f"{i},{i*2}" for i in range(20)))
    chunks = list(read_csv_chunked(path, chunk_size=7))
    total_rows = sum(c.shape[0] for _, c in chunks)
    assert total_rows == 20
    os.unlink(path)


def test_read_csv_not_found():
    with pytest.raises(FileNotFoundError):
        read_csv("/nonexistent/path/file.csv")


def test_write_csv_roundtrip():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    path = tempfile.mktemp(suffix=".csv")
    write_csv(path, data, headers=["x", "y"])
    headers, data_back = read_csv(path)
    assert headers == ["x", "y"]
    np.testing.assert_allclose(data, data_back, atol=1e-5)
    os.unlink(path)


def test_make_stream_chunks_count():
    X = np.ones((100, 3))
    y = np.zeros(100)
    chunks = make_stream_chunks(X, y, chunk_size=30)
    assert len(chunks) == 4  # 30+30+30+10


def test_make_stream_chunks_last_smaller():
    X = np.ones((25, 2))
    y = np.zeros(25)
    chunks = make_stream_chunks(X, y, chunk_size=10)
    assert chunks[-1][0].shape[0] == 5


def test_make_stream_chunks_length_mismatch():
    with pytest.raises(ValueError):
        make_stream_chunks(np.ones((10, 2)), np.zeros(9), chunk_size=5)
