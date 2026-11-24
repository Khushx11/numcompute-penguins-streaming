import numpy as np
import pytest
from pathlib import Path
from numcompute.io import read_csv, read_csv_chunked, write_csv


# ── fixtures ──────
@pytest.fixture
def simple_csv(tmp_path):
    # clean 3x3 CSV with a header row, no missing values
    f = tmp_path / "simple.csv"
    f.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
    return f


@pytest.fixture
def missing_csv(tmp_path):
    # has empty cells, "nan", and "null" to test missing value handling
    f = tmp_path / "missing.csv"
    f.write_text("a,b,c\n1,,3\nnan,5,null\n7,8,\n")
    return f


@pytest.fixture
def no_header_csv(tmp_path):
    # raw data only, no column names row
    f = tmp_path / "noheader.csv"
    f.write_text("1,2,3\n4,5,6\n")
    return f


# ── read_csv ────

def test_read_csv_shape(simple_csv):
    # 3 data rows, 3 columns -> shape should be (3, 3)
    headers, data = read_csv(simple_csv)
    assert data.shape == (3, 3)


def test_read_csv_headers(simple_csv):
    # header row should come back as a plain list of strings
    headers, data = read_csv(simple_csv)
    assert headers == ["a", "b", "c"]


def test_read_csv_values(simple_csv):
    # spot-check first and last rows to make sure values parsed correctly
    _, data = read_csv(simple_csv)
    assert np.allclose(data[0], [1.0, 2.0, 3.0])
    assert np.allclose(data[2], [7.0, 8.0, 9.0])


def test_read_csv_missing_becomes_nan(missing_csv):
    # empty string, "nan", and "null" should all land as np.nan
    _, data = read_csv(missing_csv)
    assert np.isnan(data[0, 1])
    assert np.isnan(data[1, 0])
    assert np.isnan(data[1, 2])
    assert np.isnan(data[2, 2])


def test_read_csv_no_header(no_header_csv):
    # with has_header=False, headers should be empty and all rows treated as data
    headers, data = read_csv(no_header_csv, has_header=False)
    assert headers == []
    assert data.shape == (2, 3)


def test_read_csv_file_not_found():
    # should raise FileNotFoundError for a path that doesn't exist
    with pytest.raises(FileNotFoundError):
        read_csv("definitely_does_not_exist.csv")


def test_read_csv_tab_delimiter(tmp_path):
    # passing delimiter="\t" should correctly parse tab-separated files
    f = tmp_path / "tab.csv"
    f.write_text("a\tb\n1\t2\n3\t4\n")
    headers, data = read_csv(f, delimiter="\t")
    assert headers == ["a", "b"]
    assert data.shape == (2, 2)


def test_read_csv_single_row(tmp_path):
    # edge case: only one data row should give shape (1, n_cols)
    f = tmp_path / "one.csv"
    f.write_text("x,y\n1.5,2.5\n")
    _, data = read_csv(f)
    assert data.shape == (1, 2)
    assert np.isclose(data[0, 0], 1.5)


def test_read_csv_all_missing(tmp_path):
    # every cell is a missing token, entire array should be NaN
    f = tmp_path / "allnan.csv"
    f.write_text("a,b\n,\nnan,null\n")
    _, data = read_csv(f)
    assert np.all(np.isnan(data))


# ── read_csv_chunked ──
def test_read_csv_chunked_total_rows(simple_csv):
    # stacking all chunks should give the same total rows as reading at once
    all_rows = np.vstack([chunk for _, chunk in read_csv_chunked(simple_csv, chunk_size=2)])
    assert all_rows.shape == (3, 3)


def test_read_csv_chunked_chunk_sizes(simple_csv):
    # 3 rows with chunk_size=2 -> first chunk (2,3), second chunk (1,3)
    chunks = [chunk for _, chunk in read_csv_chunked(simple_csv, chunk_size=2)]
    assert chunks[0].shape == (2, 3)
    assert chunks[1].shape == (1, 3)


def test_read_csv_chunked_headers_consistent(simple_csv):
    # every chunk should carry the same header list, not just the first one
    headers_seen = [h for h, _ in read_csv_chunked(simple_csv, chunk_size=1)]
    assert all(h == ["a", "b", "c"] for h in headers_seen)


def test_read_csv_chunked_invalid_chunk_size(simple_csv):
    # chunk_size=0 makes no sense, should raise ValueError
    with pytest.raises(ValueError):
        list(read_csv_chunked(simple_csv, chunk_size=0))


def test_read_csv_chunked_file_not_found():
    # same file-not-found behaviour as read_csv
    with pytest.raises(FileNotFoundError):
        list(read_csv_chunked("no_such_file.csv"))


# ── write_csv ────────

def test_write_csv_roundtrip(tmp_path, simple_csv):
    # write then read back should give identical headers and values
    headers, data = read_csv(simple_csv)
    out = tmp_path / "out.csv"
    write_csv(out, data, headers=headers)
    headers2, data2 = read_csv(out)
    assert headers2 == headers
    assert np.allclose(data, data2)


def test_write_csv_nan_roundtrip(tmp_path, missing_csv):
    # NaN positions should survive the write->read cycle unchanged
    _, data = read_csv(missing_csv)
    out = tmp_path / "out_nan.csv"
    write_csv(out, data)
    _, data2 = read_csv(out, has_header=False)
    assert np.array_equal(np.isnan(data), np.isnan(data2))


def test_write_csv_wrong_dimensions(tmp_path):
    # passing a 1-D array should raise ValueError
    with pytest.raises(ValueError):
        write_csv(tmp_path / "bad.csv", np.array([1.0, 2.0, 3.0]))


def test_write_csv_header_length_mismatch(tmp_path):
    # 2 headers for 3 columns should raise ValueError
    data = np.ones((3, 3))
    with pytest.raises(ValueError):
        write_csv(tmp_path / "bad.csv", data, headers=["a", "b"])


def test_write_csv_creates_parent_dirs(tmp_path):
    # should create nested directories automatically if they don't exist
    out = tmp_path / "nested" / "dir" / "out.csv"
    write_csv(out, np.ones((2, 2)))
    assert out.exists()