"""
io.py

CSV loading with a custom streaming pipeline. The chunked reader is
the main thing used by the demo and stream trainer. read_csv is there
for when you just want the whole file at once.
"""

import numpy as np
import csv
from pathlib import Path


def read_csv(filepath, delimiter=",", has_header=True, dtype=float, missing_val=np.nan):
    """
    Load an entire CSV file into a NumPy array in one shot.

    Parameters
    ----------
    filepath : str or Path
    delimiter : str, default ','
    has_header : bool, default True
    dtype : type, default float
    missing_val : any, default np.nan
        What to put in place of empty or unparseable cells.

    Returns
    -------
    headers : list of str
        Empty list if has_header is False.
    data : np.ndarray, shape (n_rows, n_cols)

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"no file at {filepath}")

    headers = []
    rows = []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            headers = next(reader)
        for line in reader:
            row = []
            for val in line:
                stripped = val.strip()
                if stripped == "" or stripped.lower() in ("na", "nan", "null", "none"):
                    row.append(missing_val)
                else:
                    try:
                        row.append(dtype(stripped))
                    except (ValueError, TypeError):
                        row.append(missing_val)
            rows.append(row)

    data = np.array(rows, dtype=float)
    return headers, data


def read_csv_chunked(
    filepath,
    chunk_size=100,
    delimiter=",",
    has_header=True,
    dtype=float,
    missing_val=np.nan,
):
    """
    Generator that yields chunks of a CSV file as NumPy arrays.

    Each call to next() returns the headers and one chunk of rows.
    Useful for simulating a streaming data feed without loading the
    whole file into memory.

    Parameters
    ----------
    filepath : str or Path
    chunk_size : int, default 100
        Number of rows per chunk.
    delimiter : str, default ','
    has_header : bool, default True
    dtype : type, default float
    missing_val : any, default np.nan

    Yields
    ------
    headers : list of str
    chunk : np.ndarray, shape (chunk_size, n_cols)
        Last chunk may be smaller if rows do not divide evenly.

    Raises
    ------
    FileNotFoundError
    ValueError
        If chunk_size is less than 1.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"no file at {filepath}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    def parse_val(val):
        stripped = val.strip()
        if stripped == "" or stripped.lower() in ("na", "nan", "null", "none"):
            return missing_val
        try:
            return dtype(stripped)
        except (ValueError, TypeError):
            return missing_val

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader) if has_header else []
        chunk = []

        for line in reader:
            chunk.append([parse_val(v) for v in line])
            if len(chunk) == chunk_size:
                yield headers, np.array(chunk, dtype=float)
                chunk = []

        # flush the last partial chunk
        if chunk:
            yield headers, np.array(chunk, dtype=float)


def write_csv(filepath, data, headers=None, delimiter=",", fmt="%.6g"):
    """
    Write a 2-D NumPy array to a CSV file.

    NaN values are written as empty strings so they round-trip cleanly
    when read back with read_csv.

    Parameters
    ----------
    filepath : str or Path
    data : np.ndarray, shape (n_rows, n_cols)
    headers : list of str, optional
    delimiter : str, default ','
    fmt : str, default '%.6g'

    Raises
    ------
    ValueError
        If data is not 2-D or header count does not match columns.
    """
    filepath = Path(filepath)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {data.shape}")
    if headers is not None and len(headers) != data.shape[1]:
        raise ValueError(
            f"headers length {len(headers)} does not match {data.shape[1]} columns"
        )

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        if headers:
            writer.writerow(headers)
        for row in data:
            writer.writerow(["" if np.isnan(v) else fmt % v for v in row])


def make_stream_chunks(X, y, chunk_size):
    """
    Split arrays X and y into a list of equal-sized chunks.

    The last chunk may be smaller if the data does not divide evenly.
    Handy for converting an in-memory dataset into a streaming simulation
    without needing a CSV file.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    chunk_size : int

    Returns
    -------
    list of (X_chunk, y_chunk) tuples

    Raises
    ------
    ValueError
        If chunk_size is less than 1 or X and y lengths do not match.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    chunks = []
    n = len(X)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append((X[start:end], y[start:end]))
    return chunks
