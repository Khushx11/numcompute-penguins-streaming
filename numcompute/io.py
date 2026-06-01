import numpy as np
import csv
from pathlib import Path


def read_csv(filepath, delimiter=",", has_header=True, dtype=float, missing_val=np.nan):
    """
    Reads a CSV file into a NumPy array.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    delimiter : str, optional
        Delimiter string, default is ','.
    has_header : bool, optional
        Whether the first row is a header, default is True.
    dtype : type, optional
        Data type to cast the values to, default is float.
    missing_val : any, optional
        Value to fill in for missing/unparseable entries, default is np.nan.

    Returns
    -------
    headers : list[str]
        List of column headers.
    data : np.ndarray, shape (n_rows, n_cols)
        The parsed data array.

    Raises
    ------
    FileNotFoundError
        If the filepath does not exist.

    Complexity
    ----------
    Time  : O(N) where N is total cells
    Space : O(N)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at {filepath}")

    headers = []
    rows = []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)

        if has_header:
            headers = next(reader)  # grab first row as column names

        for line in reader:
            row = []
            for val in line:
                stripped = val.strip()
                # treat empty strings and common null-like tokens as missing
                if stripped == "" or stripped.lower() in ("na", "nan", "null", "none"):
                    row.append(missing_val)
                else:
                    try:
                        row.append(dtype(stripped))
                    except (ValueError, TypeError):
                        row.append(missing_val)  # non-parseable values also become missing
            rows.append(row)

    data = np.array(rows, dtype=float)
    return headers, data


def read_csv_chunked(filepath, chunk_size=1000, delimiter=",", has_header=True, dtype=float, missing_val=np.nan):
    """
    Reads a CSV file into NumPy arrays in chunks using a generator.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    chunk_size : int, optional
        Number of rows per chunk, default is 1000.
    delimiter : str, optional
        Delimiter string, default is ','.
    has_header : bool, optional
        Whether the first row is a header, default is True.
    dtype : type, optional
        Data type to cast the values to, default is float.
    missing_val : any, optional
        Value to fill in for missing entries, default is np.nan.

    Yields
    ------
    headers : list[str]
        List of column headers.
    chunk_data : np.ndarray, shape (chunk_size, n_cols)
        The parsed chunk data array.

    Raises
    ------
    FileNotFoundError
        If the filepath does not exist.
    ValueError
        If chunk_size < 1.

    Complexity
    ----------
    Time  : O(N) overall
    Space : O(chunk_size * n_cols)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at {filepath}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    def parse_val(val):
        # same missing-value logic as read_csv, pulled out to avoid repeating it per row
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
                chunk = []  # reset for next chunk

        if chunk:
            yield headers, np.array(chunk, dtype=float)  # flush leftover rows


def write_csv(filepath, data, headers=None, delimiter=",", fmt="%.6g"):
    """
    Writes a 2-D NumPy array to a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the output CSV file.
    data : np.ndarray, shape (n_rows, n_cols)
        Data to write.
    headers : list[str], optional
        List of column headers.
    delimiter : str, optional
        Delimiter string, default is ','.
    fmt : str, optional
        String format for writing float values, default is '%.6g'.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If data is not 2-D or header count doesn't match columns.

    Complexity
    ----------
    Time  : O(N) where N is total cells
    Space : O(1) beyond disk buffer
    """
    filepath = Path(filepath)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {data.shape}")
    if headers is not None and len(headers) != data.shape[1]:
        raise ValueError(f"headers length {len(headers)} does not match {data.shape[1]} columns")

    filepath.parent.mkdir(parents=True, exist_ok=True)  # make sure the directory exists

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)

        if headers:
            writer.writerow(headers)

        for row in data:
            # write NaN back as empty string so it round-trips cleanly with read_csv
            writer.writerow(["" if np.isnan(v) else fmt % v for v in row])