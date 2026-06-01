import numpy as np


def _quicksort_recursive(arr):
    # base case - nothing to sort
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    # median-of-three keeps worst case rare on already-sorted input
    pivot = np.median([arr[0], arr[mid], arr[-1]])

    left   = arr[arr < pivot]
    middle = arr[arr == pivot]
    right  = arr[arr > pivot]

    return np.concatenate([
        _quicksort_recursive(left),
        middle,
        _quicksort_recursive(right)
    ])


def quicksort(arr, ascending=True):
    """
    Sort a 1-D array using quicksort with median-of-three pivot selection.

    NaN values are preserved and appended at the end regardless of sort order.

    Parameters
    ----------
    arr : array-like, shape (n,)
        Input values to sort.
    ascending : bool, optional
        Sort order. Default is True (smallest first).

    Returns
    -------
    np.ndarray, shape (n,)
        Sorted copy of arr, with NaNs at the end.

    Raises
    ------
    ValueError
        If arr is not 1-D.

    Complexity
    ----------
    Time  : O(n log n) average, O(n^2) worst case
    Space : O(n) due to recursive copies
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"arr must be 1-D, got shape {arr.shape}")

    # separate NaNs out before sorting - NaN comparisons always return False
    # which causes them to be silently dropped by the partition logic
    nan_mask = np.isnan(arr)
    clean = arr[~nan_mask]
    nans = np.full(nan_mask.sum(), np.nan)

    result = _quicksort_recursive(clean)
    result = np.concatenate([result, nans])
    return result if ascending else result[::-1]


def top_k(arr, k, largest=True):
    """
    Return the top-k values from a 1-D array.

    Parameters
    ----------
    arr : array-like, shape (n,)
        Input values.
    k : int
        How many values to return. Must be between 1 and len(arr).
    largest : bool, optional
        If True (default), return the k largest. If False, return the k smallest.

    Returns
    -------
    np.ndarray, shape (k,)
        Top-k values in descending order (largest=True) or ascending (largest=False).

    Raises
    ------
    ValueError
        If arr is not 1-D or k is out of range.

    Complexity
    ----------
    Time  : O(n log n) — sorts the full array first
    Space : O(n)
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"arr must be 1-D, got shape {arr.shape}")
    if k < 1 or k > len(arr):
        raise ValueError(f"k must be between 1 and {len(arr)}, got {k}")

    sorted_arr = _quicksort_recursive(arr.copy())
    return sorted_arr[-k:][::-1] if largest else sorted_arr[:k]


def _quickselect_recursive(arr, k):
    # single element - must be the answer
    if len(arr) == 1:
        return arr[0]

    mid = len(arr) // 2
    pivot = np.median([arr[0], arr[mid], arr[-1]])

    left   = arr[arr < pivot]
    middle = arr[arr == pivot]
    right  = arr[arr > pivot]

    if k < len(left):
        return _quickselect_recursive(left, k)
    elif k < len(left) + len(middle):
        return pivot
    else:
        return _quickselect_recursive(right, k - len(left) - len(middle))


def quickselect(arr, k):
    """
    Find the k-th smallest value in a 1-D array without fully sorting it.

    Parameters
    ----------
    arr : array-like, shape (n,)
        Input values.
    k : int
        Zero-based index of the target order statistic (0 = minimum).

    Returns
    -------
    float
        The k-th smallest value in arr.

    Raises
    ------
    ValueError
        If arr is not 1-D or k is out of range.

    Complexity
    ----------
    Time  : O(n) average, O(n^2) worst case
    Space : O(n) due to recursive copies
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"arr must be 1-D, got shape {arr.shape}")
    if k < 0 or k >= len(arr):
        raise ValueError(f"k must be between 0 and {len(arr) - 1}, got {k}")

    return _quickselect_recursive(arr.copy(), k)


def binary_search(arr, target):
    """
    Search a sorted 1-D array for target using binary search.

    Parameters
    ----------
    arr : array-like, shape (n,)
        Sorted input array.
    target : float
        Value to search for.

    Returns
    -------
    int
        Index of target in arr, or -1 if not found.

    Raises
    ------
    ValueError
        If arr is not 1-D.

    Complexity
    ----------
    Time  : O(log n)
    Space : O(1)
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"arr must be 1-D, got shape {arr.shape}")

    lo, hi = 0, len(arr) - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1
