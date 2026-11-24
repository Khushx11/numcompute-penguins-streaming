import numpy as np
import pytest
from numcompute.sort_search import quicksort, top_k, quickselect, binary_search


arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])


def test_quicksort_ascending():
    out = quicksort(arr)
    assert np.array_equal(out, np.sort(arr))


def test_quicksort_descending():
    out = quicksort(arr, ascending=False)
    assert np.array_equal(out, np.sort(arr)[::-1])


def test_quicksort_single_element():
    assert np.array_equal(quicksort(np.array([42.0])), np.array([42.0]))


def test_quicksort_duplicates():
    a = np.array([2.0, 2.0, 2.0])
    assert np.array_equal(quicksort(a), a)


def test_quicksort_non_1d():
    with pytest.raises(ValueError):
        quicksort(np.array([[1.0, 2.0]]))


def test_top_k_largest():
    out = top_k(arr, k=3, largest=True)
    assert np.array_equal(out, np.array([9.0, 5.0, 4.0]))


def test_top_k_smallest():
    out = top_k(arr, k=3, largest=False)
    assert np.array_equal(out, np.array([1.0, 1.0, 2.0]))


def test_top_k_invalid_k():
    with pytest.raises(ValueError):
        top_k(arr, k=0)


def test_top_k_k_too_large():
    with pytest.raises(ValueError):
        top_k(arr, k=100)


def test_top_k_non_1d():
    with pytest.raises(ValueError):
        top_k(np.array([[1.0, 2.0]]), k=1)


def test_quickselect_min():
    assert quickselect(arr, k=0) == 1.0


def test_quickselect_max():
    assert quickselect(arr, k=len(arr) - 1) == 9.0


def test_quickselect_middle():
    assert quickselect(arr, k=3) == 3.0


def test_quickselect_invalid_k():
    with pytest.raises(ValueError):
        quickselect(arr, k=-1)


def test_quickselect_non_1d():
    with pytest.raises(ValueError):
        quickselect(np.array([[1.0, 2.0]]), k=0)


def test_binary_search_found():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert binary_search(a, 3.0) == 2


def test_binary_search_not_found():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert binary_search(a, 9.0) == -1


def test_binary_search_first():
    a = np.array([1.0, 2.0, 3.0])
    assert binary_search(a, 1.0) == 0


def test_binary_search_last():
    a = np.array([1.0, 2.0, 3.0])
    assert binary_search(a, 3.0) == 2


def test_binary_search_non_1d():
    with pytest.raises(ValueError):
        binary_search(np.array([[1.0, 2.0]]), 1.0)


def test_quicksort_nan_preserved():
    a = np.array([3.0, np.nan, 1.0])
    out = quicksort(a)
    assert np.array_equal(out[:2], np.array([1.0, 3.0]))
    assert np.isnan(out[2])


def test_quicksort_all_nan():
    a = np.array([np.nan, np.nan])
    out = quicksort(a)
    assert len(out) == 2
    assert all(np.isnan(out))
