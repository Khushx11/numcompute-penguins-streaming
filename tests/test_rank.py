import numpy as np
import pytest
from numcompute.rank import rank_data, percentile_rank


arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])


def test_rank_data_average():
    out = rank_data(arr, method="average")
    assert np.allclose(out, [3.0, 1.5, 4.0, 1.5, 5.0])


def test_rank_data_min():
    out = rank_data(arr, method="min")
    assert np.allclose(out, [3.0, 1.0, 4.0, 1.0, 5.0])


def test_rank_data_max():
    out = rank_data(arr, method="max")
    assert np.allclose(out, [3.0, 2.0, 4.0, 2.0, 5.0])


def test_rank_data_ordinal():
    out = rank_data(arr, method="ordinal")
    assert np.allclose(out, [3.0, 1.0, 4.0, 2.0, 5.0])


def test_rank_data_no_ties():
    a = np.array([10.0, 30.0, 20.0])
    out = rank_data(a, method="average")
    assert np.allclose(out, [1.0, 3.0, 2.0])


def test_rank_data_invalid_method():
    with pytest.raises(ValueError):
        rank_data(arr, method="dense")


def test_rank_data_non_1d():
    with pytest.raises(ValueError):
        rank_data(np.array([[1.0, 2.0]]))


def test_percentile_rank_middle():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert percentile_rank(a, score=3.0) == 40.0


def test_percentile_rank_below_all():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert percentile_rank(a, score=1.0) == 0.0


def test_percentile_rank_above_all():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert percentile_rank(a, score=6.0) == 100.0


def test_percentile_rank_empty():
    with pytest.raises(ValueError):
        percentile_rank(np.array([]), score=1.0)


def test_percentile_rank_non_1d():
    with pytest.raises(ValueError):
        percentile_rank(np.array([[1.0, 2.0]]), score=1.0)


def test_rank_data_nan_propagates():
    a = np.array([1.0, np.nan, 2.0])
    out = rank_data(a)
    assert np.isnan(out[1])
    assert np.allclose(out[[0, 2]], [1.0, 2.0])


def test_rank_data_all_nan():
    a = np.array([np.nan, np.nan])
    out = rank_data(a)
    assert all(np.isnan(out))
