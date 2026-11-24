import numpy as np
import pytest
from numcompute.stats import welford, histogram, quantiles


data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])


def test_welford_mean():
    mean, _ = welford(data)
    assert np.isclose(mean, np.mean(data))


def test_welford_variance():
    _, var = welford(data)
    assert np.isclose(var, np.var(data))


def test_welford_single_element():
    mean, var = welford(np.array([5.0]))
    assert mean == 5.0
    assert var == 0.0


def test_welford_empty():
    with pytest.raises(ValueError):
        welford(np.array([]))


def test_welford_non_1d():
    with pytest.raises(ValueError):
        welford(np.array([[1.0, 2.0]]))


def test_histogram_counts_sum():
    counts, _ = histogram(data, bins=4)
    assert counts.sum() == len(data)


def test_histogram_edges_length():
    _, edges = histogram(data, bins=4)
    assert len(edges) == 5


def test_histogram_invalid_bins():
    with pytest.raises(ValueError):
        histogram(data, bins=0)


def test_histogram_custom_range():
    _, edges = histogram(data, bins=4, range_=(2.0, 9.0))
    assert edges[0] == 2.0
    assert edges[-1] == 9.0


def test_quantiles_median():
    assert np.isclose(quantiles(data, q=0.5), np.median(data))


def test_quantiles_multiple():
    out = quantiles(data, q=np.array([0.25, 0.5, 0.75]))
    assert len(out) == 3


def test_quantiles_min():
    assert np.isclose(quantiles(data, q=0.0), data.min())


def test_quantiles_max():
    assert np.isclose(quantiles(data, q=1.0), data.max())


def test_quantiles_invalid_q():
    with pytest.raises(ValueError):
        quantiles(data, q=1.5)


def test_quantiles_empty():
    with pytest.raises(ValueError):
        quantiles(np.array([]), q=0.5)