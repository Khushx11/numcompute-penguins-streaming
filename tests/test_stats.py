"""
tests for numcompute_stream/stats.py
"""

import numpy as np
import pytest
from numcompute_stream.stats import (
    RunningStats,
    MultiFeatureStats,
    StreamingHistogram,
    streaming_quantiles,
)


def test_running_stats_basic():
    rs = RunningStats()
    rs.update_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(rs.mean_ - 3.0) < 1e-9
    assert rs.n == 5


def test_running_stats_incremental_matches_batch():
    data = np.random.default_rng(0).normal(size=100)
    rs_batch = RunningStats()
    rs_batch.update_stats(data)

    rs_inc = RunningStats()
    for chunk in np.array_split(data, 10):
        rs_inc.update_stats(chunk)

    assert abs(rs_batch.mean_ - rs_inc.mean_) < 1e-9
    assert abs(rs_batch.variance() - rs_inc.variance()) < 1e-6


def test_running_stats_ignores_nan():
    rs = RunningStats()
    rs.update_stats([1.0, np.nan, 3.0])
    assert rs.n == 2
    assert abs(rs.mean_ - 2.0) < 1e-9


def test_running_stats_single_value_variance():
    rs = RunningStats()
    rs.update_stats([42.0])
    assert rs.variance() == 0.0


def test_running_stats_reset():
    rs = RunningStats()
    rs.update_stats([1.0, 2.0, 3.0])
    rs.reset()
    assert rs.n == 0
    assert rs.mean_ == 0.0


def test_running_stats_all_nan():
    rs = RunningStats()
    rs.update_stats([np.nan, np.nan])
    assert rs.n == 0


def test_multi_feature_stats_shape():
    mfs = MultiFeatureStats()
    X = np.ones((10, 4))
    mfs.update_stats(X)
    assert len(mfs.means()) == 4
    assert len(mfs.variances()) == 4


def test_multi_feature_stats_wrong_features():
    mfs = MultiFeatureStats()
    mfs.update_stats(np.ones((5, 3)))
    with pytest.raises(ValueError):
        mfs.update_stats(np.ones((5, 4)))


def test_streaming_histogram_basic():
    sh = StreamingHistogram(bins=5)
    sh.update_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sh.result()
    assert result is not None
    assert result["counts"].sum() == 5


def test_streaming_histogram_empty_chunk():
    sh = StreamingHistogram(bins=5)
    sh.update_stats([])
    assert sh.result() is None


def test_streaming_histogram_all_same_value():
    sh = StreamingHistogram(bins=5)
    sh.update_stats([3.0, 3.0, 3.0])
    result = sh.result()
    assert result["counts"].sum() == 3


def test_streaming_histogram_incremental():
    sh = StreamingHistogram(bins=10, range_=(0, 10))
    sh.update_stats([1.0, 2.0, 3.0])
    sh.update_stats([4.0, 5.0, 6.0])
    result = sh.result()
    assert result["counts"].sum() == 6


def test_streaming_histogram_reset():
    sh = StreamingHistogram(bins=5)
    sh.update_stats([1.0, 2.0, 3.0])
    sh.reset()
    assert sh.result() is None


def test_streaming_quantiles_median():
    data = np.arange(1, 10, dtype=float)
    q = streaming_quantiles(data, 0.5)
    assert abs(q - 5.0) < 1e-9


def test_streaming_quantiles_min_max():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert streaming_quantiles(data, 0.0) == 1.0
    assert streaming_quantiles(data, 1.0) == 5.0


def test_streaming_quantiles_empty_raises():
    with pytest.raises(ValueError):
        streaming_quantiles([], 0.5)


def test_streaming_quantiles_out_of_range_raises():
    with pytest.raises(ValueError):
        streaming_quantiles([1.0, 2.0], 1.5)