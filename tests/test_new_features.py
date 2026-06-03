"""
tests for EMAStats, EMAScaler, StreamingAUC, memory tracking,
tie-breaking behaviour, and read_csv_chunked specifically.
These cover the functionality added in commit 19 and 20.
"""

import numpy as np
import pytest
import tempfile
import os
from numcompute_stream.stats import EMAStats
from numcompute_stream.preprocessing import EMAScaler
from numcompute_stream.metrics import StreamingAUC
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.io import read_csv_chunked, make_stream_chunks


# -----------------------------------------------------------------------
# EMAStats
# -----------------------------------------------------------------------

def test_ema_stats_basic():
    ema = EMAStats(alpha=0.5)
    ema.update_stats([1.0, 2.0, 3.0])
    assert ema.mean_ is not None
    assert ema.fitted_


def test_ema_stats_mean_shifts_toward_recent():
    # with high alpha, the mean should move quickly toward recent chunks
    ema = EMAStats(alpha=0.9)
    ema.update_stats([0.0, 0.0, 0.0])
    ema.update_stats([100.0, 100.0, 100.0])
    # after a high-alpha update with a chunk of 100s, mean should be close to 100
    assert ema.mean_ > 50.0


def test_ema_stats_low_alpha_changes_slowly():
    ema = EMAStats(alpha=0.01)
    ema.update_stats([0.0, 0.0, 0.0])
    ema.update_stats([100.0, 100.0, 100.0])
    # low alpha means old values dominate, mean should still be close to 0
    assert ema.mean_ < 10.0


def test_ema_stats_ignores_nan():
    ema = EMAStats(alpha=0.5)
    ema.update_stats([np.nan, np.nan])
    assert not ema.fitted_


def test_ema_stats_empty_chunk():
    ema = EMAStats(alpha=0.5)
    ema.update_stats([])
    assert ema.mean_ is None


def test_ema_stats_invalid_alpha():
    with pytest.raises(ValueError):
        EMAStats(alpha=0.0)
    with pytest.raises(ValueError):
        EMAStats(alpha=1.5)


def test_ema_stats_reset():
    ema = EMAStats(alpha=0.5)
    ema.update_stats([1.0, 2.0, 3.0])
    ema.reset()
    assert ema.mean_ is None
    assert not ema.fitted_


def test_ema_stats_result_keys():
    ema = EMAStats(alpha=0.3)
    ema.update_stats([1.0, 2.0])
    result = ema.result()
    assert 'mean' in result
    assert 'variance' in result
    assert 'std' in result
    assert 'alpha' in result


# -----------------------------------------------------------------------
# EMAScaler
# -----------------------------------------------------------------------

def test_ema_scaler_basic_transform():
    sc = EMAScaler(alpha=0.5)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sc.partial_fit(X)
    Xt = sc.transform(X)
    assert Xt.shape == X.shape
    assert not np.isnan(Xt).any()


def test_ema_scaler_adapts_to_new_data():
    sc = EMAScaler(alpha=0.9)
    X1 = np.ones((10, 2)) * 0.0
    X2 = np.ones((10, 2)) * 100.0
    sc.partial_fit(X1)
    mean_after_first = sc.mean_.copy()
    sc.partial_fit(X2)
    # high alpha means mean should shift significantly toward 100
    assert np.all(sc.mean_ > mean_after_first)


def test_ema_scaler_zero_variance_no_nan():
    sc = EMAScaler(alpha=0.3)
    X = np.ones((5, 3))
    sc.partial_fit(X)
    Xt = sc.transform(X)
    assert not np.isnan(Xt).any()
    assert not np.isinf(Xt).any()


def test_ema_scaler_not_fitted_raises():
    sc = EMAScaler(alpha=0.3)
    with pytest.raises(RuntimeError):
        sc.transform(np.ones((3, 2)))


def test_ema_scaler_wrong_features_raises():
    sc = EMAScaler(alpha=0.3)
    sc.partial_fit(np.ones((5, 3)))
    with pytest.raises(ValueError):
        sc.partial_fit(np.ones((5, 4)))


def test_ema_scaler_invalid_alpha():
    with pytest.raises(ValueError):
        EMAScaler(alpha=0.0)


# -----------------------------------------------------------------------
# StreamingAUC
# -----------------------------------------------------------------------

def test_streaming_auc_perfect_separation():
    auc = StreamingAUC()
    # positive class always scores higher than negative
    auc.update([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert abs(auc.result() - 1.0) < 1e-9


def test_streaming_auc_random_is_half():
    rng = np.random.default_rng(42)
    auc = StreamingAUC()
    # random scores should give AUC close to 0.5
    for _ in range(20):
        y = rng.integers(0, 2, size=50)
        scores = rng.uniform(size=50)
        auc.update(y, scores)
    assert 0.3 < auc.result() < 0.7


def test_streaming_auc_incremental_history():
    auc = StreamingAUC()
    auc.update([1, 0], [0.9, 0.1])
    auc.update([1, 0], [0.8, 0.2])
    assert len(auc.history) == 2


def test_streaming_auc_no_positives_returns_half():
    auc = StreamingAUC()
    # no positive samples yet
    assert auc.result() == 0.5


def test_streaming_auc_shape_mismatch_raises():
    auc = StreamingAUC()
    with pytest.raises(ValueError):
        auc.update([1, 0], [0.9])


def test_streaming_auc_reset():
    auc = StreamingAUC()
    auc.update([1, 0], [0.9, 0.1])
    auc.reset()
    assert auc.result() == 0.5
    assert auc.history == []


# -----------------------------------------------------------------------
# Memory tracking in StreamTrainer
# -----------------------------------------------------------------------

def test_stream_trainer_memory_tracked():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    chunks = make_stream_chunks(X, y, chunk_size=10)

    clf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0)
    trainer = StreamTrainer(clf, classes=[0, 1])
    trainer.run(chunks)

    mem = trainer.memory_history()
    assert len(mem) == 6
    # memory should grow as buffer accumulates
    assert mem[-1] >= mem[0]


def test_stream_trainer_memory_in_log():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(int)
    chunks = make_stream_chunks(X, y, chunk_size=10)

    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    trainer = StreamTrainer(clf, classes=[0, 1])
    trainer.run(chunks)

    for entry in trainer.chunk_log:
        assert 'mem_bytes' in entry
        assert entry['mem_bytes'] >= 0


def test_stream_trainer_memory_with_pipeline():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    chunks = make_stream_chunks(X, y, chunk_size=10)

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=3, random_state=0)),
    ])
    trainer = StreamTrainer(pipe, classes=[0, 1])
    trainer.run(chunks)

    mem = trainer.memory_history()
    assert len(mem) == 4
    assert all(m >= 0 for m in mem)


# -----------------------------------------------------------------------
# Tie-breaking in decision tree splits
# -----------------------------------------------------------------------

def test_tree_handles_tied_gain():
    # perfectly symmetric data where multiple splits have equal gain
    # tree should still produce valid predictions without crashing
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [2.0, 2.0],
    ], dtype=float)
    y = np.array([0, 1, 1, 0])
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (4,)
    assert set(preds).issubset({0, 1})


def test_tree_consistent_on_ties_with_same_seed():
    # same random seed should give same split choice when gains are tied
    X = np.tile(np.arange(10, dtype=float).reshape(-1, 1), (1, 2))
    y = np.array([0, 1] * 5)
    clf1 = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf2 = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf1.fit(X, y)
    clf2.fit(X, y)
    np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))


# -----------------------------------------------------------------------
# read_csv_chunked specifically
# -----------------------------------------------------------------------

def _write_temp_csv(content):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    f.write(content)
    f.close()
    return f.name


def test_read_csv_chunked_basic():
    path = _write_temp_csv('a,b\n' + '\n'.join(f'{i},{i*2}' for i in range(15)))
    chunks = list(read_csv_chunked(path, chunk_size=5))
    assert len(chunks) == 3
    assert all(c.shape == (5, 2) for _, c in chunks)
    os.unlink(path)


def test_read_csv_chunked_last_chunk_smaller():
    path = _write_temp_csv('a,b\n' + '\n'.join(f'{i},{i}' for i in range(13)))
    chunks = list(read_csv_chunked(path, chunk_size=5))
    assert len(chunks) == 3
    assert chunks[-1][1].shape[0] == 3
    os.unlink(path)


def test_read_csv_chunked_with_missing_values():
    path = _write_temp_csv('a,b\n1,\n2,3\n,4\n5,6\n')
    chunks = list(read_csv_chunked(path, chunk_size=10))
    data = chunks[0][1]
    assert np.isnan(data[0, 1])
    assert np.isnan(data[2, 0])
    os.unlink(path)


def test_read_csv_chunked_headers_consistent():
    path = _write_temp_csv('x,y,z\n1,2,3\n4,5,6\n7,8,9\n')
    chunks = list(read_csv_chunked(path, chunk_size=2))
    # all chunks should return the same headers
    assert all(h == ['x', 'y', 'z'] for h, _ in chunks)
    os.unlink(path)


def test_read_csv_chunked_invalid_chunk_size():
    path = _write_temp_csv('a,b\n1,2\n')
    with pytest.raises(ValueError):
        list(read_csv_chunked(path, chunk_size=0))
    os.unlink(path)


def test_read_csv_chunked_file_not_found():
    with pytest.raises(FileNotFoundError):
        list(read_csv_chunked('/nonexistent/file.csv'))
