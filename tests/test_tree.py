"""
tests for numcompute_stream/tree.py
"""

import numpy as np
import pytest
from numcompute_stream.tree import DecisionTreeClassifier


def _simple_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_tree_fit_predict_basic():
    X, y = _simple_data()
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)
    assert set(preds).issubset({0, 1})


def test_tree_partial_fit_incremental():
    X, y = _simple_data()
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    for chunk in np.array_split(X, 5):
        idx = np.where(np.isin(X, chunk).all(axis=1))[0]
        clf.partial_fit(chunk, y[idx[:len(chunk)]])
    assert clf.fitted_


def test_tree_partial_fit_accumulates():
    X, y = _simple_data()
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.partial_fit(X[:50], y[:50])
    clf.partial_fit(X[50:], y[50:])
    assert len(clf._X_buf) == 100


def test_tree_predict_before_fit_raises():
    clf = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(np.ones((3, 4)))


def test_tree_nan_in_X_handled():
    X, y = _simple_data()
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.partial_fit(X_nan, y)
    preds = clf.predict(X[:5])
    assert len(preds) == 5


def test_tree_single_sample_chunk():
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.partial_fit(np.array([[1.0, 2.0]]), np.array([0]))
    clf.partial_fit(np.array([[3.0, 4.0]]), np.array([1]))
    assert clf.fitted_


def test_tree_zero_variance_feature():
    # all values in one column are identical, tree should still work
    X = np.ones((20, 3))
    X[:, 1] = np.arange(20, dtype=float)
    y = (X[:, 1] > 10).astype(int)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (20,)


def test_tree_predict_proba_sums_to_one():
    X, y = _simple_data()
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X[:10])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_tree_max_depth_respected():
    X, y = _simple_data()
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    assert clf.get_depth() <= 2


def test_tree_pure_node_becomes_leaf():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])
    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert (preds == 1).all()


def test_tree_preserves_sparse_labels_and_owns_training_batches():
    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([-2, 5, 5, 5])
    tree = DecisionTreeClassifier(max_depth=0).partial_fit(X, y, classes=[20])
    assert tree.predict([[0]])[0] == 5
    np.testing.assert_allclose(tree.predict_proba([[0]]), [[0.25, 0.75, 0]])

    X[:] = 99
    y[:] = -2
    tree.partial_fit([[4]], [5])
    np.testing.assert_array_equal(tree._X_buf[:, 0], np.arange(5))
    np.testing.assert_allclose(tree.predict_proba([[0]]), [[0.2, 0.8, 0]])
