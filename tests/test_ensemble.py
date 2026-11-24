"""
tests for numcompute_stream/ensemble.py
"""

import numpy as np
import pytest
from numcompute_stream.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    EnsembleClassifier,
)


def _data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_random_forest_basic():
    X, y = _data()
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert preds.shape == (80,)


def test_random_forest_partial_fit():
    X, y = _data()
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    rf.partial_fit(X[:40], y[:40])
    rf.partial_fit(X[40:], y[40:])
    assert rf.fitted_
    assert len(rf._X_buf) == 80


def test_random_forest_predict_proba_shape():
    X, y = _data()
    rf = RandomForestClassifier(n_estimators=5, random_state=0)
    rf.fit(X, y)
    proba = rf.predict_proba(X[:10])
    assert proba.shape[0] == 10
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_bagging_basic():
    X, y = _data()
    bag = BaggingClassifier(n_estimators=5, max_depth=3, random_state=0)
    bag.fit(X, y)
    preds = bag.predict(X)
    assert preds.shape == (80,)


def test_bagging_partial_fit():
    X, y = _data()
    bag = BaggingClassifier(n_estimators=5, random_state=0)
    bag.partial_fit(X[:40], y[:40])
    bag.partial_fit(X[40:], y[40:])
    assert bag.fitted_


def test_ensemble_classifier_random_forest():
    X, y = _data()
    clf = EnsembleClassifier(method="random_forest", n_estimators=5, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (80,)


def test_ensemble_classifier_bagging():
    X, y = _data()
    clf = EnsembleClassifier(method="bagging", n_estimators=5, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (80,)


def test_ensemble_invalid_method():
    with pytest.raises(ValueError):
        EnsembleClassifier(method="boosting")


def test_ensemble_predict_before_fit_raises():
    clf = EnsembleClassifier(method="random_forest", n_estimators=3)
    with pytest.raises(RuntimeError):
        clf.predict(np.ones((5, 4)))


@pytest.mark.parametrize("classifier", [RandomForestClassifier, BaggingClassifier])
def test_ensemble_owns_batches_and_votes_with_sparse_labels(classifier):
    X = np.arange(4.0).reshape(-1, 1)
    y = np.full(4, 5, dtype=int)
    model = classifier(n_estimators=3, random_state=0)
    model.partial_fit(X, y, classes=[-2, 5])
    X[:] = 99
    y[:] = -2
    model.partial_fit([[4]], [5])

    np.testing.assert_array_equal(model._X_buf[:, 0], np.arange(5))
    np.testing.assert_array_equal(model.predict([[0], [4]]), [5, 5])
    np.testing.assert_allclose(model.predict_proba([[0]]), [[0, 1]])
