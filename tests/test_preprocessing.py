"""
tests for numcompute_stream/preprocessing.py
"""

import numpy as np
import pytest
from numcompute_stream.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Imputer,
    OneHotEncoder,
)


def test_standard_scaler_basic():
    sc = StandardScaler()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sc.partial_fit(X)
    Xt = sc.transform(X)
    assert abs(Xt.mean(axis=0)).max() < 1e-9


def test_standard_scaler_incremental():
    sc = StandardScaler()
    X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    sc.partial_fit(X1)
    sc.partial_fit(X2)
    assert sc.fitted_


def test_standard_scaler_zero_variance():
    sc = StandardScaler()
    X = np.ones((5, 2))
    sc.partial_fit(X)
    Xt = sc.transform(X)
    # zero variance columns should not produce NaN
    assert not np.isnan(Xt).any()


def test_standard_scaler_not_fitted_raises():
    sc = StandardScaler()
    with pytest.raises(RuntimeError):
        sc.transform(np.ones((3, 2)))


def test_minmax_scaler_range():
    sc = MinMaxScaler()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sc.partial_fit(X)
    Xt = sc.transform(X)
    assert Xt.min() >= 0.0
    assert Xt.max() <= 1.0


def test_minmax_scaler_incremental_expands_range():
    sc = MinMaxScaler()
    sc.partial_fit(np.array([[1.0], [2.0]]))
    sc.partial_fit(np.array([[0.0], [3.0]]))
    assert sc.min_[0] == 0.0
    assert sc.max_[0] == 3.0


def test_imputer_mean_fills_nan():
    imp = Imputer(strategy="mean")
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    imp.partial_fit(X)
    Xt = imp.transform(X)
    assert not np.isnan(Xt).any()


def test_imputer_constant_fill():
    imp = Imputer(strategy="constant")
    X = np.array([[np.nan, 2.0], [3.0, np.nan]])
    imp.partial_fit(X, constant=-1.0)
    Xt = imp.transform(X)
    assert Xt[0, 0] == -1.0
    assert Xt[1, 1] == -1.0


def test_imputer_no_nans_unchanged():
    imp = Imputer(strategy="mean")
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    imp.partial_fit(X)
    Xt = imp.transform(X)
    np.testing.assert_array_equal(X, Xt)


def test_one_hot_encoder_basic():
    enc = OneHotEncoder()
    enc.partial_fit([0, 1, 2])
    out = enc.transform([0, 1, 2])
    assert out.shape == (3, 3)
    assert out.sum() == 3.0


def test_one_hot_encoder_incremental_categories():
    enc = OneHotEncoder()
    enc.partial_fit([0, 1])
    enc.partial_fit([2, 3])
    assert len(enc.categories_) == 4


def test_one_hot_encoder_inverse():
    enc = OneHotEncoder()
    enc.partial_fit([0, 1, 2])
    out = enc.transform([0, 1, 2])
    back = enc.inverse_transform(out)
    np.testing.assert_array_equal(back, [0, 1, 2])