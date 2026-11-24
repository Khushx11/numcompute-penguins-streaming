import numpy as np
import pytest
from numcompute.preprocessing import StandardScaler, MinMaxScaler, Imputer, OneHotEncoder


X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])


def test_standard_scaler_mean_is_zero():
    ss = StandardScaler()
    out = ss.fit_transform(X)
    assert np.allclose(out.mean(axis=0), 0.0)

def test_standard_scaler_std_is_one():
    ss = StandardScaler()
    out = ss.fit_transform(X)
    assert np.allclose(out.std(axis=0), 1.0)

def test_standard_scaler_inverse():
    ss = StandardScaler()
    out = ss.fit_transform(X)
    assert np.allclose(ss.inverse_transform(out), X)

def test_standard_scaler_not_fitted():
    with pytest.raises(RuntimeError):
        StandardScaler().transform(X)

def test_standard_scaler_non_2d():
    with pytest.raises(ValueError):
        StandardScaler().fit(np.array([1.0, 2.0, 3.0]))



def test_minmax_scaler_range():
    mm = MinMaxScaler()
    out = mm.fit_transform(X)
    assert np.allclose(out.min(axis=0), 0.0)
    assert np.allclose(out.max(axis=0), 1.0)

def test_minmax_scaler_custom_range():
    mm = MinMaxScaler(feature_range=(-1, 1))
    out = mm.fit_transform(X)
    assert np.allclose(out.min(axis=0), -1.0)
    assert np.allclose(out.max(axis=0), 1.0)

def test_minmax_scaler_inverse():
    mm = MinMaxScaler()
    out = mm.fit_transform(X)
    assert np.allclose(mm.inverse_transform(out), X)

def test_minmax_scaler_invalid_range():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0))

def test_minmax_scaler_not_fitted():
    with pytest.raises(RuntimeError):
        MinMaxScaler().transform(X)


def test_imputer_mean():
    X_miss = np.array([[1.0, np.nan],
                       [3.0, 4.0],
                       [np.nan, 6.0]])
    imp = Imputer(strategy="mean")
    out = imp.fit_transform(X_miss)
    assert not np.any(np.isnan(out))
    assert np.allclose(out[0, 1], 5.0)
    assert np.allclose(out[2, 0], 2.0)

def test_imputer_median():
    X_miss = np.array([[1.0, np.nan],
                       [3.0, 4.0],
                       [5.0, 6.0]])
    imp = Imputer(strategy="median")
    out = imp.fit_transform(X_miss)
    assert np.allclose(out[0, 1], 5.0)

def test_imputer_constant():
    X_miss = np.array([[np.nan, 2.0],
                       [3.0, 4.0]])
    imp = Imputer(strategy="constant")
    out = imp.fit_transform(X_miss, constant=-1.0)
    assert np.allclose(out[0, 0], -1.0)

def test_imputer_invalid_strategy():
    with pytest.raises(ValueError):
        Imputer(strategy="mode")

def test_imputer_not_fitted():
    with pytest.raises(RuntimeError):
        Imputer().transform(X)


def test_one_hot_shape():
    y = np.array([0, 1, 2, 1, 0])
    enc = OneHotEncoder()
    out = enc.fit_transform(y)
    assert out.shape == (5, 3)

def test_one_hot_values():
    y = np.array([0, 1, 2])
    enc = OneHotEncoder()
    out = enc.fit_transform(y)
    assert np.allclose(out, np.eye(3))

def test_one_hot_inverse():
    y = np.array([0, 1, 2, 1, 0])
    enc = OneHotEncoder()
    out = enc.fit_transform(y)
    assert np.array_equal(enc.inverse_transform(out), y)

def test_one_hot_not_fitted():
    with pytest.raises(RuntimeError):
        OneHotEncoder().transform(np.array([0, 1]))

def test_one_hot_non_1d():
    with pytest.raises(ValueError):
        OneHotEncoder().fit(np.array([[0, 1], [2, 3]]))