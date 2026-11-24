import numpy as np
import pytest
from numcompute.preprocessing import StandardScaler, MinMaxScaler, Imputer
from numcompute.pipeline import Compose, FeatureUnion


X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])


def test_compose_fit_transform_shape():
    pipe = Compose([StandardScaler()])
    out = pipe.fit_transform(X)
    assert out.shape == X.shape

def test_compose_single_step():
    pipe = Compose([StandardScaler()])
    out = pipe.fit_transform(X)
    assert np.allclose(out.mean(axis=0), 0.0)

def test_compose_two_steps():
    pipe = Compose([StandardScaler(), MinMaxScaler()])
    out = pipe.fit_transform(X)
    assert np.allclose(out.min(axis=0), 0.0)
    assert np.allclose(out.max(axis=0), 1.0)

def test_compose_transform_uses_fitted():
    pipe = Compose([StandardScaler()])
    pipe.fit(X)
    out = pipe.transform(np.array([[3.0, 4.0]]))
    assert out.shape == (1, 2)

def test_compose_empty_steps():
    with pytest.raises(ValueError):
        Compose([])

def test_compose_invalid_step():
    with pytest.raises(TypeError):
        Compose(["not_a_transformer"])

def test_compose_with_imputer():
    X_miss = np.array([[1.0, np.nan],
                       [3.0, 4.0],
                       [5.0, 6.0]])
    pipe = Compose([Imputer(strategy="mean"), StandardScaler()])
    out = pipe.fit_transform(X_miss)
    assert not np.any(np.isnan(out))
    assert out.shape == (3, 2)


def test_feature_union_output_columns():
    union = FeatureUnion([StandardScaler(), MinMaxScaler()])
    out = union.fit_transform(X)
    assert out.shape == (3, 4)

def test_feature_union_first_half():
    union = FeatureUnion([StandardScaler(), MinMaxScaler()])
    out = union.fit_transform(X)
    assert np.allclose(out[:, :2].mean(axis=0), 0.0)

def test_feature_union_second_half():
    union = FeatureUnion([StandardScaler(), MinMaxScaler()])
    out = union.fit_transform(X)
    assert np.allclose(out[:, 2:].min(axis=0), 0.0)
    assert np.allclose(out[:, 2:].max(axis=0), 1.0)

def test_feature_union_transform_uses_fitted():
    union = FeatureUnion([StandardScaler(), MinMaxScaler()])
    union.fit(X)
    out = union.transform(np.array([[3.0, 4.0]]))
    assert out.shape == (1, 4)

def test_feature_union_empty_list():
    with pytest.raises(ValueError):
        FeatureUnion([])

def test_feature_union_invalid_transformer():
    with pytest.raises(TypeError):
        FeatureUnion([42])