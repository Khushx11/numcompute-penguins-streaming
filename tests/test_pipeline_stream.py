"""
tests for numcompute_stream/pipeline.py and stream.py
"""

import numpy as np
import pytest
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.io import make_stream_chunks


def _data():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] + X[:, 2] > 0).astype(int)
    return X, y


def test_pipeline_partial_fit_predict():
    X, y = _data()
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=3, random_state=0)),
    ])
    pipe.partial_fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (60,)


def test_pipeline_score():
    X, y = _data()
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=3, random_state=0)),
    ])
    pipe.fit(X, y)
    score = pipe.score(X, y)
    assert 0.0 <= score <= 1.0


def test_pipeline_predict_before_fit_raises():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier()),
    ])
    with pytest.raises(RuntimeError):
        pipe.predict(np.ones((5, 3)))


def test_pipeline_get_step():
    sc = StandardScaler()
    pipe = Pipeline([("scale", sc), ("model", DecisionTreeClassifier())])
    assert pipe.get_step("scale") is sc


def test_pipeline_duplicate_names_raises():
    with pytest.raises(ValueError):
        Pipeline([("step", StandardScaler()), ("step", StandardScaler())])


def test_stream_trainer_basic():
    X, y = _data()
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    trainer = StreamTrainer(clf, classes=[0, 1])
    chunks = make_stream_chunks(X, y, chunk_size=15)
    trainer.run(chunks)
    assert trainer.n_chunks == 4
    assert len(trainer.chunk_log) == 4


def test_stream_trainer_accuracy_in_range():
    X, y = _data()
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    trainer = StreamTrainer(clf, classes=[0, 1])
    chunks = make_stream_chunks(X, y, chunk_size=20)
    trainer.run(chunks)
    for entry in trainer.chunk_log:
        assert 0.0 <= entry["accuracy"] <= 1.0
        assert 0.0 <= entry["cumulative_accuracy"] <= 1.0


def test_stream_trainer_reset():
    X, y = _data()
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    trainer = StreamTrainer(clf, classes=[0, 1])
    chunks = make_stream_chunks(X, y, chunk_size=20)
    trainer.run(chunks)
    trainer.reset()
    assert trainer.n_chunks == 0
    assert trainer.chunk_log == []


def test_stream_trainer_with_pipeline():
    X, y = _data()
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=3, random_state=0)),
    ])
    trainer = StreamTrainer(pipe, classes=[0, 1])
    chunks = make_stream_chunks(X, y, chunk_size=20)
    trainer.run(chunks)
    assert trainer.n_chunks == 3


def test_streaming_preprocessing_keeps_buffer_in_one_feature_space():
    first = np.array([[0.0], [2.0]])
    second = np.array([[12.0], [16.0]])
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=3, random_state=0)),
    ])
    pipe.partial_fit(first, [0, 1])
    expected = pipe.transform(np.vstack([first, second]))
    pipe.partial_fit(second, [0, 1])
    np.testing.assert_allclose(pipe._estimator._X_buf, expected)
    np.testing.assert_allclose(pipe.transform(np.vstack([first, second])), expected)


def test_failed_update_does_not_count_prequential_predictions():
    class RejectsSecondChunk:
        def partial_fit(self, X, y):
            if X[0, 0] == 1:
                raise ValueError("update failed")

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    trainer = StreamTrainer(RejectsSecondChunk())
    trainer.run_prequential([(np.array([[0.0]]), [0])])
    with pytest.raises(ValueError, match="update failed"):
        trainer.run_prequential([(np.array([[1.0]]), [0])])
    assert trainer.n_chunks == 1
    assert trainer.cumulative_total == 0
