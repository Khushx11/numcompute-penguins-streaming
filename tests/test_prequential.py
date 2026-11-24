import numpy as np
import pytest

from numcompute_stream.stream import StreamTrainer


class RecordingClassifier:
    def __init__(self):
        self.calls = []

    def partial_fit(self, X, y, classes=None):
        self.calls.append(("fit", len(y)))
        return self

    def predict(self, X):
        self.calls.append(("predict", len(X)))
        return np.zeros(len(X), dtype=int)


def test_prequential_predicts_before_refitting_current_chunk():
    model = RecordingClassifier()
    trainer = StreamTrainer(model, classes=[0, 1])
    chunks = [
        (np.array([[0.0], [1.0]]), np.array([1, 1])),
        (np.array([[2.0], [3.0]]), np.array([0, 1])),
    ]

    entries = trainer.run_prequential(chunks)
    assert model.calls == [("fit", 2), ("predict", 2), ("fit", 2)]
    assert len(entries) == 2
    assert entries[0]["evaluated"] is False
    assert np.isnan(entries[0]["accuracy"])
    assert entries[1]["evaluated"] is True
    assert entries[1]["accuracy"] == pytest.approx(0.5)
    assert entries[1]["cumulative_accuracy"] == pytest.approx(0.5)
    assert trainer.cumulative_total == 2


def test_prequential_rejects_invalid_chunks():
    model = RecordingClassifier()
    trainer = StreamTrainer(model)
    with pytest.raises(ValueError, match="matching"):
        trainer.run_prequential([(np.array([[1.0]]), np.array([]))])
    assert model.calls == []


def test_prequential_empty_stream_has_no_metrics():
    trainer = StreamTrainer(RecordingClassifier())
    assert trainer.run_prequential([]) == []
    assert trainer.cumulative_total == 0


def test_trainer_keeps_evaluation_protocols_separate():
    model = RecordingClassifier()
    trainer = StreamTrainer(model)
    chunk = (np.zeros((4, 1)), np.ones(4, dtype=int))
    trainer.fit_chunk(*chunk)
    with pytest.raises(ValueError, match="switching evaluation modes"):
        trainer.run_prequential([chunk])
    assert model.calls == [("fit", 4), ("predict", 4)]

    trainer.reset().run_prequential([chunk, chunk])
    with pytest.raises(ValueError, match="switching evaluation modes"):
        trainer.fit_chunk(*chunk)
    assert trainer.cumulative_total == 4
    assert trainer.cumulative_accuracy_history()[-1] == 0
