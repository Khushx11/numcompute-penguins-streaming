"""
tests for numcompute_stream/metrics.py
"""

import numpy as np
import pytest
from numcompute_stream.metrics import (
    AccuracyTracker,
    LossTracker,
    ClassificationReport,
    ConfusionMatrixTracker,
    RollingAccuracy,
)


def test_accuracy_tracker_perfect():
    at = AccuracyTracker()
    at.update([0, 1, 2], [0, 1, 2])
    assert at.result() == 1.0


def test_accuracy_tracker_incremental():
    at = AccuracyTracker()
    at.update([0, 1], [0, 1])
    at.update([0, 1], [1, 0])
    assert abs(at.result() - 0.5) < 1e-9


def test_accuracy_tracker_shape_mismatch():
    at = AccuracyTracker()
    with pytest.raises(ValueError):
        at.update([0, 1], [0])


def test_accuracy_tracker_reset():
    at = AccuracyTracker()
    at.update([0, 1], [0, 1])
    at.reset()
    assert at.result() == 0.0
    assert at.total == 0


def test_loss_tracker_zero_loss():
    lt = LossTracker()
    lt.update([1.0, 2.0], [1.0, 2.0])
    assert lt.result() == 0.0


def test_loss_tracker_incremental():
    lt = LossTracker()
    lt.update([0.0], [1.0])
    lt.update([0.0], [1.0])
    assert abs(lt.result() - 1.0) < 1e-9


def test_classification_report_macro_f1():
    cr = ClassificationReport()
    cr.update([0, 1, 0, 1], [0, 1, 0, 1])
    result = cr.result()
    assert abs(result["macro_f1"] - 1.0) < 1e-9


def test_classification_report_incremental():
    cr = ClassificationReport()
    cr.update([0, 1], [0, 1])
    cr.update([0, 1], [1, 0])
    result = cr.result()
    assert "macro_f1" in result


def test_confusion_matrix_tracker_shape():
    cmt = ConfusionMatrixTracker()
    cmt.update([0, 1, 2], [0, 1, 2])
    result = cmt.result()
    assert result["matrix"].shape == (3, 3)


def test_confusion_matrix_tracker_accumulates():
    cmt = ConfusionMatrixTracker()
    cmt.update([0, 1], [0, 1])
    cmt.update([0, 1], [0, 1])
    result = cmt.result()
    assert result["matrix"].sum() == 4


def test_rolling_accuracy_window():
    ra = RollingAccuracy(window=2)
    ra.update([0, 1], [0, 1])  # acc 1.0
    ra.update([0, 1], [1, 0])  # acc 0.0
    ra.update([0, 1], [0, 1])  # acc 1.0, first chunk drops out
    # window now has chunk2 (0.0) and chunk3 (1.0) = 0.5
    assert abs(ra.result() - 0.5) < 1e-9


def test_rolling_accuracy_reset():
    ra = RollingAccuracy(window=3)
    ra.update([0, 1], [0, 1])
    ra.reset()
    assert ra.result() == 0.0

def test_confusion_matrix_preserves_old_counts_when_new_class_sorts_between():
    tracker = ConfusionMatrixTracker()
    tracker.update([1, 3], [3, 1])
    tracker.update([2], [2])
    actual = tracker.result()
    np.testing.assert_array_equal(actual["classes"], [1, 2, 3])
    np.testing.assert_array_equal(actual["matrix"], [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ])


@pytest.mark.parametrize("actual,predicted", [
    ([1, 2], [1]),
    ([[1, 2]], [[1, 2]]),
    ([], []),
])
def test_confusion_matrix_rejects_invalid_chunk_shapes(actual, predicted):
    tracker = ConfusionMatrixTracker()
    with pytest.raises(ValueError):
        tracker.update(actual, predicted)
