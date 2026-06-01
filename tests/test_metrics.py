import numpy as np
import pytest
from numcompute.metrics import accuracy, f1, roc_auc, precision, recall, mse, confusion_matrix


def test_accuracy_basic():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy(y_true, y_pred) == 0.75


def test_accuracy_all_correct():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    assert accuracy(y_true, y_pred) == 1.0


def test_accuracy_length_mismatch():
    with pytest.raises(ValueError):
        accuracy(np.array([1, 2]), np.array([1]))


def test_accuracy_empty():
    with pytest.raises(ValueError):
        accuracy(np.array([]), np.array([]))


def test_f1_binary():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    assert np.isclose(f1(y_true, y_pred), 0.8)


def test_f1_macro():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert np.isclose(f1(y_true, y_pred, average="macro"), 0.7333333333333334)


def test_f1_invalid_average():
    with pytest.raises(ValueError):
        f1(np.array([0, 1]), np.array([0, 1]), average="weighted")


def test_roc_auc_basic():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    assert np.isclose(roc_auc(y_true, y_score), 0.75)


def test_roc_auc_invalid_classes():
    with pytest.raises(ValueError):
        roc_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


def test_roc_auc_length_mismatch():
    with pytest.raises(ValueError):
        roc_auc(np.array([0, 1]), np.array([0.2]))  


def test_precision_basic():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert np.isclose(precision(y_true, y_pred), 1.0)


def test_recall_basic():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert np.isclose(recall(y_true, y_pred), 0.5)


def test_mse_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 2.0])
    assert np.isclose(mse(y_true, y_pred), 1.0 / 3.0)


def test_confusion_matrix_basic():
    y_true = np.array([0, 1, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    cm = confusion_matrix(y_true, y_pred)
    assert np.array_equal(cm, np.array([[1, 1, 0], [0, 2, 0], [0, 0, 1]]))