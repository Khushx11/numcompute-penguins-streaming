import numpy as np


def accuracy(y_true, y_pred):
    """
    Computes the fraction of predictions that match the true labels.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.

    Returns
    -------
    float
        The accuracy score.

    Raises
    ------
    ValueError
        If inputs are not 1-D, shapes mismatch, or inputs are empty.

    Complexity
    ----------
    Time  : O(n)
    Space : O(1)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must both be 1-D")
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("inputs must not be empty")

    return float(np.sum(y_true == y_pred) / len(y_true))

def f1(y_true, y_pred, average="binary"):
    """
    Computes the F1 score, the harmonic mean of precision and recall.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    average : str, optional
        'binary' or 'macro', default is 'binary'.

    Returns
    -------
    float
        The F1 score.

    Raises
    ------
    ValueError
        If inputs are not 1-D, shapes mismatch, or average is invalid.

    Complexity
    ----------
    Time  : O(n * C) where C is number of classes
    Space : O(C)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must both be 1-D")
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if average not in ("binary", "macro"):
        raise ValueError("average must be 'binary' or 'macro'")

    classes = np.unique(y_true)

    def _f1_for_class(cls):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        return (2 * precision * recall / denom) if denom > 0 else 0.0

    if average == "binary":
        return _f1_for_class(classes[-1])

    return float(np.mean([_f1_for_class(c) for c in classes]))

def roc_auc(y_true, y_score):
    """
    Computes the ROC AUC score using the trapezoidal rule.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground truth binary labels.
    y_score : array-like, shape (n,)
        Target scores, can either be probability estimates or confidence values.

    Returns
    -------
    float
        The ROC AUC score.

    Raises
    ------
    ValueError
        If inputs are not 1-D, shapes mismatch, or y_true does not have exactly two classes.

    Complexity
    ----------
    Time  : O(n log n) due to sorting
    Space : O(n)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must both be 1-D")
    if len(y_true) != len(y_score):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_score)}")
    if len(np.unique(y_true)) != 2:
        raise ValueError("y_true must contain exactly two classes")

    thresholds = np.concatenate([[np.inf], np.sort(y_score)[::-1]])
    tpr_list = []
    fpr_list = []

    # find the two classes dynamically instead of hardcoding 0 and 1
    classes = np.unique(y_true)
    neg_class, pos_class = classes[0], classes[1]
    pos = np.sum(y_true == pos_class)
    neg = np.sum(y_true == neg_class)

    for thresh in thresholds:
        pred = (y_score >= thresh)
        tp = np.sum(pred & (y_true == pos_class))
        fp = np.sum(pred & (y_true == neg_class))
        tpr_list.append(tp / pos if pos > 0 else 0.0)
        fpr_list.append(fp / neg if neg > 0 else 0.0)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    order = np.argsort(fpr_arr)
    return float(np.trapezoid(tpr_arr[order], fpr_arr[order]))


def precision(y_true, y_pred, pos_label=1):
    """
    Computes the precision score.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    pos_label : int, optional
        The class to report if average='binary', default is 1.

    Returns
    -------
    float
        The precision score, 0.0 if no positive predictions.

    Complexity
    ----------
    Time  : O(n)
    Space : O(1)
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred, pos_label=1):
    """
    Computes the recall score.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    pos_label : int, optional
        The class to report if average='binary', default is 1.

    Returns
    -------
    float
        The recall score, 0.0 if no actual positives.

    Complexity
    ----------
    Time  : O(n)
    Space : O(1)
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def mse(y_true, y_pred):
    """
    Computes the mean squared error.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground truth target values.
    y_pred : array-like, shape (n,)
        Estimated target values.

    Returns
    -------
    float
        The mean squared error.

    Complexity
    ----------
    Time  : O(n)
    Space : O(n) for difference array
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix to evaluate the accuracy of a classification.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground truth class labels.
    y_pred : array-like, shape (n,)
        Predicted class labels.

    Returns
    -------
    np.ndarray, shape (n_classes, n_classes)
        Confusion matrix where rows are true classes and columns are predicted classes.

    Complexity
    ----------
    Time  : O(n log C) where C is number of distinct classes
    Space : O(C^2) for the matrix output
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    # use searchsorted to map class labels to 0..n-1 indices vectorised
    row_idx = np.searchsorted(classes, y_true)
    col_idx = np.searchsorted(classes, y_pred)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (row_idx, col_idx), 1)
    return cm