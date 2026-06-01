"""
visualise.py

Plotting functions for streaming metrics. All functions work with
plain lists or numpy arrays of values collected over time. Each one
can either display inline or save to a file depending on what you pass
for save_path.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(
    metric_values,
    title="Metric over time",
    ylabel="value",
    xlabel="chunk",
    color="steelblue",
    save_path=None,
    ax=None,
):
    """
    Line plot of a single metric tracked across streaming chunks.

    Typical use is plotting accuracy or loss after each chunk to see
    how the model improves as more data arrives.

    Parameters
    ----------
    metric_values : array-like
        One value per chunk.
    title : str
    ylabel : str
    xlabel : str, default 'chunk'
    color : str, default 'steelblue'
    save_path : str or None
        If given, saves the figure to this path instead of showing it.
    ax : matplotlib Axes or None
        If given, draws onto this axes instead of creating a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    values = np.asarray(metric_values, dtype=float)
    chunks = np.arange(1, len(values) + 1)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(chunks, values, color=color, linewidth=2, marker="o", markersize=4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    return ax


def compare_models(
    metric1,
    metric2,
    labels=("model 1", "model 2"),
    title="Model comparison",
    ylabel="accuracy",
    xlabel="chunk",
    colors=("steelblue", "tomato"),
    save_path=None,
):
    """
    Side by side line plot comparing two models on the same metric.

    Useful for showing the difference between a single decision tree
    and a random forest, or between random forest and bagging.

    Parameters
    ----------
    metric1 : array-like
        Metric values for the first model, one per chunk.
    metric2 : array-like
        Metric values for the second model, one per chunk.
    labels : tuple of str
        Legend labels for the two models.
    title : str
    ylabel : str
    xlabel : str
    colors : tuple of str
    save_path : str or None

    Returns
    -------
    matplotlib.axes.Axes
    """
    v1 = np.asarray(metric1, dtype=float)
    v2 = np.asarray(metric2, dtype=float)
    # use the shorter length if they differ so the plot does not break
    n = min(len(v1), len(v2))
    chunks = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chunks, v1[:n], color=colors[0], linewidth=2, marker="o",
            markersize=4, label=labels[0])
    ax.plot(chunks, v2[:n], color=colors[1], linewidth=2, marker="s",
            markersize=4, label=labels[1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return ax


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title="Predictions vs ground truth",
    save_path=None,
):
    """
    Scatter plot of predicted labels against true labels for a chunk.

    Points on the diagonal are correct predictions. Points off it are
    errors. Works best for small chunks where individual points are
    visible.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    title : str
    save_path : str or None

    Returns
    -------
    matplotlib.axes.Axes
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = y_true == y_pred
    wrong = ~correct

    fig, ax = plt.subplots(figsize=(6, 6))

    if correct.any():
        ax.scatter(
            y_true[correct], y_pred[correct],
            color="steelblue", alpha=0.7, label="correct", s=40
        )
    if wrong.any():
        ax.scatter(
            y_true[wrong], y_pred[wrong],
            color="tomato", alpha=0.7, label="wrong", s=40, marker="x"
        )

    # draw the perfect prediction diagonal
    all_vals = np.concatenate([y_true, y_pred])
    lo, hi = all_vals.min() - 0.5, all_vals.max() + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true label")
    ax.set_ylabel("predicted label")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return ax


def plot_confusion_matrix(
    matrix,
    classes,
    title="Confusion matrix",
    cmap="Blues",
    save_path=None,
):
    """
    Heatmap of a confusion matrix with class labels on both axes.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_classes, n_classes)
        Confusion matrix where rows are true and columns are predicted.
    classes : array-like
        Class label names for the axes.
    title : str
    cmap : str, default 'Blues'
    save_path : str or None

    Returns
    -------
    matplotlib.axes.Axes
    """
    matrix = np.asarray(matrix)
    n = len(classes)

    fig, ax = plt.subplots(figsize=(max(4, n), max(4, n)))
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    # write counts inside each cell
    thresh = matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=10,
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return ax


def plot_chunk_times(
    elapsed_times,
    title="Training time per chunk",
    save_path=None,
):
    """
    Bar chart of how long each chunk took to train.

    Good for spotting if training slows down as the buffer grows,
    which is expected with the rebuild-from-scratch tree approach.

    Parameters
    ----------
    elapsed_times : array-like
        Time in seconds per chunk, from StreamTrainer chunk_log.
    title : str
    save_path : str or None

    Returns
    -------
    matplotlib.axes.Axes
    """
    times = np.asarray(elapsed_times, dtype=float)
    chunks = np.arange(1, len(times) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(chunks, times, color="slategray", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("chunk")
    ax.set_ylabel("seconds")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return ax
