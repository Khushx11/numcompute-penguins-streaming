import numpy as np
import pytest
from numcompute.utils import (
    log_sum_exp, softmax, log_softmax,
    sigmoid, relu, leaky_relu,
    get_euclidean_dist, get_cosine_sim, get_manhattan_dist,
    iterate_batches, to_one_hot, clip_by_norm
)


# ── log_sum_exp ───────

def test_log_sum_exp_basic():
    # checks result matches naive calculation on small safe values
    x = np.array([1.0, 2.0, 3.0])
    expected = np.log(np.exp(1) + np.exp(2) + np.exp(3))
    assert np.isclose(log_sum_exp(x), expected)


def test_log_sum_exp_large_values():
    # naive exp(1000) overflows to inf, stable version should stay finite
    x = np.array([1000.0, 1000.0])
    result = log_sum_exp(x)
    assert np.isfinite(result)


def test_log_sum_exp_axis():
    # reducing along axis=1 should give one value per row
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = log_sum_exp(x, axis=1)
    assert result.shape == (2,)


# ── softmax ───

def test_softmax_sums_to_one():
    # core property: probabilities must sum to exactly 1
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(softmax(x).sum(), 1.0)


def test_softmax_large_input_stable():
    # max-shift trick should keep all outputs finite even for huge logits
    x = np.array([1000.0, 1001.0, 1002.0])
    result = softmax(x)
    assert np.all(np.isfinite(result))
    assert np.isclose(result.sum(), 1.0)


def test_softmax_all_equal():
    # equal logits should give uniform distribution
    x = np.ones(5)
    result = softmax(x)
    assert np.allclose(result, 0.2)


# ── log_softmax ───────

def test_log_softmax_matches_log_of_softmax():
    # should match log(softmax(x)) within floating point tolerance
    x = np.array([0.5, 1.5, 2.5])
    assert np.allclose(log_softmax(x), np.log(softmax(x)), atol=1e-6)


# ── sigmoid ───

def test_sigmoid_boundaries():
    # very negative -> ~0, zero -> 0.5, very positive -> ~1
    result = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.isclose(result[0], 0.0, atol=1e-6)
    assert np.isclose(result[1], 0.5)
    assert np.isclose(result[2], 1.0, atol=1e-6)


def test_sigmoid_no_warnings():
    # errstate should suppress the overflow from np.where evaluating both branches
    with np.errstate(over="raise", invalid="raise"):
        sigmoid(np.array([-1000.0, 0.0, 1000.0]))


# ── relu / leaky_relu ─────────

def test_relu_negative_zeroed():
    # negatives become 0, positives pass through unchanged
    x = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    result = relu(x)
    assert np.allclose(result, [0.0, 0.0, 0.0, 2.0, 5.0])


def test_leaky_relu_negative_slope():
    # negative side should be scaled by alpha, positive side unchanged
    x = np.array([-4.0, 0.0, 4.0])
    result = leaky_relu(x, alpha=0.1)
    assert np.allclose(result, [-0.4, 0.0, 4.0])


# ── euclidean distance ────────

def test_euclidean_dist_known_value():
    # 3-4-5 right triangle, hypotenuse should be exactly 5
    A = np.array([[0.0, 0.0]])
    B = np.array([[3.0, 4.0]])
    assert np.isclose(get_euclidean_dist(A, B)[0, 0], 5.0)


def test_euclidean_dist_self_is_zero():
    # distance from a point to itself must be 0 (allow small float residuals)
    A = np.random.randn(5, 3)
    D = get_euclidean_dist(A, A)
    assert np.allclose(np.diag(D), 0.0, atol=1e-6)


def test_euclidean_dist_dimension_mismatch():
    # should raise if A and B have different number of features
    A = np.ones((3, 2))
    B = np.ones((3, 4))
    with pytest.raises(ValueError):
        get_euclidean_dist(A, B)


# ── cosine similarity ─────────

def test_cosine_sim_identical_vectors():
    # same vector pointing same direction -> similarity of 1.0
    A = np.array([[1.0, 0.0, 0.0]])
    result = get_cosine_sim(A, A)
    assert np.isclose(result[0, 0], 1.0)


def test_cosine_sim_orthogonal_vectors():
    # perpendicular vectors have no shared direction -> similarity of 0
    A = np.array([[1.0, 0.0]])
    B = np.array([[0.0, 1.0]])
    assert np.isclose(get_cosine_sim(A, B)[0, 0], 0.0)


# ── manhattan distance ────────

def test_manhattan_dist_known_value():
    # L1 between (0,0) and (3,4) is 3+4=7, not 5 like euclidean
    A = np.array([[0.0, 0.0]])
    B = np.array([[3.0, 4.0]])
    assert np.isclose(get_manhattan_dist(A, B)[0, 0], 7.0)


# ── iterate_batches ────────

def test_iterate_batches_shapes():
    # 100 samples with batch_size=32 gives batches of 32,32,32,4
    X = np.arange(100).reshape(100, 1).astype(float)
    batches = list(iterate_batches(X, batch_size=32, shuffle=False))
    assert batches[0].shape == (32, 1)
    assert batches[-1].shape == (4, 1)


def test_iterate_batches_with_labels():
    # when y is provided each batch should be a (X_batch, y_batch) tuple
    X = np.ones((10, 2))
    y = np.arange(10)
    pairs = list(iterate_batches(X, y, batch_size=5, shuffle=False))
    assert len(pairs) == 2
    assert pairs[0][1].shape == (5,)


# ── to_one_hot ───────
def test_to_one_hot_shape_and_values():
    # each row should have exactly one 1.0 at the correct class index
    labels = np.array([0, 1, 2, 1])
    result = to_one_hot(labels, num_classes=3)
    assert result.shape == (4, 3)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert result[0, 0] == 1.0
    assert result[2, 2] == 1.0


def test_to_one_hot_negative_labels():
    # negative class indices don't make sense, should raise
    with pytest.raises(ValueError):
        to_one_hot(np.array([-1, 0, 1]))


# ── clip_by_norm ──────
def test_clip_by_norm_reduces_norm():
    # vector [3,4] has norm 5, capping at 1.0 should scale it down
    x = np.array([3.0, 4.0])
    result = clip_by_norm(x, max_norm=1.0)
    assert np.linalg.norm(result) <= 1.0 + 1e-9


def test_clip_by_norm_unchanged_when_small():
    # if norm is already under the cap, nothing should change
    x = np.array([0.1, 0.1])
    result = clip_by_norm(x, max_norm=10.0)
    assert np.allclose(result, x)