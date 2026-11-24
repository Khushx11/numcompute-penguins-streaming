from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portfolio_demo
from portfolio_demo import evaluate


def test_real_data_excludes_warmup_and_reports_buffer():
    report = evaluate("penguins", "tree", 40)
    assert report["samples"] == 342
    assert report["history"][0]["cumulative"] is None
    assert report["history"][0]["evaluated"] == 0
    last = report["history"][-1]
    assert last["evaluated"] == 302
    assert last["buffer_bytes"] == 342 * 4 * 8
    assert last["cumulative"] == pytest.approx(285 / 302)
    assert last["majority"] == pytest.approx(133 / 302)


def test_drift_stream_scores_after_first_chunk_only():
    result = evaluate("drift", "forest", 80)
    assert result["samples"] == 320
    assert [row["evaluated"] for row in result["history"]] == [0, 80, 160, 240]


def test_majority_baseline_uses_only_labels_from_earlier_chunks(monkeypatch):
    features = np.arange(80, dtype=float).reshape(-1, 1)
    labels = np.repeat([0, 1, 1, 1], 20)
    monkeypatch.setattr(
        portfolio_demo, "load_stream", lambda dataset: (features, labels, "test stream")
    )

    history = evaluate("drift", "tree", 20)["history"]

    assert [row["evaluated"] for row in history] == [0, 20, 40, 60]
    assert [row["majority"] for row in history] == [None, 0.0, 0.0, 1 / 3]
