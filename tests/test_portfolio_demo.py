from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
