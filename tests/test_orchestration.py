"""Prefect orchestration smoke test — confirms the flow actually runs
end-to-end (load -> compute -> persist) via Prefect's task/flow machinery,
not just that the underlying pipeline function works (already covered by
tests/test_pipeline.py). Uses synthetic data and a small n_users to keep this
fast; Prefect's ephemeral local server adds real overhead (~15-20s) to spin
up, so this is deliberately one test, not a full suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestration.attribution_flow import OUTPUT_DIR, attribution_pipeline_flow


@pytest.mark.timeout(90)
def test_flow_runs_end_to_end_and_persists_result():
    output_path = attribution_pipeline_flow(source="synthetic", n_users=500, seed=1)

    path = Path(output_path)
    assert path.exists()
    try:
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        assert "error" not in result
        assert result["summary"]["total_journeys"] == 500
        assert result["roi_model_used"] == "markov"
        assert len(result["roi"]) > 0
    finally:
        path.unlink(missing_ok=True)
        # OUTPUT_DIR itself is left in place (data/processed/ is a real,
        # tracked directory in the repo layout) -- only the one generated
        # file this test created is cleaned up.
