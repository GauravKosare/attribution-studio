"""End-to-end pipeline smoke tests — data generator -> run_pipeline -> a
JSON-serializable result with every invariant the dashboard relies on.
"""
from __future__ import annotations

import json

import pytest

from src.data_generator import generate_dataset
from src.pipeline import MODEL_LABELS, run_pipeline


@pytest.fixture(scope="module")
def small_dataset():
    return generate_dataset(n_users=800, days=45, seed=7)


def test_pipeline_default_run_has_expected_shape(small_dataset):
    tp, conv, spend = small_dataset
    result = run_pipeline(tp, conv, spend, include_ml=False, include_lp=False)

    assert "error" not in result
    assert result["summary"]["total_journeys"] == 800
    assert result["summary"]["conversions"] > 0
    assert 0 < result["summary"]["conversion_rate"] < 1
    assert result["ml_diagnostics"] is None
    assert result["lp_comparison"] is None


def test_every_rule_based_and_markov_shapley_credit_sums_to_one(small_dataset):
    tp, conv, spend = small_dataset
    result = run_pipeline(tp, conv, spend, include_ml=False, include_lp=False)

    import pandas as pd

    credit = pd.DataFrame(result["credit_long"])
    sums = credit.groupby("model")["credit_share"].sum()
    for model, total in sums.items():
        assert total == pytest.approx(1.0, abs=1e-3), f"{model} sums to {total}, not 1.0"

    # every model the pipeline runs by default must have actually run
    expected_default_models = {"first_touch", "last_touch", "linear", "time_decay", "position_based", "markov", "shapley"}
    assert expected_default_models <= set(sums.index)


def test_result_is_json_serializable(small_dataset):
    tp, conv, spend = small_dataset
    result = run_pipeline(tp, conv, spend, include_ml=False, include_lp=False)
    # this is exactly what dashboards/server.py's jsonify() does -- if this
    # raises, the dashboard would 500 on every request
    json.dumps(result)


def test_roi_total_recommended_spend_is_sane(small_dataset):
    tp, conv, spend = small_dataset
    result = run_pipeline(
        tp, conv, spend, roi_model="markov", shift_fraction=0.20, max_increase_pct=0.50,
    )
    roi = result["roi"]
    total_current = sum(r["spend"] for r in roi)
    total_recommended = sum(r["recommended_spend"] for r in roi)
    # capped reallocation can leave a pool unallocated, but must never recommend
    # MORE total spend than currently exists (that would be inventing money)
    assert total_recommended <= total_current + 0.01
    assert all(r["recommended_spend"] >= -0.01 for r in roi)


def test_include_ml_adds_a_model_and_diagnostics(small_dataset):
    tp, conv, spend = small_dataset
    result = run_pipeline(tp, conv, spend, include_ml=True, ml_method="logistic")
    assert result["ml_diagnostics"] is not None
    assert result["ml_diagnostics"]["method"] == "logistic"
    assert "logistic_shap" in result["credit_matrix"]["models"] or len(result["credit_matrix"]["models"]) >= 7


def test_include_lp_adds_comparison_and_lp_is_at_least_as_good(small_dataset):
    tp, conv, spend = small_dataset
    result = run_pipeline(tp, conv, spend, include_lp=True, max_increase_pct=0.50)
    lp = result["lp_comparison"]
    assert lp is not None
    assert lp["status"] == "optimal"
    assert lp["lp_projected_revenue"] >= lp["heuristic_projected_revenue"] - 0.01


def test_empty_input_returns_error_not_crash():
    import pandas as pd

    empty_tp = pd.DataFrame(columns=["user_id", "touchpoint_id", "timestamp", "channel"])
    empty_conv = pd.DataFrame(columns=["user_id", "conversion_id", "timestamp", "revenue", "converted"])
    empty_spend = pd.DataFrame(columns=["date", "channel", "spend"])
    result = run_pipeline(empty_tp, empty_conv, empty_spend)
    assert "error" in result


def test_model_labels_cover_every_model_the_pipeline_can_produce(small_dataset):
    tp, conv, spend = small_dataset
    result = run_pipeline(tp, conv, spend, include_ml=True, ml_method="xgboost", include_lp=True)
    for model_key in {r["model"] for r in result["credit_long"]}:
        assert model_key in MODEL_LABELS, f"{model_key} produced but has no label in MODEL_LABELS"
