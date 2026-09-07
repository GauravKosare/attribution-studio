"""SQL (DuckDB) vs. pandas journey extraction -- parity is the whole point of
having both implementations (Document 3 §2). These tests exist specifically to
catch regressions of the kind found during development: pandas and DuckDB
resolving simultaneous-timestamp ties differently once the `_seq` tie-breaker
was removed or a `date_diff` truncation crept back in.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_generator import generate_dataset
from src.journey_builder_sql import build_journeys_sql, verify_parity


def test_parity_on_small_synthetic_dataset():
    tp, conv, _ = generate_dataset(n_users=300, days=30, seed=3)
    report = verify_parity(tp, conv, lookback_days=30, collapse_repeats_minutes=30)
    assert report["match"], report
    assert report["path_str_mismatches"] == 0
    assert report["converted_mismatches"] == 0
    assert report["users_pandas"] == report["users_sql"] == 300


def test_parity_with_simultaneous_timestamps():
    # The exact edge case found in the real dataset: two touches for the same
    # user sharing an identical timestamp. Without the `_seq` tie-breaker in
    # both implementations, pandas and DuckDB resolve this ambiguously and
    # differently -- this test would have caught that regression directly.
    t = pd.Timestamp("2026-01-01 12:00:00")
    touchpoints = pd.DataFrame(
        [
            {"user_id": "u1", "touchpoint_id": "tp1", "timestamp": t, "channel": "Instagram"},
            {"user_id": "u1", "touchpoint_id": "tp2", "timestamp": t, "channel": "Facebook"},
            {"user_id": "u2", "touchpoint_id": "tp3", "timestamp": t, "channel": "Facebook"},
            {"user_id": "u2", "touchpoint_id": "tp4", "timestamp": t, "channel": "Instagram"},
        ]
    )
    conversions = pd.DataFrame(columns=["user_id", "conversion_id", "timestamp", "revenue", "converted"])

    from src.journey_builder import build_journeys

    pandas_result = build_journeys(touchpoints, conversions, lookback_days=30).set_index("user_id")
    sql_result = build_journeys_sql(touchpoints, conversions, lookback_days=30).set_index("user_id")

    assert pandas_result.loc["u1", "path_str"] == sql_result.loc["u1", "path_str"]
    assert pandas_result.loc["u2", "path_str"] == sql_result.loc["u2", "path_str"]
    # original input row order (Instagram before Facebook for u1) must be the
    # deterministic tie-breaker both sides land on
    assert pandas_result.loc["u1", "path_str"] == "Instagram > Facebook"
    assert pandas_result.loc["u2", "path_str"] == "Facebook > Instagram"


def test_parity_with_repeat_collapse_gap_near_threshold():
    # The date_diff('minute', ...) truncation bug specifically: a gap of
    # 30 minutes and 1 second must NOT collapse in either implementation.
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    touchpoints = pd.DataFrame(
        [
            {"user_id": "u1", "touchpoint_id": "a", "timestamp": t0, "channel": "A"},
            {"user_id": "u1", "touchpoint_id": "b", "timestamp": t0 + pd.Timedelta(minutes=30, seconds=1), "channel": "A"},
        ]
    )
    conversions = pd.DataFrame(columns=["user_id", "conversion_id", "timestamp", "revenue", "converted"])

    from src.journey_builder import build_journeys

    pandas_result = build_journeys(touchpoints, conversions, lookback_days=30, collapse_repeats_minutes=30)
    sql_result = build_journeys_sql(touchpoints, conversions, lookback_days=30, collapse_repeats_minutes=30)
    assert pandas_result.iloc[0]["n_touches"] == 2
    assert sql_result.iloc[0]["n_touches"] == 2


def test_sql_result_has_same_schema_as_pandas():
    tp, conv, _ = generate_dataset(n_users=50, days=15, seed=1)
    from src.journey_builder import build_journeys

    pandas_cols = set(build_journeys(tp, conv).columns)
    sql_cols = set(build_journeys_sql(tp, conv).columns)
    assert pandas_cols == sql_cols
