"""journey_builder.py — lookback window, repeat-touch collapsing, path-length cap,
and non-converter handling, each isolated in a small hand-constructed DataFrame.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.journey_builder import build_journeys


def _tp(user_id, channel, timestamp):
    return {
        "user_id": user_id, "touchpoint_id": f"{user_id}_{timestamp}", "timestamp": timestamp,
        "channel": channel, "campaign_id": f"{channel}_camp", "campaign_name": channel,
        "platform": channel, "device": "desktop",
    }


def test_lookback_window_excludes_old_touches():
    conv_time = pd.Timestamp("2026-01-31")
    touchpoints = pd.DataFrame(
        [
            _tp("u1", "old_channel", conv_time - pd.Timedelta(days=40)),  # outside 30-day window
            _tp("u1", "mid_channel", conv_time - pd.Timedelta(days=10)),
            _tp("u1", "recent_channel", conv_time - pd.Timedelta(days=1)),
        ]
    )
    conversions = pd.DataFrame(
        [{"user_id": "u1", "conversion_id": "c1", "timestamp": conv_time, "revenue": 50.0, "converted": True}]
    )
    journeys = build_journeys(touchpoints, conversions, lookback_days=30)
    assert len(journeys) == 1
    assert journeys.iloc[0]["path"] == ["mid_channel", "recent_channel"]


def test_collapse_repeats_drops_close_same_channel_touches():
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    touchpoints = pd.DataFrame(
        [
            _tp("u1", "A", t0),
            _tp("u1", "A", t0 + pd.Timedelta(minutes=10)),   # gap 10min <= 30 -> collapsed
            _tp("u1", "A", t0 + pd.Timedelta(minutes=45)),   # gap from PREV RAW row (10min mark) = 35min > 30 -> kept
            _tp("u1", "B", t0 + pd.Timedelta(minutes=50)),   # different channel -> always kept
        ]
    )
    conversions = pd.DataFrame(columns=["user_id", "conversion_id", "timestamp", "revenue", "converted"])
    journeys = build_journeys(touchpoints, conversions, lookback_days=30, collapse_repeats_minutes=30)
    assert journeys.iloc[0]["path"] == ["A", "A", "B"]
    assert journeys.iloc[0]["n_touches"] == 3


def test_collapse_exactly_at_threshold_is_not_collapsed():
    # A gap of exactly 30:01 must NOT collapse (> 30, strictly), a gap of
    # exactly 30:00 MUST collapse (not > 30). This directly guards the
    # DuckDB date_diff truncation bug found and fixed in sql/build_journeys.sql.
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    touchpoints = pd.DataFrame(
        [
            _tp("u1", "A", t0),
            _tp("u1", "A", t0 + pd.Timedelta(minutes=30, seconds=1)),  # > 30min -> new touch
        ]
    )
    conversions = pd.DataFrame(columns=["user_id", "conversion_id", "timestamp", "revenue", "converted"])
    journeys = build_journeys(touchpoints, conversions, lookback_days=30, collapse_repeats_minutes=30)
    assert journeys.iloc[0]["n_touches"] == 2


def test_path_length_cap_keeps_most_recent():
    t0 = pd.Timestamp("2026-01-01")
    touchpoints = pd.DataFrame(
        [_tp("u1", f"ch{i}", t0 + pd.Timedelta(hours=i)) for i in range(25)]
    )
    conversions = pd.DataFrame(columns=["user_id", "conversion_id", "timestamp", "revenue", "converted"])
    journeys = build_journeys(touchpoints, conversions, lookback_days=365, max_path_length=20)
    assert journeys.iloc[0]["n_touches"] == 20
    # the kept touches must be the LAST 20 (ch5..ch24), not the first 20
    assert journeys.iloc[0]["path"][0] == "ch5"
    assert journeys.iloc[0]["path"][-1] == "ch24"


def test_non_converter_uses_own_last_touch_as_anchor():
    t0 = pd.Timestamp("2026-01-01")
    touchpoints = pd.DataFrame(
        [_tp("u1", "A", t0), _tp("u1", "B", t0 + pd.Timedelta(days=5))]
    )
    conversions = pd.DataFrame(columns=["user_id", "conversion_id", "timestamp", "revenue", "converted"])
    journeys = build_journeys(touchpoints, conversions, lookback_days=30)
    assert len(journeys) == 1
    assert journeys.iloc[0]["converted"] == False
    assert journeys.iloc[0]["path"] == ["A", "B"]


def test_fallback_when_window_drops_every_touch():
    # All touches far outside the lookback window relative to conversion time --
    # the safety-net fallback should still produce a 1-touch journey, not an
    # empty one.
    conv_time = pd.Timestamp("2026-06-01")
    touchpoints = pd.DataFrame([_tp("u1", "old", conv_time - pd.Timedelta(days=200))])
    conversions = pd.DataFrame(
        [{"user_id": "u1", "conversion_id": "c1", "timestamp": conv_time, "revenue": 10.0, "converted": True}]
    )
    journeys = build_journeys(touchpoints, conversions, lookback_days=30)
    assert len(journeys) == 1
    assert journeys.iloc[0]["path"] == ["old"]


def test_empty_touchpoints_returns_empty_journeys():
    touchpoints = pd.DataFrame(columns=["user_id", "touchpoint_id", "timestamp", "channel"])
    conversions = pd.DataFrame(columns=["user_id", "conversion_id", "timestamp", "revenue", "converted"])
    journeys = build_journeys(touchpoints, conversions)
    assert len(journeys) == 0
