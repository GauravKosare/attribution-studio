"""Time-series forecasting — validated via walk-forward backtesting: a
synthetic series with a known, generated trend+seasonality pattern (where the
model SHOULD win) and a check that the code degrades gracefully, not
silently, on data too short to forecast reliably.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import backtest, daily_series_from_journeys, forecast_channel_metric


@pytest.fixture
def trending_seasonal_series():
    rng = np.random.default_rng(1)
    days = pd.date_range("2026-01-01", periods=120, freq="D")
    trend = np.linspace(50, 150, 120)
    weekly = 20 * np.sin(2 * np.pi * np.arange(120) / 7)
    noise = rng.normal(0, 5, 120)
    return pd.Series(trend + weekly + noise, index=days).clip(lower=0)


def test_daily_series_from_journeys_counts_touches_correctly():
    t0 = pd.Timestamp("2026-01-01")
    journeys = pd.DataFrame(
        [
            {"path": ["A", "B"], "timestamps": [t0, t0 + pd.Timedelta(days=1)], "converted": True, "journey_end": t0 + pd.Timedelta(days=1)},
            {"path": ["A"], "timestamps": [t0], "converted": False, "journey_end": t0},
            {"path": ["A", "A"], "timestamps": [t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=1, hours=2)], "converted": True, "journey_end": t0 + pd.Timedelta(days=1, hours=2)},
        ]
    )
    series = daily_series_from_journeys(journeys, "A", metric="touches")
    # day 0: journey1's A + journey2's A = 2; day 1: journey3's two A's = 2
    assert series[pd.Timestamp("2026-01-01")] == 2
    assert series[pd.Timestamp("2026-01-02")] == 2


def test_daily_series_conversions_metric():
    t0 = pd.Timestamp("2026-01-01")
    journeys = pd.DataFrame(
        [
            {"path": ["A"], "timestamps": [t0], "converted": True, "journey_end": t0},
            {"path": ["A"], "timestamps": [t0], "converted": False, "journey_end": t0},
            {"path": ["B"], "timestamps": [t0], "converted": True, "journey_end": t0},
        ]
    )
    series = daily_series_from_journeys(journeys, "A", metric="conversions")
    assert series[pd.Timestamp("2026-01-01")] == 1  # only the converted A journey counts


def test_forecast_beats_naive_with_adequate_history(trending_seasonal_series):
    result = backtest(trending_seasonal_series, holdout=14, seasonal_periods=7)
    assert "error" not in result
    assert result["beats_naive_baseline"]
    assert result["mae_model"] < result["mae_naive_baseline"] * 0.7  # not just barely


def test_confidence_interval_coverage_is_roughly_calibrated(trending_seasonal_series):
    # a well-calibrated 90% CI should cover the true value close to 90% of
    # the time on data resembling the process it was fit on
    result = backtest(trending_seasonal_series, holdout=14, seasonal_periods=7)
    assert result["ci_coverage"] >= 0.6  # generous bound -- this is a small-N check, not exact


def test_short_series_fails_gracefully_not_silently():
    tiny = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2026-01-01", periods=3))
    result = forecast_channel_metric(tiny, periods=5)
    assert "error" in result

    result2 = backtest(pd.Series(np.arange(10.0), index=pd.date_range("2026-01-01", periods=10)), holdout=7)
    assert "error" in result2  # 10 days - 7 holdout = 3 days train, below the 4-day minimum


def test_forecast_returns_nonnegative_lower_bound(trending_seasonal_series):
    result = forecast_channel_metric(trending_seasonal_series, periods=10, seasonal_periods=7)
    assert (result["lower"] >= 0).all()


def test_empty_channel_returns_empty_series():
    journeys = pd.DataFrame(
        [{"path": ["B"], "timestamps": [pd.Timestamp("2026-01-01")], "converted": True, "journey_end": pd.Timestamp("2026-01-01")}]
    )
    series = daily_series_from_journeys(journeys, "NoSuchChannel", metric="touches")
    assert series.empty
