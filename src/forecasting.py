"""Time-series forecasting for channel-level daily metrics (touches,
conversions, spend) — extends this project's snapshot-in-time attribution
numbers (one month, one set of credit shares) with a forward-looking view:
is a channel's performance a stable pattern or a one-month fluke?

Validated the same way every other model in this project is: walk-forward
backtesting against a held-out tail of the real series, compared to a naive
baseline. A forecasting model that can't beat "just repeat the last value"
has no business being presented as insight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def daily_series_from_journeys(journeys: pd.DataFrame, channel: str, metric: str = "touches") -> pd.Series:
    """Build a daily time series for one channel from the journeys table.

    `metric`: "touches" (count of touchpoints on this channel per day) or
    "conversions" (count of journeys that converted AND touched this channel,
    indexed by journey_end date). Vectorized via explode, not a per-journey
    Python loop -- same pattern as src/attribution/ml_shap.py's featurizer.
    """
    if metric == "touches":
        exploded = journeys[["path", "timestamps"]].explode(["path", "timestamps"])
        dates = exploded.loc[exploded["path"] == channel, "timestamps"].dropna()
        dates = pd.to_datetime(dates).dt.normalize()
    elif metric == "conversions":
        touched = journeys["path"].apply(lambda p: channel in p)
        rows = journeys.loc[touched & journeys["converted"]]
        dates = pd.to_datetime(rows["journey_end"]).dt.normalize()
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    if dates.empty:
        return pd.Series(dtype=float)
    counts = dates.value_counts().sort_index()
    full_range = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    return counts.reindex(full_range, fill_value=0).astype(float)


def _fit(series: pd.Series, periods: int, seasonal_periods: int | None):
    seasonal = "add" if seasonal_periods and len(series) >= 2 * seasonal_periods else None
    model = ExponentialSmoothing(
        series, trend="add", seasonal=seasonal, seasonal_periods=seasonal_periods,
        damped_trend=True, initialization_method="estimated",
    ).fit(optimized=True)
    forecast = model.forecast(periods)
    resid_std = float(np.std(model.resid)) if len(model.resid) else 0.0
    return model, forecast, resid_std


def forecast_channel_metric(
    daily_series: pd.Series, periods: int = 14, ci: float = 0.90, seasonal_periods: int | None = 7
) -> dict:
    """Forecast `periods` days forward. Returns forecast + a CI band derived
    from in-sample residual std (a simple, transparent interval — not a full
    state-space prediction interval, which statsmodels' ETS doesn't expose
    directly without a heavier SARIMAX formulation)."""
    if len(daily_series) < 4:
        return {"error": "Need at least 4 days of history to fit a trend."}

    from scipy import stats

    model, forecast, resid_std = _fit(daily_series, periods, seasonal_periods)
    z = stats.norm.ppf(0.5 + ci / 2)
    lower = forecast - z * resid_std
    upper = forecast + z * resid_std

    return {
        "forecast": forecast,
        "lower": lower.clip(lower=0),
        "upper": upper,
        "ci": ci,
        "in_sample_resid_std": resid_std,
        "trend": "damped additive",
        "seasonal_periods_used": seasonal_periods if model.model.seasonal else None,
    }


def backtest(daily_series: pd.Series, holdout: int = 7, seasonal_periods: int | None = 7) -> dict:
    """Walk-forward validation: fit on everything except the last `holdout`
    days, forecast those days, compare to what actually happened AND to a
    naive "repeat the last training value" baseline.
    """
    if len(daily_series) < holdout + 4:
        return {"error": f"Need at least {holdout + 4} days of history for a {holdout}-day backtest."}

    train, test = daily_series.iloc[:-holdout], daily_series.iloc[-holdout:]
    result = forecast_channel_metric(train, periods=holdout, ci=0.90, seasonal_periods=seasonal_periods)
    if "error" in result:
        return result

    forecast_vals = result["forecast"].to_numpy()
    test_vals = test.to_numpy()
    mae_model = float(np.mean(np.abs(forecast_vals - test_vals)))

    naive_forecast = np.full(holdout, train.iloc[-1])
    mae_naive = float(np.mean(np.abs(naive_forecast - test_vals)))

    lower_vals, upper_vals = result["lower"].to_numpy(), result["upper"].to_numpy()
    coverage = float(np.mean((test_vals >= lower_vals) & (test_vals <= upper_vals)))

    return {
        "mae_model": mae_model,
        "mae_naive_baseline": mae_naive,
        "beats_naive_baseline": mae_model < mae_naive,
        "ci_coverage": coverage,  # should be roughly >= the stated CI level if intervals are honest
        "holdout_days": holdout,
    }
