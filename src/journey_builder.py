"""Build the customer-journey table from raw touchpoints + conversions.

Works on ANY input that matches the schema in docs/02_data_dictionary.md — this is
what makes the dashboard "dynamic": swap the CSVs, everything downstream recomputes.

Vectorized (no per-row Python loops) so it scales to real datasets with hundreds of
thousands of users, not just small synthetic samples.
"""
from __future__ import annotations

import pandas as pd


def build_journeys(
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    lookback_days: int = 30,
    collapse_repeats_minutes: int = 30,
    max_path_length: int = 20,
) -> pd.DataFrame:
    """Return one row per user: ordered channel path, timestamps, conversion outcome.

    Parameters mirror the rules documented in docs/02_data_dictionary.md so behavior
    is auditable and adjustable from the dashboard UI.
    """
    tp = touchpoints.copy()
    tp["timestamp"] = pd.to_datetime(tp["timestamp"])
    # deterministic tie-breaker for touches sharing an identical timestamp (real
    # data has these): original input row order. Kept as an explicit column (not
    # just sort stability) so journey_builder_sql.py can reproduce the exact same
    # order and the two implementations agree even on ties — see sql/build_journeys.sql
    tp["_seq"] = range(len(tp))
    tp = tp.sort_values(["user_id", "timestamp", "_seq"]).reset_index(drop=True)

    conv = conversions.copy()
    if not conv.empty:
        conv["timestamp"] = pd.to_datetime(conv["timestamp"])
        conv = conv.drop_duplicates(subset="user_id", keep="first").set_index("user_id")

    # anchor time per touchpoint: this user's conversion time if converted, else
    # this user's own last-touch time (so non-converters still get a full window)
    last_touch_time = tp.groupby("user_id")["timestamp"].transform("max")
    if not conv.empty:
        conv_time_map = tp["user_id"].map(conv["timestamp"])
        anchor_time = conv_time_map.fillna(last_touch_time)
    else:
        anchor_time = last_touch_time

    window_start = anchor_time - pd.Timedelta(days=lookback_days)
    in_window = (tp["timestamp"] >= window_start) & (tp["timestamp"] <= anchor_time)
    tpw = tp[in_window]

    # safety net: a user whose window filter drops every touch keeps their last touch
    missing = tp.loc[~tp["user_id"].isin(tpw["user_id"].unique()), "user_id"].unique()
    if len(missing):
        fallback = tp[tp["user_id"].isin(missing)].groupby("user_id", sort=False).tail(1)
        tpw = pd.concat([tpw, fallback], ignore_index=True).sort_values(["user_id", "timestamp", "_seq"])

    # collapse consecutive same-channel touches within N minutes of the PREVIOUS raw
    # touch (kept or not) — equivalent to a sequential scan, done here via shift()
    g = tpw.groupby("user_id", sort=False)
    prev_channel = g["channel"].shift(1)
    prev_time = g["timestamp"].shift(1)
    gap_minutes = (tpw["timestamp"] - prev_time).dt.total_seconds() / 60.0
    new_run = (tpw["channel"] != prev_channel) | (gap_minutes > collapse_repeats_minutes) | prev_time.isna()
    tpc = tpw[new_run]

    # cap path length: keep the most recent max_path_length touches per user
    # (fully vectorized reverse-rank filter — no per-group Python loop)
    if not tpc.empty:
        rev_rank = tpc.groupby("user_id", sort=False).cumcount(ascending=False)
        tpc = tpc[rev_rank < max_path_length]

    if tpc.empty:
        return pd.DataFrame(
            columns=[
                "user_id", "path", "path_str", "timestamps", "n_touches", "converted",
                "conversion_value", "journey_start", "journey_end", "days_to_convert",
            ]
        )

    grouped = tpc.groupby("user_id", sort=False).agg(
        path=("channel", list), timestamps=("timestamp", list)
    )
    grouped["path_str"] = grouped["path"].apply(lambda p: " > ".join(p))
    grouped["n_touches"] = grouped["path"].apply(len)
    grouped["converted"] = grouped.index.isin(conv.index) if not conv.empty else False
    grouped["conversion_value"] = (
        grouped.index.map(conv["revenue"]).fillna(0.0) if not conv.empty else 0.0
    )
    grouped["journey_start"] = grouped["timestamps"].apply(lambda t: t[0] if t else pd.NaT)
    grouped["journey_end"] = grouped["timestamps"].apply(lambda t: t[-1] if t else pd.NaT)
    grouped["days_to_convert"] = (
        grouped["journey_end"] - grouped["journey_start"]
    ).dt.total_seconds() / 86400.0

    return grouped.reset_index()
