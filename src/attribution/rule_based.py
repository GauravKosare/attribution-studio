"""Rule-based attribution models — see docs/04_attribution_methodology.md section A.

Every function takes the journeys DataFrame (from journey_builder.build_journeys)
and returns a {channel: credit_share} dict normalized to sum to 1. Works on any
channel set present in the input data — nothing here is hardcoded to a fixed list.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _normalize(credit: dict[str, float]) -> dict[str, float]:
    total = sum(credit.values())
    if total <= 0:
        return {k: 0.0 for k in credit}
    return {k: v / total for k, v in credit.items()}


def first_touch(journeys: pd.DataFrame) -> dict[str, float]:
    credit = defaultdict(float)
    for _, row in journeys[journeys["converted"]].iterrows():
        if row["path"]:
            credit[row["path"][0]] += 1.0
    return _normalize(credit)


def last_touch(journeys: pd.DataFrame) -> dict[str, float]:
    credit = defaultdict(float)
    for _, row in journeys[journeys["converted"]].iterrows():
        if row["path"]:
            credit[row["path"][-1]] += 1.0
    return _normalize(credit)


def linear(journeys: pd.DataFrame) -> dict[str, float]:
    credit = defaultdict(float)
    for _, row in journeys[journeys["converted"]].iterrows():
        path = row["path"]
        if not path:
            continue
        share = 1.0 / len(path)
        for ch in path:
            credit[ch] += share
    return _normalize(credit)


def time_decay(journeys: pd.DataFrame, half_life_days: float = 7.0) -> dict[str, float]:
    credit = defaultdict(float)
    for _, row in journeys[journeys["converted"]].iterrows():
        path, ts = row["path"], row["timestamps"]
        if not path:
            continue
        end = ts[-1]
        weights = np.array(
            [0.5 ** (((end - t).total_seconds() / 86400.0) / half_life_days) for t in ts]
        )
        weights = weights / weights.sum()
        for ch, w in zip(path, weights):
            credit[ch] += w
    return _normalize(credit)


def position_based(
    journeys: pd.DataFrame, first_weight: float = 0.4, last_weight: float = 0.4
) -> dict[str, float]:
    credit = defaultdict(float)
    middle_weight = 1.0 - first_weight - last_weight
    for _, row in journeys[journeys["converted"]].iterrows():
        path = row["path"]
        n = len(path)
        if n == 0:
            continue
        if n == 1:
            credit[path[0]] += 1.0
        elif n == 2:
            credit[path[0]] += 0.5
            credit[path[1]] += 0.5
        else:
            credit[path[0]] += first_weight
            credit[path[-1]] += last_weight
            share = middle_weight / (n - 2)
            for ch in path[1:-1]:
                credit[ch] += share
    return _normalize(credit)


MODELS = {
    "first_touch": first_touch,
    "last_touch": last_touch,
    "linear": linear,
    "time_decay": time_decay,
    "position_based": position_based,
}


def run_all(journeys: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form DataFrame: channel, model, credit_share."""
    rows = []
    for name, fn in MODELS.items():
        for channel, share in fn(journeys).items():
            rows.append({"channel": channel, "model": name, "credit_share": share})
    return pd.DataFrame(rows)
