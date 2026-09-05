"""Shapley value attribution — docs/04_attribution_methodology.md section C.

v(S) = conversion rate of journeys whose touched-channel set is a subset of S.
Exact brute-force for small channel counts (<=10), sampled/permutation Shapley
above that so the dashboard stays responsive on any input size.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np
import pandas as pd


def _journey_channel_sets(journeys: pd.DataFrame) -> list[tuple[frozenset, bool]]:
    return [
        (frozenset(row["path"]), bool(row["converted"]))
        for _, row in journeys.iterrows()
        if row["path"]
    ]


def _coalition_value_fn(journey_sets: list[tuple[frozenset, bool]]):
    @lru_cache(maxsize=None)
    def v(coalition: frozenset) -> float:
        touched = [conv for chset, conv in journey_sets if chset <= coalition]
        if not touched:
            return 0.0
        return sum(touched) / len(journey_sets)

    return v


def _exact_shapley(channels: list[str], v) -> dict[str, float]:
    n = len(channels)
    phi = {c: 0.0 for c in channels}
    for c in channels:
        others = [x for x in channels if x != c]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                s = frozenset(subset)
                weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                phi[c] += weight * (v(s | {c}) - v(s))
    return phi


def _sampled_shapley(channels: list[str], v, n_samples: int = 2000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    phi = {c: 0.0 for c in channels}
    channels_arr = np.array(channels, dtype=object)
    for _ in range(n_samples):
        order = rng.permutation(channels_arr)
        coalition = frozenset()
        prev_value = v(coalition)
        for c in order:
            coalition = coalition | {c}
            new_value = v(coalition)
            phi[c] += new_value - prev_value
            prev_value = new_value
    return {c: val / n_samples for c, val in phi.items()}


def run(journeys: pd.DataFrame, exact_threshold: int = 10) -> pd.DataFrame:
    channels = sorted({ch for path in journeys["path"] for ch in path})
    journey_sets = _journey_channel_sets(journeys)
    if not channels or not journey_sets:
        return pd.DataFrame(columns=["channel", "model", "credit_share"])

    v = _coalition_value_fn(journey_sets)
    if len(channels) <= exact_threshold:
        phi = _exact_shapley(channels, v)
    else:
        phi = _sampled_shapley(channels, v)

    phi = {c: max(0.0, val) for c, val in phi.items()}
    total = sum(phi.values())
    credit = {c: (val / total if total > 0 else 0.0) for c, val in phi.items()}
    return pd.DataFrame(
        [{"channel": ch, "model": "shapley", "credit_share": s} for ch, s in credit.items()]
    )
