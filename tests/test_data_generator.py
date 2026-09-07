"""Synthetic data generator — schema and basic statistical sanity, not exact
values (the generator is stochastic by design)."""
from __future__ import annotations

import pytest

from src.data_generator import DEFAULT_CHANNELS, generate_dataset


def test_output_schema():
    tp, conv, spend = generate_dataset(n_users=200, days=20, seed=1)
    assert {"user_id", "touchpoint_id", "timestamp", "channel"} <= set(tp.columns)
    assert {"user_id", "conversion_id", "timestamp", "revenue", "converted"} <= set(conv.columns)
    assert {"date", "channel", "spend"} <= set(spend.columns)


def test_same_seed_is_reproducible():
    tp1, conv1, _ = generate_dataset(n_users=200, days=20, seed=42)
    tp2, conv2, _ = generate_dataset(n_users=200, days=20, seed=42)
    assert tp1.equals(tp2)
    assert conv1.equals(conv2)


def test_different_seed_gives_different_data():
    tp1, _, _ = generate_dataset(n_users=200, days=20, seed=1)
    tp2, _, _ = generate_dataset(n_users=200, days=20, seed=2)
    assert not tp1.equals(tp2)


def test_conversion_rate_is_plausible():
    _, conv, _ = generate_dataset(n_users=2000, days=90, seed=5, base_conversion_rate=0.06)
    n_converting_users = conv["user_id"].nunique()
    rate = n_converting_users / 2000
    # base_conversion_rate is a floor, not the exact rate (lift pushes it up) --
    # just check it's in a plausible range, not degenerate (0% or 100%)
    assert 0.02 < rate < 0.6


def test_only_requested_channels_are_used():
    subset = {"paid_search": DEFAULT_CHANNELS["paid_search"], "email": DEFAULT_CHANNELS["email"]}
    tp, _, spend = generate_dataset(n_users=300, days=20, seed=1, channels=subset)
    assert set(tp["channel"].unique()) <= set(subset.keys())
    assert set(spend["channel"].unique()) <= set(subset.keys())


def test_no_negative_spend_or_revenue():
    tp, conv, spend = generate_dataset(n_users=300, days=20, seed=1)
    assert (spend["spend"] >= 0).all()
    assert (conv["revenue"] >= 0).all()
