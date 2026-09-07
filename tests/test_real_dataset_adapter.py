"""real_dataset_adapter.py — sanity checks on the bundled real dataset and its
reshaping into the project's standard schema. Loads the actual 44MB CSV, so
this is the slowest test in the suite (a few seconds) but it's the one
guarding against silently shipping a corrupted or reformatted data file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.real_dataset_adapter import load_real_dataset

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "real_channel_journeys.csv"


@pytest.mark.skipif(not CSV_PATH.exists(), reason="real dataset not present in this checkout")
def test_load_real_dataset_shape_and_schema():
    tp, conv, spend = load_real_dataset(str(CSV_PATH))

    assert {"user_id", "touchpoint_id", "timestamp", "channel"} <= set(tp.columns)
    assert {"user_id", "conversion_id", "timestamp", "revenue", "converted"} <= set(conv.columns)
    assert {"date", "channel", "spend"} <= set(spend.columns)

    # known facts about this specific dataset (docs/05) -- catches a corrupted
    # or truncated file, or a reshaping bug, without re-deriving every number
    assert tp["user_id"].nunique() > 200_000
    assert set(tp["channel"].unique()) == {
        "Facebook", "Instagram", "Online Display", "Online Video", "Paid Search",
    }
    assert conv["converted"].all()
    assert (conv["revenue"] > 0).all()
    assert (spend["spend"] >= 0).all()


@pytest.mark.skipif(not CSV_PATH.exists(), reason="real dataset not present in this checkout")
def test_spend_is_estimated_not_fabricated_as_zero():
    _, _, spend = load_real_dataset(str(CSV_PATH))
    # every channel should have some non-zero estimated spend -- a zero-spend
    # channel would break ROAS calculations downstream (division by zero)
    totals = spend.groupby("channel")["spend"].sum()
    assert (totals > 0).all()
