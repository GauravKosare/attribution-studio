"""Shared fixtures — small, hand-computable journey sets used across test files.

These are deliberately tiny (2-3 channels, 2-5 journeys) so expected values can be
verified by hand in the test docstrings/comments, not just asserted against
whatever the code happens to currently output.
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def toy_journeys_rule_based() -> pd.DataFrame:
    """3 journeys, timestamps included for time_decay.

    J1: A -> B, converted   (day 0, day 1)
    J2: B -> C -> A, converted (day 0, day 0.5, day 1)
    J3: A, NOT converted   (excluded from all rule-based credit)
    """
    t0 = pd.Timestamp("2026-01-01")
    return pd.DataFrame(
        [
            {
                "path": ["A", "B"],
                "timestamps": [t0, t0 + pd.Timedelta(days=1)],
                "converted": True,
                "conversion_value": 10.0,
            },
            {
                "path": ["B", "C", "A"],
                "timestamps": [t0, t0 + pd.Timedelta(hours=12), t0 + pd.Timedelta(days=1)],
                "converted": True,
                "conversion_value": 20.0,
            },
            {
                "path": ["A"],
                "timestamps": [t0],
                "converted": False,
                "conversion_value": 0.0,
            },
        ]
    )


@pytest.fixture
def toy_journeys_markov() -> pd.DataFrame:
    """Textbook 2-channel example, hand-solved in docs/04 and in test_markov.py.

    U1: A, converted
    U2: A -> B, converted
    U3: B, NOT converted
    Empirical conversion rate = 2/3 — the Markov chain fit to exactly these
    transitions must reproduce that exactly (see test_markov.py for the proof).
    """
    return pd.DataFrame(
        [
            {"path": ["A"], "converted": True},
            {"path": ["A", "B"], "converted": True},
            {"path": ["B"], "converted": False},
        ]
    )


@pytest.fixture
def toy_journeys_shapley() -> pd.DataFrame:
    """Symmetric 2-channel example, hand-solved in test_shapley.py.

    J1: A, converted
    J2: B, converted
    J3: A -> B, NOT converted
    By symmetry (A and B play identical roles), Shapley credit must split 50/50.
    """
    return pd.DataFrame(
        [
            {"path": ["A"], "converted": True},
            {"path": ["B"], "converted": True},
            {"path": ["A", "B"], "converted": False},
        ]
    )
