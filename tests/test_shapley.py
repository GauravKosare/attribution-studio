"""Exact Shapley value attribution — hand-solved against a symmetric 2-channel
example (tests/conftest.py::toy_journeys_shapley).

Journeys (n=3 total, used as the denominator for v(S)):
  J1: A, converted
  J2: B, converted
  J3: A->B, NOT converted

v(S) = (count of journeys whose touched-channel-set is a subset of S that
converted) / 3:
  v({})    = 0/3 = 0            (no journey has an empty channel set)
  v({A})   = 1/3   (J1's {A} <= {A}, converted)
  v({B})   = 1/3   (J2's {B} <= {B}, converted)
  v({A,B}) = 2/3   (J1, J2, J3 all subsets; J1+J2 converted, J3 did not)

Shapley value (n=2 players, orderings A-then-B and B-then-A each weight 1/2):
  phi_A = 1/2*[v({A})-v({})] + 1/2*[v({A,B})-v({B})]
        = 1/2*(1/3) + 1/2*(1/3) = 1/3
  phi_B = 1/3 (by symmetry -- A and B play identical roles in this example)

Efficiency check: phi_A + phi_B = 2/3 = v({A,B}) exactly.
Normalized credit: A = (1/3)/(2/3) = 0.5, B = 0.5.
"""
from __future__ import annotations

import pytest

from src.attribution import shapley


def test_symmetric_example_splits_evenly(toy_journeys_shapley):
    result = shapley.run(toy_journeys_shapley)
    credit = dict(zip(result["channel"], result["credit_share"]))
    assert credit == pytest.approx({"A": 0.5, "B": 0.5}, abs=1e-9)
    assert sum(credit.values()) == pytest.approx(1.0)


def test_exact_shapley_efficiency_property():
    # Efficiency: for ANY coalition-value function, sum(phi) must equal v(grand
    # coalition) -- this is a mathematical guarantee of Shapley values, not
    # specific to this dataset. Verify it holds before normalization.
    import pandas as pd

    journeys = pd.DataFrame(
        [
            {"path": ["A"], "converted": True},
            {"path": ["B"], "converted": True},
            {"path": ["C"], "converted": False},
            {"path": ["A", "B", "C"], "converted": True},
        ]
    )
    journey_sets = shapley._journey_channel_sets(journeys)
    v = shapley._coalition_value_fn(journey_sets)
    channels = sorted({ch for p in journeys["path"] for ch in p})
    phi = shapley._exact_shapley(channels, v)
    assert sum(phi.values()) == pytest.approx(v(frozenset(channels)), abs=1e-9)


def test_null_player_gets_zero_shapley_value():
    # A channel that changes v(S) by exactly 0 no matter what coalition it
    # joins (the "null player" property) must get exactly 0 Shapley value.
    import pandas as pd

    # C never appears in any converting journey and never co-occurs with A/B,
    # so adding C to any coalition never changes the conversion rate.
    journeys = pd.DataFrame(
        [
            {"path": ["A"], "converted": True},
            {"path": ["B"], "converted": False},
            {"path": ["C"], "converted": False},
        ]
    )
    journey_sets = shapley._journey_channel_sets(journeys)
    v = shapley._coalition_value_fn(journey_sets)
    channels = sorted({ch for p in journeys["path"] for ch in p})
    phi = shapley._exact_shapley(channels, v)
    assert phi["C"] == pytest.approx(0.0, abs=1e-9)


def test_run_handles_empty_journeys_gracefully():
    import pandas as pd

    result = shapley.run(pd.DataFrame(columns=["path", "converted"]))
    assert list(result.columns) == ["channel", "model", "credit_share"]
    assert len(result) == 0
