"""Markov chain removal-effect attribution — hand-solved against a textbook
2-channel example (tests/conftest.py::toy_journeys_markov).

Journeys: U1: A (converted), U2: A->B (converted), U3: B (not converted).
Empirical conversion rate = 2/3.

Transition probabilities (counts / total from each source):
  Start -> A: 2/3   Start -> B: 1/3
  A -> Conversion: 1/2   A -> B: 1/2
  B -> Conversion: 1/2   B -> Null: 1/2

Total conversion probability from Start:
  P = P(Start->A)*[P(A->Conv) + P(A->B)*P(B->Conv)] + P(Start->B)*P(B->Conv)
    = (2/3)*(0.5 + 0.5*0.5) + (1/3)*0.5 = (2/3)*0.75 + 1/6 = 0.5 + 1/6 = 2/3

Removal effect for A: remove A, Start->A's weight (2/3) redirects to Null
(A's own out-edges vanish with the node, they are not redirected).
Remaining graph: Start->B: 1/3, Start->Null: 2/3, B->Conversion: 0.5, B->Null: 0.5
  P_without_A = (1/3)*0.5 = 1/6
  removal_effect(A) = 2/3 - 1/6 = 1/2

Removal effect for B: remove B, both edges INTO B (Start->B: 1/3, A->B: 1/2)
redirect to Null.
Remaining graph: Start->A: 2/3, Start->Null: 1/3, A->Conversion: 0.5, A->Null: 0.5
  P_without_B = (2/3)*0.5 = 1/3
  removal_effect(B) = 2/3 - 1/3 = 1/3

Normalized credit: A = (1/2) / (1/2 + 1/3) = 0.6, B = (1/3) / (5/6) = 0.4
"""
from __future__ import annotations

import pytest

from src.attribution import markov


def test_total_conversion_probability_matches_empirical_rate(toy_journeys_markov):
    graph = markov._build_graph(toy_journeys_markov)
    p = markov._total_conversion_probability(graph)
    # For a first-order chain fit to exactly these transitions, the model must
    # exactly reproduce the empirical conversion rate (2 of 3 users converted).
    assert p == pytest.approx(2 / 3, abs=1e-9)


def test_removal_effect_matches_hand_solved_values(toy_journeys_markov):
    credit = markov.removal_effect(toy_journeys_markov)
    assert credit == pytest.approx({"A": 0.6, "B": 0.4}, abs=1e-6)
    assert sum(credit.values()) == pytest.approx(1.0)


def test_run_returns_correct_shape(toy_journeys_markov):
    result = markov.run(toy_journeys_markov)
    assert set(result.columns) == {"channel", "model", "credit_share"}
    assert (result["model"] == "markov").all()
    assert result["credit_share"].sum() == pytest.approx(1.0)


def test_channel_with_no_touches_gets_zero_credit():
    # A channel that only ever appears in a journey that never reaches
    # Conversion (a dead-end) contributes 0 to conversion probability, so its
    # removal effect should be exactly 0 -- this is the "actionable divergence"
    # story documents 04/05 rely on: naive models can still give a dead-end
    # channel credit (e.g. linear/first-touch on some other journey), Markov
    # should not, when it truly never touches a converting path.
    import pandas as pd

    journeys = pd.DataFrame(
        [
            {"path": ["A"], "converted": True},
            {"path": ["DeadEnd"], "converted": False},
        ]
    )
    credit = markov.removal_effect(journeys)
    assert credit["DeadEnd"] == pytest.approx(0.0)
    assert credit["A"] == pytest.approx(1.0)
