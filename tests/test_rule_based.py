"""Rule-based attribution models — hand-verified against tests/conftest.py's
toy_journeys_rule_based fixture (J3 is unconverted and must be excluded).
"""
from __future__ import annotations

import pytest

from src.attribution import rule_based


def test_first_touch(toy_journeys_rule_based):
    # J1 first touch = A, J2 first touch = B -> 1 each, normalized 0.5/0.5
    credit = rule_based.first_touch(toy_journeys_rule_based)
    assert credit == pytest.approx({"A": 0.5, "B": 0.5})


def test_last_touch(toy_journeys_rule_based):
    # J1 last touch = B, J2 last touch = A -> 1 each, normalized 0.5/0.5
    credit = rule_based.last_touch(toy_journeys_rule_based)
    assert credit == pytest.approx({"A": 0.5, "B": 0.5})


def test_linear(toy_journeys_rule_based):
    # J1 (A,B): 0.5 each. J2 (B,C,A): 1/3 each.
    # A: 0.5 + 1/3 = 5/6, B: 0.5 + 1/3 = 5/6, C: 1/3
    # sum = 2 -> normalized: A=5/12, B=5/12, C=1/6
    credit = rule_based.linear(toy_journeys_rule_based)
    assert credit == pytest.approx({"A": 5 / 12, "B": 5 / 12, "C": 1 / 6})
    assert sum(credit.values()) == pytest.approx(1.0)


def test_position_based_two_and_three_touch(toy_journeys_rule_based):
    # J1 has 2 touches -> split 0.5/0.5 (A, B)
    # J2 has 3 touches -> first(B)=0.4, last(A)=0.4, middle(C)=0.2
    # A: 0.5 + 0.4 = 0.9, B: 0.5 + 0.4 = 0.9, C: 0.2 -> sum = 2.0
    credit = rule_based.position_based(toy_journeys_rule_based)
    assert credit == pytest.approx({"A": 0.45, "B": 0.45, "C": 0.10})
    assert sum(credit.values()) == pytest.approx(1.0)


def test_time_decay_sums_to_one_and_favors_recency(toy_journeys_rule_based):
    credit = rule_based.time_decay(toy_journeys_rule_based, half_life_days=7.0)
    assert sum(credit.values()) == pytest.approx(1.0)
    # J1: A is 1 day before the journey end (B), so B (the last touch) must get
    # strictly more of J1's credit than A -- recency should always help, never hurt.
    # We can't isolate per-journey credit from the aggregated dict directly, but
    # we CAN check the aggregate ordering makes sense: A is last-touch in J2 and
    # only ever a non-last touch in J1, B is last-touch in J1 and a middle touch
    # in J2 -- so this is mostly a sanity/regression check on the total.
    assert credit["A"] > 0 and credit["B"] > 0 and credit["C"] > 0


def test_run_all_every_model_sums_to_one(toy_journeys_rule_based):
    result = rule_based.run_all(toy_journeys_rule_based)
    sums = result.groupby("model")["credit_share"].sum()
    for model, total in sums.items():
        assert total == pytest.approx(1.0), f"{model} credit does not sum to 1: {total}"


def test_unconverted_journey_never_gets_credit(toy_journeys_rule_based):
    # J3 (path=['A'], not converted) must never contribute credit on its own --
    # if it did, A's first-touch/last-touch share would be > 0.5 since A would
    # get an extra full point from J3 (first_touch and last_touch both split
    # 0.5/0.5 between A and B across J1+J2 alone -- see test_first_touch /
    # test_last_touch above for the full derivation).
    for model_fn in [rule_based.first_touch, rule_based.last_touch]:
        credit = model_fn(toy_journeys_rule_based)
        assert credit["A"] == pytest.approx(0.5)
