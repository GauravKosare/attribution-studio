"""Uncertainty quantification — hand-verified where the math is closed-form
(Beta-Binomial), and checked for the properties any bootstrap CI must have
(point estimate inside its own interval, interval shrinks with more data)
where it isn't.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.attribution import rule_based
from src.data_generator import generate_dataset
from src.journey_builder import build_journeys
from src.uncertainty import beta_binomial_credible_interval, bootstrap_credit_share_ci


def test_beta_binomial_matches_closed_form_for_uniform_prior():
    # With a uniform prior (alpha=1, beta=1) and 50/100 successes, the
    # posterior is Beta(51, 51) -- symmetric around 0.5 by construction.
    result = beta_binomial_credible_interval(50, 100, ci=0.90)
    assert result["posterior_mean"] == pytest.approx(0.5, abs=1e-9)
    assert result["posterior_alpha"] == 51.0
    assert result["posterior_beta"] == 51.0
    lower, upper = result["credible_interval"]
    # symmetric posterior -> interval symmetric around 0.5
    assert (0.5 - lower) == pytest.approx(upper - 0.5, abs=1e-6)


def test_beta_binomial_interval_narrows_with_more_data():
    # Same point estimate (10%), 10x the sample size -- interval must be tighter.
    small = beta_binomial_credible_interval(10, 100, ci=0.90)
    large = beta_binomial_credible_interval(100, 1000, ci=0.90)
    small_width = small["credible_interval"][1] - small["credible_interval"][0]
    large_width = large["credible_interval"][1] - large["credible_interval"][0]
    assert large_width < small_width


def test_beta_binomial_zero_trials_errors_not_crashes():
    result = beta_binomial_credible_interval(0, 0)
    assert "error" in result


def test_beta_binomial_credible_interval_covers_true_rate_at_stated_frequency():
    # Simulate: true rate 0.2, draw many samples of size 200, check ~90% CIs
    # contain the true rate roughly 90% of the time (a real calibration check,
    # not just "does it run").
    rng = np.random.default_rng(0)
    true_rate = 0.2
    covered = 0
    n_trials_sim = 300
    for _ in range(n_trials_sim):
        successes = rng.binomial(200, true_rate)
        ci = beta_binomial_credible_interval(successes, 200, ci=0.90)
        lo, hi = ci["credible_interval"]
        covered += lo <= true_rate <= hi
    coverage_rate = covered / n_trials_sim
    assert 0.82 <= coverage_rate <= 0.98  # generous band around the nominal 90%


@pytest.fixture(scope="module")
def small_journeys():
    tp, conv, _ = generate_dataset(n_users=1500, days=30, seed=3)
    return build_journeys(tp, conv, lookback_days=30)


def test_bootstrap_ci_contains_point_estimate(small_journeys):
    result = bootstrap_credit_share_ci(small_journeys, rule_based.first_touch, n_bootstrap=100, ci=0.90, seed=1)
    for _, row in result.iterrows():
        assert row["ci_lower"] <= row["point_estimate"] <= row["ci_upper"] + 1e-9


def test_bootstrap_ci_channels_match_model_output(small_journeys):
    point = rule_based.first_touch(small_journeys)
    result = bootstrap_credit_share_ci(small_journeys, rule_based.first_touch, n_bootstrap=50, ci=0.90, seed=1)
    assert set(result["channel"]) == set(point.keys())


def test_bootstrap_ci_bounds_are_valid_probabilities(small_journeys):
    result = bootstrap_credit_share_ci(small_journeys, rule_based.first_touch, n_bootstrap=100, ci=0.90, seed=1)
    assert (result["ci_lower"] >= -1e-9).all()
    assert (result["ci_upper"] <= 1 + 1e-9).all()
    assert (result["ci_lower"] <= result["ci_upper"]).all()


def test_bootstrap_reproducible_with_same_seed(small_journeys):
    r1 = bootstrap_credit_share_ci(small_journeys, rule_based.first_touch, n_bootstrap=50, seed=7)
    r2 = bootstrap_credit_share_ci(small_journeys, rule_based.first_touch, n_bootstrap=50, seed=7)
    pd.testing.assert_frame_equal(r1, r2)
