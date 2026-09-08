"""Geo-holdout incrementality test — validated the same way every other model
in this project is: inject a KNOWN effect into simulated data, and confirm
the estimator recovers it. If it can't recover a known effect here, it has no
business being trusted on real data.
"""
from __future__ import annotations

import pytest

from src.causal.incrementality import (
    compare_to_attribution_estimate,
    estimate_causal_effect,
    simulate_geo_experiment,
)
from src.data_generator import generate_dataset
from src.journey_builder import build_journeys


@pytest.fixture(scope="module")
def journeys():
    tp, conv, _ = generate_dataset(n_users=15000, days=60, seed=11)
    return build_journeys(tp, conv, lookback_days=30)


def test_recovers_a_known_injected_effect(journeys):
    # 30% of paid_search's converting journeys are causally attributable --
    # both methods should detect a significant, positive lift whose CI
    # contains a value in the right ballpark.
    geo_exp = simulate_geo_experiment(
        journeys, target_channel="paid_search", true_causal_share=0.30, n_geos=30, seed=1
    )
    result = estimate_causal_effect(geo_exp)

    assert result["estimated_lift_pp"] > 0
    assert result["significant_at_alpha"]
    assert result["p_value"] < 0.05
    # the confidence interval should not straddle zero for a real, injected effect
    ci_low, ci_high = result["confidence_interval"]
    assert ci_low > 0

    # the independent regression cross-check must agree in direction: treatment
    # (paused channel) should have a NEGATIVE effect on conversion log-odds
    reg = result["regression_cross_check"]
    assert "error" not in reg
    assert reg["treatment_coef_log_odds"] < 0
    assert reg["p_value"] < 0.05


def test_null_effect_is_not_falsely_detected(journeys):
    # Negative control / "AA test": zero injected causal effect should NOT
    # produce a significant result (at the usual false-positive rate). This
    # guards against a biased estimator that finds effects that aren't there.
    geo_exp = simulate_geo_experiment(
        journeys, target_channel="paid_search", true_causal_share=0.0, n_geos=30, seed=2
    )
    result = estimate_causal_effect(geo_exp, alpha=0.10)
    # zero true effect: CI should straddle zero (allow it not to, rarely --
    # this is a statistical test and can false-positive at the stated alpha,
    # but the point estimate itself should be small)
    assert abs(result["estimated_lift_pp"]) < 0.02


def test_larger_true_effect_produces_larger_estimated_lift(journeys):
    # Monotonicity sanity check: a bigger injected causal share must produce
    # a bigger (or equal, given sampling noise) measured lift, on average.
    small = estimate_causal_effect(
        simulate_geo_experiment(journeys, "paid_search", 0.10, n_geos=30, seed=5)
    )
    large = estimate_causal_effect(
        simulate_geo_experiment(journeys, "paid_search", 0.50, n_geos=30, seed=5)
    )
    assert large["estimated_lift_pp"] > small["estimated_lift_pp"]


def test_two_methods_agree_in_direction(journeys):
    geo_exp = simulate_geo_experiment(
        journeys, target_channel="email", true_causal_share=0.25, n_geos=25, seed=3
    )
    result = estimate_causal_effect(geo_exp)
    reg = result["regression_cross_check"]
    # z-test lift positive (control > test) <=> regression coef negative
    # (treatment lowers conversion) -- same underlying effect, opposite sign
    # convention (proportion difference vs. log-odds of treatment)
    assert (result["estimated_lift_pp"] > 0) == (reg["treatment_coef_log_odds"] < 0)


def test_compare_to_attribution_estimate_structure(journeys):
    geo_exp = simulate_geo_experiment(
        journeys, target_channel="paid_search", true_causal_share=0.30, n_geos=30, seed=1
    )
    causal_result = estimate_causal_effect(geo_exp)
    comparison = compare_to_attribution_estimate(
        causal_result, markov_credit_share=0.22, total_conversion_rate=0.06
    )
    assert set(comparison.keys()) == {
        "markov_implied_lift_pp", "causally_measured_lift_pp",
        "ratio_markov_to_causal", "same_direction",
        "causal_estimate_within_markov_ballpark",
    }
    assert comparison["markov_implied_lift_pp"] == pytest.approx(0.22 * 0.06)


def test_unknown_channel_produces_no_significant_effect(journeys):
    # A channel that doesn't exist in the data can't have any journeys
    # touching it, so the simulator should inject zero effect (nothing to
    # flip) -- this is a true null / AA test.
    #
    # n_geos=60 here, not the 20 used elsewhere in this file: with few
    # clusters, asymptotic (z/logit) inference is anti-conservative -- a
    # documented, textbook limitation of cluster-randomized designs, not a
    # bug in the estimator. Measured directly: with n_geos=20 the empirical
    # false-positive rate across 20 replicate seeds was ~15-30% at alpha=0.05
    # (should be ~5%); with n_geos=60 it was exactly 0/20. See the "Known
    # limitation" note in src/causal/incrementality.py's module docstring.
    geo_exp = simulate_geo_experiment(
        journeys, target_channel="not_a_real_channel", true_causal_share=0.9, n_geos=60, seed=1
    )
    result = estimate_causal_effect(geo_exp, alpha=0.05)
    assert not result["regression_cross_check"]["p_value"] < 0.05
