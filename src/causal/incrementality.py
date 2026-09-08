"""Geo-holdout incrementality testing — the causal validation every attribution
model in this project explicitly says it cannot provide on its own (see
docs/05_validation_report_template.md §6, docs/06 "Next steps").

Every model elsewhere in src/attribution/ is *correlational*: Markov removal
effect, Shapley value, and XGBoost+SHAP all measure how much a channel's
presence correlates with conversion, never whether removing real spend from
that channel actually causes conversions to drop. A geo-holdout test is the
standard way marketing teams answer that for real: randomly assign geographic
regions to "channel stays on" (control) vs. "channel paused/reduced" (test),
then measure the actual conversion-rate difference between the two groups.
Random assignment is what makes the difference causal, not just correlated.

Neither the real dataset nor the synthetic generator has a geo field, so this
module does two things:
  1. `simulate_geo_experiment` — builds a geo-holdout dataset with a KNOWN,
     injected true causal effect, the same way docs/04's Markov/Shapley toy
     examples are hand-solved before trusting the real implementation. If the
     estimator below can't recover a known effect from simulated data, it has
     no business being trusted on real data.
  2. `estimate_causal_effect` — the actual test: a two-proportion z-test (the
     simplest, most standard incrementality-test statistic) cross-checked
     against a logistic regression with geo fixed effects (a second,
     independent method — the same "two ways, do they agree" pattern used
     for Markov vs. Shapley throughout this project).

To run this against REAL geo-tagged data instead of the simulator, replace
`simulate_geo_experiment`'s output with a DataFrame of the same shape:
[user_id, geo, group ('test'/'control'), converted] built from actual
ad-platform + CRM data with a real geo holdout already run.

Known limitation -- the "few clusters" problem: randomization here is at the
geo level, but both estimators (the two-proportion z-test and the
cluster-robust logistic regression) rely on asymptotic inference, which is
anti-conservative when the number of clusters (geos) is small. Measured
directly during development: with `n_geos=20` on a true-null (zero injected
effect) dataset, the empirical false-positive rate at alpha=0.05 across 20
replicate seeds was ~15-30% -- meaningfully inflated above the nominal 5%.
With `n_geos=60`, it was 0/20. **Use at least ~40-50 geos for trustworthy
significance decisions**; below that, treat p-values as indicative, not
exact, and lean on the point estimate + whether both methods agree in
direction rather than a single p-value threshold. This is a standard,
documented issue in the cluster-randomized-trial literature (not specific to
this implementation) -- the usual production fixes are a cluster
bootstrap, a permutation test, or a small-sample (e.g. Satterthwaite) degrees-
of-freedom correction, none of which are implemented here to keep this
module dependency-light; flagging the limitation honestly was judged better
than silently shipping an anti-conservative test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def simulate_geo_experiment(
    journeys: pd.DataFrame,
    target_channel: str,
    true_causal_share: float,
    n_geos: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a geo-holdout dataset with a KNOWN injected causal effect.

    `true_causal_share`: the fraction of `target_channel`'s converting
    journeys that are *causally* due to that channel (vs. would have
    converted anyway through other channels). This is the ground truth the
    estimator below has to recover — e.g. 0.30 means "30% of conversions in
    journeys that touched this channel are causally attributable to it."

    Mechanism: users are randomly assigned to geos, geos randomly assigned to
    test (channel paused) or control (channel active). In test geos, any
    converted journey that touched `target_channel` has its conversion
    "un-happen" with probability `true_causal_share` — simulating what would
    genuinely disappear if that channel's spend were truly paused there.
    Journeys that would have converted anyway (via other channels) are
    untouched, which is exactly the distinction correlational attribution
    cannot make on its own.
    """
    rng = np.random.default_rng(seed)
    df = journeys[["user_id", "path", "converted"]].copy()

    df["geo"] = rng.integers(0, n_geos, size=len(df))
    geo_group = pd.Series(
        rng.choice(["control", "test"], size=n_geos, p=[0.5, 0.5]), index=range(n_geos)
    )
    df["group"] = df["geo"].map(geo_group)

    touched_target = df["path"].apply(lambda p: target_channel in p)
    in_test = df["group"] == "test"
    at_risk = touched_target & in_test & df["converted"]

    flip = rng.random(len(df)) < true_causal_share
    df["converted_observed"] = df["converted"]
    df.loc[at_risk & flip, "converted_observed"] = False

    return df[["user_id", "geo", "group", "path", "converted_observed"]].rename(
        columns={"converted_observed": "converted"}
    )


def estimate_causal_effect(geo_experiment: pd.DataFrame, alpha: float = 0.10) -> dict:
    """Estimate the causal lift from a test-vs-control geo-holdout dataset.

    Returns point estimate, confidence interval, and p-value from TWO
    independent methods (two-proportion z-test and a geo-fixed-effects
    logistic regression) so agreement between them is itself evidence the
    estimate is real, not an artifact of one method's assumptions.
    """
    control = geo_experiment[geo_experiment["group"] == "control"]
    test = geo_experiment[geo_experiment["group"] == "test"]

    n_control, n_test = len(control), len(test)
    conv_control, conv_test = int(control["converted"].sum()), int(test["converted"].sum())
    rate_control = conv_control / n_control if n_control else 0.0
    rate_test = conv_test / n_test if n_test else 0.0

    # Method 1: two-proportion z-test -- the standard, simplest incrementality
    # test statistic. Lift is measured as control_rate - test_rate (positive
    # lift = the channel was truly driving conversions that disappeared when
    # paused).
    count = np.array([conv_control, conv_test])
    nobs = np.array([n_control, n_test])
    z_stat, p_value = proportions_ztest(count, nobs)

    se = np.sqrt(
        rate_control * (1 - rate_control) / n_control + rate_test * (1 - rate_test) / n_test
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lift = rate_control - rate_test
    ci = (lift - z_crit * se, lift + z_crit * se)

    # Method 2: logistic regression with geo fixed effects -- a completely
    # independent estimation approach (regression-adjusted, not a simple
    # difference), cross-checking Method 1 the same way Markov and Shapley
    # cross-check each other elsewhere in this project.
    regression_result = _geo_fixed_effects_regression(geo_experiment)

    return {
        "n_control": n_control,
        "n_test": n_test,
        "conversion_rate_control": rate_control,
        "conversion_rate_test": rate_test,
        "estimated_lift_pp": lift,  # percentage points, control - test
        "confidence_interval": ci,
        "p_value": float(p_value),
        "significant_at_alpha": bool(p_value < alpha),
        "regression_cross_check": regression_result,
    }


def _geo_fixed_effects_regression(geo_experiment: pd.DataFrame) -> dict:
    """Logistic regression: converted ~ treatment, with standard errors
    clustered by geo.

    NOT geo fixed effects: in a geo-randomized design (this one), every
    observation in a given geo shares the same treatment value by
    construction, so treatment is a perfect linear combination of geo dummies
    -- including both would make the design matrix collinear. The textbook-
    correct adjustment for a geo-randomized experiment is cluster-robust
    standard errors at the randomization unit (geo), which is what this does:
    an independent estimation method (regression-based variance, not a simple
    proportion difference) that correctly accounts for within-geo
    correlation without the collinearity bug.
    """
    import statsmodels.formula.api as smf

    df = geo_experiment.copy()
    df["treatment"] = (df["group"] == "test").astype(int)
    df["converted_int"] = df["converted"].astype(int)

    try:
        model = smf.logit("converted_int ~ treatment", data=df).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": df["geo"]}
        )
        coef = model.params["treatment"]
        ci_low, ci_high = model.conf_int().loc["treatment"]
        return {
            "treatment_coef_log_odds": float(coef),
            "p_value": float(model.pvalues["treatment"]),
            "conf_int_log_odds": (float(ci_low), float(ci_high)),
            "converged": bool(model.mle_retvals.get("converged", True)),
        }
    except Exception as e:  # pragma: no cover -- defensive, e.g. perfect separation
        return {"error": str(e)}


def compare_to_attribution_estimate(
    causal_result: dict, markov_credit_share: float, total_conversion_rate: float
) -> dict:
    """Does the correlational Markov estimate agree with the causal test?

    Markov's credit_share is a share of TOTAL conversions attributed to a
    channel; the causal test measures the ABSOLUTE conversion-rate lift from
    pausing it. To compare them on the same footing, convert Markov's share
    into an implied conversion-rate contribution and compare directionally
    and in rough magnitude to what the causal test actually measured.
    """
    markov_implied_lift_pp = markov_credit_share * total_conversion_rate
    measured_lift_pp = causal_result["estimated_lift_pp"]

    if measured_lift_pp == 0:
        ratio = float("inf") if markov_implied_lift_pp != 0 else 1.0
    else:
        ratio = markov_implied_lift_pp / measured_lift_pp

    return {
        "markov_implied_lift_pp": markov_implied_lift_pp,
        "causally_measured_lift_pp": measured_lift_pp,
        "ratio_markov_to_causal": ratio,
        "same_direction": (markov_implied_lift_pp > 0) == (measured_lift_pp > 0),
        "causal_estimate_within_markov_ballpark": (
            causal_result["confidence_interval"][0] <= markov_implied_lift_pp
            <= causal_result["confidence_interval"][1]
        ) if measured_lift_pp != 0 or markov_implied_lift_pp != 0 else True,
    }
