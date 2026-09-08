"""Uncertainty quantification — every number this project has produced so far
(credit shares, ROAS, the projected reallocation lift) has been a bare point
estimate. This module puts an honest error bar on the two kinds of numbers
that most need one, using two different, well-established methods chosen
specifically to avoid a heavy new dependency (no PyMC/Stan — see the
"Why not full Bayesian MCMC" note below):

1. **Beta-Binomial conjugate credible interval** — closed-form Bayesian
   inference (`scipy.stats.beta`, already a dependency) for a single
   proportion, e.g. "what's the credible interval on this channel's
   conversion rate?" No sampling needed; exact.
2. **Bootstrap resampling** — for anything that isn't a simple proportion
   (e.g. an attribution model's `credit_share`, which comes from a whole
   pipeline of transformations), resample journeys with replacement, rerun
   the model, and take the empirical percentile interval across resamples.
   This is model-agnostic: it works identically whether the "model" is
   first-touch or the full Markov removal-effect computation.

Why not full Bayesian MCMC (PyMC/Stan): a hierarchical Bayesian model would
be the more complete answer (and is named as a stretch option in
docs/07-adjacent methodology notes), but it's a heavy, often fragile-to-install
dependency for a marginal gain over these two methods on a project this
scoped — the same "know when not to reach for the heavier tool" judgment
already applied to gunicorn vs. a full ASGI stack, and Flask vs. React,
elsewhere in this project.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def beta_binomial_credible_interval(
    successes: int, trials: int, ci: float = 0.90, prior_alpha: float = 1.0, prior_beta: float = 1.0
) -> dict:
    """Exact Bayesian credible interval for a conversion rate (or any
    proportion), via the Beta-Binomial conjugate model.

    Default prior (alpha=1, beta=1) is uniform over [0,1] -- "no prior
    opinion." Posterior is Beta(alpha + successes, beta + trials - successes);
    its mean, and the equal-tailed credible interval, are both closed-form.
    """
    if trials == 0:
        return {"error": "No trials observed."}

    post_alpha = prior_alpha + successes
    post_beta = prior_beta + (trials - successes)
    posterior = stats.beta(post_alpha, post_beta)

    lower_tail = (1 - ci) / 2
    lower, upper = posterior.ppf(lower_tail), posterior.ppf(1 - lower_tail)

    return {
        "point_estimate": successes / trials,
        "posterior_mean": float(posterior.mean()),
        "credible_interval": (float(lower), float(upper)),
        "ci": ci,
        "posterior_alpha": post_alpha,
        "posterior_beta": post_beta,
    }


def bootstrap_credit_share_ci(
    journeys: pd.DataFrame,
    model_fn: Callable[[pd.DataFrame], dict],
    n_bootstrap: int = 200,
    ci: float = 0.90,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for ANY attribution model's credit_share
    output, by resampling journeys (with replacement) and rerunning the model.

    `model_fn`: a function like src.attribution.rule_based.first_touch or
    src.attribution.markov.removal_effect — takes a journeys DataFrame,
    returns a {channel: credit_share} dict. Works unmodified for any model in
    this project because they all share that exact interface.

    Returns a DataFrame: channel, point_estimate, ci_lower, ci_upper, std_error.
    """
    rng = np.random.default_rng(seed)
    n = len(journeys)
    point_estimate = model_fn(journeys)
    channels = sorted(point_estimate.keys())

    samples = np.zeros((n_bootstrap, len(channels)))
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = journeys.iloc[idx].reset_index(drop=True)
        credit = model_fn(resampled)
        for j, ch in enumerate(channels):
            samples[i, j] = credit.get(ch, 0.0)

    lower_pct, upper_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    rows = []
    for j, ch in enumerate(channels):
        col = samples[:, j]
        rows.append(
            {
                "channel": ch,
                "point_estimate": point_estimate[ch],
                "bootstrap_mean": float(col.mean()),
                "ci_lower": float(np.percentile(col, lower_pct)),
                "ci_upper": float(np.percentile(col, upper_pct)),
                "std_error": float(col.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)
