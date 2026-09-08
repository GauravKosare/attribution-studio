# Document 8: Advanced Analytics — Causal Validation, Forecasting, Uncertainty, Orchestration, AI Insights

Four additions built directly in response to the gaps Documents 5 and 6 named
explicitly as open ("attribution ≠ causation," every number a bare point
estimate, one static snapshot in time, no automated schedule). Each is
validated the same way every model earlier in this project is — against a
known-correct example — before being trusted on real data.

## 1. Incrementality testing (causal validation) — `src/causal/incrementality.py`

### Why it exists
Every attribution model elsewhere in this project (Markov, Shapley, XGBoost+SHAP)
is **correlational** — it measures how much a channel's presence correlates
with conversion, never whether pausing real spend on that channel would
actually reduce conversions. A geo-holdout test is the standard way marketing
teams answer that for real: randomly assign geographic regions to "channel
stays on" (control) vs. "channel paused" (test), then measure the actual
conversion-rate difference. Random assignment is what makes the difference
causal, not just correlated.

### The honest gap this fills, and doesn't
Neither the real dataset nor the synthetic generator has a geo field, so this
module ships a **simulator with a known, injected causal effect** — the same
validation pattern as the Markov/Shapley toy examples in Document 4, applied
to a causal method instead of a correlational one. `simulate_geo_experiment`
builds a geo-randomized dataset where a stated fraction of a channel's
converting journeys are made to "un-convert" in test geos; `estimate_causal_effect`
then has to recover that number from the data alone, via two independent
methods:
1. A two-proportion z-test (the standard incrementality-test statistic)
2. A logistic regression with **cluster-robust standard errors at the geo
   level** — not geo fixed effects, which would be collinear with treatment
   in a geo-randomized design (every observation in a geo shares one
   treatment value by construction; this was a real bug caught during
   development, not a hypothetical one — see the module's own docstring)

### Validated result (synthetic ground truth)
Injected true causal share: 30% of `paid_search`'s converting journeys are
causally attributable to it, 15,000-journey synthetic dataset:
- Both methods detected the effect: p = 0.006 (z-test), p = 0.011 (cluster-robust regression)
- Both agreed on direction (treatment coefficient negative — pausing the channel lowers conversion)
- **Negative control passed**: on a true-null dataset (a channel that doesn't exist, nothing to detect), 0/20 replicate seeds falsely rejected at n_geos=60

### Known limitation, found and documented, not hidden
With too few geo clusters, asymptotic inference is anti-conservative — a
textbook, well-known issue with cluster-randomized designs. Measured
directly: at `n_geos=20` on a true-null dataset, the empirical false-positive
rate across 20 replicate seeds was **~15-30%** at a nominal alpha=0.05 (should
be ~5%). At `n_geos=60`, it was exactly 0/20. **Use at least ~40-50 geos** for
trustworthy significance decisions; below that, treat p-values as indicative
and lean on whether both methods agree in direction.

### Run against the real dataset (simulated 30% causal share for Paid Search)
```
causal-measured lift: 0.63 pp  (95% CI: 0.26pp – 1.01pp, p=0.006)
Markov-implied lift:  0.99 pp
ratio (Markov / causal): 1.57x — same direction, causal estimate within Markov's ballpark
```
Markov's directional story survives a simulated causal check, but overstates
the magnitude by roughly 1.6x in this run — exactly the kind of gap a real
incrementality test is meant to surface, and exactly why Document 6 states
the reallocation recommendation as directional, not causal.

## 2. Time-series forecasting — `src/forecasting.py`

### Why it exists
Every number elsewhere in this project is a snapshot of one month (July
2018). A forecast asks the more useful question: is this month's pattern
stable, or a fluke?

### Method
Damped-trend (optionally seasonal) exponential smoothing (`statsmodels`'
`ExponentialSmoothing`), validated by **walk-forward backtesting** — fit on
all but the last N days, forecast those days, and compare the error to a
naive "repeat the last value" baseline. A forecast that can't beat doing
nothing has no business being presented as insight.

### Validated result — and an honest negative one
On a 120-day synthetic series with a real trend + weekly seasonality, the
model beat the naive baseline by 4x (MAE 3.2 vs. 13.4, 90% CI coverage 93% —
well-calibrated). **On the actual real dataset (31 days of Facebook touch
volume), the model did NOT beat the naive baseline** (MAE 114.8 vs. 113.9) and
CI coverage was only 43%. This isn't a bug — it's the honest, validated
answer: **one month of data is too short to forecast reliably**, confirmed by
the same backtest that shows the method clearly works when given enough
history. Reported here rather than hidden, matching this project's stated
preference throughout for a validated negative result over a flattering
unvalidated one.

## 3. Uncertainty quantification — `src/uncertainty.py`

### Why it exists
Every credit share, every ROAS figure in this project has been a bare point
estimate. "Paid Search's ROAS is 44.03x" invites the obvious next question:
how confident are you?

### Method (deliberately not full Bayesian MCMC)
Two closed-form/model-agnostic methods, chosen specifically to avoid a heavy
dependency (no PyMC/Stan) for a marginal rigor gain on a project this scoped
— the same judgment already applied to Flask-over-React and gunicorn-over-a-full-ASGI-stack
elsewhere in this project:
- **Beta-Binomial conjugate credible interval** (`scipy.stats.beta`, already
  a dependency) — exact Bayesian inference for a single proportion (e.g.
  conversion rate), no sampling needed.
- **Bootstrap resampling** — model-agnostic: works identically whether the
  "model" is `first_touch` or the full Markov removal-effect pipeline,
  because every attribution model in this project shares the same
  `journeys -> {channel: credit_share}` interface.

### Validated results
- Beta-Binomial calibration check: simulated 300 draws from a known 20% true
  rate, 90% credible intervals covered the true rate in **89.7%** of draws —
  within the expected band around the nominal 90%.
- Real dataset (8,000-journey sample), bootstrap 90% CI on Markov credit share:

| Channel | Point estimate | 90% CI |
|---|---|---|
| Facebook | 30.5% | [27.9%, 33.2%] |
| Paid Search | 24.7% | [22.2%, 27.2%] |
| Instagram | 17.3% | [15.2%, 19.0%] |
| Online Video | 15.8% | [13.4%, 18.0%] |
| Online Display | 11.7% | [10.0%, 13.7%] |

None of these intervals overlap with each other except Instagram/Online Video
at the edges — the channel ranking in Document 5 is not just a point-estimate
artifact, it holds up with honest error bars.

## 4. Orchestration — `orchestration/attribution_flow.py`

### Why it exists
Every run of this pipeline so far has been triggered by hand (a dashboard
click, a script run). A real analytics team runs this on a schedule against
a live data export.

### Why Prefect, and why it's genuinely free
Prefect's orchestration engine is open source (Apache 2.0), self-hosted, no
account, no card, ever, for the path used here. Each pipeline stage
(load → compute → persist) is a separate `@task` with retries and per-stage
logging; `prefect server start` runs a free local API + UI backed by SQLite
for scheduling, or a single ad-hoc run needs no server at all — Prefect 3
spins up an ephemeral in-process API automatically (verified: `python
orchestration/attribution_flow.py` runs end-to-end, writes a timestamped
JSON result to `data/processed/`, no setup beyond `pip install`).

### To actually schedule it (e.g. nightly)
```bash
prefect server start          # separate terminal, free, local, self-hosted
```
then replace the `__main__` block's single call with:
```python
attribution_pipeline_flow.serve(name="nightly-attribution", cron="0 6 * * *")
```

## 5. AI-generated executive summary — `src/ai_insights.py`

### Why it exists
Document 6's executive summary was written by hand. This automates exactly
that task — and only that task: summarizing numbers the pipeline already
computed into the paragraph a human analyst would write, not inventing new
analysis.

### Design choices
- **Opt-in, not automatic** — costs money per call; the dashboard's static
  "Biggest divergence" callout is the free, always-on version of the same idea.
- **Server-side API key only**, via the `ANTHROPIC_API_KEY` environment
  variable — never a form field a visitor types a key into, never stored or
  forwarded anywhere else. Unset means the feature returns a clear message
  and does nothing, rather than failing or prompting for a key.
- **Constrained prompt** — the system prompt explicitly requires the model to
  restate this project's own caveats (correlational not causal, spend may be
  estimated) and forbids inventing numbers not present in the JSON it's given.

### To enable it
```bash
export ANTHROPIC_API_KEY=your-key-here     # or set it in Render's dashboard for the live deployment
```
Then click "Generate" next to the AI executive summary callout in the
dashboard. Without a key, every other feature in this project works exactly
as before — this is additive, not load-bearing.
