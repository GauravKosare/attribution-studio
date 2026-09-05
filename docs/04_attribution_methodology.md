# Document 4: Attribution Modeling Methodology

For each model: what it does, the math, and the implementation approach.

## A. Rule-based models (baseline)

All operate on `journeys.path` for converted journeys only, distributing each
conversion's credit (1.0, or `conversion_value`) across the touchpoints in that path.

| Model | Rule |
|---|---|
| **First-touch** | 100% credit to `path[0]` |
| **Last-touch** | 100% credit to `path[-1]` |
| **Linear** | `1/n_touches` credit to every touch |
| **Time-decay** | Credit weighted by recency: `weight_i = 0.5 ** ((T - t_i) / half_life)`, normalized to sum to 1. Typical half-life: 7 days |
| **Position-based (U-shaped)** | 40% to first touch, 40% to last touch, remaining 20% split evenly among middle touches (if only 1–2 touches, fall back to linear) |

Implementation: a single function per model that takes a journey (list of channels)
and returns a `{channel: credit}` dict; sum across all converted journeys, then
normalize per channel to get `credit_share`.

**Why include these at all**: they're the strawman. The entire narrative value of
the project is "here's what naive attribution gets wrong, and here's the data-driven
model that corrects it." You need the baseline to show the gap.

## B. Markov chain model (the industry-standard advanced technique)

### Concept
Model the customer journey as a first-order Markov chain over states:
`{Start, channel_1, channel_2, ..., Conversion, Null}` (`Null` = did-not-convert absorbing state).

1. From the *converted and non-converted* journeys, count transitions between
   consecutive states (including `Start -> first_touch` and `last_touch -> Conversion/Null`).
2. Build the transition probability matrix `P` (row-normalized transition counts).
3. Compute **total conversion probability** = probability of reaching `Conversion` from `Start`,
   via matrix multiplication / absorbing Markov chain solution.
4. **Removal effect** for channel `c`: remove `c`'s node from the graph (redirect any
   transition that would have gone through `c` straight to `Null`), recompute total
   conversion probability `P_without_c`. Removal effect = `(P_total - P_without_c) / P_total`.
5. Normalize removal effects across channels so they sum to 1 → this is the channel's
   `credit_share` under the Markov model.

### Why it's considered the industry standard
It captures *sequence and co-occurrence* effects that rule-based models ignore: a
channel that reliably appears alongside high-converting sequences gets more credit
even if it's never the last touch, and a channel that only ever appears in dead-end
journeys gets appropriately little.

### Implementation notes
- Use `NetworkX.DiGraph` for the state graph — nodes = channels + Start/Conversion/Null,
  edge weight = transition probability.
- Compute total conversion probability with a direct absorbing-Markov-chain matrix solve
  (`numpy.linalg`) rather than simulation for speed and determinism; Monte Carlo
  simulation is a fine sanity check/second implementation to validate against.
- Higher-order Markov chains (2nd/3rd order — state = last 2–3 channels) are a
  legitimate stretch goal; they capture more sequence nuance but need much more data
  to estimate reliably. Mention this tradeoff explicitly in the writeup.

## C. Shapley value attribution

### Concept
Borrowed from cooperative game theory. Treat each channel as a "player" and each
possible **subset (coalition) of channels** as a "team." The value of a coalition
`v(S)` = conversion rate (or count) of journeys whose channel-set is a subset of `S`
(or: fraction of conversions achievable using only channels in `S`).

Shapley value for channel `c`:
```
φ_c = Σ over all subsets S not containing c:
        [ |S|! (n - |S| - 1)! / n! ] * ( v(S ∪ {c}) - v(S) )
```
i.e., the average marginal contribution of adding `c` across every possible ordering
of channels joining a coalition.

### Why it matters
It's the only model with a formal fairness guarantee (efficiency, symmetry, additivity,
null-player) — credit sums exactly to total conversions, and a channel that changes
nothing when added gets exactly zero credit. This is the strongest "I understand the
theory" talking point in an interview.

### Implementation notes
- With ≤ 8–10 unique channels, exact Shapley by brute-force over all `2^n` subsets is
  computationally fine — implement it directly with `itertools.combinations` to prove
  you understand the formula before reaching for a shortcut.
- Define `v(S)` practically as: conversion rate of journeys whose **touched-channel set**
  is a subset of `S`, or use a simplified pairwise/first-order approximation if you
  want to avoid combinatorial explosion on journey-set matching — document whichever
  you pick and why.
- For channel counts beyond ~10, use **sampled/permutation Shapley** (randomly sample
  orderings, average marginal contributions) instead of exact enumeration.

## D. ML-based attribution (logistic regression / XGBoost + SHAP) — implemented

**Status: built**, `src/attribution/ml_shap.py`, wired into the dashboard as an
opt-in "Advanced model" (off by default — it costs ~10–20s vs. <1s for everything
else, so it's a checkbox, not a default).

### Concept
1. Feature-engineer each journey into a fixed-width vector: channel presence/count/
   recency-weighted presence per channel (e.g. `has_email`, `count_email`,
   `recency_email`) — vectorized via explode+groupby, not a per-journey Python loop.
2. Train a classifier (`LogisticRegression` baseline, `XGBClassifier` by default —
   captures non-linear channel-interaction effects) to predict `converted`, on a
   75/25 train/test split.
3. Use `shap.TreeExplainer` (XGBoost) or `shap.LinearExplainer` (logistic) to get
   per-feature SHAP values on the held-out test set, then sum `|SHAP|` across each
   channel's three features and normalize → `credit_share`.

### Why include it
This is the piece that connects classical marketing attribution to modern ML
explainability — SHAP *is* an approximation of Shapley values applied to a model's
output, so this is the "apply the same game-theoretic idea to a predictive model"
version of the exact-Shapley implementation in §C. It also naturally produces a
model-quality artifact (test AUC) alongside the attribution numbers — the dashboard
surfaces this explicitly (with a warning if AUC is close to 0.5) so the credit
numbers aren't presented with more confidence than the underlying model earns.

### Result on the real dataset (20K-user sample, July 2018)
XGBoost test AUC **0.685** — meaningfully above random, a reasonable third
data-driven lens. Its channel credit mostly agrees with Markov/Shapley on
direction (Instagram and Online Video both pulled toward each other relative to
last-touch) but the *magnitude* differs more than Markov vs. Shapley do — expected,
since it's a different mechanism (predictive model + explainability) rather than
two variations on the same removal/coalition logic.

## E. Cross-model comparison (feeds into Document 5)
Build one table: rows = channels, columns = each model's `credit_share`. The
interesting finding is *always* the divergence — which channel is over/under-credited
by last-touch relative to Markov/Shapley — because that's the actionable insight.
