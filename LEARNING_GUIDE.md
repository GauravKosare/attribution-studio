# Learning Guide

Concepts in the order you'll need them, matched to the step in
[`STEP_BY_STEP_GUIDE.md`](STEP_BY_STEP_GUIDE.md) that requires them. Don't front-load
all of this — learn each block right before the step that uses it.

## Before Step 1: Marketing attribution fundamentals
- What a "touchpoint" and "conversion window" mean in marketing analytics
- Why last-touch/first-touch attribution is the default in most ad platforms (it's
  what Google Ads/Meta report natively) and why that biases budget decisions
- Skim: Google's own explainer on attribution models (search "Google Ads attribution
  models comparison") — useful for vocabulary, not for the math

## Before Step 2: SQL for sessionization / journey building
- Window functions: `LAG`/`LEAD`, `ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp)`
- `ARRAY_AGG` / `LIST` aggregation to build the `path` column from ordered rows
- Session/lookback-window logic: filtering touchpoints to N days before conversion
- If SQL is new to you: practice window functions specifically — that's 90% of what
  journey-building needs

## Before Step 3: Just careful reading of Document 4 §A
- No new math — these are the "obvious" attribution rules. The only subtlety is
  time-decay's exponential weighting; make sure you understand half-life before coding it

## Before Step 4: Markov chains
- Core concept: states, transition probabilities, absorbing states
- "Absorbing Markov chain" specifically — Conversion and Null are absorbing states,
  everything else transitions onward. Look up the "fundamental matrix" method for
  computing absorption probabilities analytically (this is what makes your removal-effect
  computation exact rather than simulated)
- Concept of "removal effect" is specific to attribution — not standard Markov-chain
  vocabulary, so search "Markov chain attribution removal effect marketing" rather than
  a general Markov chains course for this part
- Practical: work through one tiny hand-computed example (3 channels, ~10 journeys) on
  paper before trusting your code — this is the single highest-leverage learning step
  in the whole project

## Before Step 5: Shapley values
- Cooperative game theory basics: players, coalitions, characteristic function `v(S)`
- The Shapley formula itself (Document 4 §C) — work through a 3-player toy example by
  hand (there are many worked examples online for "Shapley value 3 player example")
- Why it satisfies efficiency/symmetry/null-player/additivity — you don't need to prove
  these, but you should be able to explain in one sentence each what they mean, since
  this is the property that makes Shapley the "fair" attribution method
- Marketing-specific framing: search "Shapley value marketing attribution" for
  worked examples using channels-as-players, closer to what you're building than a
  generic game theory resource

## Before Step 6 (optional ML path): SHAP values
- Difference between SHAP and raw feature importance (SHAP is locally accurate and
  additive per-prediction, not just a global ranking)
- `shap.TreeExplainer` for tree models is fast/exact; `shap.LinearExplainer` for
  logistic regression; avoid `KernelExplainer` unless necessary (it's slow)
- Conceptual link back to Shapley values (Step 5) — SHAP *is* an approximation of
  Shapley values applied to ML model outputs. Noticing and being able to explain this
  connection is a strong interview signal

## Before Step 8: ROI / budget optimization basics
- ROAS (return on ad spend) definition and how it differs from ROI
- Diminishing returns / saturation curves in marketing response — even a simple
  mention that "ROAS isn't linear in spend" in your writeup shows maturity
- If attempting the optimizer stretch goal: basic linear programming — objective
  function, constraints, `scipy.optimize.linprog` usage

## Before Step 9: Sankey diagrams and dashboarding
- Plotly `go.Sankey` — nodes, links, values; how to bucket long-tail paths into an
  "Other" node so the diagram stays readable
- Streamlit basics if going that route: `st.dataframe`, `st.plotly_chart`, caching
  with `@st.cache_data`

## Throughout: the interview narrative
- Be able to explain, without notes, why Markov/Shapley usually diverge from last-touch
  in the *same direction* — favoring earlier/assist channels — and why that's not a
  coincidence (last-touch structurally can't credit anything but the final touch)
- Be able to name the biggest limitation of your own project unprompted (see
  Document 5 §6) — this is what separates "I followed a tutorial" from "I understand
  what I built"

## Suggested learning resources (general, not linked — search these terms)
- "Markov chain attribution model marketing Python tutorial"
- "Shapley value attribution marketing channels worked example"
- "GA4 BigQuery export schema" (if going the real-data route)
- "SHAP values explained" (the shap library's own docs/readme are unusually good)
- Kaggle: search "multi touch attribution dataset" for a possible real/semi-real dataset
