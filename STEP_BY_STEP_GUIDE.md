# Step-by-Step Build Guide

Follow in order. Each step names its inputs, outputs, and the document(s) it feeds.
Roughly a 2–3 week project at a few hours/day; compress or stretch as needed.

## Step 0 — Setup (30 min)
- [ ] `git init`, create the folder structure (already scaffolded in this repo)
- [ ] Set up a Python virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy networkx shap scikit-learn xgboost plotly duckdb streamlit
```
- [ ] Create a GitHub repo, push the scaffold, so every later step is a visible commit

## Step 1 — Get or generate data (2–3 days)
- [ ] Decide: real GA4/CRM export vs. synthetic generation (default: synthetic — see
      [`docs/01_project_charter.md`](docs/01_project_charter.md))
- [ ] If synthetic: write `src/generate_synthetic_data.py` producing
      `touchpoints.csv`, `conversions.csv`, `channel_spend.csv` per the schema in
      [`docs/02_data_dictionary.md`](docs/02_data_dictionary.md). Encode deliberate
      channel-role bias (some channels skew early-funnel, some late-funnel) — this is
      what makes the later model comparison interesting.
- [ ] If real: pull GA4 BigQuery export or Kaggle multi-touch-attribution dataset,
      reshape into the same schema.
- [ ] Sanity-check: row counts, date range, no duplicate touchpoint IDs, conversion
      rate looks plausible (1–10%, not 90%)

## Step 2 — Build the journey table (2 days)
- [ ] Write `sql/build_journeys.sql` (run against DuckDB/SQLite) implementing the
      lookback window and session-collapsing rules from Document 2
- [ ] Load into pandas, materialize `data/processed/journeys.parquet`
- [ ] Validate: path-length histogram looks like a decaying distribution (most
      journeys short, a long tail of longer ones), non-converters are present

## Step 3 — Rule-based attribution models (2 days)
- [ ] Implement `src/attribution/rule_based.py`: first-touch, last-touch, linear,
      time-decay, position-based — see Document 4 §A for exact formulas
- [ ] Unit-test each on 2–3 hand-computed toy journeys before running on the full table
- [ ] Output `channel_credit.csv` rows for each model

## Step 4 — Markov chain model (3 days)
- [ ] Build the transition graph with NetworkX (`src/attribution/markov.py`)
- [ ] Compute total conversion probability via matrix solve
- [ ] Implement removal effect per channel; normalize to `credit_share`
- [ ] Validate against a Monte Carlo simulation of the same chain (should roughly match)

## Step 5 — Shapley value model (2 days)
- [ ] Implement exact brute-force Shapley (`src/attribution/shapley.py`) for your
      channel set — see Document 4 §C for the formula and `v(S)` definition
- [ ] If channel count is large, add sampled/permutation Shapley as a fallback
- [ ] Sanity check: credit shares sum to 1; a channel appearing only in dead-end
      journeys should get ~0

## Step 6 — (Optional) ML + SHAP model (2 days)
- [ ] Feature-engineer journeys into a fixed vector (channel presence/count/recency)
- [ ] Train logistic regression baseline, then XGBoost
- [ ] Compute SHAP values, aggregate by channel
- [ ] Report model AUC alongside attribution — this is a nice added credibility signal

## Step 7 — Comparison & validation (1–2 days)
- [ ] Fill in [`docs/05_validation_report_template.md`](docs/05_validation_report_template.md)
      with real numbers from Steps 3–6
- [ ] Run every sanity check listed in that document
- [ ] Write the divergence analysis — this is your interview story, don't skip it

## Step 8 — ROI & budget optimization (2 days)
- [ ] Implement `src/roi.py`: join credit shares with `channel_spend.csv`, compute ROAS
- [ ] Implement a simple reallocation simulation (proportional shift toward
      above-average ROAS channels under fixed total budget); stretch: constrained
      optimizer with `scipy.optimize.linprog`
- [ ] Populate the ROI table in [`docs/06_business_recommendation_template.md`](docs/06_business_recommendation_template.md)

## Step 9 — Visualization (2–3 days)
- [ ] Sankey diagram of top N journey paths (Plotly `go.Sankey`) — use `path_str`
      grouped by frequency
- [ ] Channel-credit comparison bar chart across all models (grouped bar, Plotly)
- [ ] ROI table with recommended budget shift, rendered nicely (Plotly table or
      Streamlit dataframe)
- [ ] Assemble into a Streamlit app (`dashboards/app.py`) or Power BI report

## Step 10 — Write it up (1 day)
- [ ] Finish [`docs/06_business_recommendation_template.md`](docs/06_business_recommendation_template.md)
      as a clean 1-pager
- [ ] Write a top-level project summary (README already scaffolded) with screenshots
      of the Sankey and comparison chart
- [ ] Practice the 60-second interview pitch described in Document 1

## Step 11 — Polish for portfolio (ongoing)
- [ ] Clean commit history / README badges
- [ ] Deploy the Streamlit dashboard (Streamlit Community Cloud is free) so it's a live link
- [ ] Add a short Loom/GIF walkthrough to the README
