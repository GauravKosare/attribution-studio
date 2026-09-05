# Document 3: Technical Architecture

## Pipeline overview

```
[Data Source]                 [Structuring]              [Modeling]                    [Output]
Real dataset / GA4 / CRM  --> SQL (DuckDB)  ─┐            Rule-based (5 models)   --->  channel_credit
  or synthetic generator      pandas        ─┴─ journeys  Markov chain                  roi_summary
                              (cross-checked,              Shapley value                 Sankey + charts
                               0 mismatches)                XGBoost + SHAP (opt-in)      Flask+HTML dashboard
                                                            ROI: heuristic + LP (opt-in)
```

## Layer-by-layer

### 1. Data collection
- **Real-world option**: Google Analytics 4 (BigQuery export) or Google Ads / Meta Ads
  Marketing APIs for spend; CRM export (HubSpot/Salesforce) for offline touches.
- **Portfolio option (recommended to start)**: Python-generated synthetic dataset with
  realistic channel-role bias (see Document 2). Fastest path to a working end-to-end
  pipeline; swap in real data later without changing anything downstream.
- Either way, land raw exports in `data/raw/` as CSV/Parquet.

### 2. Data structuring — **two implementations, cross-checked against each other**
- `src/journey_builder.py` — pandas/vectorized (shift/cumcount, no per-row Python
  loop). This is what the live dashboard calls, because it needs to stay fast under
  a request.
- `sql/build_journeys.sql` — the same logic in SQL (DuckDB dialect: window
  functions for the lookback join, `LAG()` for repeat-touch collapsing, `LIST(...
  ORDER BY timestamp)` for path aggregation), run via `src/journey_builder_sql.py`.
  Written to demonstrate the SQL-skill angle explicitly, and to serve as an
  independent correctness check on the pandas version — see `verify_parity()` in
  that module.
- **Parity result** (full real dataset, 232,691 users): **0 mismatches** on path
  order and conversion outcome between the two implementations, once both used the
  same deterministic tie-breaker (`_seq`, the original row order) for the ~0.05%
  of users whose touches share an identical timestamp — see the comments in
  `sql/build_journeys.sql` for why that tie-breaker is necessary at all (pandas'
  sort and DuckDB's `ORDER BY` otherwise resolve exact ties differently).

### 3. Attribution modeling (`src/attribution/`)
- `rule_based.py` — first-touch, last-touch, linear, time-decay, position-based (U-shaped)
- `markov.py` — build transition matrix between channel-states (+ `Start`/`Conversion`/`Null`
  absorbing states) with NetworkX or plain numpy; compute total conversion probability;
  compute removal effect per channel by zeroing its transitions and re-running the chain
- `shapley.py` — compute each channel's marginal contribution across all coalition
  subsets of channels present in the data (use itertools + memoized coalition conversion
  rates; for larger channel sets use an approximation like Shapley via ordering sampling)
- `ml_shap.py` — logistic regression or XGBoost on touchpoint-presence features
  predicting conversion, then SHAP values as a fourth attribution lens (see
  Document 4 §D — implemented, opt-in in the dashboard due to its runtime cost)

### 4. ROI & budget optimization (`src/roi.py`)
- Join `channel_credit` (any model) with `channel_spend` aggregated by channel
- `ROAS = attributed_revenue / spend` per channel per model
- Reallocation simulation: shift spend from below-average-ROAS channels to
  above-average-ROAS channels (proportional to ROAS), under a total-budget
  constraint — with a `max_increase_pct` cap (default 50%) on how much any single
  channel's spend can grow in one step, via water-filling: a channel that hits its
  cap stops absorbing more of the pool, and the remainder redistributes among the
  still-uncapped winners. Without this cap, a small-spend/high-ROAS channel can
  mechanically get proposed a 100%+ increase purely because it started small — not
  a realistic first move (this happened in the first pass of Document 6 before the
  cap was added: Paid Search was proposed +120%). If every winner hits its cap
  before the full pool is placed, the leftover is left unallocated rather than
  forced onto some channel — surfaced as `unallocated_pool` and shown as a warning
  banner in the dashboard when non-zero.
- **`src/roi_optimizer.py` — a second, LP-exact reallocation**, alongside the
  heuristic above (opt-in in the dashboard, both are shown side by side for
  comparison when enabled). Maximizes `Σ roas_i × spend_i` subject to total spend
  held constant and the same per-channel bounds, via `scipy.optimize.linprog`
  (HiGHS solver). Because the objective and constraints are linear, this is exactly
  solvable — no iteration needed — and it *correctly* concentrates budget into the
  single highest-ROAS channel up to its cap before touching the next, rather than
  spreading share proportionally like the heuristic. That's mathematically optimal
  under the model's stated linear-ROAS assumption, but the more aggressive
  reallocation it produces is also a *more* extreme extrapolation of that
  assumption — see Document 6 for why the heuristic, not the LP result, is what's
  actually recommended as a first move, with the LP result shown for comparison.

### 5. Visualization — shipped as a Flask + hand-written HTML/JS dashboard
- Sankey diagram of journey paths — Plotly.js (`Sankey` trace)
- Channel-credit comparison bar chart + heatmap table across all models — Plotly.js
- ROI table with recommended shift, LP-vs-heuristic comparison — Plotly.js + plain
  HTML tables, styled directly (no component library)
- `dashboards/server.py` (Flask API: `/api/run`, `/api/upload`, `/api/meta`) +
  `dashboards/index.html` (single-file frontend, dark/light theme, live sliders) —
  see the "Frontend options considered" note below for why this over Streamlit/Dash/Power BI

## Tech stack summary
| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | Standard for this kind of modeling work |
| Data wrangling | pandas, DuckDB | pandas for the live dashboard path (fast, no extra dependency in the request path); DuckDB for the SQL journey-extraction demonstration, cross-checked against pandas for parity |
| Graph/Markov | NetworkX | Clean transition-matrix + graph API, plus you get the Sankey-adjacent graph structure for free |
| Shapley | itertools + custom, and `shap` for the ML-based variant | Exact Shapley over ~5-8 channels is tractable by brute force; implemented directly to show the math before reaching for a shortcut |
| ML | scikit-learn / XGBoost + `shap` | Fourth attribution lens (Document 4 §D) — implemented, opt-in due to runtime cost |
| Optimization | `scipy.optimize.linprog` (HiGHS) | Exact LP solution for budget reallocation, shown alongside the simpler capped heuristic for comparison (Document 3 §4) |
| Backend | Flask | Thin JSON API in front of `src/pipeline.py` — no ORM/templating needed since the frontend is a static page calling the API |
| Frontend | Hand-written HTML/CSS/JS + Plotly.js | Full control over layout/interaction that a default Streamlit theme can't match, zero build tooling |
| Notebook env | Jupyter / VS Code notebooks | Exploratory work before promoting logic into `src/` |
| Version control | Git + GitHub | Portfolio visibility |

See [`SERVICE_PROVIDERS.md`](../SERVICE_PROVIDERS.md) for the full reasoning behind each choice
and free-tier/alternative options.

## Data flow diagram (textual)
```
touchpoints ─┐
conversions ─┼─> journey_builder.py (pandas, live) ──────┐
             │   journey_builder_sql.py (DuckDB, parity)  │
             ┘                                            v
                                                       journeys ─┬─> rule_based.py ──┐
                                                                 ├─> markov.py       ├─> channel_credit ─┐
                                                                 ├─> shapley.py      ┘                    ├─> roi.py (heuristic)     ─┐
                                                                 └─> ml_shap.py (opt-in) ──────────────────┘   roi_optimizer.py (opt-in)├─> dashboard
                                                       channel_spend ──────────────────────────────────────────────────────────────────┘
```
