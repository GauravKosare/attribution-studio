# Attribution Studio — Multi-Touch Attribution Modeling for Marketing Spend

[![tests](https://github.com/GauravKosare/attribution-studio/actions/workflows/tests.yml/badge.svg)](https://github.com/GauravKosare/attribution-studio/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A working analytics system that answers a question every marketing team argues
about: which channel actually deserves credit for a conversion, and where should
budget move as a result?**

Give it real customer-journey data (or generate synthetic data, or upload your
own CSVs), and it reconstructs every customer's path across channels, scores each
channel's credit with **8 different attribution models** — from naive last-touch
to a Markov-chain removal-effect model to Shapley values to an XGBoost+SHAP
classifier — and turns the disagreement between those models into a concrete,
capped budget-reallocation recommendation with a real linear-programming
comparison alongside it. Everything recomputes live in a dashboard; nothing here
is a static report.

**[Live results below](#real-world-results) come from a real, anonymized
232,691-user customer-journey dataset — not synthetic placeholder numbers.**

### **[→ Try it live: attribution-studio.onrender.com](https://attribution-studio.onrender.com)**
Hosted on Render's free tier, so two things to expect: the first request after
a period of inactivity takes ~30–50s (free instances spin down when idle, then
cold-start), and larger sample sizes are noticeably slower than running this
locally (shared/throttled free-tier CPU) — the live demo defaults to a 5,000-user
sample for that reason; push it higher once it's loaded once and warmed up.

---

## Why this exists

Marketing budget decisions are usually made on **last-touch attribution** —
whichever channel gets the click right before a purchase gets 100% of the credit,
because that's what ad platforms report natively. That systematically overvalues
bottom-funnel channels (paid search retargeting, email) and undervalues
awareness/discovery channels (social, display, video) that set up conversions they
never get credited for.

This project exists to make that gap *visible and quantified*, not just asserted.
It was built to:
1. **Learn, hands-on, the industry-standard advanced attribution techniques** —
   Markov chain removal effect and Shapley value attribution — that most "intro to
   marketing analytics" content describes in a paragraph but rarely implements
   from scratch.
2. **Build a genuinely dynamic system**, not a one-off notebook — the same
   pipeline has to work identically on a real dataset, a synthetic one, or
   whatever CSV a user uploads.
3. **Practice the full analytics loop**, not just modeling: extraction (SQL),
   modeling (5 rule-based + Markov + Shapley + ML/SHAP), decision-making (ROI +
   two different budget-reallocation strategies), and communication (an
   executive-facing recommendation document with explicit caveats).
4. **Produce something that looks and runs like a real product**, because that's
   what actually differentiates a portfolio piece in a stack of similar
   attribution-modeling projects — see [`docs/07_frontend_options.md`](docs/07_frontend_options.md)
   for the honest reasoning behind the frontend choice.

## What problem it solves

> "Last-touch attribution said Paid Search deserved 22% of credit. But it's
> getting only 8.6% of spend. Meanwhile Online Video absorbs 35% of spend but a
> Markov removal-effect analysis — which measures how much conversion rate
> actually drops if you remove a channel from the journey graph — shows it earns
> only 19% of credit. That gap is real money sitting in the wrong place."

This project builds the actual pipeline that produces that sentence from data,
instead of asserting it.

## Skills this project is built to demonstrate (and did, in practice)
- **Marketing analytics domain knowledge**: attribution modeling theory, ROI/ROAS,
  the difference between correlational attribution and causal incrementality
- **Applied game theory**: implementing exact Shapley value attribution from the
  combinatorial formula, not from a library — and connecting it explicitly to
  SHAP (SHAP *is* an approximate Shapley value applied to an ML model's output)
- **Graph/Markov modeling**: building a transition graph, solving absorbing-state
  conversion probability via the fundamental matrix method, computing removal effect
- **Applied ML**: feature engineering from event sequences, XGBoost, SHAP
  explainability, and — importantly — reporting model quality (AUC) alongside
  its output rather than presenting every model's numbers with equal confidence
- **Operations research**: formulating budget reallocation as a linear program
  and solving it exactly with `scipy.optimize.linprog`, then honestly comparing
  it against a simpler heuristic instead of assuming "more optimal = better"
- **Data engineering**: a from-scratch vectorized pandas pipeline *and* an
  independent SQL (DuckDB) implementation of the same logic, cross-validated
  against each other down to the exact row order of tied timestamps
- **Full-stack delivery**: a Flask API + hand-written dashboard front end, not
  just a notebook — see [Real-world results](#real-world-results) for what it produces

---

## Real-world results

Every number below is from an actual run of this pipeline against the real
dataset described in [Data](#data) — not illustrative placeholders. Reproduce
with:
```python
from src.real_dataset_adapter import load_real_dataset
from src.pipeline import run_pipeline
tp, conv, spend = load_real_dataset("data/raw/real_channel_journeys.csv")
result = run_pipeline(tp, conv, spend, lookback_days=30, collapse_minutes=30,
                       roi_model="markov", shift_fraction=0.20, max_increase_pct=0.50)
```

### Dataset summary (full population, 232,691 journeys)
| Metric | Value |
|---|---|
| Date range | 2018-07-01 to 2018-07-31 |
| Total journeys | 232,691 |
| Conversions | 10,222 (4.4% conversion rate) |
| Median path length | 1 touch (mean 2.03, max 20, capped) |
| Median time-to-convert | 2.0 days |
| Channels | Facebook, Instagram, Online Display, Online Video, Paid Search |

### Channel credit across all rule-based + data-driven models
| Channel | First-touch | Last-touch | Linear | Time-decay | Position-based | Markov | Shapley |
|---|---|---|---|---|---|---|---|
| Facebook | 30.6% | 30.8% | 30.6% | 30.7% | 30.7% | 30.4% | 28.6% |
| Instagram | 14.0% | 13.6% | 14.1% | 14.0% | 13.9% | **17.7%** | 16.2% |
| Online Display | 10.8% | 10.3% | 10.5% | 10.4% | 10.6% | 10.7% | 10.6% |
| Online Video | 21.4% | 22.9% | 22.1% | 22.4% | 22.1% | **18.9%** | 21.2% |
| Paid Search | 23.2% | 22.4% | 22.8% | 22.6% | 22.8% | 22.3% | 23.4% |

**Headline finding**: last-touch over-credits Instagram by **+4.1 points** vs.
Markov, and under-credits nothing as sharply as it *mis-funds* Paid Search — which
earns 22.3% of Markov-modeled credit on just 8.6% of estimated spend. All credit
shares independently verified to sum to 1.0 per model (max rounding error 0.0001).

### ROI & capped budget reallocation (Markov model, +50% max increase per channel)
| Channel | Est. spend | Spend % | Credit % | ROAS | Recommended | Δ |
|---|---|---|---|---|---|---|
| Online Video | $1,319 | 35.2% | 18.9% | 9.18x | $1,055 | −20% |
| Facebook | $1,278 | 34.1% | 30.4% | 15.19x | $1,023 | −20% |
| Instagram | $584 | 15.6% | 17.7% | 19.35x | $467 | −20% |
| Paid Search | $323 | 8.6% | 22.3% | **44.03x** | $485 | **+50% (capped)** |
| Online Display | $241 | 6.4% | 10.7% | 28.32x | $362 | **+50% (capped)** |

Without the +50% cap, the naive proportional math would have proposed **+120%**
for Paid Search — an unrealistic overnight jump purely because it started
small. The cap redistributes via water-filling and leaves $354 (9.5%) honestly
unallocated rather than forcing it somewhere unrealistic.

### Heuristic vs. LP-optimal reallocation (same budget, same caps)
| Approach | Projected revenue | Lift vs. current ($63,874) |
|---|---|---|
| Current allocation | $63,874 | — |
| Capped heuristic (shipped recommendation) | $65,842 | **+3.1%** |
| LP-exact optimum (`scipy.optimize.linprog`) | $75,298 | +17.9% |

The LP solver is *mathematically optimal* given the model's linear-ROAS
assumption — it correctly concentrates budget into the single highest-ROAS
channel up to its cap before touching the next. But that also makes it a more
extreme extrapolation of an assumption that's already shaky at scale, which is
exactly why **the capped heuristic, not the LP result, is what this project
recommends as an actual first move** — see
[`docs/06_business_recommendation_template.md`](docs/06_business_recommendation_template.md)
for the full reasoning, not just the number.

### SQL vs. pandas journey extraction — parity check
Two independent implementations of the same journey-building logic
(`src/journey_builder.py` in pandas, `sql/build_journeys.sql` in DuckDB SQL),
run against the full 232,691-user dataset:

```
{'users_pandas': 232691, 'users_sql': 232691, 'users_in_common': 232691,
 'path_str_mismatches': 0, 'converted_mismatches': 0, 'match': True}
```

**Exact agreement**, including the ~0.05% of users whose touches share an
identical timestamp — which required adding an explicit, documented tie-breaker
(original row order) since pandas and DuckDB otherwise resolve exact ties
differently. That edge case, found and fixed during development, is exactly the
kind of thing a single implementation would never have caught.

### ML + SHAP attribution (XGBoost, 20K-user sample — opt-in due to runtime cost)
```
{'auc': 0.685, 'method': 'xgboost', 'n_train': 15000, 'n_test': 5000}
```
Test AUC comfortably above random (0.5), so its SHAP-derived channel credit is a
reasonable fourth data-driven lens — and it broadly agrees with Markov/Shapley on
*direction* (Instagram and Online Video pulled toward each other relative to
last-touch), while differing more in magnitude, which is expected from a
genuinely different mechanism (predictive model + explainability, vs.
removal-effect/coalition-value logic).

---

## Tech stack (why, not just what)

| Layer | Choice | Why |
|---|---|---|
| Data | Real anonymized dataset (bundled) + synthetic generator + CSV upload | Same pipeline, three ways to feed it — see [Data](#data) |
| Journey extraction | pandas (vectorized) **and** DuckDB SQL, cross-checked | Live dashboard needs pandas' speed; SQL demonstrates the skill and doubles as a correctness check |
| Attribution models | NetworkX (Markov), `itertools` (exact Shapley), XGBoost + SHAP | Implemented from the underlying math/formulas, not black-box libraries, wherever it mattered for understanding |
| Budget optimization | A capped proportional heuristic **and** `scipy.optimize.linprog` | Built both, then reasoned honestly about which one to trust — see the [results above](#heuristic-vs-lp-optimal-reallocation-same-budget-same-caps) |
| Backend | Flask | Thin JSON API in front of one pipeline function — no unnecessary framework weight |
| Frontend | Hand-written HTML/CSS/JS + Plotly.js | Full design control for a single-page tool, no build toolchain — see [`docs/07_frontend_options.md`](docs/07_frontend_options.md) for the full comparison against Streamlit/Dash/React/Power BI |

Full reasoning, alternatives considered, and free-tier notes for every tool:
[`SERVICE_PROVIDERS.md`](SERVICE_PROVIDERS.md).

## Data

`data/raw/real_channel_journeys.csv` is a **real, anonymized multi-channel
customer-journey dataset** — ~240K users, 586K impression/conversion events
across 5 real channels (Facebook, Instagram, Paid Search, Online Video, Online
Display), July 2018. It's the dashboard's default data source. Channel spend
isn't part of the public dataset, so it's estimated from impression volume via
documented CPM assumptions (`src/real_dataset_adapter.py`) — flagged everywhere
it's shown. You can also generate synthetic data or upload your own CSVs (schema:
[`docs/02_data_dictionary.md`](docs/02_data_dictionary.md)) from the same dashboard.

## Run it

```bash
git clone https://github.com/GauravKosare/attribution-studio.git
cd attribution-studio
pip install -r requirements.txt
python dashboards/server.py
```
Open **http://localhost:5050** — it loads with the real dataset by default and
recomputes live as you change the sidebar controls.

## Tests

```bash
python -m pytest tests/ -v
```
52 tests, ~15s. Not smoke tests — every attribution model is checked against a
hand-computed toy example with a worked derivation in the test file's docstring
(e.g. `tests/test_markov.py` hand-solves a 2-channel absorbing chain and asserts
the code reproduces it to 1e-6), and `tests/test_journey_builder_sql.py` exists
specifically to catch the exact class of bug that was found and fixed during
development (pandas vs. DuckDB resolving simultaneous-timestamp ties
differently). Runs automatically on every push via
[GitHub Actions](.github/workflows/tests.yml) — Python 3.11 and 3.12, plus a
smoke test that the Flask app actually boots.

## The 6 core documents + 1

| # | Document | Purpose |
|---|----------|---------|
| 1 | [`docs/01_project_charter.md`](docs/01_project_charter.md) | Problem statement, scope, objectives, success metrics — the PRD |
| 2 | [`docs/02_data_dictionary.md`](docs/02_data_dictionary.md) | Every table/field, source system, and the customer-journey schema |
| 3 | [`docs/03_technical_architecture.md`](docs/03_technical_architecture.md) | Pipeline design, tech stack, data flow diagram |
| 4 | [`docs/04_attribution_methodology.md`](docs/04_attribution_methodology.md) | How each attribution model works, math + code approach |
| 5 | [`docs/05_validation_report_template.md`](docs/05_validation_report_template.md) | Model comparison + validation, filled with real results |
| 6 | [`docs/06_business_recommendation_template.md`](docs/06_business_recommendation_template.md) | Executive-facing ROI/budget reallocation recommendation |
| 7 | [`docs/07_frontend_options.md`](docs/07_frontend_options.md) | Frontend tech-stack tradeoffs, argued without bias toward what was shipped |

Also: [`STEP_BY_STEP_GUIDE.md`](STEP_BY_STEP_GUIDE.md) (build plan),
[`LEARNING_GUIDE.md`](LEARNING_GUIDE.md) (concepts in the order you need them),
[`SERVICE_PROVIDERS.md`](SERVICE_PROVIDERS.md) (every tool + why).

## Repo layout

```
data/raw/                      # real dataset + synthetic/uploaded data
sql/build_journeys.sql          # SQL journey extraction (DuckDB)
src/
  data_generator.py             # synthetic touchpoint/conversion/spend generator
  real_dataset_adapter.py       # loads + reshapes the bundled real dataset
  journey_builder.py            # vectorized pandas journey extraction (live path)
  journey_builder_sql.py         # SQL journey extraction + parity check vs. pandas
  attribution/
    rule_based.py                # first/last-touch, linear, time-decay, position-based
    markov.py                    # Markov chain + removal effect
    shapley.py                   # exact / sampled Shapley value
    ml_shap.py                    # XGBoost / logistic regression + SHAP
  roi.py                        # ROI/ROAS + capped heuristic reallocation
  roi_optimizer.py               # LP-exact reallocation (scipy.optimize.linprog)
  pipeline.py                   # single function: data -> full result dict
dashboards/
  server.py                     # Flask API (/api/run, /api/upload, /api/meta)
  index.html                    # dashboard UI (dark/light, Plotly.js, no build step)
  app.py                        # earlier Streamlit version, kept for comparison
docs/                          # the 7 documents above
tests/                         # pytest suite, 52 tests -- hand-verified expected
                                # values, not just smoke tests (see Tests above)
.github/workflows/tests.yml    # CI: runs the suite on every push (Python 3.11 + 3.12)
```

## Status
Everything described above is implemented, tested, and verified against the real
dataset — not aspirational. What's still open (and why) is tracked candidly in
each document's own notes rather than hidden: e.g. Document 6 states plainly that
its "Next steps" (an incrementality test, replacing estimated spend with real
platform spend) are not yet done.

## License
[MIT](LICENSE) — the code is free to use, modify, and learn from. The bundled
real dataset has its own notice in the LICENSE file; verify its terms
independently before any commercial redistribution of the data itself.
