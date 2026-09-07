# Service Providers & Tools — What and Why

Every external service/tool this project can use, why it's the right pick, free-tier
notes, and alternatives. Organized by pipeline stage.

## Data source

| Option | Why | Notes |
|---|---|---|
| **Bundled real dataset** (`data/raw/real_channel_journeys.csv`) — used by the dashboard by default | Real, anonymized customer-journey data (~240K users, 586K events, 5 real channels: Facebook, Instagram, Paid Search, Online Video, Online Display, July 2018) — no signup, no API key, plugs straight into the pipeline | Free, public. Mirrored via [AjNavneet/MultiTouch-Attribution-Marketing-Spend-Optimization](https://github.com/AjNavneet/MultiTouch-Attribution-Marketing-Spend-Optimization) (`input/attribution_data.csv`), a commonly used public marketing-attribution dataset. Channel spend isn't part of the public data — `src/real_dataset_adapter.py` estimates it from impression volume x documented CPM assumptions; treat ROI/ROAS from this source as directional, not audited |
| **Criteo Attribution Modeling for Bidding Dataset** | A real ad-tech company's own click/conversion data, 16.5M rows, released specifically for attribution-modeling research (AdKDD 2017) | Free, [huggingface.co/datasets/criteo/criteo-attribution-dataset](https://huggingface.co/datasets/criteo/criteo-attribution-dataset), CC-BY-NC-SA-4.0. Schema is campaign/click-level (no named channels like "email"/"social") — good alternative if you want to demonstrate handling messier real ad-platform data, more setup work than the bundled dataset |
| **Synthetic data (Python-generated)** | No access barrier, no PII risk, full control over ground truth so you can validate your models against a known-correct answer | Free. Document explicitly as synthetic in every deliverable — credibility matters more than realism here |
| **Google Analytics 4 (GA4) + BigQuery export** | Industry-standard web analytics; free BigQuery export is the real shape of touchpoint data most companies actually have | Free tier: GA4 is free; BigQuery has a free monthly query quota (1TB) that's more than enough for a portfolio dataset. Needs a GA4 property with export configured — set this up on a personal site/blog if you don't have one at work |
| **Kaggle multi-touch attribution datasets** | Real(ish) journey-level data without needing your own tracked property | Free. Search "multi touch attribution" on Kaggle. Quality/schema varies — expect to reshape it |
| **Google Ads / Meta Ads Marketing APIs** | For real channel spend figures if you have an active ad account | Free API access, but requires an active advertiser account with spend history — not realistic for most portfolio builds unless you already run ads |
| **CRM exports (HubSpot/Salesforce)** | Offline touchpoints (email opens, sales calls) that ad-platform data misses | Free-tier HubSpot works for a demo CRM if you want to simulate a fuller funnel |

## Data structuring

| Tool | Why | Alternative |
|---|---|---|
| **DuckDB** | Embedded, zero-setup SQL engine that runs directly on CSV/Parquet files — lets you write real analytical SQL (window functions, `ARRAY_AGG`) without standing up a database server | SQLite (simpler, but weaker window-function/array support); a real Postgres instance if you want to demonstrate that specifically |
| **pandas** | Standard for iteration once data is in Python; easiest place to prototype journey logic before finalizing as SQL | polars (faster on large data, newer skill to show off if relevant to target roles) |

## Attribution modeling

| Tool | Why | Alternative |
|---|---|---|
| **NetworkX** | Clean, well-documented graph API for building the Markov transition graph; the resulting graph structure is directly reusable for diagramming | Roll your own transition-matrix code with pure numpy if you want to demonstrate you don't need a library for the math |
| **numpy** | Matrix solve for absorbing Markov chain conversion probabilities | — |
| **Plain Python (itertools)** | Exact Shapley computation via brute-force subset enumeration — deliberately not using a library here proves you understand the formula | `shapley` PyPI packages exist but hide the math; use them only for cross-checking your own implementation |
| **scikit-learn** | Logistic regression baseline for the optional ML attribution lens; universally recognized, minimal setup | — |
| **XGBoost** | Handles channel-interaction effects (non-linear combinations) better than logistic regression; also one of the most commonly requested skills in analytics/DS job postings | LightGBM (comparable, faster on larger data) |
| **SHAP (`shap` library)** | The standard tool for explaining ML model predictions feature-by-feature; ties your ML attribution lens back to the same game-theoretic foundation as your custom Shapley implementation — a nice narrative through-line | — |

## ROI / optimization

| Tool | Why | Alternative |
|---|---|---|
| **scipy.optimize.linprog** (`src/roi_optimizer.py`) | Exact LP solver (HiGHS backend) for budget reallocation — implemented alongside the simpler capped-proportional heuristic in `src/roi.py`, shown side by side in the dashboard so the tradeoff (LP is optimal under the model's assumptions but more aggressive; the heuristic is more conservative) is visible rather than asserted | PuLP (more readable constraint syntax if the LP gets complex) |
| **DuckDB** (`sql/build_journeys.sql`, `src/journey_builder_sql.py`) | SQL-based journey extraction — window functions, `LAG()`-based dedup, `LIST(... ORDER BY)` aggregation — cross-checked against the pandas implementation for parity (0 mismatches on the full real dataset, once both used the same deterministic tie-breaker for simultaneous-timestamp touches) | A real warehouse (Postgres/Snowflake/BigQuery) if you want to run this against a live table instead of an in-memory frame — the SQL only needs the syntax notes at the bottom of `build_journeys.sql` adjusted |
| **XGBoost + SHAP** (`src/attribution/ml_shap.py`) | Fourth attribution lens: predict conversion from channel-presence features, explain with SHAP — ties classical attribution back to modern ML explainability, and surfaces a trust signal (test AUC) alongside the numbers | Logistic regression is included as a faster, more stable baseline (`method="logistic"`) |

## Visualization / dashboarding

| Tool | Why | Alternative |
|---|---|---|
| **Plotly.js** | Native, well-supported Sankey diagram support — no other common library makes this as easy; interactive out of the box; used both server-side (Python) and here client-side via CDN for the live dashboard charts | matplotlib + a manual Sankey layout (much more work, not worth it here) |
| **Flask** (`dashboards/server.py`) | Thin JSON API in front of `src/pipeline.py` — three routes (`/api/meta`, `/api/run`, `/api/upload`), no ORM/templating needed since the frontend is a static page that calls the API | FastAPI (comparable, adds async + auto-docs if the project grows) |
| **Hand-written HTML/CSS/JS dashboard** (`dashboards/index.html`) | Full control over layout and interaction (sidebar controls, live sliders, dark/light theme, KPI cards, heatmap table) that a default Streamlit theme can't match, with zero build tooling — one file, open `index.html` served by Flask, no npm/webpack | Streamlit (much faster to scaffold, but harder to make look like a real product UI); a React/Vite frontend (more powerful component model, real build step, worth it only if the project grows past a single page) |
| **Power BI** | If the target roles/companies are enterprise BI-heavy, a `.pbix` report demonstrates a skill this project's stack doesn't — DAX, relationships, enterprise dashboarding conventions | Tableau (comparable positioning, different tool familiarity depending on target employer) |

**This project ships with the Flask + hand-written HTML dashboard** — it's the best
fit for a portfolio piece meant to look and feel like a real product, not a data-science
notebook. Add Power BI only if the specific jobs you're targeting list it explicitly —
it's a different skill from the rest of this project's stack.

## Testing / CI

| Tool | Why | Notes |
|---|---|---|
| **pytest** (`tests/`) | Standard Python test runner; used for hand-verified attribution-math tests (each toy example is solved by hand in the test docstring, not just asserted against current code output), regression tests for the SQL/pandas parity bug found during development, and end-to-end pipeline smoke tests | Free. 52 tests, ~15s locally — see `README.md` "Tests" section |
| **GitHub Actions** (`.github/workflows/tests.yml`) | Runs the full suite on every push/PR across Python 3.11 and 3.12, plus a smoke test that the Flask app actually boots | Free for public repos, no setup beyond the workflow file |

## Hosting / delivery

| Tool | Why | Notes |
|---|---|---|
| **GitHub** | Version control + the actual portfolio artifact recruiters look at | Free, essential |
| **Render / Railway / Fly.io free tier** | Hosting for the Flask dashboard (not Streamlit Community Cloud — that's Streamlit-specific and this project moved to a hand-written Flask+HTML frontend, see Document 3 §5) | Any of the three has a free tier that runs a small Flask app fine; the real dataset (44MB CSV) fits within typical free-tier disk/memory limits |
| **Jupyter / VS Code notebooks** | Where the exploratory modeling work (Steps 3–6) actually happens before being refactored into `src/` modules | Google Colab if you want free GPU/cloud compute and easy sharing — not needed for this project's scale, but zero-friction if you prefer not to manage a local environment |

## What you deliberately do NOT need
- No paid ad-platform spend (synthetic spend figures are fine and clearly labeled)
- No cloud data warehouse (BigQuery free tier or local DuckDB covers this scale)
- No GPU/managed ML platform — everything here trains in seconds to minutes locally
- No enterprise CDP or identity-resolution vendor — assume `user_id` is already resolved
  (explicitly scoped out in Document 1)
