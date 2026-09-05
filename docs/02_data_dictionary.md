# Document 2: Data Dictionary & Journey Schema

## 1. Raw touchpoint data (`data/raw/touchpoints.csv`)
One row per touchpoint (event), not per user. This is the shape GA4 exports, ad-platform
API pulls, and CRM activity logs all roughly converge to.

| Field | Type | Description |
|---|---|---|
| `user_id` | string | Anonymized/hashed user or client ID |
| `touchpoint_id` | string | Unique ID for this event |
| `timestamp` | datetime (UTC) | When the touchpoint occurred |
| `channel` | string | One of: `paid_search`, `paid_social`, `email`, `organic_search`, `referral`, `direct`, `display` |
| `campaign_id` | string | Campaign identifier (nullable for organic/direct) |
| `campaign_name` | string | Human-readable campaign name |
| `platform` | string | e.g. `google_ads`, `meta_ads`, `mailchimp`, `organic` |
| `device` | string | `desktop` / `mobile` / `tablet` |
| `cost` | float | Cost attributable to this impression/click, if known (else null — spend usually comes in aggregate, see §3) |

## 2. Conversion events (`data/raw/conversions.csv`)
| Field | Type | Description |
|---|---|---|
| `user_id` | string | Matches touchpoints.user_id |
| `conversion_id` | string | Unique ID |
| `timestamp` | datetime | When conversion occurred |
| `revenue` | float | Order value / conversion value |
| `converted` | bool | True for this table by definition; used after join to label non-converters |

## 3. Channel spend (`data/raw/channel_spend.csv`)
Aggregate, not touchpoint-level — this is what you actually get from ad platforms.

| Field | Type | Description |
|---|---|---|
| `date` | date | Day of spend |
| `channel` | string | Matches touchpoints.channel |
| `campaign_id` | string | Nullable |
| `spend` | float | Amount spent that day |

## 4. Structured customer journey table (`data/processed/journeys.parquet`)
This is the output of the data-structuring step (Step 2) — one row per user, built by
grouping and ordering touchpoints. This is the table every attribution model reads from.

| Field | Type | Description |
|---|---|---|
| `user_id` | string | Primary key |
| `path` | list[string] | Ordered list of channels touched, e.g. `["organic_search","paid_social","email","paid_search"]` |
| `path_str` | string | Same, joined with `" > "` for readability/grouping (e.g. Sankey input) |
| `timestamps` | list[datetime] | Parallel list to `path` |
| `n_touches` | int | `len(path)` |
| `converted` | bool | Did this journey end in conversion |
| `conversion_value` | float | Revenue if converted, else 0 |
| `journey_start` | datetime | First touch |
| `journey_end` | datetime | Last touch or conversion time |
| `days_to_convert` | float | `journey_end - journey_start` in days |

### Journey construction rules (define these explicitly, they change results)
- **Lookback window**: touchpoints count toward a journey only if within N days of
  conversion (industry default: 30–90 days). Document whichever you pick.
- **Session vs. touchpoint**: decide whether repeated touches of the *same* channel
  within a short window (e.g. 30 min) collapse into one touchpoint. Recommended: collapse.
- **Non-converters**: keep them. Markov chain and Shapley both need the full population
  (including non-converting paths) to compute removal effect / marginal contribution correctly.
- **Path length cap**: cap at a reasonable max (e.g. 20) to avoid pathological outliers
  from bots/crawlers skewing the state graph.

## 5. Model output tables
- `channel_credit.csv` — `channel, model_name, credit_share` (one row per channel per model)
- `roi_summary.csv` — `channel, credit_share, spend, attributed_conversions, attributed_revenue, roas, recommended_spend, delta_spend`

## Notes on synthetic data generation
If using synthetic data (see Document 1), generate it so it has real-world structure,
not uniform randomness:
- Give each channel a different "role" bias (e.g. `organic_search`/`paid_social` skew
  toward journey start, `email`/`paid_search` skew toward journey end, `direct` skews
  toward end for returning users).
- Vary path length (Poisson-ish, mean ~3, long tail to ~10).
- Make conversion probability depend on the *combination* of channels touched, not just
  count — this is what makes Markov/Shapley diverge interestingly from last-touch, and
  is the whole point of the project.
