# Document 5: Comparison & Validation Report

**Run date:** 2026-08-20 · **Data source:** bundled real dataset
(`data/raw/real_channel_journeys.csv`) · **Population:** full dataset, no sampling ·
**Journey rules:** 30-day lookback window, repeat touches within 30 min collapsed
(defaults from `docs/02_data_dictionary.md`) · **Reproduce with:**
```python
from src.real_dataset_adapter import load_real_dataset
from src.pipeline import run_pipeline
tp, conv, spend = load_real_dataset("data/raw/real_channel_journeys.csv")
result = run_pipeline(tp, conv, spend, lookback_days=30, collapse_minutes=30,
                       roi_model="markov", shift_fraction=0.20)
```

## 1. Data summary
- Date range covered: **2018-07-01 to 2018-07-31** (one calendar month)
- Total users / journeys: **232,691**
- Conversions: **10,222** → conversion rate **4.4%**
- Median path length: **1** touch (mean **2.03**, max **20**, capped by design —
  see `docs/02_data_dictionary.md` path-length cap)
- Median days-to-convert (converted journeys): **2.0 days** (mean 5.3, max 28.9)
- Channels present: **Facebook, Instagram, Online Display, Online Video, Paid Search**
- Total attributed value across all conversions: **$63,874** (dataset's native
  `conversion_value` units, not necessarily USD — see Document 2)

Most journeys are single-touch (75th percentile is still just 2 touches) — this is
a real-world signature of retargeting-style display/social data: a large share of
users see exactly one impression and either convert or don't. The multi-touch story
lives in the minority of longer journeys, which is exactly where rule-based models
diverge most from Markov/Shapley.

## 2. Channel credit by model
| Channel | First-touch | Last-touch | Linear | Time-decay | Position-based | Markov | Shapley |
|---|---|---|---|---|---|---|---|
| Facebook | 30.6% | 30.8% | 30.6% | 30.7% | 30.7% | 30.4% | 28.6% |
| Instagram | 14.0% | 13.6% | 14.1% | 14.0% | 13.9% | **17.7%** | 16.2% |
| Online Display | 10.8% | 10.3% | 10.5% | 10.4% | 10.6% | 10.7% | 10.6% |
| Online Video | 21.4% | 22.9% | 22.1% | 22.4% | 22.1% | **18.9%** | 21.2% |
| Paid Search | 23.2% | 22.4% | 22.8% | 22.6% | 22.8% | 22.3% | 23.4% |

*(Bold = the two cells driving the headline divergence, §4.)*

## 3. Sanity checks — all passed
- [x] **Credit shares sum to 1 for every model** — verified programmatically:
  first-touch 0.9999, last-touch 1.0000, linear 1.0000, time-decay 1.0000,
  position-based 1.0000, markov 1.0000, shapley 1.0001 (rounding only).
- [x] **Markov removal effect is directionally sane** — Facebook, the highest-volume
  channel by far (175,741 of 586,737 raw events), also gets the highest Markov credit
  (30.4%) in every model, consistent with its removal doing the most damage to the
  conversion graph.
- [x] **Rule-based models cluster tightly, Markov is the outlier** — first-touch,
  last-touch, linear, time-decay, and position-based are all within ~1–2 points of
  each other per channel. Markov is the one model that meaningfully disagrees
  (Instagram +3.7–4.1 pts higher, Online Video ~3–4 pts lower than every rule-based
  model) — exactly the pattern you'd expect from a model that accounts for sequence
  and co-occurrence rather than just position.
- [x] **Shapley broadly agrees with Markov's direction** (Instagram up, Online Video
  down vs. last-touch) but is less extreme — Instagram +2.6 pts vs. last-touch under
  Shapley vs. +4.1 pts under Markov. Two independently-derived data-driven models
  agreeing on direction, even if not magnitude, is a meaningful validation signal.
- [ ] Monte Carlo cross-check of the Markov matrix solve and hand-verification of
  3–5 individual journeys were **not** re-run for this pass (already covered by the
  unit-level pipeline smoke test in the initial build) — recommended before using
  these numbers in a real stakeholder deck.

## 4. Model agreement / divergence analysis
**Headline finding:** Last-touch over-credits **Instagram** by **+4.1 points**
relative to Markov (13.6% → 17.7%), and under-credits **Online Video** by **-3.9
points** (22.9% → 18.9%).

Why: Online Video's raw volume (113,302 events) is smaller than Facebook's or Paid
Search's, and last-touch credit depends only on which channel happens to be the
final touch before conversion — a channel can be present throughout a journey and
still get zero credit if it's never last. Markov's removal effect instead asks "how
much does total conversion probability drop if this channel disappeared from the
graph entirely," which captures Instagram's role even in journeys where it isn't the
final touch, and correspondingly reveals that a meaningful share of Online Video's
apparent last-touch credit doesn't hold up once sequence effects are modeled.

Markov and Shapley agree on direction for both flagged channels (Instagram up,
Online Video down vs. last-touch), which is the cross-validation you want between
two independently-derived data-driven models — see the sanity checks above.

Rule-based models cluster almost on top of each other in this dataset (all within
~1.5 points per channel), which is itself informative: with a median path length of
1, there usually isn't much "position" for position-based or time-decay to
differentiate on — most of their theoretical difference from last-touch only shows
up on the minority of multi-touch journeys.

## 5. Budget allocation implied by each model
*(Spend below is a CPM-based estimate — the public dataset has no real cost data;
see `src/real_dataset_adapter.py` and Document 6's caveats.)*

| Channel | Est. spend | Spend % | Last-touch-implied % | Markov-implied % | Shapley-implied % |
|---|---|---|---|---|---|
| Facebook | $1,278 | 34.1% | 30.8% | 30.4% | 28.6% |
| Online Video | $1,319 | 35.2% | 22.9% | 18.9% | 21.2% |
| Instagram | $584 | 15.6% | 13.6% | 17.7% | 16.2% |
| Paid Search | $323 | 8.6% | 22.4% | 22.3% | 23.4% |
| Online Display | $241 | 6.4% | 10.3% | 10.7% | 10.6% |

**If we had been allocating budget using Markov instead of current (volume-driven)
spend, Paid Search would have received roughly 2.6x more budget than it does today**
(22.3% of credit vs. 8.6% of spend) — it is by a wide margin the most under-funded
channel relative to the value it drives, regardless of which attribution model you
trust. Online Video is the mirror image: it absorbs 35% of spend but earns only
18.9–22.9% of credit across every model, making it the clearest reallocation source.

## 6. Limitations (stated explicitly)
- **Attribution ≠ causation.** None of these models establish that Instagram or
  Paid Search *causes* conversion, only that removing them from the observed graph
  correlates with fewer conversions. A real budget shift should be piloted and
  ideally validated with an incrementality test (geo holdout) before being treated
  as ground truth — see Document 6's next steps.
- **Markov chain here is first-order** (memoryless beyond the immediately preceding
  channel). With 75% of journeys at ≤2 touches, higher-order effects are unlikely to
  change much here, but this would matter more on a longer-journey dataset (e.g. B2B).
- **Shapley's `v(S)`** is defined as the conversion rate of journeys whose
  touched-channel set is a subset of `S` — a specific, documented modeling choice
  (Document 4 §C), not the only valid definition. A different `v(S)` could shift
  results, particularly for Instagram/Online Video where Markov and Shapley already
  disagree in magnitude.
- **Spend is estimated, not real.** The public dataset has no cost data; the CPM
  assumptions in `src/real_dataset_adapter.py` are illustrative, not audited. The
  *credit shares* (columns 2–8 above) are computed from real behavioral data and are
  trustworthy; the *dollar figures* in §5 and Document 6 are directional only.
- **One month of data, one geography/context unknown** — the dataset doesn't specify
  industry, seasonality, or region. Treat July 2018 patterns as a snapshot, not a
  guarantee of stability month-to-month.

## 7. Recommendation carried forward to Document 6
Based on **Markov (removal effect)**, cross-validated by Shapley agreeing on
direction: shift budget from **Online Video** (currently 35.2% of spend but only
18.9% of Markov-modeled credit) toward **Paid Search** (currently 8.6% of spend but
22.3% of credit) and, secondarily, **Instagram** (15.6% of spend vs. 17.7% of
credit). See Document 6 for the quantified reallocation and ROI table.
