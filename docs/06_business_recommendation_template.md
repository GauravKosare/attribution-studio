# Document 6: Business Recommendation & ROI Report

The executive-facing deliverable. Methodology and caveats live in Document 5, linked, not repeated.

---

## Marketing Attribution & Budget Reallocation — Executive Summary

**Period covered:** July 1–31, 2018
**Prepared by:** Attribution Studio pipeline (`src/pipeline.py`), full population (232,691 journeys)
**Model used for recommendation:** Markov chain (removal effect), cross-validated against Shapley value
**Reallocation settings:** 20% shift from below-average-ROAS channels, capped at **+50% max
increase per channel** (`max_increase_pct=0.50` in `src/roi.py::recommend_reallocation`) —
see the note below the ROI table for why the cap exists and what it costs.

### The headline
> Last-touch attribution (today's default) **under-credits Instagram by 4.1 points**
> (13.6% vs. 17.7% under Markov) and **over-credits Online Video by 3.9 points**
> (22.9% vs. 18.9%). Shapley, an independently derived model, agrees on the direction
> for both. This credit-side finding rests only on behavioral data and no cost
> assumption — it is the robust core of this report.
>
> Converting that into a *budget* move additionally requires channel spend, which this
> dataset does not contain (it is estimated — see the spend caveat). A
> [CPM sensitivity check](09_robustness_cpm_sensitivity.md) shows the reallocation
> direction is stable for only one channel: **Online Display** is genuinely under-funded
> across essentially all plausible CPM assumptions. The larger "Paid Search is starved"
> figure is an artifact of one weak CPM input and reverses under most alternatives —
> treat the dollar table below as a demonstration of the capped-reallocation *mechanism*,
> not as an audited recommendation.

### ROI table
*(Spend is a CPM-based estimate — see caveat below the table.)*

| Channel | Spend (est.) | Spend % | Credit % (Markov) | Attr. conversions | Attr. revenue | ROAS | Recommended spend | Δ spend |
|---|---|---|---|---|---|---|---|---|
| Online Video | $1,319 | 35.2% | 18.9% | 1,937 | $12,102 | 9.18x | $1,055 | −$264 (−20%) |
| Facebook | $1,278 | 34.1% | 30.4% | 3,108 | $19,419 | 15.19x | $1,023 | −$256 (−20%) |
| Instagram | $584 | 15.6% | 17.7% | 1,807 | $11,293 | 19.35x | $467 | −$117 (−20%) |
| Paid Search | $323 | 8.6% | 22.3% | 2,277 | $14,228 | **44.03x** | $485 | +$162 (**+50%, capped**) |
| Online Display | $241 | 6.4% | 10.7% | 1,093 | $6,832 | 28.32x | $362 | +$121 (**+50%, capped**) |
| **Total** | **$3,745** | 100% | 100% | 10,222 | $63,874 | — | **$3,391** | **−$354 (unallocated)** |

**Spend caveat:** this dataset (a real, anonymized customer-journey dataset — see
Document 5) does not include real ad-platform cost data. Spend above is estimated
from impression volume × documented CPM assumptions (`src/real_dataset_adapter.py`).
**Credit %, attributed conversions, and ROAS *ranking* are computed from real
behavioral data and are trustworthy; absolute dollar figures are illustrative.**

**Why the total doesn't rebalance to exactly $3,745:** with only two channels
(Paid Search, Online Display) above the average-ROAS threshold, the full 20% pool
pulled from the three underperformers ($636) can't be placed into just two channels
without either of them growing more than 50% — so $354 of that pool is deliberately
left unallocated rather than forced onto Paid Search or Online Display as an
unrealistic jump. Total recommended spend is $3,391, about 9.5% below current. This
is a property of the cap, not a bug: see `unallocated_pool` in `src/roi.py`, which
the dashboard also surfaces as a warning banner when it's non-zero.

### Recommended actions

*Ranked by how much each survives the spend-estimation caveat.*

1. **Rebalance credit expectations, not budget, first.** The reliable finding is that
   Instagram is doing 4 points more work than last-touch reporting shows, and Online
   Video 4 points less. Before moving a dollar, stop judging Instagram on last-click
   and stop rewarding Online Video for it. This costs nothing and needs no cost data.
2. **Increase** spend on **Online Display** (from $241 to $362 est., **+50%, capped**)
   — second-highest modeled ROAS (28.3x) and the one channel that comes out under-funded
   across essentially every plausible CPM assumption (small base, genuinely low CPM,
   ~10.7% Markov credit).
3. **Decrease** spend on **Online Video** (−20%, from $1,319 to $1,055) — largest
   spend share (35.2%), lowest ROAS (9.2x), and the channel last-touch over-credits
   most relative to Markov (see Document 5 §4). Robust in direction.
4. **Treat the Paid Search "+50%" line as conditional.** Its modeled ROAS (44x) is
   the highest in the table, but that number divides real credit by *estimated* spend,
   and Paid Search's estimated spend share (8.6%) is the least defensible cell in the
   dataset. Only raise Paid Search budget after checking its real platform cost.
5. **Decrease** spend on **Facebook** (−20%) — a reasonable performer, but below the
   average ROAS threshold the reallocation logic uses. *Note:* the model also flags
   Instagram for a −20% cut on the same ROAS-threshold logic, which sits awkwardly
   next to the credit finding that Instagram is under-recognised — a good example of
   why the ROAS-threshold heuristic is a starting point, not the last word.
6. **Leave the unallocated remainder ($354, ~9.5% of current spend) unallocated this
   cycle** rather than force it past the +50% cap. Revisit next cycle once real cost
   data and this round's results are in.

### Expected impact
- **Projected revenue lift: ~3.1%** ($63,874 → ~$65,841), computed by applying each
  channel's *current* ROAS to its *capped* recommended spend level
  (`Σ recommended_spend_i × roas_i`).
- **This is deliberately the conservative number.** The same reallocation *without*
  the +50% cap projects a ~24% lift — but that figure assumes Paid Search converts
  just as efficiently at more than double its current spend, which is not credible.
  Capping the per-channel increase produces a smaller but far more defensible
  projection, at the cost of leaving some of the theoretical pool unallocated.
- **Simplifying assumption stated plainly either way:** this assumes constant
  marginal ROAS within the (now bounded) spend range. A capped +50% increase is a
  much smaller extrapolation than an uncapped +120%, which is the entire point of
  capping — the projection is more conservative *because* it's more believable, not
  despite it.
- **Confidence level: Directional, not causal.** This is a reallocation
  recommendation based on correlational attribution (Markov removal effect,
  cross-validated by Shapley agreeing on direction — Document 5 §4), not a
  guaranteed outcome. Recommend piloting on a smaller, staged shift before a full
  reallocation, and pairing with an incrementality test to validate causally.

### What this does NOT tell us
- Whether Paid Search's high modeled ROAS reflects a genuinely more efficient
  channel, or a measurement artifact of how last-click-adjacent behavior gets
  recorded in this dataset.
- The real diminishing-returns curve for Paid Search or Online Display even at the
  capped +50% — the linear-ROAS assumption is still a placeholder within that range,
  just a smaller extrapolation than the uncapped version.
- What to do with the $354 left unallocated — reinvesting it into a *third* channel,
  spreading it evenly, or simply not spending it are all reasonable options this
  report doesn't adjudicate between.
- Whether the underlying spend figures are even close to real — they're CPM-based
  estimates on a dataset with no disclosed cost data (Document 5 §6). Before acting
  on this in a real business, replace `channel_spend.csv` with actual platform spend
  and re-run the pipeline; the credit-share numbers won't change, but every dollar
  figure in this document will.
- Anything about new customer acquisition vs. re-engaging existing customers — this
  dataset doesn't distinguish the two.

### Next steps
- [x] ~~Cap the maximum per-channel % increase~~ — done (`max_increase_pct=0.50` in
  `src/roi.py`, exposed as a live slider in the dashboard's ROI panel).
- [ ] Replace estimated spend with real platform spend data the moment it's
  available, and re-run — this document's credit-share findings are robust to that
  swap, but every dollar figure here should be treated as a placeholder until then.
- [ ] Decide what to do with the unallocated pool (reinvest, hold, or spread it) once
  stakeholders weigh in — the model deliberately doesn't decide this for you.
- [ ] Re-run attribution monthly (or as new data arrives) to check whether channel
  credit is stable or drifting.
- [ ] Design a lightweight incrementality test (geo holdout) for Paid Search and
  Online Display — the two channels this report recommends funding more — before
  committing to a larger, permanent budget shift or raising the cap.
