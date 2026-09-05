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
> Last-touch attribution (today's default) over-credits **Instagram** by **+4.1 pts**
> and under-credits nothing as sharply as it *mis-funds* **Paid Search**, which earns
> **22.3%** of Markov-modeled conversion credit on just **8.6%** of spend. Shifting
> spend away from the two lowest-ROAS channels (Online Video, Facebook) toward Paid
> Search and Online Display — capped at +50% growth per channel so no channel gets an
> unrealistic overnight jump — is projected, under a simplifying linear-ROAS
> assumption, to lift attributed revenue by **~3%**. That's the trustworthy,
> conservative estimate; see Expected impact for why the *uncapped* version of this
> same reallocation would have overstated the case.

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
1. **Increase** spend on **Paid Search** (from $323 to $485 est., **+50%, capped**)
   — it has by far the highest modeled ROAS (44.0x) and is currently the most
   under-funded channel relative to the conversion credit it earns. The model's
   uncapped math would have proposed +120%; +50% is the realistic cycle-one move.
2. **Increase** spend on **Online Display** (from $241 to $362 est., **+50%,
   capped**) — second-highest ROAS (28.3x), also under-funded (uncapped math: +103%).
3. **Decrease** spend on **Online Video** (−20%, from $1,319 to $1,055) — largest
   spend share (35.2%) but lowest ROAS (9.2x) and the channel most over-credited by
   last-touch relative to Markov (see Document 5 §4).
4. **Decrease** spend on **Facebook** and **Instagram** (−20% each) — both
   reasonable performers, but below the average ROAS threshold the model's
   reallocation logic uses to distinguish winners from losers.
5. **Leave $354 (9.5% of current spend) unallocated this cycle** rather than force
   it onto Paid Search or Online Display beyond the +50% cap. Revisit next cycle
   once this round's results are in — if Paid Search sustains its ROAS at the new
   spend level, raise the cap or run a second +50% step rather than jumping straight
   to the uncapped +120%.

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
