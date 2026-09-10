# Document 9: Robustness — how much does the estimated spend assumption move the conclusion?

The headline finding has two parts. They do not rest on the same footing, and this document
separates them.

## Part 1 — model disagreement (no cost assumption involved)

| Channel | Last-touch | Markov | Last-touch − Markov |
|---|---|---|---|
| Facebook | 30.8% | 30.4% | +0.4 |
| **Instagram** | **13.6%** | **17.7%** | **−4.1** |
| Online Display | 10.3% | 10.7% | −0.4 |
| **Online Video** | **22.9%** | **19.0%** | **+3.9** |
| Paid Search | 22.4% | 22.3% | +0.1 |

This comes purely from journey structure. Relative to Markov, last-touch **under-credits
Instagram** by 4.1 points and **over-credits Online Video** by 3.9. Shapley agrees on the
direction for both; the five rule-based models sit within ~1.5 points of each other. No
spend, cost, or CPM input touches any of these numbers — this is the robust core of the
project's finding.

## Part 2 — "channel X is under/over-funded" (entirely a function of the CPM table)

The public dataset has no cost data, so channel spend is estimated as
`impressions × assumed CPM` (`src/real_dataset_adapter.py`, `DEFAULT_CPM`). The
"Paid Search is under-funded" line in earlier drafts of Document 6 depended heavily on one
assumption: Paid Search is given a CPM of \$2.2, versus \$7.5–12 for the display/social
channels — even though Paid Search carries **25.8% of all impressions** in the data.

### Test: Monte Carlo over plausible CPM ranges

50,000 draws, each channel's CPM sampled uniformly from a plausible 2018-era range
(`scratchpad`/repo script). Funding gap = Markov credit share − spend share, in points.

| Scenario | Paid Search verdict | P(Paid Search is the *most* under-funded channel) |
|---|---|---|
| Original point CPMs | under-funded by 13.7 pts | — |
| Equal CPM for every channel | **over-funded by 3.5 pts** | — |
| MC, wide ranges (Paid Search effective CPM \$10–80) | over-funded in **99.7%** of draws | **0.00** |
| MC, conservative ranges (Paid Search kept cheap, \$3–30) | under-funded in only 13% of draws | 0.06 |

The project's own stated derivation for the Paid Search figure — "CPC × average CTR" —
actually implies an *effective CPM* of roughly \$20–100 for realistic search click-through
rates, i.e. Paid Search should be **expensive** per impression, not cheap. The \$2.2 value
looks like an error, and the honest scenarios are the ones where the "under-funded" call
disappears.

### The one spend-dependent call that survives

**Online Display** stays under-funded in >99% of all scenarios sampled — small channel,
genuinely low CPM, ~10.7% Markov credit against ~6–12% spend. That is a ~\$120 reallocation,
not a headline.

## Conclusion

- **Lead with Part 1.** It is assumption-free and cross-validated by an independent model.
- **Treat every spend-derived number (ROAS, the dollar reallocation, "under-funded") as a
  demonstration of the *mechanism*, not an audited recommendation** — with Online Display
  as the single directionally-robust move. Documents 5 and 6 are written accordingly.
- Replacing estimated spend with real platform cost data is the highest-value single
  improvement to the ROI half of this project, and is listed as such in Document 6's next
  steps.

*This check also surfaced a labelling bug (the `divergence` field and several docs said
"over-credits" where they meant "under-credits"); fixed across `src/pipeline.py`, the
dashboard, and Documents 5–6.*
