# Document 1: Project Charter

## Problem statement
Marketing spend is split across channels (paid search, paid social, email, referral,
organic, direct) but budget decisions are usually made on **last-touch attribution**
— whichever channel gets the click right before a purchase gets 100% of the credit.
This systematically overvalues bottom-funnel channels (paid search retargeting, email)
and undervalues awareness/discovery channels (social, display), leading to misallocated
budget.

## Objective
Build a system that:
1. Reconstructs each customer's full journey of touchpoints before conversion.
2. Scores channel credit using multiple attribution models — simple rule-based
   baselines and two data-driven/algorithmic models (Markov chain removal effect,
   Shapley value).
3. Combines attribution credit with actual channel spend to compute ROI per channel.
4. Recommends a reallocated budget split and quantifies the expected lift.
5. Presents all of the above through visuals a marketing stakeholder can act on
   without reading code.

## In scope
- Touchpoint-level customer journey data (channel, campaign, timestamp, conversion flag)
- Rule-based models: first-touch, last-touch, linear, time-decay, position-based (U-shaped)
- Markov chain attribution with removal effect
- Shapley value attribution
- Optional: logistic regression / XGBoost + SHAP as a third data-driven lens
- ROI computation and budget reallocation simulation
- Sankey diagram of journey paths, channel-credit comparison chart, ROI table

## Out of scope (v1)
- Real-time/streaming attribution
- Cross-device identity resolution (assume a single `user_id` is already resolved)
- Incrementality testing / geo holdouts (mention as "next step," don't build)
- Privacy-sensitive PII handling — use synthetic or fully anonymized data only

## Data reality check
Real ad-platform and CRM journey-level data is rarely public. Default plan: **generate
a realistic synthetic dataset** (documented explicitly as synthetic) that mimics GA4 /
CRM export shape. If you have access to a real GA4 property or a Kaggle multi-touch
attribution dataset, swap the ingestion step only — everything downstream is unchanged.
See [`docs/02_data_dictionary.md`](02_data_dictionary.md) for the exact schema, and
[`SERVICE_PROVIDERS.md`](../SERVICE_PROVIDERS.md) for data source options.

## Success metrics (what "done" looks like)
- [ ] A journeys table with realistic path-length distribution (most conversions have 2–6 touches)
- [ ] 5 rule-based models + Markov + Shapley all implemented and producing channel-credit tables
- [ ] A written comparison showing how much recommendations diverge between last-touch and Markov/Shapley (this divergence *is* the interview story)
- [ ] An ROI table with a recommended budget shift and a stated expected-conversion-lift estimate
- [ ] Sankey diagram + channel comparison bar chart + ROI table, in a dashboard or notebook
- [ ] A 1-page executive summary (Document 6) a non-technical stakeholder could act on

## Stakeholder framing (for interview storytelling)
Practice being able to say, in under 60 seconds:
> "Last-touch attribution said Paid Search deserved 45% of credit. The Markov removal-effect
> model — which measures how much conversion rate actually drops if you remove a channel from
> the journey graph — showed Paid Search only drives 22%, and Email/Retargeting was
> free-riding on demand that Social and Organic actually created. Reallocating budget
> accordingly, holding total spend constant, projects a lift of X% in conversions."

## Timeline (reference — see STEP_BY_STEP_GUIDE.md for detail)
| Phase | Duration |
|---|---|
| Data collection/generation | 2–3 days |
| Data structuring (journeys) | 2 days |
| Rule-based models | 2 days |
| Markov + Shapley | 4–5 days |
| ROI/budget optimization | 2 days |
| Visualization + writeup | 3 days |
