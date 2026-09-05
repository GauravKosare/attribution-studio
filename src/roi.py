"""ROI computation and budget reallocation simulation — docs/03 §4, docs/06.

Works on whatever credit table + spend table are handed to it; no hardcoded channels.
"""
from __future__ import annotations

import pandas as pd


def compute_roi(
    credit_df: pd.DataFrame,
    channel_spend: pd.DataFrame,
    total_conversions: int,
    total_revenue: float,
    model: str,
) -> pd.DataFrame:
    """credit_df: long-form [channel, model, credit_share]. Returns per-channel ROI table."""
    sub = credit_df[credit_df["model"] == model][["channel", "credit_share"]].copy()
    spend_by_channel = channel_spend.groupby("channel")["spend"].sum().rename("spend")

    roi = sub.merge(spend_by_channel, on="channel", how="outer").fillna(0.0)
    roi["attributed_conversions"] = roi["credit_share"] * total_conversions
    roi["attributed_revenue"] = roi["credit_share"] * total_revenue
    roi["roas"] = roi.apply(
        lambda r: (r["attributed_revenue"] / r["spend"]) if r["spend"] > 0 else float("nan"), axis=1
    )
    roi["spend_share"] = roi["spend"] / roi["spend"].sum() if roi["spend"].sum() > 0 else 0.0
    return roi.sort_values("attributed_revenue", ascending=False).reset_index(drop=True)


def recommend_reallocation(
    roi: pd.DataFrame,
    shift_fraction: float = 0.20,
    max_increase_pct: float | None = 0.50,
) -> pd.DataFrame:
    """Proportionally shift `shift_fraction` of total spend from below-average-ROAS
    channels toward above-average-ROAS channels, holding total spend constant.

    `max_increase_pct` caps how much any single channel's spend can grow in one
    step (e.g. 0.50 = at most +50%). Without a cap, a small-spend/high-ROAS channel
    can mechanically get proposed a 100%+ increase just because it started small —
    not a realistic first move. When a winner hits its cap, the capped portion of
    its share is redistributed (proportionally, by ROAS) among the remaining
    uncapped winners via water-filling. If every winner is capped before the full
    pool is placed, the leftover is left unallocated (total spend then drops
    slightly below the current total, rather than forcing an unrealistic increase
    onto some channel) — check `unallocated_pool` in the returned metadata if that
    matters for your use case.
    """
    out = roi.copy()
    valid_roas = out["roas"].replace([float("inf")], pd.NA).dropna()
    avg_roas = valid_roas.mean() if len(valid_roas) else 0.0

    out["roas_filled"] = out["roas"].fillna(0.0)
    out["above_avg"] = out["roas_filled"] > avg_roas

    pool = (out.loc[~out["above_avg"], "spend"] * shift_fraction).sum()

    out["recommended_spend"] = out["spend"].astype(float)
    out.loc[~out["above_avg"], "recommended_spend"] = out.loc[~out["above_avg"], "spend"] * (1 - shift_fraction)

    winner_idx = list(out.index[out["above_avg"]])
    if winner_idx and pool > 0:
        cap_spend = (
            {i: out.at[i, "spend"] * (1 + max_increase_pct) for i in winner_idx}
            if max_increase_pct is not None
            else {i: float("inf") for i in winner_idx}
        )
        alloc = {i: 0.0 for i in winner_idx}
        remaining_idx = list(winner_idx)
        remaining_pool = pool

        # water-filling: distribute remaining_pool by ROAS weight each round; any
        # channel whose share would exceed its headroom to cap gets filled exactly
        # to cap and drops out, and the next round redistributes among the rest
        while remaining_idx and remaining_pool > 1e-9:
            weights = out.loc[remaining_idx, "roas_filled"]
            weight_sum = weights.sum()
            share = (
                weights / weight_sum
                if weight_sum > 0
                else pd.Series(1.0 / len(remaining_idx), index=remaining_idx)
            )

            headroom = {i: max(0.0, cap_spend[i] - out.at[i, "spend"] - alloc[i]) for i in remaining_idx}
            newly_capped = [i for i in remaining_idx if share[i] * remaining_pool >= headroom[i] - 1e-9]

            if not newly_capped:
                for i in remaining_idx:
                    alloc[i] += share[i] * remaining_pool
                remaining_pool = 0.0
                break

            used_this_round = 0.0
            for i in remaining_idx:
                if i in newly_capped:
                    alloc[i] += headroom[i]
                    used_this_round += headroom[i]
                else:
                    given = share[i] * remaining_pool
                    alloc[i] += given
                    used_this_round += given
            remaining_pool -= used_this_round
            remaining_idx = [i for i in remaining_idx if i not in newly_capped]

        for i in winner_idx:
            out.at[i, "recommended_spend"] = out.at[i, "spend"] + alloc[i]

        unallocated_pool = max(0.0, remaining_pool)
    else:
        unallocated_pool = pool

    out["delta_spend"] = out["recommended_spend"] - out["spend"]
    out["delta_spend_pct"] = out.apply(
        lambda r: (r["delta_spend"] / r["spend"] * 100) if r["spend"] > 0 else float("nan"), axis=1
    )
    out.attrs["unallocated_pool"] = unallocated_pool
    return out.drop(columns=["above_avg", "roas_filled"])
