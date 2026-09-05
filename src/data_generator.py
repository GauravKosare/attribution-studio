"""Synthetic touchpoint/conversion/spend generator.

Produces data with deliberate channel-role bias (some channels skew early-funnel,
some late-funnel) so that rule-based and data-driven attribution models diverge in
realistic, explainable ways. Used when the user has no real journey data to upload.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CHANNELS = {
    # channel: (funnel_position bias 0=early..1=late, base conversion lift)
    "organic_search": 0.15,
    "paid_social": 0.20,
    "display": 0.10,
    "paid_search": 0.75,
    "email": 0.80,
    "referral": 0.45,
    "direct": 0.90,
}


def generate_dataset(
    n_users: int = 5000,
    channels: dict[str, float] | None = None,
    start_date: str = "2026-01-01",
    days: int = 90,
    base_conversion_rate: float = 0.06,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate (touchpoints, conversions, channel_spend) DataFrames.

    Journey length is Poisson-ish with a long tail; channel choice at each step is
    weighted by funnel position so early- and late-funnel channels cluster where
    they realistically would. Conversion probability depends on the *combination*
    of channels touched (not just count), which is what makes Markov/Shapley
    diverge interestingly from last-touch.
    """
    rng = np.random.default_rng(seed)
    channels = channels or DEFAULT_CHANNELS
    channel_names = list(channels.keys())
    late_bias = np.array([channels[c] for c in channel_names])

    start = pd.Timestamp(start_date)
    touchpoints_rows = []
    conversions_rows = []

    for uid in range(n_users):
        user_id = f"user_{uid:06d}"
        n_touches = max(1, min(12, rng.poisson(2.5) + 1))
        journey_start = start + pd.Timedelta(days=rng.uniform(0, days - 7))

        picked = []
        for step in range(n_touches):
            progress = step / max(1, n_touches - 1)
            weights = np.exp(-((late_bias - progress) ** 2) / 0.15)
            weights = weights / weights.sum()
            ch = rng.choice(channel_names, p=weights)
            picked.append(ch)

        gap_hours = rng.exponential(20, size=n_touches).cumsum()
        timestamps = [journey_start + pd.Timedelta(hours=h) for h in gap_hours]

        for ch, ts in zip(picked, timestamps):
            touchpoints_rows.append(
                {
                    "user_id": user_id,
                    "touchpoint_id": f"{user_id}_{ts.value}",
                    "timestamp": ts,
                    "channel": ch,
                    "campaign_id": f"{ch}_camp_{rng.integers(1, 4)}",
                    "campaign_name": f"{ch.title()} Campaign {rng.integers(1, 4)}",
                    "platform": ch,
                    "device": rng.choice(["desktop", "mobile", "tablet"], p=[0.45, 0.45, 0.10]),
                }
            )

        # conversion probability: base rate + lift from late-funnel channels present
        # and a bonus for touching 3+ distinct channels (multi-touch synergy)
        lift = sum(channels[c] for c in set(picked)) / len(channels)
        synergy = 0.03 if len(set(picked)) >= 3 else 0.0
        p_convert = min(0.95, base_conversion_rate + lift * 0.25 + synergy)

        if rng.random() < p_convert:
            conv_ts = timestamps[-1] + pd.Timedelta(hours=rng.exponential(4))
            revenue = float(rng.normal(85, 25))
            conversions_rows.append(
                {
                    "user_id": user_id,
                    "conversion_id": f"conv_{uid:06d}",
                    "timestamp": conv_ts,
                    "revenue": max(10.0, round(revenue, 2)),
                    "converted": True,
                }
            )

    touchpoints = pd.DataFrame(touchpoints_rows)
    conversions = pd.DataFrame(conversions_rows)

    # aggregate daily spend per channel, roughly proportional to touchpoint volume
    # plus noise, so spend and volume aren't perfectly correlated (mimics real life)
    tp_counts = touchpoints.groupby("channel").size()
    spend_rows = []
    dates = pd.date_range(start, periods=days, freq="D")
    for ch in channel_names:
        daily_base = (tp_counts.get(ch, 10) / days) * rng.uniform(3, 8)
        for d in dates:
            spend_rows.append(
                {
                    "date": d,
                    "channel": ch,
                    "campaign_id": f"{ch}_camp_1",
                    "spend": max(0.0, round(float(rng.normal(daily_base, daily_base * 0.25)), 2)),
                }
            )
    channel_spend = pd.DataFrame(spend_rows)

    return touchpoints, conversions, channel_spend


if __name__ == "__main__":
    tp, conv, spend = generate_dataset()
    tp.to_csv("data/raw/touchpoints.csv", index=False)
    conv.to_csv("data/raw/conversions.csv", index=False)
    spend.to_csv("data/raw/channel_spend.csv", index=False)
    print(f"touchpoints={len(tp)} conversions={len(conv)} spend_rows={len(spend)}")
