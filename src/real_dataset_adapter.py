"""Adapter for the bundled real-world dataset (data/raw/real_channel_journeys.csv).

Source: an anonymized real multi-channel customer-journey / attribution dataset
(cookie-level impressions + conversions across Facebook, Paid Search, Online Video,
Instagram, Online Display; ~240K users, July 2018). Original schema:
`cookie, time, interaction, conversion, conversion_value, channel`.
Mirror used: github.com/AjNavneet/MultiTouch-Attribution-Marketing-Spend-Optimization
(input/attribution_data.csv) — a commonly used public marketing-attribution dataset.

This module reshapes it into our standard schema (docs/02_data_dictionary.md) so it
plugs into the same pipeline as synthetic or uploaded data. Real ad spend is NOT part
of the public dataset, so channel spend is *estimated* from impression volume using
documented, editable CPM assumptions — this is flagged clearly wherever it's surfaced.
"""
from __future__ import annotations

import pandas as pd

# Rough, documented CPM ($ per 1000 impressions) assumptions per channel, used only to
# derive an illustrative spend figure since the public dataset has no real cost data.
DEFAULT_CPM = {
    "Facebook": 7.5,
    "Instagram": 8.0,
    "Paid Search": 2.2,  # modeled as an effective CPM equivalent of CPC * avg CTR
    "Online Video": 12.0,
    "Online Display": 3.5,
}


def load_real_dataset(
    csv_path: str, cpm: dict[str, float] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (touchpoints, conversions, channel_spend) in our standard schema."""
    raw = pd.read_csv(csv_path)
    raw["time"] = pd.to_datetime(raw["time"])
    cpm = cpm or DEFAULT_CPM

    impressions = raw[raw["interaction"] == "impression"].copy()
    touchpoints = pd.DataFrame(
        {
            "user_id": impressions["cookie"],
            "touchpoint_id": impressions.index.astype(str),
            "timestamp": impressions["time"],
            "channel": impressions["channel"],
            "campaign_id": impressions["channel"] + "_camp",
            "campaign_name": impressions["channel"],
            "platform": impressions["channel"],
            "device": "unknown",
        }
    )

    conv_rows = raw[raw["interaction"] == "conversion"].copy()
    conv_rows = conv_rows.drop_duplicates(subset="cookie", keep="first")
    conversions = pd.DataFrame(
        {
            "user_id": conv_rows["cookie"],
            "conversion_id": [f"conv_{i}" for i in range(len(conv_rows))],
            "timestamp": conv_rows["time"],
            "revenue": conv_rows["conversion_value"].astype(float),
            "converted": True,
        }
    )

    daily = (
        touchpoints.assign(date=touchpoints["timestamp"].dt.floor("D"))
        .groupby(["date", "channel"])
        .size()
        .rename("impressions")
        .reset_index()
    )
    daily["spend"] = daily.apply(
        lambda r: round(r["impressions"] / 1000.0 * cpm.get(r["channel"], 5.0), 2), axis=1
    )
    channel_spend = daily[["date", "channel", "spend"]].copy()
    channel_spend["campaign_id"] = channel_spend["channel"] + "_camp"

    return touchpoints, conversions, channel_spend


DATASET_DESCRIPTION = (
    "Real anonymized customer-journey dataset: ~240K users, 586K impression/conversion "
    "events across 5 real ad channels (Facebook, Instagram, Paid Search, Online Video, "
    "Online Display), July 2018. Channel spend is an ILLUSTRATIVE ESTIMATE derived from "
    "impression volume x assumed CPM (the public dataset has no real cost data) — treat "
    "ROI/ROAS figures from this source as directional, not audited."
)
