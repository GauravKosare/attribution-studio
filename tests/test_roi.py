"""ROI computation and capped reallocation — regression-tested against the exact
figures verified by hand in docs/06_business_recommendation_template.md (the
real dataset's 5-channel ROI table), plus edge cases for the water-filling cap.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.roi import recommend_reallocation

# The real dataset's channel/spend/ROAS figures (docs/06), used as a fixed
# regression fixture -- if these numbers ever drift, either the model changed
# (expected -- update the doc) or the reallocation math broke (a real bug).
REAL_ROI_FIXTURE = pd.DataFrame(
    [
        {"channel": "Online Video", "spend": 1318.73, "roas": 9.1774},
        {"channel": "Facebook", "spend": 1278.29, "roas": 15.1912},
        {"channel": "Instagram", "spend": 583.68, "roas": 19.3479},
        {"channel": "Paid Search", "spend": 323.16, "roas": 44.0263},
        {"channel": "Online Display", "spend": 241.21, "roas": 28.3247},
    ]
)


def test_capped_reallocation_matches_hand_verified_figures():
    out = recommend_reallocation(REAL_ROI_FIXTURE.copy(), shift_fraction=0.20, max_increase_pct=0.50)
    by_channel = out.set_index("channel")["recommended_spend"]

    # Below-average-ROAS channels always get a flat -20% (the shift_fraction),
    # uncapped by max_increase_pct since they're decreasing, not increasing.
    assert by_channel["Online Video"] == pytest.approx(1054.984, abs=0.01)
    assert by_channel["Facebook"] == pytest.approx(1022.632, abs=0.01)
    assert by_channel["Instagram"] == pytest.approx(466.944, abs=0.01)

    # Above-average-ROAS channels are capped at exactly +50% -- with only two
    # winners and a $636.14 pool from three losers, both winners hit their cap
    # before the pool is exhausted (see the unallocated_pool assertion below).
    assert by_channel["Paid Search"] == pytest.approx(323.16 * 1.5, abs=0.01)
    assert by_channel["Online Display"] == pytest.approx(241.21 * 1.5, abs=0.01)


def test_unallocated_pool_when_winners_cap_out():
    out = recommend_reallocation(REAL_ROI_FIXTURE.copy(), shift_fraction=0.20, max_increase_pct=0.50)
    # pool = 0.20 * (1318.73 + 1278.29 + 583.68) = 636.14
    # used  = (484.74-323.16) + (361.815-241.21) = 161.58 + 120.605 = 282.185
    # unallocated = 636.14 - 282.185 = 353.955
    assert out.attrs["unallocated_pool"] == pytest.approx(353.955, abs=0.01)


def test_no_cap_produces_larger_uncapped_increase():
    out = recommend_reallocation(REAL_ROI_FIXTURE.copy(), shift_fraction=0.20, max_increase_pct=None)
    by_channel = out.set_index("channel")["recommended_spend"]
    # without a cap, total spend is exactly conserved (no unallocated pool)
    assert by_channel.sum() == pytest.approx(REAL_ROI_FIXTURE["spend"].sum(), abs=0.01)
    # and Paid Search's increase must be LARGER than the capped +50% case
    assert by_channel["Paid Search"] > 323.16 * 1.5


def test_tight_cap_produces_larger_unallocated_pool():
    loose = recommend_reallocation(REAL_ROI_FIXTURE.copy(), shift_fraction=0.20, max_increase_pct=0.50)
    tight = recommend_reallocation(REAL_ROI_FIXTURE.copy(), shift_fraction=0.20, max_increase_pct=0.10)
    assert tight.attrs["unallocated_pool"] > loose.attrs["unallocated_pool"]
    # every winner's recommended spend must respect its own cap exactly
    for row in tight.itertuples():
        if row.channel in ("Paid Search", "Online Display"):
            assert row.recommended_spend <= row.spend * 1.10 + 0.01


def test_recommended_spend_never_goes_negative():
    # a pathological case: shift_fraction=1.0 (100% of loser spend moved out)
    out = recommend_reallocation(REAL_ROI_FIXTURE.copy(), shift_fraction=1.0, max_increase_pct=None)
    assert (out["recommended_spend"] >= 0).all()


def test_zero_spend_channel_does_not_crash():
    df = pd.DataFrame(
        [
            {"channel": "A", "spend": 0.0, "roas": float("nan")},
            {"channel": "B", "spend": 100.0, "roas": 5.0},
        ]
    )
    out = recommend_reallocation(df, shift_fraction=0.2, max_increase_pct=0.5)
    assert not out["recommended_spend"].isna().any()
