"""LP-exact reallocation (scipy.optimize.linprog) — regression-tested against
the exact figures verified by hand and cross-checked against the dashboard's
live output (docs/06 "Heuristic vs. LP-optimal reallocation").
"""
from __future__ import annotations

import pytest

from src.roi_optimizer import optimize_reallocation
from tests.test_roi import REAL_ROI_FIXTURE


def test_lp_optimal_matches_hand_verified_figures():
    out = optimize_reallocation(REAL_ROI_FIXTURE.copy(), max_increase_pct=0.50, max_decrease_pct=0.50)
    assert out.attrs["lp_status"] == "optimal"

    by_channel = out.set_index("channel")["recommended_spend_lp"]
    # The LP fills bounds strictly in ROAS order: the two highest-ROAS channels
    # (Paid Search 44.03x, Online Display 28.32x) hit their +50% cap, and the
    # lowest-ROAS channel (Online Video 9.18x) is pushed to its -50% floor.
    assert by_channel["Paid Search"] == pytest.approx(323.16 * 1.5, abs=0.01)
    assert by_channel["Online Display"] == pytest.approx(241.21 * 1.5, abs=0.01)
    assert by_channel["Online Video"] == pytest.approx(1318.73 * 0.5, abs=0.01)
    # Instagram (19.35x, the next-highest ROAS after the two winners above)
    # also hits its own +50% cap -- there's enough budget freed by Online
    # Video's floor to fully fund it too.
    assert by_channel["Instagram"] == pytest.approx(583.68 * 1.5, abs=0.01)
    # Facebook (15.19x, the LOWEST of the remaining channels) absorbs whatever
    # budget is left after every other bound is pinned -- NOT its own +50% cap.
    # This is the one channel whose recommended spend sits strictly between its
    # bounds, which is exactly what "the LP concentrates budget by ROAS rank
    # until the budget constraint binds" predicts for a 5-channel problem.
    remaining_for_facebook = REAL_ROI_FIXTURE["spend"].sum() - (
        by_channel["Paid Search"] + by_channel["Online Display"]
        + by_channel["Online Video"] + by_channel["Instagram"]
    )
    assert by_channel["Facebook"] == pytest.approx(remaining_for_facebook, abs=0.01)
    assert 1278.29 * 0.5 < by_channel["Facebook"] < 1278.29 * 1.5


def test_lp_conserves_total_budget_exactly():
    out = optimize_reallocation(REAL_ROI_FIXTURE.copy(), max_increase_pct=0.50, max_decrease_pct=0.50)
    assert out["recommended_spend_lp"].sum() == pytest.approx(REAL_ROI_FIXTURE["spend"].sum(), abs=0.01)


def test_lp_projected_revenue_exceeds_heuristic():
    # The LP is mathematically optimal given the linear-ROAS assumption, so its
    # objective value must be >= any feasible alternative, including the
    # capped-proportional heuristic's allocation -- this is the core claim
    # docs/06 makes and the dashboard displays; verify it's actually true.
    from src.roi import recommend_reallocation

    lp = optimize_reallocation(REAL_ROI_FIXTURE.copy(), max_increase_pct=0.50, max_decrease_pct=0.50)
    heuristic = recommend_reallocation(REAL_ROI_FIXTURE.copy(), shift_fraction=0.20, max_increase_pct=0.50)

    lp_revenue = (lp["recommended_spend_lp"] * lp["roas"]).sum()
    heuristic_revenue = (heuristic["recommended_spend"] * heuristic["roas"]).sum()
    assert lp_revenue >= heuristic_revenue


def test_infeasible_bounds_falls_back_gracefully():
    # Force every channel's LOWER bound to 150% of current spend (negative
    # max_decrease_pct = a floor ABOVE current spend) with no upper cap. Since
    # the equality constraint requires sum(x) == current total spend, but every
    # x_i is forced >= 1.5x its current spend, the sum can never reach the
    # required total -- genuinely infeasible. Must not crash; must fall back to
    # current spend rather than propagate a scipy exception.
    out = optimize_reallocation(REAL_ROI_FIXTURE.copy(), max_increase_pct=None, max_decrease_pct=-0.5)
    assert "infeasible" in out.attrs["lp_status"]
    assert not out["recommended_spend_lp"].isna().any()
    assert (out["recommended_spend_lp"] == out["spend"]).all()
