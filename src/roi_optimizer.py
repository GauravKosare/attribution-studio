"""Constrained linear-programming budget optimizer — the "real optimizer" stretch
goal from docs/03_technical_architecture.md, alongside the simpler proportional
heuristic in roi.py::recommend_reallocation.

Formulation: maximize modeled attributed revenue (credit_share x total_revenue is
already ~linear in spend under the same "current ROAS holds at new spend" caveat
every model in this project states explicitly), subject to:
  - total spend held constant (equality constraint)
  - each channel's spend bounded within [current * (1 - max_decrease_pct),
    current * (1 + max_increase_pct)] — the same kind of realism cap as roi.py's
    water-filling heuristic, expressed here as LP bounds instead of an iterative
    redistribution rule.

Because the objective is linear in spend (coefficients = each channel's current
ROAS) and every constraint is linear, this reduces to a small linear program —
scipy.optimize.linprog's HiGHS solver finds the exact optimum instantly, no
iteration needed. It will, correctly, put as much of the movable budget as the
bounds allow into the single highest-ROAS channel first, then the next highest,
and so on — the textbook LP solution to a knapsack-shaped problem. Compare this to
the heuristic's proportional-by-ROAS split: the LP is optimal *given the stated
assumptions*, the heuristic is more conservative (spreads share among several
above-average channels rather than concentrating it in the single best one). Which
one to trust more is a judgment call — see docs/06 for how this project frames it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def optimize_reallocation(
    roi: pd.DataFrame,
    max_increase_pct: float | None = 0.50,
    max_decrease_pct: float | None = 0.50,
) -> pd.DataFrame:
    """LP-optimal reallocation of the same total budget, maximizing modeled revenue.

    Returns a copy of `roi` with `recommended_spend_lp`, `delta_spend_lp`,
    `delta_spend_pct_lp` columns added, plus `.attrs["lp_status"]` describing
    solver success/failure so a caller can fall back to the heuristic if the LP
    is infeasible (e.g. bounds too tight to hit the required total).
    """
    out = roi.copy()
    n = len(out)
    spend = out["spend"].to_numpy(dtype=float)
    roas = out["roas"].fillna(0.0).replace([np.inf, -np.inf], 0.0).to_numpy(dtype=float)
    total_budget = spend.sum()

    # linprog minimizes, so negate the objective to maximize sum(roas_i * x_i)
    c = -roas

    lower = spend * (1 - max_decrease_pct) if max_decrease_pct is not None else np.zeros(n)
    upper = spend * (1 + max_increase_pct) if max_increase_pct is not None else None
    lower = np.maximum(lower, 0.0)
    bounds = list(zip(lower, upper if upper is not None else [None] * n))

    # equality constraint: total spend unchanged
    A_eq = [np.ones(n)]
    b_eq = [total_budget]

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        recommended = result.x
        status = "optimal"
    else:
        # infeasible bounds (e.g. sum of lower bounds already exceeds budget) —
        # fall back to current spend rather than crash the pipeline
        recommended = spend.copy()
        status = f"infeasible: {result.message}"

    out["recommended_spend_lp"] = recommended
    out["delta_spend_lp"] = out["recommended_spend_lp"] - out["spend"]
    out["delta_spend_pct_lp"] = out.apply(
        lambda r: (r["delta_spend_lp"] / r["spend"] * 100) if r["spend"] > 0 else float("nan"), axis=1
    )
    out.attrs["lp_status"] = status
    out.attrs["lp_objective_value"] = float(-result.fun) if result.success else None
    return out
