"""Single entry point tying together journey building, every attribution model,
and ROI/reallocation — returns one JSON-serializable dict. Used by the dashboard
server so the whole computation is one function call, testable outside Flask too.
"""
from __future__ import annotations

import pandas as pd

from src.attribution import markov, ml_shap, rule_based, shapley
from src.journey_builder import build_journeys
from src.roi import compute_roi, recommend_reallocation
from src.roi_optimizer import optimize_reallocation

MODEL_LABELS = {
    "first_touch": "First-touch",
    "last_touch": "Last-touch",
    "linear": "Linear",
    "time_decay": "Time-decay",
    "position_based": "Position-based (U)",
    "markov": "Markov (removal effect)",
    "shapley": "Shapley value",
    "xgboost_shap": "XGBoost + SHAP",
    "logistic_shap": "Logistic + SHAP",
}


def _sankey_from_journeys(journeys: pd.DataFrame, top_n: int = 20) -> dict:
    converted = journeys[journeys["converted"]]
    if converted.empty:
        return {"nodes": [], "links": []}
    counts = converted["path_str"].value_counts()
    top_paths = set(counts.head(top_n).index)

    nodes: list[str] = []
    node_idx: dict[str, int] = {}

    def get_node(label: str) -> int:
        if label not in node_idx:
            node_idx[label] = len(nodes)
            nodes.append(label)
        return node_idx[label]

    links: dict[tuple[int, int], int] = {}
    for path_str, count in counts.items():
        path = path_str.split(" > ") if path_str else []
        if not path:
            continue
        display_path = path if path_str in top_paths else ["Other paths"]
        display_path = display_path[:6]
        prev_label = None
        for i, ch in enumerate(display_path):
            label = f"{i + 1}. {ch}"
            src = get_node(prev_label) if prev_label else get_node(label)
            dst = get_node(label)
            if prev_label is not None:
                key = (src, dst)
                links[key] = links.get(key, 0) + int(count)
            prev_label = label

    link_list = [{"source": s, "target": t, "value": v} for (s, t), v in links.items()]
    return {"nodes": nodes, "links": link_list}


def run_pipeline(
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    channel_spend: pd.DataFrame,
    lookback_days: int = 30,
    collapse_minutes: int = 30,
    roi_model: str = "markov",
    shift_fraction: float = 0.20,
    max_increase_pct: float | None = 0.50,
    sankey_top_n: int = 20,
    include_ml: bool = False,
    ml_method: str = "xgboost",
    include_lp: bool = False,
) -> dict:
    journeys = build_journeys(
        touchpoints, conversions,
        lookback_days=lookback_days, collapse_repeats_minutes=collapse_minutes,
    )
    if journeys.empty:
        return {"error": "No journeys could be built from this data."}

    rule_credit = rule_based.run_all(journeys)
    markov_credit = markov.run(journeys)
    shapley_credit = shapley.run(journeys)
    credit_frames = [rule_credit, markov_credit, shapley_credit]

    ml_diagnostics = None
    if include_ml:
        ml_credit, ml_diagnostics = ml_shap.run(journeys, method=ml_method)
        if not ml_credit.empty:
            credit_frames.append(ml_credit)

    all_credit = pd.concat(credit_frames, ignore_index=True)
    all_credit["model_label"] = all_credit["model"].map(MODEL_LABELS)

    pivot = all_credit.pivot_table(
        index="channel", columns="model_label", values="credit_share", fill_value=0.0
    )
    ordered_cols = [v for v in MODEL_LABELS.values() if v in pivot.columns]
    pivot = pivot[ordered_cols]

    divergence = None
    if "Last-touch" in pivot.columns and "Markov (removal effect)" in pivot.columns:
        diff = (pivot["Markov (removal effect)"] - pivot["Last-touch"]).sort_values()
        divergence = {
            "under_credited": diff.index[0],
            "under_pts": round(float(diff.iloc[0]) * 100, 1),
            "over_credited": diff.index[-1],
            "over_pts": round(float(diff.iloc[-1]) * 100, 1),
        }

    total_conversions = int(journeys["converted"].sum())
    total_revenue = float(journeys["conversion_value"].sum())
    roi = compute_roi(all_credit, channel_spend, total_conversions, total_revenue, roi_model)
    roi = recommend_reallocation(roi, shift_fraction=shift_fraction, max_increase_pct=max_increase_pct)
    unallocated_pool = roi.attrs.get("unallocated_pool", 0.0)

    lp_comparison = None
    if include_lp:
        roi_lp = optimize_reallocation(
            roi, max_increase_pct=max_increase_pct, max_decrease_pct=max_increase_pct
        )
        heuristic_projected = float((roi["recommended_spend"] * roi["roas"].fillna(0)).sum())
        lp_projected = float((roi_lp["recommended_spend_lp"] * roi_lp["roas"].fillna(0)).sum())
        lp_comparison = {
            "status": roi_lp.attrs.get("lp_status"),
            "rows": roi_lp[["channel", "recommended_spend_lp", "delta_spend_lp", "delta_spend_pct_lp"]]
            .round(4).to_dict(orient="records"),
            "heuristic_projected_revenue": round(heuristic_projected, 2),
            "lp_projected_revenue": round(lp_projected, 2),
            "current_modeled_revenue": round(float((roi["spend"] * roi["roas"].fillna(0)).sum()), 2),
        }

    return {
        "summary": {
            "total_journeys": int(len(journeys)),
            "conversions": total_conversions,
            "conversion_rate": float(journeys["converted"].mean()),
            "median_path_length": float(journeys["n_touches"].median()),
            "total_revenue": total_revenue,
            "channels": sorted({ch for p in journeys["path"] for ch in p}),
        },
        "credit_matrix": {
            "channels": pivot.index.tolist(),
            "models": pivot.columns.tolist(),
            "values": pivot.round(4).values.tolist(),
        },
        "credit_long": all_credit[["channel", "model", "model_label", "credit_share"]]
        .round(4)
        .to_dict(orient="records"),
        "divergence": divergence,
        "sankey": _sankey_from_journeys(journeys, top_n=sankey_top_n),
        "roi": roi.round(4).to_dict(orient="records"),
        "roi_model_used": roi_model,
        "max_increase_pct": max_increase_pct,
        "unallocated_pool": round(float(unallocated_pool), 2),
        "model_labels": MODEL_LABELS,
        "ml_diagnostics": ml_diagnostics,
        "lp_comparison": lp_comparison,
    }
