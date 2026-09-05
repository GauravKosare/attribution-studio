"""ML-based attribution: logistic regression + XGBoost, explained with SHAP —
docs/04_attribution_methodology.md section D.

Feature-engineers each journey into a fixed-width vector per channel (presence,
touch count, recency-weighted presence), trains a classifier to predict
`converted`, then aggregates SHAP values by channel into a credit-share table with
the same shape as every other model in src/attribution/ — so it drops straight
into src/pipeline.py alongside rule-based, Markov, and Shapley.

Why this belongs next to the exact Shapley implementation (shapley.py): SHAP *is*
an approximation of Shapley values applied to a model's output, so this module is
the "apply the same game-theoretic idea to a predictive model" version of the same
concept — worth noting explicitly, it's a strong point in an interview.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _featurize(journeys: pd.DataFrame, channels: list[str], half_life_days: float = 7.0) -> pd.DataFrame:
    """One row per journey: presence / count / recency-weighted-presence per channel.

    Vectorized via explode + groupby (no per-journey Python loop) so it scales to
    a real-dataset-sized sample without becoming the pipeline's bottleneck.
    """
    j = journeys[["path", "timestamps"]].copy()
    j["journey_end"] = j["timestamps"].apply(lambda t: t[-1] if t else pd.NaT)

    exploded = j.explode(["path", "timestamps"], ignore_index=False).dropna(subset=["path"])
    exploded = exploded.rename(columns={"path": "channel", "timestamps": "touch_time"})
    days_before_end = (exploded["journey_end"] - exploded["touch_time"]).dt.total_seconds() / 86400.0
    exploded["recency_weight"] = 0.5 ** (days_before_end / half_life_days)

    agg = exploded.groupby([exploded.index, "channel"]).agg(
        count=("channel", "size"), recency=("recency_weight", "sum")
    )
    counts = agg["count"].unstack(fill_value=0).reindex(columns=channels, fill_value=0)
    recency = agg["recency"].unstack(fill_value=0.0).reindex(columns=channels, fill_value=0.0)

    X = pd.DataFrame(index=journeys.index)
    for c in channels:
        cnt = counts[c].reindex(journeys.index, fill_value=0) if c in counts.columns else pd.Series(0, index=journeys.index)
        rec = recency[c].reindex(journeys.index, fill_value=0.0) if c in recency.columns else pd.Series(0.0, index=journeys.index)
        X[f"has_{c}"] = (cnt > 0).astype(int)
        X[f"count_{c}"] = cnt.astype(int)
        X[f"recency_{c}"] = rec.astype(float)

    y = journeys["converted"].astype(int)
    return X, y


def run(
    journeys: pd.DataFrame,
    method: str = "xgboost",
    min_conversions: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Returns (credit_df, diagnostics). credit_df: [channel, model, credit_share].

    `method`: "xgboost" (default, captures channel-interaction effects) or
    "logistic" (baseline, faster, more stable on small samples).
    diagnostics: {"auc": ..., "n_train": ..., "method": ..., "warning": str|None}
    so the caller (dashboard) can show a trust signal alongside the numbers —
    a model with poor AUC shouldn't be presented with the same confidence as one
    that discriminates well.
    """
    channels = sorted({ch for p in journeys["path"] for ch in p})
    model_name = "xgboost_shap" if method == "xgboost" else "logistic_shap"

    if not channels or journeys["converted"].sum() < min_conversions:
        return (
            pd.DataFrame(columns=["channel", "model", "credit_share"]),
            {"auc": None, "n_train": len(journeys), "method": method,
             "warning": f"Too few conversions (<{min_conversions}) to fit a reliable model."},
        )

    X, y = _featurize(journeys, channels)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    import shap

    if method == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)

    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else None
    except Exception:
        auc = None

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_by_feature = pd.Series(mean_abs_shap, index=X.columns)

    channel_importance = {}
    for c in channels:
        # sum |SHAP| across this channel's 3 features (has/count/recency) — a
        # simple, defensible way to collapse feature-level SHAP to channel-level
        channel_importance[c] = float(
            shap_by_feature.get(f"has_{c}", 0)
            + shap_by_feature.get(f"count_{c}", 0)
            + shap_by_feature.get(f"recency_{c}", 0)
        )

    total = sum(channel_importance.values())
    credit = {c: (v / total if total > 0 else 0.0) for c, v in channel_importance.items()}

    credit_df = pd.DataFrame(
        [{"channel": ch, "model": model_name, "credit_share": s} for ch, s in credit.items()]
    )
    diagnostics = {
        "auc": auc, "n_train": len(X_train), "n_test": len(X_test),
        "method": method, "warning": None if (auc is None or auc >= 0.55) else
        f"AUC {auc:.2f} is close to random (0.5) — treat this model's attribution with low confidence.",
    }
    return credit_df, diagnostics
