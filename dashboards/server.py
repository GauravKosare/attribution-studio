"""Backend for the dynamic attribution dashboard.

Serves the polished HTML/JS frontend and a small JSON API that (re)computes the
full attribution pipeline on demand from any of three input modes: the bundled
real-world dataset, a synthetic generator with live parameters, or user-uploaded
CSVs. Nothing about the analysis is precomputed/static — every request re-runs
journey building + all attribution models + ROI on whatever data was requested.

Run with:
    python dashboards/server.py
Then open http://localhost:5050
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generator import DEFAULT_CHANNELS, generate_dataset
from src.pipeline import run_pipeline
from src.real_dataset_adapter import DATASET_DESCRIPTION, load_real_dataset

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))

REAL_CSV = ROOT / "data" / "raw" / "real_channel_journeys.csv"
_real_cache: dict | None = None


def _get_real_data():
    global _real_cache
    if _real_cache is None:
        tp, conv, spend = load_real_dataset(str(REAL_CSV))
        _real_cache = {"touchpoints": tp, "conversions": conv, "spend": spend}
    return _real_cache["touchpoints"], _real_cache["conversions"], _real_cache["spend"]


def _subsample_by_user(tp: pd.DataFrame, conv: pd.DataFrame, n_users: int, seed: int = 7):
    users = tp["user_id"].unique()
    if n_users >= len(users):
        return tp, conv
    rng = pd.Series(users).sample(n=n_users, random_state=seed)
    keep = set(rng.tolist())
    return tp[tp["user_id"].isin(keep)], conv[conv["user_id"].isin(keep)]


@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.get("/api/meta")
def meta():
    return jsonify(
        {
            "channels_default": list(DEFAULT_CHANNELS.keys()),
            "real_dataset_description": DATASET_DESCRIPTION,
        }
    )


@app.post("/api/run")
def run():
    body = request.get_json(force=True)
    source = body.get("source", "synthetic")
    lookback_days = int(body.get("lookback_days", 30))
    collapse_minutes = int(body.get("collapse_minutes", 30))
    roi_model = body.get("roi_model", "markov")
    shift_fraction = float(body.get("shift_fraction", 0.20))
    max_increase_raw = body.get("max_increase_pct", 0.50)
    max_increase_pct = None if max_increase_raw in (None, "none", "None") else float(max_increase_raw)
    sankey_top_n = int(body.get("sankey_top_n", 20))
    include_ml = bool(body.get("include_ml", False))
    ml_method = body.get("ml_method", "xgboost")
    include_lp = bool(body.get("include_lp", False))

    if source == "real":
        tp, conv, spend = _get_real_data()
        n_users = int(body.get("n_users", 5000))
        tp, conv = _subsample_by_user(tp, conv, n_users)
    else:
        params = body.get("synthetic", {})
        channels = params.get("channels") or list(DEFAULT_CHANNELS.keys())
        channels_cfg = {c: DEFAULT_CHANNELS[c] for c in channels if c in DEFAULT_CHANNELS}
        tp, conv, spend = generate_dataset(
            n_users=int(params.get("n_users", 5000)),
            channels=channels_cfg or DEFAULT_CHANNELS,
            days=int(params.get("days", 90)),
            base_conversion_rate=float(params.get("base_conversion_rate", 0.06)),
            seed=int(params.get("seed", 42)),
        )

    result = run_pipeline(
        tp, conv, spend,
        lookback_days=lookback_days, collapse_minutes=collapse_minutes,
        roi_model=roi_model, shift_fraction=shift_fraction,
        max_increase_pct=max_increase_pct, sankey_top_n=sankey_top_n,
        include_ml=include_ml, ml_method=ml_method, include_lp=include_lp,
    )
    result["source"] = source
    return jsonify(result)


@app.post("/api/upload")
def upload():
    tp_file = request.files.get("touchpoints")
    conv_file = request.files.get("conversions")
    spend_file = request.files.get("channel_spend")
    if not (tp_file and conv_file and spend_file):
        return jsonify({"error": "Upload all three CSVs: touchpoints, conversions, channel_spend."}), 400

    tp = pd.read_csv(tp_file)
    conv = pd.read_csv(conv_file)
    spend = pd.read_csv(spend_file)

    lookback_days = int(request.form.get("lookback_days", 30))
    collapse_minutes = int(request.form.get("collapse_minutes", 30))
    roi_model = request.form.get("roi_model", "markov")
    shift_fraction = float(request.form.get("shift_fraction", 0.20))
    max_increase_raw = request.form.get("max_increase_pct", "0.50")
    max_increase_pct = None if max_increase_raw in (None, "none", "None", "") else float(max_increase_raw)
    include_ml = request.form.get("include_ml", "false").lower() == "true"
    include_lp = request.form.get("include_lp", "false").lower() == "true"

    result = run_pipeline(
        tp, conv, spend,
        lookback_days=lookback_days, collapse_minutes=collapse_minutes,
        roi_model=roi_model, shift_fraction=shift_fraction, max_increase_pct=max_increase_pct,
        include_ml=include_ml, include_lp=include_lp,
    )
    result["source"] = "upload"
    return jsonify(result)


if __name__ == "__main__":
    import os

    # Local dev: `python dashboards/server.py` — debug/reload on, default port 5050.
    # Production (Render, etc.) runs this module via gunicorn instead, which
    # never executes this block; PORT is respected here only for parity if
    # something does invoke this entrypoint directly in a hosted environment.
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
