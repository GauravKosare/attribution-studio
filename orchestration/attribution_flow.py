"""Prefect orchestration for the attribution pipeline — turns the ad-hoc
"click Run analysis in the dashboard" workflow into a schedulable,
observable, retryable job, the way a real analytics team would run this
nightly/weekly against a live data export instead of by hand.

Why Prefect, and why this counts as genuinely free: Prefect's orchestration
engine is open source (Apache 2.0) and runs entirely self-hosted — no
Prefect Cloud account, no card, ever, for the local/self-hosted path used
here. `prefect server start` runs a local API + UI backed by SQLite; running
a flow directly (as this file's `__main__` block does) doesn't even need
that — Prefect 3 spins up an ephemeral in-process API automatically for a
single ad-hoc run. Scheduling (§ bottom of this file) is the only piece that
benefits from `prefect server start` running in the background.

Each pipeline stage is a separate `@task` — not because any one of them is
slow enough to need it in isolation, but because that's what makes retries,
per-stage logging, and (if this fed a real ad-platform API instead of a
bundled CSV) per-stage failure isolation possible: a flaky network call to
pull yesterday's spend data can retry without re-running the whole
attribution computation.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generator import generate_dataset
from src.pipeline import run_pipeline
from src.real_dataset_adapter import load_real_dataset

OUTPUT_DIR = ROOT / "data" / "processed"


@task(retries=2, retry_delay_seconds=10, log_prints=True)
def load_data_task(source: str, n_users: int, seed: int = 42):
    """Stage 1: data acquisition. Retries because this is the stage that
    would hit a real, flaky network call in production (an ad-platform API
    pull) — the local CSV/generator path here never actually fails, but the
    retry wrapper is what should be here regardless of today's data source.
    """
    print(f"Loading data: source={source}, n_users={n_users}")
    if source == "real":
        tp, conv, spend = load_real_dataset(str(ROOT / "data" / "raw" / "real_channel_journeys.csv"))
        if n_users < tp["user_id"].nunique():
            users = tp["user_id"].unique()
            import numpy as np

            keep = set(np.random.default_rng(seed).choice(users, size=n_users, replace=False))
            tp = tp[tp["user_id"].isin(keep)]
            conv = conv[conv["user_id"].isin(keep)]
    else:
        tp, conv, spend = generate_dataset(n_users=n_users, seed=seed)
    print(f"Loaded {len(tp)} touchpoints, {len(conv)} conversions")
    return tp, conv, spend


@task(retries=1, log_prints=True)
def run_pipeline_task(tp, conv, spend, roi_model: str, shift_fraction: float, max_increase_pct: float):
    """Stage 2: the actual analysis — journey extraction through every
    attribution model through ROI/reallocation, via the same src/pipeline.py
    the live dashboard calls. One retry: this stage is pure computation (no
    network), so a transient failure here is more likely a real bug than
    something a retry fixes — but a single retry still guards against, e.g.,
    a flaky worker running low on memory during a large run.
    """
    logger = get_run_logger()
    result = run_pipeline(
        tp, conv, spend, roi_model=roi_model,
        shift_fraction=shift_fraction, max_increase_pct=max_increase_pct,
    )
    if "error" in result:
        raise RuntimeError(f"Pipeline returned an error: {result['error']}")
    logger.info(
        f"Pipeline complete: {result['summary']['total_journeys']} journeys, "
        f"{result['summary']['conversions']} conversions, "
        f"divergence={result.get('divergence')}"
    )
    return result


@task(log_prints=True)
def persist_result_task(result: dict, run_label: str):
    """Stage 3: write the result somewhere durable. A real deployment would
    write to a warehouse table or object store; here it's a timestamped JSON
    file under data/processed/, which is exactly what a BI tool or the next
    pipeline run downstream would read from.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"attribution_result_{run_label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote result to {out_path}")
    return str(out_path)


@flow(name="attribution-pipeline", log_prints=True)
def attribution_pipeline_flow(
    source: str = "real",
    n_users: int = 5000,
    roi_model: str = "markov",
    shift_fraction: float = 0.20,
    max_increase_pct: float = 0.50,
    seed: int = 42,
) -> str:
    """The full orchestrated run: load -> compute -> persist."""
    tp, conv, spend = load_data_task(source, n_users, seed)
    result = run_pipeline_task(tp, conv, spend, roi_model, shift_fraction, max_increase_pct)
    run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return persist_result_task(result, run_label)


if __name__ == "__main__":
    # Ad-hoc single run, no Prefect server needed -- Prefect 3 spins up an
    # ephemeral in-process API automatically for this.
    #     python orchestration/attribution_flow.py
    output_path = attribution_pipeline_flow()
    print(f"\nDone. Result written to: {output_path}")

    # To actually SCHEDULE this (e.g. nightly), run in a separate terminal:
    #     prefect server start                    # free, local, self-hosted
    # then, in this file, replace the __main__ block above with:
    #     attribution_pipeline_flow.serve(
    #         name="nightly-attribution",
    #         cron="0 6 * * *",                    # 6am daily
    #     )
    # and run `python orchestration/attribution_flow.py` once to register the
    # schedule -- it then runs automatically as long as the serve() process
    # (or a Prefect worker) stays running. No cloud account, ever, for this path.
