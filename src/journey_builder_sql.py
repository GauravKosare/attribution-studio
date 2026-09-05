"""SQL-based journey extraction (DuckDB) — the SQL counterpart to journey_builder.py.

Runs sql/build_journeys.sql against in-memory DuckDB tables registered from the
touchpoints/conversions DataFrames, and returns a journeys DataFrame in the same
shape as journey_builder.build_journeys(). The two implementations are meant to be
cross-checked against each other (see verify_parity() below) — same lookback
window, same repeat-touch collapsing rule, same path-length cap, same output.

Why both exist: journey_builder.py is what the live dashboard uses (it needs to be
fast and callable from Python without a DuckDB dependency in the request path).
This module exists to demonstrate the same logic in SQL — window functions,
LIST aggregation, LAG-based dedup — the skill Document 3 calls for, and as a
correctness check: if pandas and SQL versions disagree, one of them has a bug.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SQL_PATH = Path(__file__).resolve().parents[1] / "sql" / "build_journeys.sql"


def build_journeys_sql(
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    lookback_days: int = 30,
    collapse_repeats_minutes: int = 30,
    max_path_length: int = 20,
) -> pd.DataFrame:
    """Same contract as journey_builder.build_journeys, computed entirely in SQL."""
    tp = touchpoints.copy()
    tp["timestamp"] = pd.to_datetime(tp["timestamp"])
    # same deterministic tie-breaker as journey_builder.py: original input row
    # order, used to resolve touches sharing an identical timestamp identically
    # in both implementations (real data has these — see sql/build_journeys.sql)
    tp["_seq"] = range(len(tp))

    conv = conversions.copy()
    if conv.empty:
        # DuckDB needs a typed empty table to LEFT JOIN against
        conv = pd.DataFrame(
            {"user_id": pd.Series(dtype="object"), "timestamp": pd.Series(dtype="datetime64[ns]"),
             "revenue": pd.Series(dtype="float64")}
        )
    else:
        conv = conv.copy()
        conv["timestamp"] = pd.to_datetime(conv["timestamp"])

    sql_template = SQL_PATH.read_text(encoding="utf-8")
    sql = sql_template.format(
        lookback_days=int(lookback_days),
        collapse_repeats_minutes=int(collapse_repeats_minutes),
        max_path_length=int(max_path_length),
    )

    con = duckdb.connect(database=":memory:")
    con.register("touchpoints", tp)
    con.register("conversions", conv)
    result = con.execute(sql).fetchdf()
    con.close()

    if result.empty:
        return result
    result["days_to_convert"] = result["days_to_convert"].astype(float)
    result["n_touches"] = result["n_touches"].astype(int)
    result["converted"] = result["converted"].astype(bool)
    return result


def verify_parity(
    touchpoints: pd.DataFrame,
    conversions: pd.DataFrame,
    lookback_days: int = 30,
    collapse_repeats_minutes: int = 30,
    max_path_length: int = 20,
) -> dict:
    """Cross-check the SQL and pandas journey builders against each other.

    Returns a small report dict rather than asserting, so it can be surfaced in a
    notebook or CI log without crashing a pipeline run over a one-off mismatch.
    """
    from src.journey_builder import build_journeys as build_journeys_pandas

    j_py = build_journeys_pandas(
        touchpoints, conversions, lookback_days, collapse_repeats_minutes, max_path_length
    ).set_index("user_id").sort_index()
    j_sql = build_journeys_sql(
        touchpoints, conversions, lookback_days, collapse_repeats_minutes, max_path_length
    ).set_index("user_id").sort_index()

    common = j_py.index.intersection(j_sql.index)
    path_mismatches = int((j_py.loc[common, "path_str"] != j_sql.loc[common, "path_str"]).sum())
    converted_mismatches = int((j_py.loc[common, "converted"] != j_sql.loc[common, "converted"]).sum())

    return {
        "users_pandas": len(j_py),
        "users_sql": len(j_sql),
        "users_in_common": len(common),
        "path_str_mismatches": path_mismatches,
        "converted_mismatches": converted_mismatches,
        "match": path_mismatches == 0 and converted_mismatches == 0 and len(j_py) == len(j_sql),
    }


if __name__ == "__main__":
    from src.data_generator import generate_dataset

    tp, conv, _ = generate_dataset(n_users=2000, seed=1)
    report = verify_parity(tp, conv)
    print(report)
