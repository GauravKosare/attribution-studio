-- Customer-journey extraction, in SQL (DuckDB dialect — window functions + LIST
-- aggregation work the same way in Postgres/Snowflake/BigQuery with only minor
-- syntax differences noted inline).
--
-- Mirrors src/journey_builder.py exactly (same lookback window, same repeat-touch
-- collapsing rule, same path-length cap) so the two are cross-checkable against
-- each other — see src/journey_builder_sql.py for the runner + parity test.
--
-- Params: {lookback_days}, {collapse_repeats_minutes}, {max_path_length} are
-- plain Python str.format() placeholders, substituted by src/journey_builder_sql.py
-- before this file is executed (defaults match docs/02_data_dictionary.md: 30, 30, 20).
-- To run this file as-is in any SQL client instead, replace the three placeholders
-- with literal numbers first.
--
-- Expects two tables already registered in the DuckDB connection:
--   touchpoints(user_id, touchpoint_id, timestamp, channel, campaign_id, ..., _seq)
--   conversions(user_id, conversion_id, timestamp, revenue, converted)
--
-- `_seq` (added by journey_builder_sql.py) is the touchpoints table's original row
-- order, used as a deterministic tie-breaker everywhere two touches for the same
-- user share an identical timestamp — real ad-event data has these. Without it,
-- SQL's ORDER BY and pandas' sort would each pick an arbitrary (and different)
-- order among tied rows; with it, both implementations agree exactly, which is
-- what src/journey_builder_sql.py::verify_parity() checks.

-- Step 1: anchor each touchpoint to "the moment its journey ends" — the user's
-- conversion time if they converted, else their own last touch (so non-converters
-- still get a full window). This is a window function over touchpoints joined to
-- conversions, not a per-row Python loop.
CREATE OR REPLACE TEMP TABLE touchpoints_anchored AS
SELECT
    t.*,
    COALESCE(c.timestamp, MAX(t.timestamp) OVER (PARTITION BY t.user_id)) AS anchor_time,
    (c.user_id IS NOT NULL) AS converted,
    COALESCE(c.revenue, 0.0) AS conversion_value
FROM touchpoints t
LEFT JOIN (
    -- de-duplicate to one conversion per user (first conversion in period), same
    -- as journey_builder.py's drop_duplicates(subset="user_id", keep="first")
    SELECT user_id, timestamp, revenue,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS rn
    FROM conversions
) c ON c.user_id = t.user_id AND c.rn = 1;

-- Step 2: keep only touches inside the lookback window ending at the anchor time.
-- A user whose window drops every touch (edge case: all touches older than the
-- window) falls back to their single most recent touch, via UNION ALL + ROW_NUMBER.
CREATE OR REPLACE TEMP TABLE touchpoints_windowed AS
WITH in_window AS (
    SELECT *
    FROM touchpoints_anchored
    WHERE timestamp >= anchor_time - INTERVAL ({lookback_days}) DAY
      AND timestamp <= anchor_time
),
fallback AS (
    SELECT *
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp DESC, _seq DESC) AS rn
        FROM touchpoints_anchored
        WHERE user_id NOT IN (SELECT DISTINCT user_id FROM in_window)
    )
    WHERE rn = 1
)
SELECT * FROM in_window
UNION ALL
SELECT * EXCLUDE (rn) FROM fallback;

-- Step 3: collapse consecutive same-channel touches within N minutes of the
-- previous raw touch — the SQL equivalent of journey_builder.py's shift()-based
-- dedup. LAG() over (user_id ORDER BY timestamp) gives the previous row; a row
-- starts a "new run" if the channel changed or the gap exceeds the threshold.
CREATE OR REPLACE TEMP TABLE touchpoints_deduped AS
WITH lagged AS (
    SELECT
        *,
        LAG(channel)   OVER (PARTITION BY user_id ORDER BY timestamp, _seq) AS prev_channel,
        LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp, _seq) AS prev_time
    FROM touchpoints_windowed
)
SELECT * EXCLUDE (prev_channel, prev_time)
FROM lagged
WHERE prev_time IS NULL
   OR channel <> prev_channel
   -- date_diff('minute', ...) truncates to whole minutes; use seconds/60.0 for a
   -- continuous gap matching journey_builder.py's total_seconds()/60.0 exactly
   -- (a 30:01 gap must count as > 30 minutes, not collapse to 30)
   OR date_diff('second', prev_time, timestamp) / 60.0 > {collapse_repeats_minutes};

-- Step 4: cap path length to the most recent N touches per user (reverse rank
-- filter — same trick used in the vectorized pandas version, no per-user loop).
CREATE OR REPLACE TEMP TABLE touchpoints_capped AS
SELECT * EXCLUDE (rev_rank)
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp DESC, _seq DESC) AS rev_rank
    FROM touchpoints_deduped
)
WHERE rev_rank <= {max_path_length};

-- Step 5: aggregate into one row per user — the final journeys table, same shape
-- as journey_builder.build_journeys()'s output.
SELECT
    user_id,
    LIST(channel ORDER BY timestamp, _seq)                         AS path,
    ARRAY_TO_STRING(LIST(channel ORDER BY timestamp, _seq), ' > ')  AS path_str,
    LIST(timestamp ORDER BY timestamp, _seq)                        AS timestamps,
    COUNT(*)                                                  AS n_touches,
    BOOL_OR(converted)                                        AS converted,
    MAX(conversion_value)                                     AS conversion_value,
    MIN(timestamp)                                            AS journey_start,
    MAX(timestamp)                                            AS journey_end,
    date_diff('second', MIN(timestamp), MAX(timestamp)) / 86400.0 AS days_to_convert
FROM touchpoints_capped
GROUP BY user_id;

-- Portability notes for other warehouses:
--  * Postgres: identical syntax; INTERVAL literal works the same.
--  * Snowflake: use DATEDIFF(minute, prev_time, timestamp) instead of date_diff(...).
--  * BigQuery: use ARRAY_AGG(channel ORDER BY timestamp) instead of LIST(...), and
--    TIMESTAMP_DIFF(timestamp, prev_time, MINUTE) instead of date_diff(...).
