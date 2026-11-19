#!/usr/bin/env python3
"""Step 03 - official labeler -> outcome tables (data/interim/labels/).

Source: labelers/full_labeler.parquet (official Bluesky labeler did:plc:ar7c4by46qjdydhdevvrndac, 2023-04 .. 2026-07-01).
official_labeler.parquet (2026-04-01 .. 2026-07-01) is used only as a consistency check.

Outputs:
  account_label_events.parquet  did, val, ts, neg, exp_ts, seq          (labels on the account itself, uri = did)
  record_label_events.parquet   did, collection, rkey, val, ts, neg     (labels on posts/profiles etc., did parsed from at-uri)
  account_outcomes.parquet      one row per account with any account-level label:
      takedown_active_at_start   already taken down when the window starts (exclude from "new takedown" analyses)
      n_takedowns_before         takedown labels applied before the window (repeat offender history)
      takedown_ts                first !takedown applied inside the window  <- main outcome time
      takedown_reversed_ts       first negation after takedown_ts (appeal won / mistake)
      takedown_active_at_end     state at window end
      suspend_ts, suspend_reversed_ts, n_suspends
      needs_review_ts            first needs-review flag in the window (platform's own triage signal)
      n_needs_review, needs_review_before_takedown
      first_<label>_ts           first application inside the window for reason labels (spam, impersonation, ...)
      reason_labels              non-"!" account labels applied from 30 days before to 1 day after takedown_ts
  label_summary.txt             counts for the log / paper
"""
from __future__ import annotations

import argparse
import time

from bskymod.common import connect, get_logger, load_config, sql_str

REASONS = ["spam", "impersonation", "rude", "intolerant", "threat", "misleading", "scam", "self-harm",
           "sexual", "porn", "nudity", "graphic-media", "!hide", "!warn", "inauthentic", "engagement-farming"]


def col(v: str) -> str:
    return "first_" + v.replace("!", "").replace("-", "_") + "_ts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--start", default="2026-04-01", help="outcome window start (UTC)")
    ap.add_argument("--end", default="2026-07-01", help="outcome window end (UTC), labels available until here")
    ap.add_argument("--out-dir", default="labels", help="folder under data/interim (e.g. labels_hist, labels_post)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw, out = cfg["paths"]["raw"], cfg["paths"]["interim"] / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("03", cfg["paths"]["results"] / "logs" / f"03_{args.out_dir}.log")
    con = connect(cfg)
    full = raw / "labelers" / "full_labeler.parquet"
    official = raw / "labelers" / "official_labeler.parquet"
    S, E = f"TIMESTAMPTZ '{args.start} 00:00:00+00'", f"TIMESTAMPTZ '{args.end} 00:00:00+00'"
    did_ = cfg.get("official_labeler_did", "did:plc:ar7c4by46qjdydhdevvrndac")
    t0 = time.time()

    con.execute(f"""CREATE TEMP TABLE lab AS
        SELECT seq, CAST(ts AS TIMESTAMPTZ) AS ts, val, neg, uri, TRY_CAST(exp AS TIMESTAMPTZ) AS exp_ts
        FROM read_parquet('{sql_str(str(full))}') WHERE src = '{did_}'""")
    n_all = con.execute("SELECT count(*) FROM lab").fetchone()[0]
    log.info(f"loaded {n_all:,} labels from {full.name}")

    # consistency: official file should equal full file inside its window
    if official.exists():
        a, b = con.execute(f"""SELECT
            (SELECT count(*) FROM lab WHERE ts >= {S} AND ts < {E}),
            (SELECT count(*) FROM read_parquet('{sql_str(str(official))}') WHERE ts >= {S} AND ts < {E})""").fetchone()
        log.info(f"consistency: full_labeler rows in window = {a:,}, official_labeler rows in window = {b:,} ({'MATCH' if a == b else 'DIFFERENT - check'})")

    def write(name, sql):
        dst = out / f"{name}.parquet"
        con.execute(f"COPY ({sql}) TO '{sql_str(str(dst))}' (FORMAT parquet, COMPRESSION zstd)")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{sql_str(str(dst))}')").fetchone()[0]
        log.info(f"{name}: {n:,} rows")
        return dst

    ev = write("account_label_events", """
        SELECT uri AS did, val, ts, neg, exp_ts, seq FROM lab WHERE uri LIKE 'did:%' ORDER BY did, ts, seq""")
    write("record_label_events", """
        SELECT regexp_extract(uri, '^at://(did:[^/]+)/', 1) AS did,
               regexp_extract(uri, '^at://did:[^/]+/([^/]+)/', 1) AS collection,
               regexp_extract(uri, '/([^/]+)$', 1) AS rkey, val, ts, neg
        FROM lab WHERE uri LIKE 'at://%'""")

    E_ = f"read_parquet('{sql_str(str(ev))}')"
    reason_cols = ",\n".join(
        f"min(ts) FILTER (val = '{v}' AND NOT neg AND ts >= {S} AND ts < {E}) AS {col(v)}" for v in REASONS)
    con.execute(f"""CREATE TEMP TABLE o AS
        WITH td AS (SELECT * FROM {E_} WHERE val = '!takedown'),
        base AS (
          SELECT did,
            coalesce(NOT arg_max(neg, (ts, seq)) FILTER (ts < {S}), false)                 AS takedown_active_at_start,
            count(*) FILTER (NOT neg AND ts < {S})                                       AS n_takedowns_before,
            min(ts) FILTER (NOT neg AND ts >= {S} AND ts < {E})                           AS takedown_ts,
            coalesce(NOT arg_max(neg, (ts, seq)) FILTER (ts < {E}), false)                 AS takedown_active_at_end
          FROM td GROUP BY did),
        rev AS (
          SELECT b.did, min(t.ts) AS takedown_reversed_ts
          FROM base b JOIN td t ON t.did = b.did AND t.neg AND t.ts > b.takedown_ts AND t.ts < {E}
          GROUP BY b.did),
        other AS (
          SELECT did,
            min(ts) FILTER (val = '!suspend' AND NOT neg AND ts >= {S} AND ts < {E})       AS suspend_ts,
            count(*) FILTER (val = '!suspend' AND NOT neg AND ts >= {S} AND ts < {E})      AS n_suspends,
            min(ts) FILTER (val = 'needs-review' AND NOT neg AND ts >= {S} AND ts < {E})   AS needs_review_ts,
            count(*) FILTER (val = 'needs-review' AND NOT neg AND ts >= {S} AND ts < {E})  AS n_needs_review,
            {reason_cols}
          FROM {E_} GROUP BY did)
        SELECT o.did,
               coalesce(b.takedown_active_at_start, false) AS takedown_active_at_start,
               coalesce(b.n_takedowns_before, 0) AS n_takedowns_before,
               b.takedown_ts, r.takedown_reversed_ts,
               coalesce(b.takedown_active_at_end, false) AS takedown_active_at_end,
               o.* EXCLUDE (did)
        FROM other o LEFT JOIN base b USING (did) LEFT JOIN rev r USING (did)""")

    # suspend reversal + reason labels around the takedown
    con.execute(f"""CREATE TEMP TABLE o2 AS
        SELECT o.*,
          (SELECT min(e.ts) FROM {E_} e WHERE e.did = o.did AND e.val = '!suspend' AND e.neg AND e.ts > o.suspend_ts) AS suspend_reversed_ts,
          o.needs_review_ts IS NOT NULL AND o.takedown_ts IS NOT NULL AND o.needs_review_ts <= o.takedown_ts AS needs_review_before_takedown,
          (SELECT list(DISTINCT e.val ORDER BY e.val) FROM {E_} e
             WHERE e.did = o.did AND NOT e.neg AND e.val NOT LIKE '!%' AND e.val <> 'needs-review'
               AND e.ts BETWEEN o.takedown_ts - INTERVAL 30 DAY AND o.takedown_ts + INTERVAL 1 DAY) AS reason_labels
        FROM o""")
    write("account_outcomes", "SELECT * FROM o2 ORDER BY did")

    # summary for the log and the paper
    q = lambda s: con.execute(s).fetchone()
    lines = [f"window {args.start} .. {args.end} (UTC)"]
    lines.append("accounts with any account-level label: {:,}".format(q("SELECT count(*) FROM o2")[0]))
    lines.append("already taken down at window start: {:,}".format(q("SELECT count(*) FROM o2 WHERE takedown_active_at_start")[0]))
    lines.append("NEW takedowns in window (not active at start): {:,}".format(
        q("SELECT count(*) FROM o2 WHERE takedown_ts IS NOT NULL AND NOT takedown_active_at_start")[0]))
    lines.append("  of which reversed before window end: {:,}".format(
        q("SELECT count(*) FROM o2 WHERE takedown_ts IS NOT NULL AND NOT takedown_active_at_start AND takedown_reversed_ts IS NOT NULL")[0]))
    lines.append("  of which repeat offenders (takedown before window): {:,}".format(
        q("SELECT count(*) FROM o2 WHERE takedown_ts IS NOT NULL AND NOT takedown_active_at_start AND n_takedowns_before > 0")[0]))
    lines.append("  of which had needs-review first: {:,}".format(
        q("SELECT count(*) FROM o2 WHERE takedown_ts IS NOT NULL AND NOT takedown_active_at_start AND needs_review_before_takedown")[0]))
    lines.append("accounts with needs-review in window: {:,}".format(q("SELECT count(*) FROM o2 WHERE needs_review_ts IS NOT NULL")[0]))
    lines.append("  of which taken down later in window: {:,}".format(
        q("SELECT count(*) FROM o2 WHERE needs_review_ts IS NOT NULL AND takedown_ts >= needs_review_ts")[0]))
    lines.append("accounts suspended in window: {:,}".format(q("SELECT count(*) FROM o2 WHERE suspend_ts IS NOT NULL")[0]))
    lines.append("takedowns per month (new): " + str(con.execute(f"""
        SELECT strftime(takedown_ts, '%Y-%m') m, count(*) FROM o2
        WHERE takedown_ts IS NOT NULL AND NOT takedown_active_at_start GROUP BY 1 ORDER BY 1""").fetchall()))
    lines.append("median hours needs-review -> takedown: " + str(q(f"""
        SELECT median(epoch(takedown_ts - needs_review_ts) / 3600) FROM o2 WHERE needs_review_before_takedown""")[0]))
    lines.append("top reason labels at takedown: " + str(con.execute("""
        SELECT r, count(*) n FROM (SELECT unnest(reason_labels) r FROM o2 WHERE takedown_ts IS NOT NULL)
        GROUP BY 1 ORDER BY n DESC LIMIT 15""").fetchall()))
    lines.append("takedowns with no reason label: {:,}".format(
        q("SELECT count(*) FROM o2 WHERE takedown_ts IS NOT NULL AND (reason_labels IS NULL OR len(reason_labels) = 0)")[0]))
    (out / "label_summary.txt").write_text("\n".join(lines) + "\n")
    for l in lines:
        log.info(l)
    log.info(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
