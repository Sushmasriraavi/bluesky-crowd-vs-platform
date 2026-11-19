#!/usr/bin/env python3
"""Step 07 - new accounts: no own records in the snapshot, first seen in the firehose (snapshot + firehose join).

Most takedowns hit accounts created after the snapshot crawl, so they can only be studied from the firehose.
For each new account:
  first_seen = first firehose operation (on/after --fh-start)
  index time = first_seen + 24 h  (features use only the first 24 h; outcome counted after the index)
  y_takedown = first official !takedown after the index, until --end (>= 30 days follow-up because index < fh end)
  instant takedown = taken down within the first 24 h (reported separately, not predictable by anyone)

Features (first 24 h only)
  new_act_*    posts, reply/quote/link/mention/tag/media ratios, duplicate ratio, text length, languages,
               reply targets, follows/blocks/likes/reposts made, deletes, profile set up (avatar / name)
  new_rcv_*    likes, reposts and replies received
  new_crowd_*  distinct blockers and mod-list additions received in the first 24 h (NOT model inputs for RQ1)

Re-run with --fh-start 2026-01-01 once the Jan-Mar firehose is converted (step 01) and step 04 is re-run in merged mode.

Outputs
  data/processed/accounts_new.parquet
  results/S2/T31_new_accounts.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--fh-start", default="2026-04-01", help="first firehose day")
    ap.add_argument("--fh-end", default="2026-06-01", help="end of firehose coverage (exclusive)")
    ap.add_argument("--end", default="2026-07-01", help="end of label outcome window")
    ap.add_argument("--hours", type=int, default=24, help="observation window after first seen")
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    P.mkdir(parents=True, exist_ok=True)
    log = get_logger("07", cfg["paths"]["results"] / "logs" / "07_new_accounts.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    FH = I / "firehose"
    S = f"TIMESTAMPTZ '{args.fh_start} 00:00:00+00'"
    E = f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    END = f"TIMESTAMPTZ '{args.end} 00:00:00+00'"
    H = f"INTERVAL {args.hours} HOUR"

    # ---- cohort: first seen in the firehose window, no own records in the snapshot
    con.execute(f"""CREATE TEMP TABLE cohort AS
        WITH fs AS (SELECT did, min(t) AS first_seen FROM {rp(FH / 'ops' / '*' / '*.parquet')}
                    WHERE t >= {S} AND t < {E} GROUP BY did)
        SELECT fs.did, fs.first_seen, fs.first_seen + {H} AS index_t
        FROM fs
        WHERE NOT EXISTS (SELECT 1 FROM {rp(I / 'snapshot' / 'repo_rev.parquet')} r WHERE r.did = fs.did)
          AND fs.first_seen + {H} < {E}""")
    n = con.execute("SELECT count(*) FROM cohort").fetchone()[0]
    log.info(f"new accounts (no snapshot records, first seen {args.fh_start}..{args.fh_end}): {n:,}")

    # ---- labels: takedown state before first seen, instant takedowns, outcome after index
    ev = rp(I / "labels_hist" / "account_label_events.parquet")
    con.execute(f"""CREATE TEMP TABLE lab AS
        SELECT c.did,
          coalesce(NOT arg_max(e.neg, (e.ts, e.seq)) FILTER (e.val = '!takedown' AND e.ts < c.first_seen), false) AS down_before,
          min(e.ts) FILTER (e.val = '!takedown' AND NOT e.neg AND e.ts >= c.first_seen AND e.ts < c.index_t) AS instant_takedown_ts,
          min(e.ts) FILTER (e.val = '!takedown' AND NOT e.neg AND e.ts >= c.index_t AND e.ts < {END}) AS y_takedown_ts,
          min(e.ts) FILTER (e.val = 'needs-review' AND NOT e.neg AND e.ts >= c.first_seen AND e.ts < c.index_t) AS needs_review_in_window_ts,
          min(e.ts) FILTER (e.val = 'needs-review' AND NOT e.neg AND e.ts >= c.index_t AND e.ts < {END}) AS y_needs_review_ts,
          min(e.ts) FILTER (e.val = '!suspend' AND NOT e.neg AND e.ts >= c.index_t AND e.ts < {END}) AS y_suspend_ts
        FROM cohort c JOIN {ev} e ON e.did = c.did GROUP BY c.did""")

    # ---- own activity in the first 24 h
    con.execute(f"""CREATE TEMP TABLE pa AS
        SELECT c.did, count(*) new_act_posts,
          avg((p.reply_parent_uri IS NOT NULL)::INT) new_act_reply_ratio,
          avg((p.quote_uri IS NOT NULL)::INT) new_act_quote_ratio,
          avg((len(p.link_uris) > 0 OR p.external_uri IS NOT NULL)::INT) new_act_link_ratio,
          avg((len(p.mention_dids) > 0)::INT) new_act_mention_ratio,
          avg((coalesce(p.n_tags, 0) > 0)::INT) new_act_tag_ratio,
          avg((coalesce(p.embed_type, '') LIKE '%images%' OR coalesce(p.embed_type, '') LIKE '%video%')::INT) new_act_media_ratio,
          1 - count(DISTINCT hash(p.text)) / count(*) new_act_duplicate_ratio,
          avg(length(coalesce(p.text, ''))) new_act_mean_text_len,
          count(DISTINCT p.lang) new_act_n_langs, mode(p.lang) new_act_top_lang,
          count(DISTINCT regexp_extract(p.reply_parent_uri, 'at://(did:[^/]+)/', 1)) FILTER (p.reply_parent_uri IS NOT NULL) new_act_reply_targets
        FROM cohort c JOIN {rp(FH / 'posts' / '*' / '*.parquet')} p ON p.did = c.did AND p.t >= c.first_seen AND p.t < c.index_t
        GROUP BY c.did""")
    con.execute(f"""CREATE TEMP TABLE oa AS
        SELECT c.did, count(*) new_act_ops,
          count(*) FILTER (o.operation = 'create' AND o.collection = 'app.bsky.graph.follow') new_act_follows_made,
          count(*) FILTER (o.operation = 'create' AND o.collection = 'app.bsky.graph.block') new_act_blocks_made,
          count(*) FILTER (o.operation = 'create' AND o.collection = 'app.bsky.feed.like') new_act_likes_made,
          count(*) FILTER (o.operation = 'create' AND o.collection = 'app.bsky.feed.repost') new_act_reposts_made,
          count(*) FILTER (o.operation = 'delete') new_act_deletes,
          count(*) FILTER (o.collection = 'app.bsky.actor.profile') new_act_profile_edits,
          count(DISTINCT date_trunc('hour', o.t)) new_act_active_hours
        FROM cohort c JOIN {rp(FH / 'ops' / '*' / '*.parquet')} o ON o.did = c.did AND o.t >= c.first_seen AND o.t < c.index_t
        GROUP BY c.did""")
    con.execute(f"""CREATE TEMP TABLE pr AS
        SELECT c.did, bool_or(pf.has_avatar) new_act_has_avatar, bool_or(pf.display_name IS NOT NULL) new_act_has_name,
               bool_or(length(coalesce(pf.description, '')) > 0) new_act_has_bio
        FROM cohort c JOIN {rp(FH / 'profiles' / '*' / '*.parquet')} pf ON pf.did = c.did AND pf.t >= c.first_seen AND pf.t < c.index_t
        GROUP BY c.did""")
    log.info("own activity done")

    # ---- received in the first 24 h
    con.execute(f"""CREATE TEMP TABLE rc AS
        SELECT c.did, count(*) FILTER (i.collection = 'app.bsky.feed.like') new_rcv_likes,
                      count(*) FILTER (i.collection = 'app.bsky.feed.repost') new_rcv_reposts
        FROM cohort c JOIN {rp(FH / 'interactions' / '*' / '*.parquet')} i
          ON i.subject_did = c.did AND i.t >= c.first_seen AND i.t < c.index_t AND i.did <> c.did
        GROUP BY c.did""")
    con.execute(f"""CREATE TEMP TABLE rr AS
        SELECT c.did, count(*) new_rcv_replies, count(DISTINCT p.did) new_rcv_repliers
        FROM (SELECT did, t, regexp_extract(reply_parent_uri, 'at://(did:[^/]+)/', 1) AS parent_did
              FROM {rp(FH / 'posts' / '*' / '*.parquet')} WHERE reply_parent_uri IS NOT NULL) p
        JOIN cohort c ON p.parent_did = c.did AND p.t >= c.first_seen AND p.t < c.index_t AND p.did <> c.did
        GROUP BY c.did""")
    con.execute(f"""CREATE TEMP TABLE cr AS
        WITH b AS (SELECT c.did, count(DISTINCT e.blocker_did) n_blk FROM cohort c
                   JOIN {rp(I / 'crowd' / 'block_edges.parquet')} e
                     ON e.subject_did = c.did AND e.t_created >= c.first_seen AND e.t_created < c.index_t
                   GROUP BY c.did),
             l AS (SELECT c.did, count(DISTINCT e.list_uri) n_lst FROM cohort c
                   JOIN {rp(I / 'crowd' / 'listitem_edges.parquet')} e
                     ON e.subject_did = c.did AND e.t_added >= c.first_seen AND e.t_added < c.index_t
                   GROUP BY c.did)
        SELECT coalesce(b.did, l.did) did, coalesce(n_blk, 0) new_crowd_blockers_24h, coalesce(n_lst, 0) new_crowd_lists_24h
        FROM b FULL JOIN l USING (did)""")
    log.info("received + crowd done")

    out = P / "accounts_new.parquet"
    con.execute(f"""COPY (
        SELECT c.did, c.first_seen, c.index_t,
               pa.* EXCLUDE (did), oa.* EXCLUDE (did), pr.* EXCLUDE (did),
               coalesce(rc.new_rcv_likes, 0) new_rcv_likes, coalesce(rc.new_rcv_reposts, 0) new_rcv_reposts,
               coalesce(rr.new_rcv_replies, 0) new_rcv_replies, coalesce(rr.new_rcv_repliers, 0) new_rcv_repliers,
               coalesce(cr.new_crowd_blockers_24h, 0) new_crowd_blockers_24h, coalesce(cr.new_crowd_lists_24h, 0) new_crowd_lists_24h,
               l.instant_takedown_ts, l.instant_takedown_ts IS NOT NULL AS instant_takedown,
               l.needs_review_in_window_ts,
               l.y_takedown_ts, l.y_takedown_ts IS NOT NULL AS y_takedown, l.y_needs_review_ts, l.y_suspend_ts
        FROM cohort c
        LEFT JOIN lab l USING (did) LEFT JOIN pa USING (did) LEFT JOIN oa USING (did) LEFT JOIN pr USING (did)
        LEFT JOIN rc USING (did) LEFT JOIN rr USING (did) LEFT JOIN cr USING (did)
        WHERE NOT coalesce(l.down_before, false)
    ) TO '{sql_str(str(out))}' (FORMAT parquet, COMPRESSION zstd)""")
    log.info(f"written -> {out}")

    t = con.execute(f"""SELECT
        count(*) AS new_accounts,
        count(*) FILTER (instant_takedown) AS instant_takedowns_first_24h,
        count(*) FILTER (NOT instant_takedown AND y_takedown) AS taken_down_after_24h,
        count(*) FILTER (NOT instant_takedown AND coalesce(new_act_posts, 0) > 0) AS posted_in_first_24h,
        count(*) FILTER (NOT instant_takedown AND new_crowd_blockers_24h > 0) AS blocked_in_first_24h,
        count(*) FILTER (NOT instant_takedown AND new_crowd_blockers_24h > 0 AND y_takedown) AS blocked_24h_and_taken_down,
        count(*) FILTER (NOT instant_takedown AND needs_review_in_window_ts IS NOT NULL) AS needs_review_in_first_24h
      FROM {rp(out)}""").df().T.reset_index()
    t.columns = ["measure", "accounts"]
    viz.table(t, cfg["paths"]["results"] / "S2", "T31_new_accounts")
    for r in t.itertuples():
        log.info(f"{r.measure}: {r.accounts:,}")


if __name__ == "__main__":
    main()
