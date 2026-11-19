#!/usr/bin/env python3
"""Step 06 - one row per account as it looked at the snapshot time T0 (2026-04-09 00:00 UTC), plus outcomes after T0.

Everything in the feature columns uses ONLY records created before T0 (no leakage).
Outcomes come from the official labeler after T0 (T0 .. 2026-07-01).

Column groups (prefix tells the group):
  act_   own activity in the 90/30/7 days before T0 (posts, replies, quotes, links, mentions, media, duplicates,
         reply targets, replies to accounts they do not follow, active days, languages)
  net_   followers / following, follows made and followers gained in the last 30 days, blocks made, lists owned
  acc_   account age, days since last commit, PDS, prior labels and prior takedowns (history before T0)
  crowd_ crowd signals before T0 (blockers total / 30d / 7d, first block, mod lists, list reach) - NOT used as
         inputs to the RQ1 models; used for RQ2/RQ3 and as the RQ1 "block burst" label
  y_     outcomes after T0: takedown, needs-review, suspension (+ times)

Outputs
  data/interim/features/posts90.parquet       posts created in the 90 days before T0 (for features + toxicity)
  data/interim/features/text_sample.parquet   up to 20 most recent texts per account (input for Detoxify)
  data/processed/accounts_T0.parquet          the feature/outcome table
  results/S2/T30_population.csv               population sizes and outcome rates
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--t0", default="2026-04-09", help="snapshot time (UTC date)")
    ap.add_argument("--end", default="2026-07-01", help="end of outcome window")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    F = I / "features"
    F.mkdir(parents=True, exist_ok=True)
    P.mkdir(parents=True, exist_ok=True)
    log = get_logger("06", cfg["paths"]["results"] / "logs" / "06_features.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    T0 = f"TIMESTAMPTZ '{args.t0} 00:00:00+00'"
    END = f"TIMESTAMPTZ '{args.end} 00:00:00+00'"
    D = lambda n: f"({T0} - INTERVAL {n} DAY)"
    S = I / "snapshot"

    def build(path, sql, what):
        t0 = time.time()
        if path.exists() and not args.force:
            log.info(f"{what}: already built, skipping")
            return
        tmp = path.with_suffix(".tmp")
        con.execute(f"COPY ({sql}) TO '{sql_str(str(tmp))}' (FORMAT parquet, COMPRESSION zstd)")
        tmp.rename(path)
        n = con.execute(f"SELECT count(*) FROM {rp(path)}").fetchone()[0]
        log.info(f"{what}: {n:,} rows ({time.time()-t0:.0f}s)")

    # ---------- 1. posts of the last 90 days before T0 (one scan of the 2.4B-post snapshot)
    posts90 = F / "posts90.parquet"
    build(posts90, f"""
        WITH p AS (
          SELECT repo_id, rkey, content,
                 TRY_CAST(json_extract_string(content, '$.createdAt') AS TIMESTAMPTZ) AS created_at
          FROM {rp(I / 'snapshot_fixed' / 'post.parquet')})
        SELECT r.did, p.rkey, p.created_at,
               json_extract_string(p.content, '$.text') AS text,
               json_extract_string(p.content, '$.langs[0]') AS lang,
               regexp_extract(json_extract_string(p.content, '$.reply.parent.uri'), 'at://(did:[^/]+)/', 1) AS parent_did,
               json_extract_string(p.content, '$.reply.parent.uri') IS NOT NULL AS is_reply,
               p.content LIKE '%app.bsky.embed.record%' AS is_quote,
               p.content LIKE '%richtext.facet#link%' OR p.content LIKE '%app.bsky.embed.external%' AS has_link,
               p.content LIKE '%richtext.facet#mention%' AS has_mention,
               p.content LIKE '%richtext.facet#tag%' AS has_tag,
               p.content LIKE '%app.bsky.embed.images%' OR p.content LIKE '%app.bsky.embed.video%' AS has_media
        FROM p JOIN {rp(S / 'repos.parquet')} r USING (repo_id)
        WHERE p.created_at >= {D(90)} AND p.created_at < {T0}""", "posts90")

    # ---------- 2. text sample for toxicity (most recent 20 non-empty texts per account)
    build(F / "text_sample.parquet", f"""
        SELECT did, rkey, created_at, text FROM {rp(posts90)}
        WHERE length(coalesce(text, '')) >= 3
        QUALIFY row_number() OVER (PARTITION BY did ORDER BY created_at DESC) <= 20""", "text_sample")

    # ---------- 3. activity aggregates
    con.execute(f"""CREATE TEMP TABLE act AS
        SELECT did,
          count(*) act_posts_90d,
          count(*) FILTER (created_at >= {D(30)}) act_posts_30d,
          count(*) FILTER (created_at >= {D(7)}) act_posts_7d,
          avg(is_reply::INT) act_reply_ratio, avg(is_quote::INT) act_quote_ratio,
          avg(has_link::INT) act_link_ratio, avg(has_mention::INT) act_mention_ratio,
          avg(has_tag::INT) act_tag_ratio, avg(has_media::INT) act_media_ratio,
          1 - count(DISTINCT hash(text)) / count(*) act_duplicate_ratio,
          avg(length(coalesce(text, ''))) act_mean_text_len,
          count(DISTINCT CAST(created_at AS DATE)) act_active_days_90d,
          count(*) / greatest(count(DISTINCT CAST(created_at AS DATE)), 1) act_posts_per_active_day,
          count(DISTINCT lang) act_n_langs, mode(lang) act_top_lang,
          count(DISTINCT parent_did) FILTER (is_reply) act_reply_targets
        FROM {rp(posts90)} GROUP BY did""")
    # how concentrated replies are on the top target (harassment pattern) and replies to non-followed accounts
    con.execute(f"""CREATE TEMP TABLE rep AS
        WITH rc AS (SELECT did, parent_did, count(*) n FROM {rp(posts90)}
                    WHERE is_reply AND parent_did IS NOT NULL AND parent_did <> did GROUP BY ALL),
             fol AS (SELECT follower_did AS did, subject_did AS parent_did FROM {rp(S / 'follows.parquet')}
                     WHERE created_at < {T0} OR created_at IS NULL)
        SELECT rc.did,
               max(rc.n) / sum(rc.n) AS act_reply_top_target_share,
               sum(rc.n) FILTER (fol.did IS NULL) / sum(rc.n) AS act_reply_to_nonfollowed_ratio
        FROM rc LEFT JOIN fol ON fol.did = rc.did AND fol.parent_did = rc.parent_did
        GROUP BY rc.did""")
    log.info("activity aggregates done")

    # ---------- 4. network + own moderation actions
    con.execute(f"""CREATE TEMP TABLE net AS
        WITH fo AS (SELECT follower_did did, count(*) net_following,
                           count(*) FILTER (created_at >= {D(30)} AND created_at < {T0}) net_follows_made_30d,
                           min(created_at) FILTER (created_at > TIMESTAMPTZ '2023-01-01 00:00:00+00') first_follow_at
                    FROM {rp(S / 'follows.parquet')} GROUP BY 1),
             fe AS (SELECT subject_did did, count(*) net_followers,
                           count(*) FILTER (created_at >= {D(30)} AND created_at < {T0}) net_followers_gained_30d
                    FROM {rp(S / 'follows.parquet')} GROUP BY 1),
             bm AS (SELECT blocker_did did, count(*) net_blocks_made,
                           count(*) FILTER (created_at >= {D(30)} AND created_at < {T0}) net_blocks_made_30d,
                           min(created_at) FILTER (created_at > TIMESTAMPTZ '2023-01-01 00:00:00+00') first_block_made_at
                    FROM {rp(S / 'blocks.parquet')} GROUP BY 1),
             lo AS (SELECT owner_did did, count(*) FILTER (purpose LIKE '%modlist') net_modlists_owned,
                           count(*) FILTER (purpose LIKE '%curatelist') net_curatelists_owned
                    FROM {rp(S / 'lists.parquet')} GROUP BY 1)
        SELECT coalesce(fo.did, fe.did, bm.did, lo.did) did, coalesce(net_following, 0) net_following,
               coalesce(net_followers, 0) net_followers, coalesce(net_follows_made_30d, 0) net_follows_made_30d,
               coalesce(net_followers_gained_30d, 0) net_followers_gained_30d,
               coalesce(net_blocks_made, 0) net_blocks_made, coalesce(net_blocks_made_30d, 0) net_blocks_made_30d,
               coalesce(net_modlists_owned, 0) net_modlists_owned, coalesce(net_curatelists_owned, 0) net_curatelists_owned,
               first_follow_at, first_block_made_at
        FROM fo FULL JOIN fe USING (did) FULL JOIN bm USING (did) FULL JOIN lo USING (did)""")
    log.info("network aggregates done")

    # ---------- 5. labels before T0 and outcomes after T0
    ev = rp(I / "labels_hist" / "account_label_events.parquet")
    rl = rp(I / "labels_hist" / "record_label_events.parquet")
    con.execute(f"""CREATE TEMP TABLE lab AS
        SELECT did,
          coalesce(NOT arg_max(neg, (ts, seq)) FILTER (val = '!takedown' AND ts < {T0}), false) AS takedown_active_at_t0,
          count(*) FILTER (val = '!takedown' AND NOT neg AND ts < {T0}) AS acc_prior_takedowns,
          count(*) FILTER (val = 'needs-review' AND NOT neg AND ts < {T0}) AS acc_prior_needs_review,
          count(*) FILTER (val NOT LIKE '!%' AND val <> 'needs-review' AND NOT neg AND ts < {T0}) AS acc_prior_account_labels,
          min(ts) FILTER (val = '!takedown' AND NOT neg AND ts >= {T0} AND ts < {END}) AS y_takedown_ts,
          min(ts) FILTER (val = 'needs-review' AND NOT neg AND ts >= {T0} AND ts < {END}) AS y_needs_review_ts,
          min(ts) FILTER (val = '!suspend' AND NOT neg AND ts >= {T0} AND ts < {END}) AS y_suspend_ts
        FROM {ev} GROUP BY did""")
    con.execute(f"""CREATE TEMP TABLE rlab AS
        SELECT did, count(*) FILTER (NOT neg AND val NOT LIKE '!%' AND ts >= {D(90)} AND ts < {T0}) AS acc_post_labels_90d,
                    count(*) FILTER (NOT neg AND val LIKE '!%' AND ts < {T0}) AS acc_post_takedowns_before
        FROM {rl} GROUP BY did""")

    # ---------- 6. crowd signals before T0 (from snapshot block / list records)
    be = rp(I / "crowd_snapshot" / "block_edges.parquet")
    con.execute(f"""CREATE TEMP TABLE crowd AS
        WITH pr AS (SELECT subject_did did, blocker_did, min(t_created) t FROM {be}
                    WHERE subject_did <> blocker_did AND (t_created < {T0} OR t_created IS NULL) GROUP BY ALL)
        SELECT did, count(*) crowd_blockers, count(*) FILTER (t >= {D(30)}) crowd_blockers_30d,
               count(*) FILTER (t >= {D(7)}) crowd_blockers_7d, min(t) crowd_first_block_at
        FROM pr GROUP BY did""")
    sig = rp(I / "crowd_snapshot" / "subject_signals.parquet")

    # ---------- 7. assemble
    out = P / "accounts_T0.parquet"
    con.execute(f"""COPY (
        WITH base AS (SELECT r.did, r.pds AS acc_pds, rr.last_commit_ts FROM {rp(S / 'repos.parquet')} r
                      LEFT JOIN {rp(S / 'repo_rev.parquet')} rr USING (did))
        SELECT b.did,
               -- account
               least(ph.first_post_at, n.first_follow_at, n.first_block_made_at) AS acc_first_seen,
               date_diff('day', least(ph.first_post_at, n.first_follow_at, n.first_block_made_at), {T0}) AS acc_age_days,
               date_diff('day', b.last_commit_ts, {T0}) AS acc_days_since_last_commit,
               b.acc_pds, coalesce(ph.n_posts, 0) AS acc_posts_alltime, coalesce(ph.n_replies, 0) AS acc_replies_alltime,
               coalesce(l.acc_prior_takedowns, 0) acc_prior_takedowns, coalesce(l.acc_prior_needs_review, 0) acc_prior_needs_review,
               coalesce(l.acc_prior_account_labels, 0) acc_prior_account_labels,
               coalesce(rl.acc_post_labels_90d, 0) acc_post_labels_90d, coalesce(rl.acc_post_takedowns_before, 0) acc_post_takedowns_before,
               -- activity
               a.* EXCLUDE (did), rp_.act_reply_top_target_share, rp_.act_reply_to_nonfollowed_ratio,
               -- network
               n.* EXCLUDE (did, first_follow_at, first_block_made_at),
               CASE WHEN coalesce(n.net_following, 0) > 0 THEN ln(1 + coalesce(n.net_followers, 0)) - ln(1 + n.net_following) END AS net_log_follower_ratio,
               -- crowd (not model inputs for RQ1)
               coalesce(c.crowd_blockers, 0) crowd_blockers, coalesce(c.crowd_blockers_30d, 0) crowd_blockers_30d,
               coalesce(c.crowd_blockers_7d, 0) crowd_blockers_7d, c.crowd_first_block_at,
               s.t_k1 AS crowd_t_k1, s.t_k5 AS crowd_t_k5, s.vel_1d AS crowd_vel_1d, s.vel_7d AS crowd_vel_7d,
               coalesce(s.n_modlists, 0) crowd_modlists, coalesce(s.modlist_reach, 0) crowd_modlist_reach,
               coalesce(s.n_list_owners, 0) crowd_list_owners,
               -- outcomes
               l.y_takedown_ts, l.y_takedown_ts IS NOT NULL AS y_takedown,
               l.y_needs_review_ts, l.y_suspend_ts,
               coalesce(c.crowd_blockers_30d, 0) >= 5 AS y_block_burst_30d,
               -- flags
               coalesce(a.act_posts_90d, 0) > 0 OR coalesce(n.net_follows_made_30d, 0) > 0
                 OR coalesce(n.net_blocks_made_30d, 0) > 0 AS active_90d
        FROM base b
        LEFT JOIN {rp(S / 'post_history.parquet')} ph USING (did)
        LEFT JOIN act a USING (did) LEFT JOIN rep rp_ USING (did) LEFT JOIN net n USING (did)
        LEFT JOIN lab l USING (did) LEFT JOIN rlab rl USING (did) LEFT JOIN crowd c USING (did)
        LEFT JOIN {sig} s USING (did)
        WHERE NOT coalesce(l.takedown_active_at_t0, false)
    ) TO '{sql_str(str(out))}' (FORMAT parquet, COMPRESSION zstd)""")
    log.info(f"accounts_T0 written -> {out}")

    # ---------- 8. population summary
    t = con.execute(f"""SELECT
        count(*) AS accounts_not_taken_down_at_t0,
        count(*) FILTER (active_90d) AS active_90d,
        count(*) FILTER (y_takedown) AS taken_down_after_t0,
        count(*) FILTER (y_takedown AND active_90d) AS taken_down_after_t0_active,
        count(*) FILTER (y_needs_review_ts IS NOT NULL) AS needs_review_after_t0,
        count(*) FILTER (y_block_burst_30d) AS block_burst_30d_before_t0,
        count(*) FILTER (y_block_burst_30d AND y_takedown) AS block_burst_and_taken_down,
        count(*) FILTER (crowd_blockers > 0) AS ever_blocked,
        count(*) FILTER (crowd_blockers > 0 AND y_takedown) AS ever_blocked_and_taken_down
      FROM {rp(out)}""").df().T.reset_index()
    t.columns = ["measure", "accounts"]
    viz.table(t, cfg["paths"]["results"] / "S2", "T30_population")
    for r in t.itertuples():
        log.info(f"{r.measure}: {r.accounts:,}")


if __name__ == "__main__":
    main()
