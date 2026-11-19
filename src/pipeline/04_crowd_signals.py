#!/usr/bin/env python3
"""Step 04 - crowd signals per account (data/interim/crowd/).

Combines the snapshot (state up to ~2026-04-08) with the firehose (2026-04-01 .. 2026-05-31):
a record seen in both is the same record (same author + rkey) and counted once; firehose deletes
remove it (unblock / removed from list / unsubscribed).

Time of a record = firehose relay time if we saw it live, else the client createdAt from the
snapshot (kept only if plausible: 2023-02-01 .. crawl end); otherwise unknown -> treated as
"before the observation window" (it existed in the snapshot).

Outputs
  block_edges.parquet     blocker_did, subject_did, rkey, t_created, t_removed, in_snapshot, in_firehose
  listitem_edges.parquet  owner_did, list_uri, subject_did, rkey, t_added, t_removed
  listsub_edges.parquet   subscriber_did, list_uri, rkey, t_sub, t_unsub
  lists_all.parquet       list_uri, owner_did, purpose, name, n_subscribers (active at end)
  subject_signals.parquet one row per account that was ever blocked or put on a list:
      n_blockers, n_blockers_pre, n_blockers_win, n_unblocked,
      t_k1, t_k5, t_k20 (time the account reached its 1st/5th/20th distinct blocker; PRE = before window),
      vel_1d, vel_3d, vel_7d (new distinct blockers in the first 1/3/7 days after t_k1),
      n_modlists, n_modlists_pre, t_list1, modlist_reach (sum of active subscribers of those lists),
      n_curatelists, n_list_owners
  account_activity.parquet one row per DID active in the firehose:
      first_t, last_t, n_ops, n_posts, n_replies, n_follows, n_blocks_made, n_active_days
  merge_report.txt         how many records matched between snapshot and firehose, unmatched deletes
"""
from __future__ import annotations

import argparse
import time

from bskymod.common import connect, get_logger, load_config, sql_str

PRE = "TIMESTAMPTZ '2000-01-01 00:00:00+00'"   # sentinel: existed before the window, exact time unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--fh-end", default="2026-06-01", help="end of firehose coverage (exclusive)")
    ap.add_argument("--crawl-end", default="2026-04-09", help="latest plausible snapshot createdAt")
    ap.add_argument("--parts", nargs="*", default=["edges", "signals", "activity"])
    ap.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    ap.add_argument("--source", choices=["merged", "snapshot"], default="merged",
                    help="snapshot = snapshot records only (history since 2023, ends at crawl); merged = snapshot + firehose")
    args = ap.parse_args()

    cfg = load_config(args.config)
    I = cfg["paths"]["interim"]
    snap, fh = I / "snapshot", I / "firehose"
    out = I / ("crowd_snapshot" if args.source == "snapshot" else "crowd")
    if args.source == "snapshot":
        # history window: from labeler/platform start to crawl end; firehose not used
        args.start = args.start if args.start != "2026-04-01" else "2023-02-01"
        args.fh_end = args.crawl_end
        args.parts = [p for p in args.parts if p != "activity"]
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("04", cfg["paths"]["results"] / "logs" / f"04_{out.name}.log")
    con = connect(cfg)
    S = f"TIMESTAMPTZ '{args.start} 00:00:00+00'"
    E = f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    C = f"TIMESTAMPTZ '{args.crawl_end} 00:00:00+00'"
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    graph, ops = rp(fh / "graph" / "*" / "*.parquet"), rp(fh / "ops" / "*" / "*.parquet")
    plaus = lambda c: f"CASE WHEN CAST({c} AS TIMESTAMPTZ) >= TIMESTAMPTZ '2023-02-01 00:00:00+00' AND CAST({c} AS TIMESTAMPTZ) < {C} THEN CAST({c} AS TIMESTAMPTZ) END"
    report = []

    def write(name, sql):
        t0 = time.time()
        dst = out / f"{name}.parquet"
        if dst.exists() and not args.force:
            log.info(f"{name}: already built, skipping (use --force to rebuild)")
            return dst
        con.execute(f"COPY ({sql}) TO '{sql_str(str(dst))}' (FORMAT parquet, COMPRESSION zstd)")
        n = con.execute(f"SELECT count(*) FROM {rp(dst)}").fetchone()[0]
        log.info(f"{name}: {n:,} rows ({time.time()-t0:.0f}s)")
        return dst

    if "edges" in args.parts:
        log.info("loading firehose graph creates and deletes")
        no_fh = " WHERE false" if args.source == "snapshot" else ""
        con.execute(f"""CREATE TEMP TABLE g AS SELECT did, CAST(t AS TIMESTAMPTZ) AS t, rkey, collection, CAST(subject AS VARCHAR) AS subject, CAST(list_uri AS VARCHAR) AS list_uri, CAST(list_name AS VARCHAR) AS list_name, CAST(list_purpose AS VARCHAR) AS list_purpose
                        FROM {graph}{no_fh}""")
        con.execute(f"""CREATE TEMP TABLE d AS SELECT did, rkey, collection, CAST(min(t) AS TIMESTAMPTZ) AS t FROM {ops}
                        WHERE operation = 'delete' AND collection IN
                          ('app.bsky.graph.block','app.bsky.graph.listitem','app.bsky.graph.listblock','app.bsky.graph.list')
                          {'AND false' if args.source == 'snapshot' else ''}
                        GROUP BY ALL""")

        def merge(name, snap_sql, coll, who, extra_cols, t_new, t_del):
            """Full outer join snapshot <-> firehose creates on (author, rkey); attach first delete."""
            ex_f = ", ".join(f"any_value({c}) AS {a}" for c, a in extra_cols)
            ex_c = ", ".join(f"coalesce(f.{a}, s.{a}) AS {a}" for _, a in extra_cols)
            sql = f"""
              WITH s AS ({snap_sql}),
                   f AS (SELECT did AS {who}, rkey, min(t) AS t_fh, {ex_f} FROM g WHERE collection = '{coll}' GROUP BY did, rkey)
              SELECT coalesce(f.{who}, s.{who}) AS {who}, coalesce(f.rkey, s.rkey) AS rkey, {ex_c},
                     coalesce(f.t_fh, s.t_client) AS {t_new},
                     s.rkey IS NOT NULL AS in_snapshot, f.rkey IS NOT NULL AS in_firehose,
                     dl.t AS {t_del}
              FROM s FULL OUTER JOIN f ON s.{who} = f.{who} AND s.rkey = f.rkey
              LEFT JOIN (SELECT did, rkey, t FROM d WHERE collection = '{coll}') dl
                     ON dl.did = coalesce(f.{who}, s.{who}) AND dl.rkey = coalesce(f.rkey, s.rkey)"""
            dst = write(name, sql)
            both, s_only, f_only, rem = con.execute(f"""SELECT count(*) FILTER (in_snapshot AND in_firehose),
                count(*) FILTER (in_snapshot AND NOT in_firehose), count(*) FILTER (in_firehose AND NOT in_snapshot),
                count(*) FILTER ({t_del} IS NOT NULL) FROM {rp(dst)}""").fetchone()
            unmatched = con.execute(f"""SELECT count(*) FROM d WHERE collection = '{coll}' AND NOT EXISTS
                (SELECT 1 FROM {rp(dst)} e WHERE e.{who} = d.did AND e.rkey = d.rkey)""").fetchone()[0]
            report.append(f"{name}: snapshot+firehose {both:,} | snapshot only {s_only:,} | firehose only {f_only:,} | "
                          f"removed in window {rem:,} | deletes with no matching record {unmatched:,}")
            log.info(report[-1])

        merge("block_edges",
              f"SELECT blocker_did, rkey, subject_did, {plaus('created_at')} AS t_client FROM {rp(snap / 'blocks.parquet')}",
              "app.bsky.graph.block", "blocker_did", [("subject", "subject_did")], "t_created", "t_removed")
        merge("listitem_edges",
              f"SELECT owner_did, rkey, list_uri, subject_did, {plaus('created_at')} AS t_client FROM {rp(snap / 'listitems.parquet')}",
              "app.bsky.graph.listitem", "owner_did", [("list_uri", "list_uri"), ("subject", "subject_did")], "t_added", "t_removed")
        merge("listsub_edges",
              f"SELECT subscriber_did, rkey, list_uri, {plaus('created_at')} AS t_client FROM {rp(snap / 'listblocks.parquet')}",
              "app.bsky.graph.listblock", "subscriber_did", [("subject", "list_uri")], "t_sub", "t_unsub")

        write("lists_all", f"""
            WITH s AS (SELECT list_uri, owner_did, purpose, "name" AS list_name FROM {rp(snap / 'lists.parquet')}),
                 f AS (SELECT 'at://' || did || '/app.bsky.graph.list/' || rkey AS list_uri, did AS owner_did,
                              any_value(list_purpose) AS purpose, any_value(list_name) AS list_name
                       FROM g WHERE collection = 'app.bsky.graph.list' GROUP BY ALL),
                 l AS (SELECT coalesce(f.list_uri, s.list_uri) AS list_uri, coalesce(f.owner_did, s.owner_did) AS owner_did,
                              coalesce(f.purpose, s.purpose) AS purpose, coalesce(f.list_name, s.list_name) AS list_name
                       FROM s FULL OUTER JOIN f USING (list_uri)),
                 subs AS (SELECT list_uri, count(DISTINCT subscriber_did) AS n_subs FROM {rp(out / 'listsub_edges.parquet')}
                          WHERE t_unsub IS NULL GROUP BY 1)
            SELECT l.*, coalesce(subs.n_subs, 0) AS n_subscribers FROM l LEFT JOIN subs USING (list_uri)""")
        (out / "merge_report.txt").write_text("\n".join(report) + "\n")

    if "signals" in args.parts:
        log.info("computing per-account crowd signals")
        con.execute(f"""CREATE TEMP TABLE pairs AS
            SELECT subject_did, blocker_did,
                   min(coalesce(t_created, {PRE})) AS t_first,
                   bool_and(t_removed IS NOT NULL) AS all_removed
            FROM {rp(out / 'block_edges.parquet')}
            WHERE subject_did LIKE 'did:%' AND subject_did <> blocker_did
              AND coalesce(t_created, {PRE}) < {E}
            GROUP BY ALL""")
        con.execute("""CREATE TEMP TABLE ranked AS
            SELECT *, row_number() OVER (PARTITION BY subject_did ORDER BY t_first, blocker_did) AS rn,
                      min(t_first) OVER (PARTITION BY subject_did) AS t0
            FROM pairs""")
        con.execute(f"""CREATE TEMP TABLE bsig AS
            SELECT subject_did,
                   count(*) n_blockers,
                   count(*) FILTER (t_first < {S}) n_blockers_pre,
                   count(*) FILTER (t_first >= {S}) n_blockers_win,
                   count(*) FILTER (all_removed) n_unblocked,
                   min(t_first) FILTER (rn = 1)  t_k1,
                   min(t_first) FILTER (rn = 5)  t_k5,
                   min(t_first) FILTER (rn = 20) t_k20,
                   -- velocity only defined when the first block falls inside the window
                   CASE WHEN min(t0) >= {S} THEN count(*) FILTER (t_first < t0 + INTERVAL 1 DAY) END vel_1d,
                   CASE WHEN min(t0) >= {S} THEN count(*) FILTER (t_first < t0 + INTERVAL 3 DAY) END vel_3d,
                   CASE WHEN min(t0) >= {S} THEN count(*) FILTER (t_first < t0 + INTERVAL 7 DAY) END vel_7d
            FROM ranked GROUP BY subject_did""")
        con.execute(f"""CREATE TEMP TABLE lsig AS
            WITH li AS (SELECT e.subject_did, e.list_uri, e.owner_did, min(coalesce(e.t_added, {PRE})) t_add,
                               bool_and(e.t_removed IS NOT NULL) removed
                        FROM {rp(out / 'listitem_edges.parquet')} e
                        WHERE e.subject_did LIKE 'did:%' AND coalesce(e.t_added, {PRE}) < {E} GROUP BY ALL)
            SELECT li.subject_did,
                   count(DISTINCT li.list_uri) FILTER (l.purpose LIKE '%modlist' AND NOT li.removed) n_modlists,
                   count(DISTINCT li.list_uri) FILTER (l.purpose LIKE '%modlist' AND li.t_add < {S}) n_modlists_pre,
                   min(li.t_add) FILTER (l.purpose LIKE '%modlist') t_list1,
                   sum(l.n_subscribers) FILTER (l.purpose LIKE '%modlist' AND NOT li.removed) modlist_reach,
                   count(DISTINCT li.list_uri) FILTER (l.purpose LIKE '%curatelist') n_curatelists,
                   count(DISTINCT li.owner_did) FILTER (l.purpose LIKE '%modlist') n_list_owners
            FROM li LEFT JOIN {rp(out / 'lists_all.parquet')} l USING (list_uri)
            GROUP BY 1""")
        write("subject_signals", """
            SELECT coalesce(b.subject_did, l.subject_did) AS did,
                   coalesce(n_blockers, 0) n_blockers, coalesce(n_blockers_pre, 0) n_blockers_pre,
                   coalesce(n_blockers_win, 0) n_blockers_win, coalesce(n_unblocked, 0) n_unblocked,
                   t_k1, t_k5, t_k20, vel_1d, vel_3d, vel_7d,
                   coalesce(n_modlists, 0) n_modlists, coalesce(n_modlists_pre, 0) n_modlists_pre, t_list1,
                   coalesce(modlist_reach, 0) modlist_reach, coalesce(n_curatelists, 0) n_curatelists,
                   coalesce(n_list_owners, 0) n_list_owners
            FROM bsig b FULL OUTER JOIN lsig l ON b.subject_did = l.subject_did""")

    if "activity" in args.parts:
        log.info("computing firehose activity per account")
        posts = rp(fh / "posts" / "*" / "*.parquet")
        write("account_activity", f"""
            WITH o AS (SELECT did, min(t) first_t, max(t) last_t, count(*) n_ops,
                              count(*) FILTER (operation = 'create' AND collection = 'app.bsky.feed.post') n_posts,
                              count(*) FILTER (operation = 'create' AND collection = 'app.bsky.graph.follow') n_follows,
                              count(*) FILTER (operation = 'create' AND collection = 'app.bsky.graph.block') n_blocks_made,
                              count(DISTINCT CAST(t AS DATE)) n_active_days
                       FROM {ops} WHERE t >= {S} AND t < {E} GROUP BY did),
                 r AS (SELECT did, count(*) n_replies FROM {posts} WHERE reply_parent_uri IS NOT NULL GROUP BY did)
            SELECT o.*, coalesce(r.n_replies, 0) n_replies FROM o LEFT JOIN r USING (did)""")
    log.info("done")


if __name__ == "__main__":
    main()
