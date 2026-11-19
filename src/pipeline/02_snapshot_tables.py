#!/usr/bin/env python3
"""Step 02 - snapshot -> DID-keyed tables (data/interim/snapshot/).

The snapshot (crawl ending ~2026-04-08) is our "before" picture of the network.
Every snapshot file has: repo_id, collection, rkey, at_rev, content(JSON). repos.parquet maps repo_id -> did.
Records authored by accounts taken down before the crawl are missing (noted by the data provider), so for those accounts only
incoming edges (blocks/follows/list memberships by others) are reliable.

Parts (run all, or pick with --parts):
  repos         repo_id, did, pds, tombstoned_at, last_error
  blocks        blocker_did, rkey, subject_did, created_at                      (~189M)
  listitems     owner_did, list_uri, subject_did, created_at                    (~57M)
  listblocks    subscriber_did, list_uri, created_at                            (~4.6M)
  lists         list_uri, owner_did, name, purpose, created_at                  (~1.3M, from leo_v2 fixed file)
  profiles      did, display_name, description, has_avatar, has_banner, self_labels, created_at
  follows       follower_did, subject_did, created_at                           (~2.6B, big)
  follow_counts did, n_following, n_followers   (as of snapshot)
  post_history  did, n_posts, n_replies, n_quotes, first_post_at, last_post_at, top_lang   (~2.4B posts, slowest)

Usage:
  python3 src/pipeline/02_snapshot_tables.py                               # everything
  python3 src/pipeline/02_snapshot_tables.py --parts repos blocks lists    # some parts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bskymod.common import connect, get_logger, load_config, sql_str

ALL_PARTS = ["repos", "lists", "blocks", "listitems", "listblocks", "profiles",
             "follows", "follow_counts", "post_history"]


def find_file(raw: Path, names: list[str]) -> Path:
    """First existing file among candidate relative paths (fixed copies in extra_leo_v2 win)."""
    for n in names:
        p = raw / n
        if p.exists():
            return p
    raise FileNotFoundError(f"none of {names} under {raw}")


def fix_extension_types(path: Path, cache_dir: Path, log) -> Path:
    """Write a cached copy DuckDB can read: arrow.json extension columns become plain strings.
    Row groups that cannot be decompressed (corrupted on disk) are skipped and reported in
    <cache_dir>/_bad_rowgroups.json, so no data loss is silent."""
    f = pq.ParquetFile(path)
    ext_cols = [fld.name for fld in f.schema_arrow if isinstance(fld.type, pa.BaseExtensionType)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / path.name
    report_file = cache_dir / "_bad_rowgroups.json"
    report = json.loads(report_file.read_text()) if report_file.exists() else {}
    if out.exists() and out.stat().st_mtime > path.stat().st_mtime and path.name in report:
        return out
    log.info(f"{path.name}: checking {f.num_row_groups} row groups and writing a clean copy {out}"
             + (f" (columns {ext_cols} converted from extension type)" if ext_cols else ""))
    schema = pa.schema([pa.field(fld.name, fld.type.storage_type) if fld.name in ext_cols else fld
                        for fld in f.schema_arrow])
    tmp = out.with_suffix(".tmp")
    bad, rows_ok, t0 = [], 0, time.time()
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for i in range(f.num_row_groups):
            try:
                t = f.read_row_group(i)
            except Exception as e:  # corrupted block on disk
                n = f.metadata.row_group(i).num_rows
                bad.append({"row_group": i, "rows": n, "error": f"{type(e).__name__}: {e}"[:200]})
                log.warning(f"{path.name}: row group {i} unreadable ({n:,} rows skipped): {type(e).__name__}: {e}")
                continue
            cols = [pa.chunked_array([c.storage if isinstance(c, pa.ExtensionArray) else c for c in t.column(n).chunks],
                                     type=schema.field(n).type) for n in t.column_names]
            w.write_table(pa.Table.from_arrays(cols, schema=schema))
            rows_ok += t.num_rows
            if (i + 1) % 500 == 0:
                log.info(f"  {path.name}: {i+1}/{f.num_row_groups} row groups ({time.time()-t0:.0f}s)")
    tmp.rename(out)
    total = f.metadata.num_rows
    report[path.name] = {"source": str(path), "row_groups": f.num_row_groups, "bad_row_groups": bad,
                         "rows_total": total, "rows_kept": rows_ok, "rows_lost": total - rows_ok,
                         "pct_lost": round(100 * (total - rows_ok) / max(total, 1), 4)}
    report_file.write_text(json.dumps(report, indent=1))
    msg = f"{path.name}: kept {rows_ok:,} of {total:,} rows"
    (log.warning if bad else log.info)(msg + (f"; LOST {total-rows_ok:,} rows in {len(bad)} corrupted row groups" if bad else ""))
    return out


def ts(expr: str) -> str:
    return f"TRY_CAST({expr} AS TIMESTAMPTZ)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--parts", nargs="*", default=ALL_PARTS, choices=ALL_PARTS)
    ap.add_argument("--force", action="store_true", help="rebuild parts that already exist")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw, out = cfg["paths"]["raw"], cfg["paths"]["interim"] / "snapshot"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("02", cfg["paths"]["results"] / "logs" / "02_snapshot.log")
    con = connect(cfg)
    fixed = cfg["paths"]["interim"] / "snapshot_fixed"

    src = {
        "repos": find_file(raw, ["extra_leo_v2/snapshot/repos.parquet", "snapshot/repos.parquet"]),
        "block": find_file(raw, ["snapshot/block.parquet", "snapshot/app.bsky.graph.block.parquet"]),
        "follow": find_file(raw, ["snapshot/follow.parquet", "snapshot/app.bsky.graph.follow.parquet"]),
        "list": find_file(raw, ["extra_leo_v2/snapshot/app.bsky.graph.list.parquet", "snapshot/list.parquet"]),
        "listitem": find_file(raw, ["snapshot/listitem.parquet", "snapshot/app.bsky.graph.listitem.parquet"]),
        "listblock": find_file(raw, ["snapshot/listblock.parquet", "snapshot/app.bsky.graph.listblock.parquet"]),
        "profile": find_file(raw, ["snapshot/profile.parquet", "snapshot/app.bsky.actor.profile.parquet"]),
        "post": find_file(raw, ["snapshot/post.parquet", "snapshot/app.bsky.feed.post.parquet"]),
    }

    def rp(key: str) -> str:
        p = src[key] if key == "repos" else fix_extension_types(src[key], fixed, log)
        return f"read_parquet('{sql_str(str(p))}')"

    def write(name: str, sql: str):
        t0 = time.time()
        dst = out / f"{name}.parquet"
        if dst.exists() and not args.force:
            n = con.execute(f"SELECT count(*) FROM read_parquet('{sql_str(str(dst))}')").fetchone()[0]
            log.info(f"{name}: already built ({n:,} rows), skipping (use --force to rebuild)")
            return n
        tmp = out / f"{name}.parquet.tmp"
        con.execute(f"COPY ({sql}) TO '{sql_str(str(tmp))}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)")
        tmp.rename(dst)
        n = con.execute(f"SELECT count(*) FROM read_parquet('{sql_str(str(dst))}')").fetchone()[0]
        log.info(f"{name}: {n:,} rows in {time.time()-t0:.0f}s -> {dst}")
        return n

    # repo_id -> did lookup used by every part
    log.info(f"loading repos from {src['repos']}")
    con.execute(f"CREATE TEMP TABLE repo AS SELECT id AS repo_id, did FROM {rp('repos')}")
    summary = {}

    if "repos" in args.parts:
        summary["repos"] = write("repos", f"SELECT id AS repo_id, did, pds, tombstoned_at, last_error FROM {rp('repos')}")

    if "lists" in args.parts:
        summary["lists"] = write("lists", f"""
            SELECT 'at://' || r.did || '/app.bsky.graph.list/' || s.rkey AS list_uri, r.did AS owner_did, s.rkey,
                   json_extract_string(s.content, '$.name')        AS name,
                   json_extract_string(s.content, '$.purpose')     AS purpose,
                   json_extract_string(s.content, '$.description') AS description,
                   {ts("json_extract_string(s.content, '$.createdAt')")} AS created_at
            FROM {rp('list')} s JOIN repo r USING (repo_id)""")

    if "blocks" in args.parts:
        summary["blocks"] = write("blocks", f"""
            SELECT r.did AS blocker_did, s.rkey,
                   json_extract_string(s.content, '$.subject') AS subject_did,
                   {ts("json_extract_string(s.content, '$.createdAt')")} AS created_at
            FROM {rp('block')} s JOIN repo r USING (repo_id)""")

    if "listitems" in args.parts:
        summary["listitems"] = write("listitems", f"""
            SELECT r.did AS owner_did, s.rkey,
                   json_extract_string(s.content, '$.list')    AS list_uri,
                   json_extract_string(s.content, '$.subject') AS subject_did,
                   {ts("json_extract_string(s.content, '$.createdAt')")} AS created_at
            FROM {rp('listitem')} s JOIN repo r USING (repo_id)""")

    if "listblocks" in args.parts:
        summary["listblocks"] = write("listblocks", f"""
            SELECT r.did AS subscriber_did, s.rkey,
                   json_extract_string(s.content, '$.subject') AS list_uri,
                   {ts("json_extract_string(s.content, '$.createdAt')")} AS created_at
            FROM {rp('listblock')} s JOIN repo r USING (repo_id)""")

    if "profiles" in args.parts:
        summary["profiles"] = write("profiles", f"""
            SELECT r.did,
                   json_extract_string(s.content, '$.displayName') AS display_name,
                   json_extract_string(s.content, '$.description') AS description,
                   json_extract(s.content, '$.avatar') IS NOT NULL AS has_avatar,
                   json_extract(s.content, '$.banner') IS NOT NULL AS has_banner,
                   CAST(json_extract(s.content, '$.labels.values') AS VARCHAR) AS self_labels,
                   json_extract_string(s.content, '$.pinnedPost.uri') AS pinned_post_uri,
                   {ts("json_extract_string(s.content, '$.createdAt')")} AS created_at
            FROM {rp('profile')} s JOIN repo r USING (repo_id)
            WHERE s.rkey = 'self'""")

    if "follows" in args.parts:
        summary["follows"] = write("follows", f"""
            SELECT r.did AS follower_did,
                   json_extract_string(s.content, '$.subject') AS subject_did,
                   {ts("json_extract_string(s.content, '$.createdAt')")} AS created_at
            FROM {rp('follow')} s JOIN repo r USING (repo_id)""")

    if "follow_counts" in args.parts:
        f = out / "follows.parquet"
        if not f.exists():
            raise SystemExit("follow_counts needs follows.parquet; run --parts follows first")
        summary["follow_counts"] = write("follow_counts", f"""
            WITH fo AS (SELECT follower_did AS did, count(*) n_following FROM read_parquet('{sql_str(str(f))}') GROUP BY 1),
                 fe AS (SELECT subject_did  AS did, count(*) n_followers FROM read_parquet('{sql_str(str(f))}') GROUP BY 1)
            SELECT did, coalesce(n_following, 0) n_following, coalesce(n_followers, 0) n_followers
            FROM fo FULL OUTER JOIN fe USING (did)""")

    if "post_history" in args.parts:
        summary["post_history"] = write("post_history", f"""
            WITH p AS (
              SELECT repo_id,
                     {ts("json_extract_string(content, '$.createdAt')")} AS created_at,
                     json_extract(content, '$.reply') IS NOT NULL AS is_reply,
                     json_extract_string(content, '$.embed.record.uri') IS NOT NULL
                       OR json_extract_string(content, '$.embed.record.record.uri') IS NOT NULL AS is_quote,
                     json_extract_string(content, '$.langs[0]') AS lang
              FROM {rp('post')})
            SELECT r.did, count(*) n_posts, count(*) FILTER (is_reply) n_replies, count(*) FILTER (is_quote) n_quotes,
                   min(created_at) FILTER (created_at > TIMESTAMPTZ '2023-01-01') AS first_post_at,
                   max(created_at) FILTER (created_at < TIMESTAMPTZ '2026-07-01') AS last_post_at,
                   mode(lang) AS top_lang, count(DISTINCT lang) AS n_langs
            FROM p JOIN repo r USING (repo_id) GROUP BY r.did""")

    rep = fixed / "_bad_rowgroups.json"
    if rep.exists():
        for k, v in json.loads(rep.read_text()).items():
            if v["rows_lost"]:
                log.warning(f"DATA LOSS {k}: {v['rows_lost']:,} rows ({v['pct_lost']}%) in row groups {[b['row_group'] for b in v['bad_row_groups']]}")
    log.info(f"done: {summary}")


if __name__ == "__main__":
    main()
