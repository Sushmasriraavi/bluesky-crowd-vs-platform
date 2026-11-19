#!/usr/bin/env python3
"""Step 01 - convert firehose day files (.json.gz) into compact parquet tables.

One pass per day file. Output (hive-partitioned by day) under data/interim/firehose/:
  ops/          every operation: did, t, seq, operation, collection, rkey          (activity, deletes/unblocks)
  posts/        post creates: text, lang, reply parent/root, quote, embed type, links, mentions
  graph/        block / follow / listitem / listblock / list creates: subject, list, list name/purpose
  interactions/ like / repost creates: subject_did (whose post was liked/reposted)
  profiles/     profile record updates: display name, description, avatar present

Times: `t` is the relay event time (field "time", UTC). record.createdAt is client-set and kept as created_at.
Truncated files (2026-05-17) are salvaged: read until the gzip error, last partial line dropped,
and the day is marked partial in the manifest.

Usage:
  python3 src/pipeline/01_firehose_to_parquet.py --days 2026-05-03          # one day (test first!)
  python3 src/pipeline/01_firehose_to_parquet.py --workers 8                 # all days, 8 in parallel
Re-running skips days that are already done (use --force to redo).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from bskymod.common import connect, get_logger, load_config, sql_str

DAY_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

COLUMNS = (
    "{'did': 'VARCHAR', 'time': 'VARCHAR', 'received': 'VARCHAR', 'seq': 'BIGINT', "
    "'commit': 'STRUCT(rev VARCHAR, ops STRUCT(operation VARCHAR, collection VARCHAR, rkey VARCHAR, record JSON)[])'}"
)

GRAPH = ("app.bsky.graph.block", "app.bsky.graph.follow", "app.bsky.graph.listitem",
         "app.bsky.graph.listblock", "app.bsky.graph.list")

TABLES = {
    "ops": """
        SELECT did, t, seq, operation, collection, rkey FROM ev""",
    "posts": """
        SELECT did, t, rkey,
          json_extract_string(rec, '$.createdAt')                       AS created_at,
          json_extract_string(rec, '$.text')                            AS text,
          json_extract_string(rec, '$.langs[0]')                        AS lang,
          json_array_length(rec, '$.langs')                             AS n_langs,
          json_extract_string(rec, '$.reply.parent.uri')                AS reply_parent_uri,
          json_extract_string(rec, '$.reply.root.uri')                  AS reply_root_uri,
          json_extract_string(json_extract(rec, '$.embed'), '$."$type"') AS embed_type,
          coalesce(json_extract_string(rec, '$.embed.record.record.uri'),
                   json_extract_string(rec, '$.embed.record.uri'))       AS quote_uri,
          json_extract_string(rec, '$.embed.external.uri')              AS external_uri,
          regexp_extract_all(CAST(json_extract(rec, '$.facets') AS VARCHAR),
                             '"uri":\\s*"([^"]+)"', 1)                   AS link_uris,
          regexp_extract_all(CAST(json_extract(rec, '$.facets') AS VARCHAR),
                             '"did":\\s*"(did:[^"]+)"', 1)               AS mention_dids,
          array_length(regexp_extract_all(CAST(json_extract(rec, '$.facets') AS VARCHAR),
                             'richtext\\.facet#tag'))                   AS n_tags,
          json_extract(rec, '$.labels') IS NOT NULL                     AS has_self_label
        FROM ev WHERE operation = 'create' AND collection = 'app.bsky.feed.post'""",
    "graph": f"""
        SELECT did, t, rkey, collection,
          json_extract_string(rec, '$.createdAt') AS created_at,
          json_extract_string(rec, '$.subject')   AS subject,
          json_extract_string(rec, '$.list')      AS list_uri,
          json_extract_string(rec, '$.name')      AS list_name,
          json_extract_string(rec, '$.purpose')   AS list_purpose
        FROM ev WHERE operation = 'create' AND collection IN {GRAPH}""",
    "interactions": """
        SELECT did, t, collection,
          regexp_extract(json_extract_string(rec, '$.subject.uri'), 'at://(did:[^/]+)/', 1) AS subject_did,
          json_extract_string(rec, '$.subject.uri') AS subject_uri
        FROM ev WHERE operation = 'create'
          AND collection IN ('app.bsky.feed.like', 'app.bsky.feed.repost')""",
    "profiles": """
        SELECT did, t, operation,
          json_extract_string(rec, '$.displayName') AS display_name,
          json_extract_string(rec, '$.description') AS description,
          json_extract(rec, '$.avatar') IS NOT NULL AS has_avatar,
          json_extract(rec, '$.labels') IS NOT NULL AS has_self_label
        FROM ev WHERE collection = 'app.bsky.actor.profile'""",
}


def salvage(src: Path, dst: Path, log) -> dict:
    """Copy readable lines of a truncated .json.gz into plain .json; drop the last (partial) line."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n, prev, err = 0, None, None
    with gzip.open(src, "rb") as f, open(dst, "wb") as out:
        try:
            for line in f:
                if prev is not None:
                    out.write(prev)
                    n += 1
                prev = line
        except (EOFError, OSError, zlib.error) as e:
            err = f"{type(e).__name__}: {e}"
        if prev is not None and err is None:  # file was complete after all
            out.write(prev)
            n += 1
    # make sure the last kept line is valid json; otherwise trim it
    log.info(f"salvaged {n:,} lines from {src.name} ({err or 'no error'})")
    return {"salvaged_lines": n, "salvage_error": err}


def gzip_ok(path: Path) -> bool:
    import subprocess
    return subprocess.run(["gzip", "-t", str(path)], capture_output=True).returncode == 0


def convert_day(day: str, src: str, cfg_path: str | None, threads: int, mem: str, force: bool, check: bool) -> dict:
    cfg = load_config(cfg_path)
    out_root = cfg["paths"]["interim"] / "firehose"
    log = get_logger(f"fh-{day}", cfg["paths"]["results"] / "logs" / f"01_firehose_{day}.log")
    done_flag = out_root / "_done" / f"{day}.json"
    if done_flag.exists() and not force:
        return json.loads(done_flag.read_text()) | {"skipped": True}

    t0 = time.time()
    src = Path(src)
    info = {"day": day, "source": str(src), "partial": False}
    read_path, compression = src, "gzip"
    if check and not gzip_ok(src):
        tmp = cfg["paths"]["interim"] / "salvage" / f"{day}.json"
        info |= salvage(src, tmp, log) | {"partial": True}
        read_path, compression = tmp, "uncompressed"

    con = connect(cfg)
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{mem}'")
    day_tmp = cfg["paths"]["interim"] / "duckdb_tmp" / day  # separate spill folder per parallel worker
    day_tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{sql_str(str(day_tmp))}'")
    log.info(f"reading {read_path}")
    con.execute(f"""
        CREATE TEMP TABLE ev AS
        SELECT did,
               coalesce(TRY_CAST("time" AS TIMESTAMPTZ), TRY_CAST(received AS TIMESTAMPTZ)) AS t,
               TRY_CAST("time" AS TIMESTAMPTZ) IS NULL AS bad_time,
               "time" AS raw_time,
               seq,
               op.operation AS operation, op.collection AS collection, op.rkey AS rkey, op.record AS rec
        FROM (
          SELECT did, "time", received, seq, unnest("commit".ops) AS op
          FROM read_json('{sql_str(str(read_path))}', format='newline_delimited', compression='{compression}',
                         columns={COLUMNS}, maximum_object_size=268435456, ignore_errors=true)
          WHERE "commit" IS NOT NULL
        )""")
    counts = {}
    for name, sql in TABLES.items():
        d = out_root / name / f"day={day}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        con.execute(f"COPY ({sql}) TO '{sql_str(str(d / 'part.parquet'))}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)")
        counts[name] = con.execute(f"SELECT count(*) FROM read_parquet('{sql_str(str(d / 'part.parquet'))}')").fetchone()[0]
    n_bad, n_unfixable, bad_ex = con.execute(
        "SELECT count(DISTINCT seq) FILTER (bad_time), count(DISTINCT seq) FILTER (t IS NULL), "
        "(list(DISTINCT raw_time) FILTER (bad_time))[1:5] FROM ev").fetchone()
    if n_bad:
        log.warning(f"{day}: {n_bad} events with unreadable 'time' (received used instead; {n_unfixable} still null). examples: {bad_ex}")
    mn, mx, nlines = con.execute("SELECT CAST(min(t) AS VARCHAR), CAST(max(t) AS VARCHAR), count(DISTINCT seq) FROM ev").fetchone()
    top = con.execute("SELECT collection, count(*) n FROM ev GROUP BY 1 ORDER BY n DESC LIMIT 12").fetchall()
    con.close()
    shutil.rmtree(day_tmp, ignore_errors=True)
    if read_path != src:
        read_path.unlink(missing_ok=True)  # salvaged plain copy no longer needed

    info |= {"bad_time_events": n_bad, "null_time_events": n_unfixable, "bad_time_examples": bad_ex,
             "rows": counts, "events": nlines, "t_min": str(mn), "t_max": str(mx),
             "top_collections": {c: n for c, n in top}, "seconds": round(time.time() - t0)}
    done_flag.parent.mkdir(parents=True, exist_ok=True)
    done_flag.write_text(json.dumps(info, indent=1))
    log.info(f"{day} done in {info['seconds']}s: {counts}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--days", nargs="*", help="YYYY-MM-DD ... (default: all)")
    ap.add_argument("--workers", type=int, default=8, help="days converted in parallel")
    ap.add_argument("--threads", type=int, default=8, help="DuckDB threads per day")
    ap.add_argument("--mem", default="100GB", help="DuckDB memory per day")
    ap.add_argument("--no-check", action="store_true", help="skip gzip -t before converting")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger("01", cfg["paths"]["results"] / "logs" / "01_firehose.log")
    fh_dirs = [p for p in cfg["paths"]["raw"].iterdir() if p.name == "firehose" or p.name.startswith("extra_")]
    files = {}
    for d in fh_dirs:
        for p in sorted(d.rglob("*.json.gz")):
            if "firehose" not in str(p):
                continue
            m = DAY_RE.search(p.name) or DAY_RE.search(str(p).replace("/", "-"))
            if m:
                day = "-".join(m.groups())
                files.setdefault(day, p)  # main firehose folder wins; extra copies ignored
    if args.days:
        files = {d: files[d] for d in args.days if d in files}
    log.info(f"{len(files)} day files to convert with {args.workers} workers")

    results = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(convert_day, d, str(p), args.config, args.threads, args.mem, args.force, not args.no_check): d
                for d, p in sorted(files.items())}
        for f in as_completed(futs):
            d = futs[f]
            try:
                r = f.result()
                results.append(r)
                log.info(f"[{len(results)}/{len(files)}] {d}: " + ("skipped (already done)" if r.get("skipped") else f"{r['seconds']}s, posts={r['rows']['posts']:,}, ops={r['rows']['ops']:,}" + (" PARTIAL" if r["partial"] else "")))
            except Exception as e:
                log.error(f"{d} FAILED: {type(e).__name__}: {e}")
                results.append({"day": d, "error": str(e)})

    # manifest over all done days
    done_dir = cfg["paths"]["interim"] / "firehose" / "_done"
    rows = [json.loads(p.read_text()) for p in sorted(done_dir.glob("*.json"))]
    man = cfg["paths"]["results"] / "inventory" / "firehose_manifest.tsv"
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "w") as f:
        f.write("day\tpartial\tbad_time\tevents\tops\tposts\tgraph\tinteractions\tprofiles\tt_min\tt_max\tseconds\n")
        for r in rows:
            c = r["rows"]
            f.write(f"{r['day']}\t{r['partial']}\t{r.get('bad_time_events', 0)}\t{r['events']}\t{c['ops']}\t{c['posts']}\t{c['graph']}\t{c['interactions']}\t{c['profiles']}\t{r['t_min']}\t{r['t_max']}\t{r['seconds']}\n")
    failed = [r["day"] for r in results if "error" in r]
    log.info(f"manifest -> {man}; done days: {len(rows)}; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
