#!/usr/bin/env python3
"""Step 0 - data inventory (read-only).

What it checks:
  1. every file under data/raw: size, type, readable or broken
  2. parquet schema + row count for each snapshot / labeler file
  3. firehose: schema consistency across days, day coverage, missing days
  4. sample rows, time ranges, top values of type-like columns
  5. official labeler: label values, negations, account vs post targets
  6. firehose/2026/05/17/missing.json.gz: readable, line count, keys

Outputs go to results/inventory/: report.md (send this back), files.csv, broken.csv, schemas.csv.

Usage:
  python src/pipeline/00_inventory.py                 # full
  python src/pipeline/00_inventory.py --quick         # skip full scans of big snapshot files
  python src/pipeline/00_inventory.py --sample-day 2026-05-03
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from bskymod.common import connect, get_logger, load_config, qident, sql_str

TIME_NAME = re.compile(r"(created|time|_at$|^ts$|cts|date|indexed)", re.I)
CAT_NAME = re.compile(r"(type|kind|collection|action|^op|status|^val$|event|lang|nsid|path|src|neg|active)", re.I)
DAY_RE = re.compile(r"(\d{4})[/-](\d{2})[/-](\d{2})")


def human(n: int) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def walk(raw: Path):
    rows = []
    for dirpath, _, files in os.walk(raw, followlinks=True):
        for fn in files:
            p = Path(dirpath) / fn
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            rows.append({"path": str(p.relative_to(raw)), "size": size, "ext": ".json.gz" if fn.endswith(".json.gz") else p.suffix})
    return pd.DataFrame(rows, columns=["path", "size", "ext"])


def parquet_meta(raw: Path, files: pd.DataFrame, log):
    meta, broken = [], []
    pfiles = files[files.ext.str.endswith(".parquet")]
    t0 = time.time()
    for i, rel in enumerate(pfiles.path):
        try:
            pf = pq.ParquetFile(raw / rel)
            sch = pf.schema_arrow
            meta.append({
                "path": rel,
                "rows": pf.metadata.num_rows,
                "row_groups": pf.metadata.num_row_groups,
                "n_cols": len(sch),
                "schema_sig": "|".join(f"{f.name}:{f.type}" for f in sch),
            })
        except Exception as e:  # corrupted / truncated file
            broken.append({"path": rel, "error": f"{type(e).__name__}: {e}"[:300]})
        if (i + 1) % 500 == 0:
            log.info(f"  metadata {i+1}/{len(pfiles)} ({time.time()-t0:.0f}s)")
    return pd.DataFrame(meta), pd.DataFrame(broken, columns=["path", "error"])


def profile(con, path: Path, full_scan: bool, max_cat_cols: int = 12) -> list[str]:
    """Return markdown lines describing one parquet file (or glob)."""
    src = f"read_parquet('{sql_str(str(path))}', union_by_name=true)"
    out = []
    desc = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchdf()
    out.append("Columns:")
    for _, r in desc.iterrows():
        out.append(f"    {r['column_name']}: {r['column_type']}")

    sample = con.execute(f"SELECT * FROM {src} LIMIT 3").fetchdf()
    out.append("Sample rows (values cut to 120 chars):")
    for _, r in sample.iterrows():
        d = {k: (str(v)[:120]) for k, v in r.items()}
        out.append("    " + json.dumps(d, ensure_ascii=False)[:1500])

    if not full_scan:
        out.append("(full scan skipped: --quick)")
        return out

    cols = list(desc.column_name)
    types = dict(zip(desc.column_name, desc.column_type))
    # null counts
    nulls = con.execute(
        "SELECT " + ", ".join(f"count(*) - count({qident(c)}) AS {qident(c)}" for c in cols) + f" FROM {src}"
    ).fetchdf().iloc[0]
    nz = {c: int(v) for c, v in nulls.items() if int(v) > 0}
    out.append(f"Null counts (non-zero only): {nz if nz else 'none'}")

    # time ranges
    tcols = [c for c in cols if TIME_NAME.search(c) and not types[c].startswith(("STRUCT", "MAP")) and "[]" not in types[c]]
    for c in tcols:
        mn, mx = con.execute(f"SELECT min({qident(c)}), max({qident(c)}) FROM {src}").fetchone()
        out.append(f"Range {c}: {mn}  ->  {mx}")

    # categorical tops
    ccols = [c for c in cols if CAT_NAME.search(c) and (types[c] in ("VARCHAR", "BOOLEAN") or types[c].startswith("INT"))]
    for c in ccols[:max_cat_cols]:
        nd = con.execute(f"SELECT approx_count_distinct({qident(c)}) FROM {src}").fetchone()[0]
        top = con.execute(
            f"SELECT {qident(c)} AS v, count(*) AS n FROM {src} GROUP BY 1 ORDER BY n DESC LIMIT 25"
        ).fetchall()
        out.append(f"Top values {c} (~{nd} distinct): " + "; ".join(f"{v}={n}" for v, n in top))
    return out


def labeler_extras(con, path: Path) -> list[str]:
    src = f"read_parquet('{sql_str(str(path))}')"
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    out = []
    if "val" in cols:
        neg = "neg" if "neg" in cols else "NULL"
        rows = con.execute(
            f"SELECT val, count(*) n, sum(CASE WHEN {neg} THEN 1 ELSE 0 END) n_neg FROM {src} GROUP BY 1 ORDER BY n DESC"
        ).fetchall()
        out.append("All label values (value: count, negations):")
        out += [f"    {v}: {n} (neg {nn})" for v, n, nn in rows]
    if "uri" in cols:
        rows = con.execute(
            f"""SELECT CASE WHEN uri LIKE 'did:%' THEN 'account (did)'
                            WHEN uri LIKE 'at://%' THEN 'record (at-uri)' ELSE 'other' END t, count(*)
                FROM {src} GROUP BY 1"""
        ).fetchall()
        out.append(f"Label targets: {dict(rows)}")
        if "val" in cols:
            rows = con.execute(
                f"SELECT val, count(DISTINCT uri) FROM {src} WHERE uri LIKE 'did:%' GROUP BY 1 ORDER BY 2 DESC LIMIT 40"
            ).fetchall()
            out.append("Distinct accounts per account-level label value: " + "; ".join(f"{v}={n}" for v, n in rows))
    return out


def _flat(obj, prefix="", depth=0, out=None):
    """Flatten nested dict keys up to depth 3 -> {dotted.key: value}."""
    out = {} if out is None else out
    if isinstance(obj, dict) and depth < 3:
        for k, v in obj.items():
            _flat(v, f"{prefix}{k}.", depth + 1, out)
    elif isinstance(obj, list) and obj and depth < 3:
        out[prefix[:-1] + "[]"] = f"<list len {len(obj)}>"
        _flat(obj[0], prefix[:-1] + "[0].", depth + 1, out)
    else:
        out[prefix[:-1]] = obj
    return out


def sample_json_gz(p: Path, n_lines: int) -> list[str]:
    """Read only the first n_lines of a .json.gz file and describe its structure."""
    out, keys, vals, firsts, n, bad = [], Counter(), defaultdict(Counter), [], 0, 0
    try:
        with gzip.open(p, "rt", errors="replace") as f:
            for line in f:
                n += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                if len(firsts) < 3:
                    firsts.append(line.strip()[:1500])
                for k, v in _flat(obj).items():
                    keys[k] += 1
                    if isinstance(v, (str, bool, int)):
                        if isinstance(v, str) and (len(v) > 60 or v.startswith(("did:", "at://", "baf"))):
                            continue
                        vals[k][str(v)] += 1
                if n >= n_lines:
                    break
    except Exception as e:
        return [f"BROKEN while reading first lines ({type(e).__name__}: {e})"]
    out.append(f"lines read: {n}, not-json: {bad}")
    out.append("First lines (cut to 1500 chars):")
    out += ["    " + x for x in firsts]
    out.append("Keys (flattened, depth<=3) with how many of the sampled lines have them:")
    out += [f"    {k}: {c}" for k, c in sorted(keys.items())]
    out.append("Low-cardinality fields (top values in sample):")
    for k, c in sorted(vals.items()):
        if 1 <= len(c) <= 60:
            out.append(f"    {k}: " + "; ".join(f"{v}={m}" for v, m in c.most_common(15)))
    tk = [k for k in keys if TIME_NAME.search(k.split(".")[-1])]
    return out + [f"time-like keys: {tk}"]


def gzip_test(paths: list[Path], workers: int, log) -> dict:
    """Full integrity check: 'gzip -t' on every file, in parallel."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        r = subprocess.run(["gzip", "-t", str(p)], capture_output=True, text=True)
        return p, r.returncode, r.stderr.strip()[:300]

    res, t0 = {}, time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, (p, rc, err) in enumerate(ex.map(one, paths)):
            res[p] = "OK" if rc == 0 else f"BROKEN: {err}"
            if (i + 1) % 10 == 0:
                log.info(f"  gzip -t {i+1}/{len(paths)} ({time.time()-t0:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--quick", action="store_true", help="skip full scans of snapshot files")
    ap.add_argument("--sample-day", help="firehose day to profile in depth, YYYY-MM-DD")
    ap.add_argument("--sample-lines", type=int, default=20000, help="lines to read from a json.gz day file")
    ap.add_argument("--check-gz", action="store_true", help="full gzip integrity test of every .json.gz (slow, parallel)")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = cfg["paths"]["raw"]
    out_dir = cfg["paths"]["results"] / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("inventory", cfg["paths"]["results"] / "logs" / "00_inventory.log")
    con = connect(cfg)
    R = ["# Data inventory", f"raw dir: {raw.resolve()}", ""]

    # 1. files
    log.info("walking raw dir")
    files = walk(raw)
    files.to_csv(out_dir / "files.csv", index=False)
    files["top"] = files.path.str.split("/").str[0]
    R.append("## 1. Files")
    for top, g in files.groupby("top"):
        R.append(f"- {top}: {len(g)} files, {human(int(g['size'].clip(lower=0).sum()))}, types {dict(Counter(g.ext))}")
    zero = files[files["size"] <= 0]
    if len(zero):
        R.append(f"- WARNING {len(zero)} empty/unreadable files, e.g. {list(zero.path[:5])}")

    # 2. parquet metadata / integrity
    log.info("reading parquet metadata")
    meta, broken = parquet_meta(raw, files, log)
    meta.to_csv(out_dir / "schemas.csv", index=False)
    broken.to_csv(out_dir / "broken.csv", index=False)
    R += ["", "## 2. Integrity", f"parquet files OK: {len(meta)}, broken: {len(broken)}"]
    R += [f"- BROKEN {r.path}: {r.error}" for r in broken.itertuples()][:50]

    # 3. snapshot + labelers, per file
    is_fh = meta.path.str.contains(r"(?:^|/)firehose/") & meta.path.str.contains(r"\d{4}/\d{2}/\d{2}")
    nonfh = meta[~is_fh]
    tops = sorted(set(nonfh.path.map(lambda p: "/".join(p.split("/")[:-1]) or ".")))
    for section in tops:
        m = nonfh[nonfh.path.map(lambda p: ("/".join(p.split("/")[:-1]) or ".") == section)]
        if m.empty:
            R += ["", f"## {section}: no parquet files found"]
            continue
        R += ["", f"## 3. {section}"]
        for r in m.sort_values("path").itertuples():
            log.info(f"profiling {r.path}")
            R += ["", f"### {r.path}  ({r.rows:,} rows, {r.row_groups} row groups)"]
            try:
                R += profile(con, raw / r.path, full_scan=("labelers" in section or not args.quick))
                if "labelers" in section:
                    R += labeler_extras(con, raw / r.path)
            except Exception as e:
                R.append(f"PROFILE FAILED: {type(e).__name__}: {e}")

    # 4. firehose
    fh = meta[is_fh].copy()
    R += ["", "## 4. Firehose"]
    if fh.empty:
        R.append("no firehose parquet found")
    else:
        fh["day"] = fh.path.map(lambda p: (lambda m: date(*map(int, m.groups())) if m else None)(DAY_RE.search(p)))
        days = sorted(d for d in fh.day.unique() if d)
        R.append(f"files: {len(fh)}, rows: {fh.rows.sum():,}, days with data: {len(days)}, from {days[0]} to {days[-1]}")
        per_month = Counter(d.strftime("%Y-%m") for d in days)
        R.append(f"days per month: {dict(sorted(per_month.items()))}")
        full = {days[0] + timedelta(i) for i in range((days[-1] - days[0]).days + 1)}
        missing = sorted(full - set(days))
        R.append(f"missing days in range: {[str(d) for d in missing] if missing else 'none'}")
        perday = fh.groupby("day").rows.sum()
        R.append(f"rows per day: median {int(perday.median()):,}, min {int(perday.min()):,} ({perday.idxmin()}), max {int(perday.max()):,} ({perday.idxmax()})")
        low = perday[perday < 0.5 * perday.median()]
        if len(low):
            R.append(f"days with < 50% of median rows (possible gaps): {[str(d) for d in low.index]}")
        sigs = fh.schema_sig.value_counts()
        R.append(f"distinct schemas across firehose files: {len(sigs)}")
        for i, (sig, n) in enumerate(sigs.items()):
            ex = fh[fh.schema_sig == sig].path.iloc[0]
            R.append(f"  schema {i+1} ({n} files, e.g. {ex}): {sig[:600]}")

        sd = date.fromisoformat(args.sample_day) if args.sample_day else next((d for d in days if d.month == 5 and d.day == 3), days[len(days) // 2])
        day_files = fh[fh.day == sd].path.tolist()
        R += ["", f"### Sample day {sd} ({len(day_files)} files) - in-depth profile"]
        if day_files:
            day_dir = (raw / day_files[0]).parent
            log.info(f"profiling firehose day {sd}")
            try:
                R += profile(con, day_dir / "*.parquet", full_scan=True, max_cat_cols=20)
            except Exception as e:
                R.append(f"PROFILE FAILED: {type(e).__name__}: {e}")

    # 5. json.gz files (firehose days are stored like this)
    jg = files[files.ext == ".json.gz"].copy()
    if len(jg):
        R += ["", "## 5. json.gz files (firehose)"]
        jg["day"] = jg.path.map(lambda p: (lambda m: date(*map(int, m.groups())) if m else None)(DAY_RE.search(p)))
        R.append(f"files: {len(jg)}, total size {human(int(jg['size'].sum()))}")
        for top, g in jg.groupby(jg.path.str.split("/").str[0]):
            d = sorted(x for x in g.day if x)
            R.append(f"- {top}: {len(g)} files, {human(int(g['size'].sum()))}" + (f", days {d[0]} -> {d[-1]}" if d else ""))
            if d:
                full = {d[0] + timedelta(i) for i in range((d[-1] - d[0]).days + 1)}
                miss = sorted(full - set(d))
                R.append(f"  missing days: {[str(x) for x in miss] if miss else 'none'}")
                dup = [str(k) for k, v in Counter(d).items() if v > 1]
                if dup:
                    R.append(f"  days with more than one file: {dup}")
        sz = jg[jg.day.notna()].groupby("day")["size"].sum()
        if len(sz):
            R.append(f"size per day: median {human(int(sz.median()))}, min {human(int(sz.min()))} ({sz.idxmin()}), max {human(int(sz.max()))} ({sz.idxmax()})")
            small = sz[sz < 0.5 * sz.median()]
            if len(small):
                R.append(f"days < 50% of median size (possible gaps): {[str(x) for x in small.index]}")
        R.append("all files: " + "; ".join(f"{r.path}={human(int(r.size))}" for r in jg.sort_values("path").itertuples()))

        main_days = jg[jg.day.notna() & ~jg.path.str.startswith("extra_")]
        sd = date.fromisoformat(args.sample_day) if args.sample_day else date(2026, 5, 3)
        pick = main_days[main_days.day == sd]
        if pick.empty and len(main_days):
            pick = main_days.iloc[[len(main_days) // 2]]
        for rel in list(pick.path) + list(jg[jg.path.str.startswith("extra_")].path):
            log.info(f"sampling {rel} (first {args.sample_lines} lines)")
            R += ["", f"### sample of {rel}"] + sample_json_gz(raw / rel, args.sample_lines)

        if args.check_gz:
            log.info(f"full gzip test of {len(jg)} files with {args.workers} workers")
            res = gzip_test([raw / p for p in jg.path], args.workers, log)
            bad = {str(p.relative_to(raw)): v for p, v in res.items() if v != "OK"}
            R += ["", f"### gzip integrity: {len(res) - len(bad)} OK, {len(bad)} broken"] + [f"- {k}: {v}" for k, v in bad.items()]
        else:
            R.append("(full gzip integrity test not run; use --check-gz)")

    (out_dir / "report.md").write_text("\n".join(R) + "\n")
    log.info(f"done -> {out_dir/'report.md'}")


if __name__ == "__main__":
    main()
