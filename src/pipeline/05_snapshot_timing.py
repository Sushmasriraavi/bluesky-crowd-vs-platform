#!/usr/bin/env python3
"""Step 05 - when was each repo crawled, and how complete is the snapshot over time? (snapshot only)

Every snapshot record carries at_rev = the repo revision at crawl time. A revision is a TID whose first
53 bits are a microsecond timestamp, so decoding it gives the time of the repo's LAST COMMIT before the
crawl (a lower bound on the crawl time). Knowing this per account tells us up to when its records are
complete, which every snapshot time window depends on.

Outputs
  data/interim/snapshot/repo_rev.parquet   repo_id, did, at_rev, last_commit_ts
  results/S0/F20_last_commit_time           distribution of last-commit times (shows the crawl period)
  results/S0/F21_records_per_month          blocks / follows / list items created per month, 2023-2026
  results/S0/T20_snapshot_timing.csv        key dates (percentiles of last commit, max createdAt per table)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow as pa

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str

B32 = "234567abcdefghijklmnopqrstuvwxyz"


def tid_to_us(revs: pa.Array) -> np.ndarray:
    """Vectorised TID decode -> microseconds since epoch (NaN for malformed)."""
    s = pd.Series(revs.to_pylist(), dtype="object")
    ok = s.str.len().eq(13).fillna(False).to_numpy()
    out = np.full(len(s), np.nan)
    if ok.any():
        b = np.frombuffer("".join(s[ok]).encode("ascii"), dtype=np.uint8).reshape(-1, 13)
        lut = np.full(256, 255, dtype=np.uint8)
        for i, c in enumerate(B32):
            lut[ord(c)] = i
        idx = lut[b].astype(np.uint64)
        good = (idx != 255).all(axis=1) & (idx[:, 0] < 16)
        val = np.zeros(len(idx), dtype=np.uint64)
        for i in range(13):
            val = (val << np.uint64(5)) | idx[:, i]
        us = (val >> np.uint64(10)).astype(np.float64)
        us[~good] = np.nan
        out[ok] = us
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, R = cfg["paths"]["interim"], cfg["paths"]["results"] / "S0"
    log = get_logger("05", cfg["paths"]["results"] / "logs" / "05_snapshot_timing.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    fixed = I / "snapshot_fixed"

    # one revision per repo: all records of a repo share the rev of the crawl
    srcs = [p for p in ["block.parquet", "follow.parquet", "listitem.parquet", "listblock.parquet",
                        "app.bsky.graph.list.parquet", "post.parquet"] if (fixed / p).exists()]
    log.info(f"reading at_rev from {srcs}")
    union = " UNION ALL ".join(f"SELECT repo_id, max(at_rev) AS at_rev FROM {rp(fixed / p)} GROUP BY repo_id" for p in srcs)
    res = con.execute(f"""SELECT r.repo_id, r.did, x.at_rev
        FROM (SELECT repo_id, max(at_rev) AS at_rev FROM ({union}) GROUP BY repo_id) x
        JOIN {rp(I / 'snapshot' / 'repos.parquet')} r USING (repo_id)""")
    for m in ("to_arrow_table", "fetch_arrow_table", "arrow"):  # name differs across DuckDB versions
        if hasattr(res, m):
            rev = getattr(res, m)()
            break
    rev = rev.read_all() if hasattr(rev, "read_all") else rev
    us = tid_to_us(rev.column("at_rev"))
    ts = pd.to_datetime(us, unit="us", utc=True)
    rev = rev.append_column("last_commit_ts", pa.array(ts, type=pa.timestamp("us", tz="UTC")))
    con.register("rev_tbl", rev)
    dst = I / "snapshot" / "repo_rev.parquet"
    con.execute(f"COPY (SELECT * FROM rev_tbl) TO '{sql_str(str(dst))}' (FORMAT parquet, COMPRESSION zstd)")
    log.info(f"repo_rev: {rev.num_rows:,} repos, {np.isnan(us).sum():,} undecodable revs")

    # ---- T20 key dates
    lc = pd.Series(ts).dropna()
    rows = [{"measure": f"last commit p{int(q*100)}", "value": str(lc.quantile(q))} for q in [.01, .1, .5, .9, .99]]
    rows.append({"measure": "last commit max", "value": str(lc.max())})
    for t in ["blocks", "follows", "listitems", "listblocks"]:
        mx = con.execute(f"""SELECT max(created_at) FROM {rp(I / 'snapshot' / f'{t}.parquet')}
                             WHERE created_at < TIMESTAMPTZ '2026-07-01 00:00:00+00'""").fetchone()[0]
        rows.append({"measure": f"max createdAt {t}", "value": str(mx)})
    viz.table(pd.DataFrame(rows), R, "T20_snapshot_timing")

    # ---- F20 last-commit time (only the recent tail is interesting: it shows the crawl)
    recent = lc[lc >= pd.Timestamp("2025-12-01", tz="UTC")]
    daily = recent.dt.floor("D").value_counts().sort_index().rename_axis("day").reset_index(name="repos")
    viz.table(daily, R, "F20_last_commit_time")
    f, ax = viz.fig(7.5, 3.3)
    ax.bar(daily.day, daily.repos, width=0.9, color=viz.SERIES[0])
    viz.dates(ax)
    ax.set_ylabel("repos (last commit that day)")
    ax.set_title("When each repo last changed before it was crawled")
    viz.save(f, R, "F20_last_commit_time",
             f"{len(lc):,} repos. The right edge marks the crawl; older dates are inactive accounts. Earlier dates not shown.")

    # ---- F21 records created per month (completeness over time)
    m = con.execute(f"""
        SELECT 'blocks' kind, date_trunc('month', created_at) AS mon, count(*) n FROM {rp(I / 'snapshot' / 'blocks.parquet')} GROUP BY ALL
        UNION ALL SELECT 'follows', date_trunc('month', created_at), count(*) FROM {rp(I / 'snapshot' / 'follows.parquet')} GROUP BY ALL
        UNION ALL SELECT 'list items', date_trunc('month', created_at), count(*) FROM {rp(I / 'snapshot' / 'listitems.parquet')} GROUP BY ALL""").df()
    m = m[(m.mon >= pd.Timestamp("2023-02-01", tz="UTC")) & (m.mon < pd.Timestamp("2026-07-01", tz="UTC"))].sort_values("mon")
    viz.table(m, R, "F21_records_per_month")
    f, ax = viz.fig(7.5, 3.5)
    for i, k in enumerate(["follows", "blocks", "list items"]):
        s = m[m.kind == k]
        ax.plot(s.mon, s.n, color=viz.SERIES[i], label=k, marker="o", ms=3)
    ax.set_yscale("log")
    viz.dates(ax)
    ax.set_ylabel("records created per month (log)")
    ax.set_title("Snapshot records by creation month")
    ax.legend(loc="lower right", ncol=3)
    viz.save(f, R, "F21_records_per_month", "Records still present at crawl time; deleted records (unblocks, unfollows) are not in the snapshot.")
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
