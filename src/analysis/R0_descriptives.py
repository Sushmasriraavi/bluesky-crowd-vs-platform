#!/usr/bin/env python3
"""R0 - data and platform-enforcement descriptives (results/R0/).

  T01_data_sources.csv           what data we have (rows, period) - Table 1 of the paper
  T02_outcomes.csv               takedown / review / suspension counts in the window
  F01_firehose_coverage          events per hour over Apr-May with collection gaps shaded
  F02_daily_platform_actions     new takedowns, first needs-review flags, suspensions per day
  F03_review_to_takedown_lag     how long from Bluesky's needs-review flag to takedown (ECDF)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str

GAPS = [("2026-04-16 07:00", "2026-04-16 10:00"), ("2026-05-17 15:28", "2026-05-18 00:00"),
        ("2026-05-18 23:00", "2026-05-19 01:00")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-07-01")
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, R = cfg["paths"]["interim"], cfg["paths"]["results"] / "R0"
    log = get_logger("R0", cfg["paths"]["results"] / "logs" / "R0.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    q = lambda s: con.execute(s).df()
    S, E = f"TIMESTAMPTZ '{args.start} 00:00:00+00'", f"TIMESTAMPTZ '{args.end} 00:00:00+00'"
    gaps = [(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")) for a, b in GAPS]
    O = rp(I / "labels" / "account_outcomes.parquet")

    # ---- T01 data sources
    rows = []
    for name, path, desc in [
        ("snapshot repos", "snapshot/repos.parquet", "accounts (repo id -> DID)"),
        ("snapshot blocks", "snapshot/blocks.parquet", "block records existing at crawl"),
        ("snapshot follows", "snapshot/follows.parquet", "follow records"),
        ("snapshot lists", "snapshot/lists.parquet", "lists (mod + curate)"),
        ("snapshot list items", "snapshot/listitems.parquet", "accounts placed on lists"),
        ("snapshot list subscriptions", "snapshot/listblocks.parquet", "users blocking via a mod list"),
        ("snapshot post history", "snapshot/post_history.parquet", "accounts with posts (2.4B posts summarised)"),
        ("firehose posts", "firehose/posts/*/*.parquet", "post creates, 2026-04-01..05-31"),
        ("firehose graph", "firehose/graph/*/*.parquet", "block/follow/list creates"),
        ("firehose likes+reposts", "firehose/interactions/*/*.parquet", "interactions"),
        ("firehose all operations", "firehose/ops/*/*.parquet", "every repo operation"),
        ("labels on accounts", "labels/account_label_events.parquet", "official labeler, 2023-04..2026-07-01"),
        ("labels on records", "labels/record_label_events.parquet", "official labeler, posts/profiles"),
    ]:
        try:
            n = con.execute(f"SELECT count(*) FROM {rp(I / path)}").fetchone()[0]
        except Exception as e:
            n = None
            log.warning(f"{name}: {e}")
        rows.append({"source": name, "rows": n, "content": desc})
    viz.table(pd.DataFrame(rows), R, "T01_data_sources")
    log.info("T01 written")

    # ---- T02 outcomes
    t = q(f"""SELECT
        count(*) FILTER (takedown_ts IS NOT NULL AND NOT takedown_active_at_start) AS new_takedowns,
        count(*) FILTER (takedown_active_at_start) AS already_taken_down_at_start,
        count(*) FILTER (takedown_ts IS NOT NULL AND NOT takedown_active_at_start AND takedown_reversed_ts IS NOT NULL) AS takedowns_reversed,
        count(*) FILTER (takedown_ts IS NOT NULL AND NOT takedown_active_at_start AND n_takedowns_before > 0) AS repeat_offenders,
        count(*) FILTER (needs_review_ts IS NOT NULL) AS flagged_needs_review,
        count(*) FILTER (needs_review_ts IS NOT NULL AND takedown_ts >= needs_review_ts) AS flagged_then_taken_down,
        count(*) FILTER (takedown_ts IS NOT NULL AND NOT takedown_active_at_start AND needs_review_before_takedown) AS takedowns_after_review_flag,
        count(*) FILTER (suspend_ts IS NOT NULL) AS suspended
      FROM {O}""").T.reset_index()
    t.columns = ["measure", "accounts"]
    viz.table(t, R, "T02_outcomes")

    # ---- F01 coverage
    h = q(f"""SELECT date_trunc('hour', t) AS hour, count(*) AS events
              FROM {rp(I / 'firehose' / 'ops' / '*' / '*.parquet')}
              WHERE t >= TIMESTAMPTZ '2026-04-01 00:00:00+00' AND t < TIMESTAMPTZ '2026-06-01 00:00:00+00'
              GROUP BY 1 ORDER BY 1""")
    viz.table(h, R, "F01_firehose_coverage")
    f, ax = viz.fig(8, 3.2)
    viz.shade_gaps(ax, gaps)
    ax.plot(h.hour, h.events / 1e6, color=viz.SERIES[0], lw=1.2, label="events per hour")
    ax.set_ylabel("events per hour (millions)")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_ylim(bottom=0)
    viz.dates(ax)
    ax.set_title("Firehose coverage, April-May 2026")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1.16), ncol=2)
    viz.save(f, R, "F01_firehose_coverage", "Grey: hours with no data (collector gaps). Times in UTC.")

    # ---- F02 daily platform actions
    d = q(f"""WITH x AS (
          SELECT CAST(takedown_ts AS DATE) AS "day", 'new takedowns' AS kind FROM {O} WHERE takedown_ts IS NOT NULL AND NOT takedown_active_at_start
          UNION ALL SELECT CAST(needs_review_ts AS DATE), 'first needs-review flag' FROM {O} WHERE needs_review_ts IS NOT NULL
          UNION ALL SELECT CAST(suspend_ts AS DATE), 'suspensions' FROM {O} WHERE suspend_ts IS NOT NULL)
        SELECT "day", kind, count(*) n FROM x WHERE "day" >= DATE '{args.start}' AND "day" < DATE '{args.end}' GROUP BY ALL ORDER BY "day" """)
    viz.table(d, R, "F02_daily_platform_actions")
    f, ax = viz.fig(8, 3.4)
    for i, k in enumerate(["new takedowns", "first needs-review flag", "suspensions"]):
        s = d[d.kind == k]
        ax.plot(pd.to_datetime(s.day), s.n, color=viz.SERIES[i], label=k)
    ax.set_ylabel("accounts per day")
    viz.dates(ax)
    ax.set_ylim(bottom=0)
    ax.set_title("Platform actions per day (official Bluesky labeler)")
    ax.legend(loc="upper left", ncol=3)
    viz.save(f, R, "F02_daily_platform_actions", "Account-level labels only; new takedowns exclude accounts already taken down on 1 April.")

    # ---- F03 review -> takedown lag
    lag = q(f"""SELECT epoch(takedown_ts - needs_review_ts) / 3600.0 AS hours FROM {O}
                WHERE needs_review_before_takedown AND NOT takedown_active_at_start""")
    qs = lag.hours.quantile([.1, .25, .5, .75, .9]).rename("hours").reset_index().rename(columns={"index": "quantile"})
    viz.table(qs, R, "F03_review_to_takedown_lag_quantiles")
    f, ax = viz.fig(6.5, 3.4)
    viz.ecdf(ax, lag.hours.clip(lower=1 / 60), viz.SERIES[0], "taken down after a needs-review flag", logx=True)
    med = lag.hours.median()
    ax.axvline(med, color=viz.MUTED, lw=1, ls="--")
    ax.text(med, 0.05, f"  median {med:.1f} h", color=viz.INK2, fontsize=9)
    ax.set_xlabel("hours from first needs-review flag to takedown (log scale)")
    ax.set_title("Once Bluesky flags an account, takedown follows fast")
    ax.legend(loc="upper left")
    viz.save(f, R, "F03_review_to_takedown_lag")
    log.info(f"R0 done -> {R}")


if __name__ == "__main__":
    main()
