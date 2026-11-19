#!/usr/bin/env python3
"""Probe: what does the official labeler do besides takedowns? (results/RQ1/probe/)

Quick look before redesigning the RQ1 platform outcome. Window = T0 .. labels end (2026-07-01).
  P1_account_labels.csv   account-level label values: events, accounts (non-negated)
  P2_post_labels.csv      record-level (post/profile) label values: events, accounts
  P3_takedown_reasons.csv for accounts newly taken down in the window: share with each reason label (30 d before .. 1 d after)
  P4_group_actions.csv    established cohort, 4 crowd x takedown groups (k=5, 30 d): share of accounts that got each
                          account label / post label within 30 days and by labels end
"""
from __future__ import annotations

import argparse

import pandas as pd

from bskymod.common import connect, get_logger, load_config, sql_str

GROUPS = ["1 blocked + taken down", "2 blocked only (crowd)", "3 taken down only (platform)", "4 neither"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--t0", default="2026-04-09")
    ap.add_argument("--end", default="2026-07-01")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    R = cfg["paths"]["results"] / "RQ1" / "probe"
    R.mkdir(parents=True, exist_ok=True)
    log = get_logger("D1", cfg["paths"]["results"] / "logs" / "D1_probe.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    T0, END = f"TIMESTAMPTZ '{args.t0} 00:00:00+00'", f"TIMESTAMPTZ '{args.end} 00:00:00+00'"
    L = I / "labels"
    ae, re_ = rp(L / "account_label_events.parquet"), rp(L / "record_label_events.parquet")

    p1 = con.execute(f"""SELECT val, count(*) AS events, count(DISTINCT did) AS accounts FROM {ae}
                         WHERE NOT neg AND ts >= {T0} AND ts < {END} GROUP BY 1 ORDER BY accounts DESC""").df()
    p1.to_csv(R / "P1_account_labels.csv", index=False)
    p2 = con.execute(f"""SELECT val, collection, count(*) AS events, count(DISTINCT did) AS accounts FROM {re_}
                         WHERE NOT neg AND ts >= {T0} AND ts < {END} GROUP BY 1, 2 ORDER BY accounts DESC""").df()
    p2.to_csv(R / "P2_post_labels.csv", index=False)

    p3 = con.execute(f"""
        WITH td AS (SELECT did, reason_labels FROM {rp(L / 'account_outcomes.parquet')}
                    WHERE takedown_ts >= {T0} AND takedown_ts < {END} AND NOT takedown_active_at_start),
             n AS (SELECT count(*) AS n, count(*) FILTER (len(coalesce(reason_labels, [])) > 0) AS with_reason FROM td)
        SELECT 'ANY reason label' AS reason, with_reason AS accounts, with_reason / n AS share FROM n
        UNION ALL
        SELECT r AS reason, count(*) AS accounts, count(*) / any_value(n.n) AS share
        FROM td, n, unnest(td.reason_labels) AS u(r) GROUP BY r ORDER BY accounts DESC""").df()
    p3.to_csv(R / "P3_takedown_reasons.csv", index=False)

    con.execute(f"""CREATE TEMP TABLE g AS
        WITH a AS (SELECT did, y_takedown_ts FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d),
             b AS (SELECT e.subject_did AS did, count(DISTINCT e.blocker_did) AS n
                   FROM {rp(I / 'crowd' / 'block_edges.parquet')} e SEMI JOIN a ON a.did = e.subject_did
                   WHERE e.t_created > {T0} AND e.t_created <= {T0} + INTERVAL 30 DAY GROUP BY 1)
        SELECT a.did, coalesce(b.n, 0) >= {args.k} AS bl,
               coalesce(a.y_takedown_ts <= {T0} + INTERVAL 30 DAY, false) AS td
        FROM a LEFT JOIN b USING (did)""")
    grp = con.execute("""SELECT CASE WHEN bl AND td THEN 1 WHEN bl THEN 2 WHEN td THEN 3 ELSE 4 END AS gi,
                                count(*) AS n FROM g GROUP BY 1""").df().set_index("gi").n
    rows = []
    for level, src in [("account", ae), ("post", re_)]:
        for horizon, until in [("30d", f"{T0} + INTERVAL 30 DAY"), ("to_labels_end", END)]:
            d = con.execute(f"""
                SELECT CASE WHEN g.bl AND g.td THEN 1 WHEN g.bl THEN 2 WHEN g.td THEN 3 ELSE 4 END AS gi,
                       l.val, count(DISTINCT l.did) AS accounts
                FROM g JOIN {src} l ON l.did = g.did
                WHERE NOT l.neg AND l.ts > {T0} AND l.ts <= {until} GROUP BY ALL""").df()
            anyv = con.execute(f"""
                SELECT CASE WHEN g.bl AND g.td THEN 1 WHEN g.bl THEN 2 WHEN g.td THEN 3 ELSE 4 END AS gi,
                       'ANY non-takedown label' AS val, count(DISTINCT l.did) AS accounts
                FROM g JOIN {src} l ON l.did = g.did
                WHERE NOT l.neg AND l.ts > {T0} AND l.ts <= {until} AND l.val <> '!takedown' GROUP BY ALL""").df()
            d = pd.concat([d, anyv])
            d["level"], d["horizon"] = level, horizon
            rows.append(d)
    t = pd.concat(rows)
    t["group"] = t.gi.map(dict(enumerate(GROUPS, 1)))
    t["group_size"] = t.gi.map(grp)
    t["share"] = t.accounts / t.group_size
    t = t.pivot_table(index=["level", "horizon", "val"], columns="group", values="share").fillna(0).reset_index()
    t = t.sort_values(["level", "horizon", GROUPS[1]], ascending=[True, True, False])
    t.to_csv(R / "P4_group_actions.csv", index=False)

    pd.set_option("display.width", 220)
    log.info(f"group sizes: { {GROUPS[i - 1]: int(v) for i, v in grp.items()} }")
    log.info(f"P1 account labels\n{p1.head(30).to_string()}")
    log.info(f"P2 post labels\n{p2.head(30).to_string()}")
    log.info(f"P3 takedown reasons\n{p3.head(20).to_string()}")
    log.info(f"P4 (30d, account level)\n{t[(t.level == 'account') & (t.horizon == '30d')].head(20).to_string()}")
    log.info(f"P4 (30d, post level)\n{t[(t.level == 'post') & (t.horizon == '30d')].head(20).to_string()}")
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
