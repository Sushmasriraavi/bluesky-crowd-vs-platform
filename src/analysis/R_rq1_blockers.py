#!/usr/bin/env python3
"""RQ1: WHO is the crowd? Ordinary blockers vs heavy (bulk) blockers vs moderation lists, and a burst measure
that is relative to the account's visibility.  (results/RQ1/blockers/)

Heavy blocker = an account that had already made more than --heavy blocks before T0 (default 1,000; this small group
makes a large share of all blocks). Everyone else = ordinary blocker.
Crowd signals in (index, index + 30 d]:
  burst_all          5+ distinct new blockers (the RQ1 definition)
  burst_ordinary     5+ ordinary blockers
  burst_heavy_only   5+ blockers in total but fewer than 5 ordinary ones (the burst exists only because of bulk blockers)
  modlist            added to at least one moderation list
  modlist_only       added to a moderation list without any burst
  burst_relative     (established) blockers in the top 5% of the account's own follower band (and 2+):
                     "blocked far more than accounts of the same size"
For each signal x platform action: P(action | signal), P(action | no signal), lift (95% CI) and the share of the
platform's actions that the signal covers.

Outputs
  T80_blocker_mix_<cohort>       how much of the blocking comes from heavy blockers
  T81_signal_lift_<cohort>       lift + coverage for every crowd signal x platform action
  F80_signal_lift_<cohort>       ordinary burst vs heavy-only burst vs moderation list, per action
  T82_relative_burst             established: relative burst vs raw burst per follower band
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str
from bskymod.outcomes import ACTIONS, ORDER, flags_sql

FBANDS = [(0, 9, "0-9"), (10, 99, "10-99"), (100, 999, "100-999"), (1000, 9999, "1k-10k"), (10000, 10**12, "10k+")]
SIGNALS = ["burst_all", "burst_ordinary", "burst_heavy_only", "modlist", "modlist_only", "burst_relative"]


def lift(y, s):
    a, n1, c, n0 = int((y & s).sum()), int(s.sum()), int((y & ~s).sum()), int((~s).sum())
    out = {"accounts_signal": n1, "actioned_signal": a, "actioned_no_signal": c,
           "p_action_given_signal": a / n1 if n1 else np.nan, "p_action_given_no_signal": c / n0 if n0 else np.nan,
           "share_of_actions_covered": a / (a + c) if a + c else np.nan,
           "lift": np.nan, "lift_low": np.nan, "lift_high": np.nan}
    if n1 and n0 and a >= 3:
        aa, cc = (a, c) if c else (a + 0.5, c + 0.5)
        rr = (aa / n1) / (cc / n0)
        se = np.sqrt(1 / aa - 1 / n1 + 1 / cc - 1 / n0)
        out.update(lift=rr, lift_low=rr * np.exp(-1.96 * se), lift_high=rr * np.exp(1.96 * se))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--t0", default="2026-04-09")
    ap.add_argument("--fh-end", default="2026-06-01")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--heavy", type=int, default=1000, help="blocks made before T0 to count as a heavy blocker")
    ap.add_argument("--cohorts", nargs="*", default=["established", "new"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    R = cfg["paths"]["results"] / "RQ1" / "blockers"
    log = get_logger("RQ1blk", cfg["paths"]["results"] / "logs" / "R_rq1_blockers.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    H = "INTERVAL 30 DAY"
    T0 = f"TIMESTAMPTZ '{args.t0} 00:00:00+00'"
    FHE = f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    L, C = I / "labels", I / "crowd"
    ae, re_ = rp(L / "account_label_events.parquet"), rp(L / "record_label_events.parquet")
    be, li, la = rp(C / "block_edges.parquet"), rp(C / "listitem_edges.parquet"), rp(C / "lists_all.parquet")
    pd.set_option("display.width", 250)

    # heavy blockers: blocks made before T0 (snapshot blocks without a usable time count as "before")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE heavy AS
        SELECT blocker_did AS did, count(*) AS blocks_before_t0 FROM {be}
        WHERE coalesce(t_created, {T0} - INTERVAL 1 DAY) < {T0}
        GROUP BY 1 HAVING count(*) > {args.heavy}""")
    nh = con.execute("SELECT count(*), coalesce(sum(blocks_before_t0), 0) FROM heavy").fetchone()
    tot = con.execute(f"SELECT count(*) FROM {be} WHERE coalesce(t_created, {T0} - INTERVAL 1 DAY) < {T0}").fetchone()[0]
    log.info(f"heavy blockers (> {args.heavy} blocks before T0): {nh[0]:,} accounts, {nh[1]:,} of {tot:,} blocks ({nh[1] / max(tot, 1):.1%})")

    base = {"established": f"SELECT did, {T0} AS index_t, coalesce(net_followers, 0) AS followers "
                           f"FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d",
            "new": f"SELECT did, index_t, NULL::BIGINT AS followers FROM {rp(P / 'accounts_new.parquet')} "
                   f"WHERE NOT instant_takedown AND index_t + {H} <= {FHE}"}
    mix_rows = []
    for cohort in args.cohorts:
        con.execute(f"CREATE OR REPLACE TEMP TABLE c AS {base[cohort]}")
        df = con.execute(f"""
            WITH b AS (SELECT c.did, count(DISTINCT e.blocker_did) AS n_all,
                              count(DISTINCT e.blocker_did) FILTER (h.did IS NULL) AS n_ord,
                              count(DISTINCT e.blocker_did) FILTER (h.did IS NOT NULL) AS n_heavy
                       FROM c JOIN {be} e ON e.subject_did = c.did AND e.t_created > c.index_t AND e.t_created <= c.index_t + {H}
                       LEFT JOIN heavy h ON h.did = e.blocker_did GROUP BY 1),
                 m AS (SELECT c.did, count(DISTINCT i.list_uri) AS n_modlists FROM c JOIN {li} i ON i.subject_did = c.did
                       AND i.t_added > c.index_t AND i.t_added <= c.index_t + {H}
                       JOIN {la} l ON l.list_uri = i.list_uri AND l.purpose LIKE '%modlist' GROUP BY 1)
            SELECT c.did, c.followers, coalesce(b.n_all, 0) n_all, coalesce(b.n_ord, 0) n_ord, coalesce(b.n_heavy, 0) n_heavy,
                   coalesce(m.n_modlists, 0) n_modlists
            FROM c LEFT JOIN b USING (did) LEFT JOIN m USING (did)""").df()
        df = df.merge(con.execute(flags_sql("SELECT did, index_t FROM c", ae, re_, f"c.index_t + {H}")).df(), on="did")
        k = args.k
        df["burst_all"] = df.n_all >= k
        df["burst_ordinary"] = df.n_ord >= k
        df["burst_heavy_only"] = (df.n_all >= k) & (df.n_ord < k)
        df["modlist"] = df.n_modlists > 0
        df["modlist_only"] = df.modlist & (df.n_all < k)
        sigs = SIGNALS[:5]
        if cohort == "established":
            df["band"] = pd.cut(df.followers, [-1] + [b[1] for b in FBANDS], labels=[b[2] for b in FBANDS])
            thr = df.groupby("band", observed=True).n_all.quantile(0.95)
            df["band_p95"] = df.band.map(thr).astype(float)
            df["burst_relative"] = (df.n_all > df.band_p95) & (df.n_all >= 2)
            sigs = SIGNALS
            rel = []
            for lab in [b[2] for b in FBANDS]:
                d = df[df.band == lab]
                for a in ["abuse", "review", "takedown"]:
                    for s in ["burst_all", "burst_relative"]:
                        rel.append({"followers": lab, "p95_blockers_in_band": thr.get(lab), "action": a, "signal": s,
                                    **lift(d[f"act_{a}"], d[s])})
            viz.table(pd.DataFrame(rel), R, "T82_relative_burst")

        blocked = df[df.n_all > 0]
        mix_rows.append({"cohort": cohort, "accounts": len(df), "accounts_blocked_30d": len(blocked),
                         "blocks_from_heavy_share": df.n_heavy.sum() / max(df.n_all.sum(), 1),
                         "burst_all": int(df.burst_all.sum()), "burst_ordinary": int(df.burst_ordinary.sum()),
                         "burst_heavy_only": int(df.burst_heavy_only.sum()), "modlist": int(df.modlist.sum()),
                         "modlist_only": int(df.modlist_only.sum()),
                         "burst_relative": int(df.burst_relative.sum()) if "burst_relative" in df else np.nan,
                         "heavy_blockers": nh[0], "heavy_threshold": args.heavy})
        rows = [{"signal": s, "action": a, "description": ACTIONS[a][0], **lift(df[f"act_{a}"], df[s])}
                for s in sigs for a in ORDER]
        t = pd.DataFrame(rows)
        viz.table(t, R, f"T81_signal_lift_{cohort}")
        log.info(f"[{cohort}] mix {mix_rows[-1]}")
        log.info(f"[{cohort}] lift / coverage\n" + t.pivot(index="action", columns="signal", values="lift")
                 .reindex(ORDER).round(1).to_string() + "\n" +
                 t.pivot(index="action", columns="signal", values="share_of_actions_covered").reindex(ORDER).round(3).to_string())

        order = ORDER[::-1]
        f, (a1, a2) = viz.plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
        y = np.arange(len(order))
        for i, (s, col, lab) in enumerate([("burst_ordinary", viz.SERIES[0], "burst from ordinary blockers"),
                                           ("burst_heavy_only", viz.SERIES[1], "burst only from heavy blockers"),
                                           ("modlist_only", viz.SERIES[2], "moderation list, no burst")]):
            d = t[t.signal == s].set_index("action").reindex(order)
            yy = y + 0.22 - 0.22 * i
            a1.errorbar(d.lift, yy, xerr=[(d.lift - d.lift_low).clip(lower=0), (d.lift_high - d.lift).clip(lower=0)],
                        fmt="o", ms=5, capsize=2, color=col, label=lab)
            a2.barh(yy, d.p_action_given_signal * 100, height=0.2, color=col)
        a1.set_xscale("log")
        a1.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}"))
        a1.axvline(1, color=viz.MUTED, lw=1, ls=":")
        a1.set_yticks(y, [ACTIONS[a][0] for a in order], fontsize=8)
        a1.set_xlabel("lift (log scale)")
        a1.set_title("Which crowd signal tracks each platform action?", fontsize=10)
        a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
        a2.set_xscale("symlog", linthresh=1)
        a2.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}"))
        a2.set_xlabel("% of signalled accounts that got this action")
        a2.set_title("Precision of each signal", fontsize=10)
        for ax in (a1, a2):
            ax.grid(axis="y", visible=False)
        viz.save(f, R, f"F80_signal_lift_{cohort}", f"{cohort} accounts, 30 days. Heavy blocker = more than {args.heavy:,} "
                 "blocks made before 9 April 2026. Whiskers: 95% CI.")
    viz.table(pd.DataFrame(mix_rows), R, "T80_blocker_mix")
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
