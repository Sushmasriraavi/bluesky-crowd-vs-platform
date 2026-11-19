#!/usr/bin/env python3
"""RQ2 extensions (results/RQ2/extra/): the trade-offs and dynamics behind the crowd early warning.

1. Speed vs accuracy: alert after k = 1, 2, 3, 5, 10, 20 ordinary blockers. For each platform action:
   precision (share actioned within 30 days after the alert), coverage (share of the platform's actions that the
   alert came before) and median lead time. Fewer blockers = earlier but noisier.
2. Event study: new blockers per account per day, from 14 days before to 14 days after the first platform action,
   ordinary and heavy blockers separately. Rising before day 0 = the crowd anticipates the platform;
   rising after day 0 = the crowd reacts to the platform's label.
3. Review queue with an EXACT random baseline: for each day, the share of open (not yet actioned) cohort accounts
   that get the action in the next 30 days, against the crowd-ranked queue (top N by ordinary blockers, last 7 days).
4. Lead time by account size (established): crowd-first share and median lead time per follower band.

Outputs: T96_k_tradeoff_<cohort>, F96_k_tradeoff, T97_event_<cohort>, F97_event_<cohort>, T98_queue_exact_<cohort>,
         F98_queue_exact, T99_lead_by_followers, F99_lead_by_followers
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str
from bskymod.outcomes import ACTIONS, events_sql

warnings.filterwarnings("ignore", category=DeprecationWarning)
KS = [1, 2, 3, 5, 10, 20]
ACTS = ["abuse", "suspend", "takedown", "review"]
MAIN = ["abuse", "suspend", "takedown"]
QUEUE_N = [100, 1000, 10000]
FBANDS = [(0, 99, "<100"), (100, 999, "100-999"), (1000, 10**12, "1k+")]
EW = 14   # event window in days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--t0", default="2026-04-09")
    ap.add_argument("--fh-end", default="2026-06-01")
    ap.add_argument("--heavy", type=int, default=1000)
    ap.add_argument("--k-main", type=int, default=5)
    ap.add_argument("--queue-window", type=int, default=7)
    ap.add_argument("--cohorts", nargs="*", default=["established", "new"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    R = cfg["paths"]["results"] / "RQ2" / "extra"
    log = get_logger("13", cfg["paths"]["results"] / "logs" / "13_rq2_extra.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    T0 = f"TIMESTAMPTZ '{args.t0} 00:00:00+00'"
    FHE = f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    END = f"({FHE} + INTERVAL 30 DAY)"
    L, C = I / "labels", I / "crowd"
    ae, re_ = rp(L / "account_label_events.parquet"), rp(L / "record_label_events.parquet")
    be = rp(C / "block_edges.parquet")
    t0, fhe = pd.Timestamp(args.t0, tz="UTC"), pd.Timestamp(args.fh_end, tz="UTC")
    day = pd.Timedelta(days=1)
    pd.set_option("display.width", 250)

    con.execute(f"""CREATE OR REPLACE TEMP TABLE heavy AS
        SELECT blocker_did AS did FROM {be} WHERE coalesce(t_created, {T0} - INTERVAL 1 DAY) < {T0}
        GROUP BY 1 HAVING count(*) > {args.heavy}""")
    base = {
        "established": f"SELECT did, {T0} AS start_t, coalesce(net_followers, 0) AS followers "
                       f"FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d",
        "new": f"SELECT did, first_seen AS start_t, NULL::BIGINT AS followers FROM {rp(P / 'accounts_new.parquet')} "
               f"WHERE NOT instant_takedown AND first_seen < {FHE}"}
    trade_all, queue_all = [], []

    for cohort in args.cohorts:
        con.execute(f"CREATE OR REPLACE TEMP TABLE c AS {base[cohort]}")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE bl AS
            SELECT e.subject_did AS did, e.blocker_did, min(e.t_created) AS t, h.did IS NOT NULL AS is_heavy
            FROM c JOIN {be} e ON e.subject_did = c.did AND e.t_created > c.start_t AND e.t_created < {FHE}
            LEFT JOIN heavy h ON h.did = e.blocker_did GROUP BY e.subject_did, e.blocker_did, h.did""")
        kcols = ", ".join(f"min(t) FILTER (rn = {k}) AS alert_k{k}" for k in KS)
        aggs = ", ".join(f"min(e.ts) FILTER ({ACTIONS[a][1]}) AS ts_{a}" for a in ACTS)
        df = con.execute(f"""
            WITH o AS (SELECT did, t, row_number() OVER (PARTITION BY did ORDER BY t, blocker_did) rn FROM bl WHERE NOT is_heavy),
                 k AS (SELECT did, {kcols}, min(t) AS first_ord FROM o GROUP BY did),
                 e AS (SELECT e.did, e.val, e.lvl, e.ts FROM {events_sql(ae, re_)} e
                       JOIN c ON c.did = e.did AND e.ts > c.start_t AND e.ts <= {END}),
                 f AS (SELECT did, {aggs} FROM e GROUP BY did)
            SELECT c.*, k.* EXCLUDE (did), f.* EXCLUDE (did)
            FROM c LEFT JOIN k USING (did) LEFT JOIN f USING (did)""").df()
        log.info(f"[{cohort}] {len(df):,} accounts; alerts per k: "
                 + ", ".join(f"k={k}: {df[f'alert_k{k}'].notna().sum():,}" for k in KS))

        # ---------- 1. speed vs accuracy
        rows = []
        for a in ACTS:
            ts = df[f"ts_{a}"]
            inwin = ts.notna() & (ts < fhe)
            for k in KS:
                al = df[f"alert_k{k}"]
                # precision: alerted, not yet actioned at alert, actioned within 30 days
                open_ = al.notna() & (ts.isna() | (ts > al))
                hit = open_ & ts.notna() & (ts <= al + 30 * day)
                # coverage + lead: platform actions in window that the alert came before
                cf = inwin & al.notna() & (al < ts)
                lead = (ts[cf] - al[cf]).dt.total_seconds() / 86400
                rows.append({"cohort": cohort, "action": a, "k": k, "alerts": int(al.notna().sum()),
                             "precision_30d": hit.sum() / max(open_.sum(), 1),
                             "coverage_crowd_first": cf.sum() / max(inwin.sum(), 1),
                             "lead_median_days": lead.median(), "lead_q25": lead.quantile(.25), "lead_q75": lead.quantile(.75),
                             "actions_in_window": int(inwin.sum())})
        tr = pd.DataFrame(rows)
        viz.table(tr, R, f"T96_k_tradeoff_{cohort}")
        trade_all.append(tr)
        log.info(f"[{cohort}] speed vs accuracy\n{tr.round(4).to_string()}")

        # ---------- 2. event study around the first platform action
        con.register("acts", df[["did"] + [f"ts_{a}" for a in ACTS]])
        ev_rows = []
        for a in ACTS:
            n_acc = con.execute(f"""SELECT count(*) FROM acts WHERE ts_{a} >= {T0} + INTERVAL {EW} DAY
                                    AND ts_{a} < {FHE} - INTERVAL {EW} DAY""").fetchone()[0]
            if n_acc < 30:
                continue
            d = con.execute(f"""
                WITH x AS (SELECT did, ts_{a} AS ts FROM acts
                           WHERE ts_{a} >= {T0} + INTERVAL {EW} DAY AND ts_{a} < {FHE} - INTERVAL {EW} DAY)
                SELECT CAST(floor(date_diff('second', x.ts, b.t) / 86400.0) AS INT) AS rel_day, b.is_heavy,
                       count(*) AS blocks
                FROM x JOIN bl b ON b.did = x.did
                 AND b.t >= x.ts - INTERVAL {EW} DAY AND b.t < x.ts + INTERVAL {EW} DAY
                GROUP BY ALL""").df()
            for h in [False, True]:
                s = d[d.is_heavy == h].set_index("rel_day").blocks.reindex(range(-EW, EW), fill_value=0)
                for rd, v in s.items():
                    ev_rows.append({"action": a, "blocker": "heavy" if h else "ordinary", "rel_day": rd,
                                    "blocks_per_account": v / n_acc, "accounts": n_acc})
        ev = pd.DataFrame(ev_rows)
        if len(ev):
            viz.table(ev, R, f"T97_event_{cohort}")
            acts = [a for a in ACTS if a in set(ev.action)]
            f, axes = viz.plt.subplots(1, len(acts), figsize=(3.2 * len(acts), 3.4), squeeze=False)
            for ax, a in zip(axes.flat, acts):
                for bt, colr in [("ordinary", viz.SERIES[0]), ("heavy", viz.SERIES[1])]:
                    s = ev[(ev.action == a) & (ev.blocker == bt)]
                    ax.plot(s.rel_day, s.blocks_per_account, color=colr, marker="o", ms=2.5, label=f"{bt} blockers")
                ax.axvline(0, color=viz.MUTED, lw=1, ls=":")
                n = int(ev[ev.action == a].accounts.iloc[0])
                ax.set_title(f"{ACTIONS[a][0].split(' (')[0]} (n={n:,})", fontsize=9)
                ax.set_xlabel("days from platform action", fontsize=8)
                ax.tick_params(labelsize=8)
            axes.flat[0].set_ylabel("new blockers per account per day", fontsize=8)
            axes.flat[0].legend(fontsize=7)
            viz.save(f, R, f"F97_event_{cohort}", f"{cohort} accounts whose first action fell between {EW} days after the start "
                     f"and {EW} days before {args.fh_end}. Before 0 = anticipation, after 0 = reaction.")
            pre = ev[ev.rel_day < 0].groupby(["action", "blocker"]).blocks_per_account.sum()
            post = ev[ev.rel_day >= 0].groupby(["action", "blocker"]).blocks_per_account.sum()
            log.info(f"[{cohort}] event study: blockers per account in the {EW} days before / after\n"
                     + pd.DataFrame({"before": pre, "after": post}).round(3).to_string())

        # ---------- 3. review queue with an exact random baseline
        W = args.queue_window
        q = con.execute(f"""
            WITH days AS (SELECT unnest(generate_series(CAST(date_trunc('day', {T0}) AS TIMESTAMPTZ) + INTERVAL {W} DAY,
                                                         {FHE} - INTERVAL 1 DAY, INTERVAL 1 DAY)) AS day),
                 sc AS (SELECT d.day, b.did, count(*) AS score FROM days d JOIN bl b
                        ON NOT b.is_heavy AND b.t >= d.day - INTERVAL {W} DAY AND b.t < d.day GROUP BY 1, 2),
                 rk AS (SELECT *, row_number() OVER (PARTITION BY day ORDER BY score DESC, hash(did)) AS rn FROM sc)
            SELECT day, did, score FROM rk WHERE rn <= {max(QUEUE_N) * 3}""").df()
        days_list = sorted(q.day.unique())
        rows = []
        st = df.start_t
        for a in MAIN + ["review"]:
            col = f"ts_{a}"
            ts = df[col]
            # exact random baseline: average over days of P(action in (day, day+30] | open on that day)
            rnd = []
            for dday in days_list:
                dday = pd.Timestamp(dday)
                open_ = (st <= dday) & (ts.isna() | (ts > dday))
                hit = open_ & ts.notna() & (ts <= dday + 30 * day)
                rnd.append(hit.sum() / max(open_.sum(), 1))
            rand_p = float(np.mean(rnd))
            m = q.merge(df[["did", col]], on="did", how="left")
            m = m[m[col].isna() | (m[col] > m.day)]
            m["rank_open"] = m.groupby("day").score.rank(method="first", ascending=False)
            total = ts.notna() & (ts > t0 + W * day) & (ts < fhe + 30 * day)
            for N in QUEUE_N:
                s = m[m.rank_open <= N].sort_values("day").drop_duplicates("did")
                hit = s[col].notna() & (s[col] <= s.day + 30 * day)
                lead = (s.loc[hit, col] - s.loc[hit, "day"]).dt.total_seconds() / 86400
                rows.append({"cohort": cohort, "action": a, "queue_size_per_day": N, "accounts_flagged": len(s),
                             "actioned_within_30d": int(hit.sum()), "precision": hit.mean(), "random_precision": rand_p,
                             "lift_vs_random": hit.mean() / rand_p if rand_p else np.nan,
                             "median_days_before_action": lead.median(),
                             "share_of_all_actions_found_early": hit.sum() / max(total.sum(), 1)})
        qt = pd.DataFrame(rows)
        viz.table(qt, R, f"T98_queue_exact_{cohort}")
        queue_all.append(qt)
        log.info(f"[{cohort}] queue with exact random baseline\n{qt.round(4).to_string()}")

        # ---------- 4. lead time by account size (established)
        if cohort == "established":
            al = df[f"alert_k{args.k_main}"]
            rows = []
            for lo, hi, lab in FBANDS:
                band = df.followers.between(lo, hi)
                for a in MAIN:
                    ts = df[f"ts_{a}"]
                    inwin = band & ts.notna() & (ts < fhe)
                    cf = inwin & al.notna() & (al < ts)
                    lead = (ts[cf] - al[cf]).dt.total_seconds() / 86400
                    rows.append({"followers": lab, "action": a, "actions": int(inwin.sum()),
                                 "crowd_first_share": cf.sum() / max(inwin.sum(), 1), "lead_median_days": lead.median(),
                                 "lead_q25": lead.quantile(.25), "lead_q75": lead.quantile(.75)})
            lf = pd.DataFrame(rows)
            viz.table(lf, R, "T99_lead_by_followers")
            log.info(f"lead time by followers\n{lf.round(3).to_string()}")
            f, (a1, a2) = viz.plt.subplots(1, 2, figsize=(10, 3.4))
            bands = [b[2] for b in FBANDS]
            for i, a in enumerate(MAIN):
                d = lf[lf.action == a].set_index("followers").reindex(bands)
                x = np.arange(len(bands)) + (i - 1) * 0.25
                a1.bar(x, d.crowd_first_share * 100, width=0.25, color=viz.SERIES[i], label=ACTIONS[a][0].split(" (")[0])
                a2.errorbar(x, d.lead_median_days, yerr=[(d.lead_median_days - d.lead_q25).clip(lower=0),
                            (d.lead_q75 - d.lead_median_days).clip(lower=0)], fmt="o", color=viz.SERIES[i], capsize=2)
            for ax in (a1, a2):
                ax.set_xticks(range(len(bands)), bands)
                ax.set_xlabel("followers at 9 April")
                ax.grid(axis="x", visible=False)
            a1.set_ylabel("% of actions where the crowd was first")
            a2.set_ylabel("days the crowd was earlier (median, IQR)")
            a1.set_title("Crowd first, by account size", fontsize=10)
            a2.set_title("Lead time, by account size", fontsize=10)
            a1.legend(fontsize=7)
            viz.save(f, R, "F99_lead_by_followers", f"Established accounts, alert = {args.k_main} ordinary blockers.")

    # ---------- figures across cohorts
    tr = pd.concat(trade_all)
    f, axes = viz.plt.subplots(1, len(args.cohorts), figsize=(5.6 * len(args.cohorts), 3.8), squeeze=False)
    for ax, cohort in zip(axes.flat, args.cohorts):
        for i, a in enumerate(MAIN):
            d = tr[(tr.cohort == cohort) & (tr.action == a)].sort_values("k")
            ax.plot(d.lead_median_days, d.precision_30d * 100, marker="o", color=viz.SERIES[i], label=ACTIONS[a][0].split(" (")[0])
            for _, r in d.iterrows():
                if pd.notna(r.lead_median_days):
                    ax.annotate(f"k={r.k}", (r.lead_median_days, r.precision_30d * 100), fontsize=6.5,
                                xytext=(3, 3), textcoords="offset points", color=viz.INK2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(viz.plt.NullFormatter())
        ax.set_xlabel("median days the crowd alert came before the action")
        ax.set_ylabel("% of alerted accounts actioned within 30 days")
        ax.set_title(f"Speed vs accuracy of the crowd alert - {cohort}", fontsize=10)
        ax.legend(fontsize=7)
    viz.save(f, R, "F96_k_tradeoff", "Each point: alert after k ordinary blockers. Up = more accurate, right = earlier.")

    qt = pd.concat(queue_all)
    f, ax = viz.fig(7.4, 3.8)
    for cohort, ls in zip(args.cohorts, ["-", "--"]):
        for i, a in enumerate(MAIN):
            d = qt[(qt.cohort == cohort) & (qt.action == a)]
            ax.plot(d.queue_size_per_day, d.lift_vs_random, marker="o", ls=ls, color=viz.SERIES[i],
                    label=f"{ACTIONS[a][0].split(' (')[0]} ({cohort})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.axhline(1, color=viz.MUTED, lw=1, ls=":")
    ax.set_xlabel("accounts reviewed per day")
    ax.set_ylabel("precision / random precision")
    ax.set_title("How much better than random is a crowd-ranked queue?", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    viz.save(f, R, "F98_queue_exact", "Random = average daily share of open accounts that get the action within 30 days.")
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
