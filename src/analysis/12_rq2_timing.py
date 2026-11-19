#!/usr/bin/env python3
"""RQ2 - does the crowd act BEFORE the platform, by how much, and could a crowd-ranked review queue find cases earlier?

Built on RQ1: the crowd signal is an ALERT = the moment an account reaches k distinct ORDINARY blockers
(blockers with <= --heavy blocks made before T0), counted from the cohort start. Variants: k-th blocker of any kind,
and bursts reached only thanks to heavy blockers. Platform actions as in RQ1 (src/bskymod/outcomes.py).
Block times are exact only in the firehose period, so alerts are taken in [start, fh_end); actions are observed until
the end of the label data (fh_end + 30 days), so every alert has a full 30-day follow-up.

Cohorts: established (start = T0 = 9 April 2026, accounts active in the 90 days before) and new (start = first seen,
not taken down in the first 24 h).

Outputs (results/RQ2/)
  T90_who_first_<cohort>      for platform actions in [start, fh_end): crowd first / platform first / platform only
  F90_who_first               the same as stacked bars
  T91_lead_time_<cohort>      crowd-first cases: days from alert to action (median, IQR, share > 1 / 7 days)
  F91_lead_time               distribution of lead time per action
  T92_after_alert_<cohort>    cumulative share actioned 1/7/30 days after an alert (ordinary / any / heavy-only)
                              against the 30-day rate of accounts that never got an alert
  F92_after_alert_<cohort>    Kaplan-Meier cumulative incidence after alert, per action
  T93_queue_<cohort>          daily review queue: top-N accounts by ordinary blockers in the last 7 days
                              (not yet actioned) -> precision within 30 days, median days before the action,
                              share of all actions found early, against a random queue
  F93_queue                   precision and early share vs queue size
  T94_lang_timing_<cohort>    crowd-first share and median lead time per language (abuse labels, takedowns)
  T95_cox_<cohort>_<action>   among alerted accounts: what makes the platform act faster (hazard ratios)
  F95_cox_<cohort>_<action>
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

TIMING_ACTIONS = ["abuse", "suspend", "post_removed", "review", "takedown", "adult"]
MAIN = ["abuse", "suspend", "takedown"]
QUEUE_N = [100, 1000, 10000]


def norm_lang(col):
    return f"nullif(lower(split_part(replace(CAST({col} AS VARCHAR), '_', '-'), '-', 1)), '')"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--t0", default="2026-04-09")
    ap.add_argument("--fh-end", default="2026-06-01")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--heavy", type=int, default=1000)
    ap.add_argument("--queue-window", type=int, default=7, help="days of ordinary blocks used to rank the queue")
    ap.add_argument("--cohorts", nargs="*", default=["established", "new"])
    ap.add_argument("--cox-rows", type=int, default=500_000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    R = cfg["paths"]["results"] / "RQ2"
    log = get_logger("12", cfg["paths"]["results"] / "logs" / "12_rq2_timing.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    T0 = f"TIMESTAMPTZ '{args.t0} 00:00:00+00'"
    FHE = f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    END = f"({FHE} + INTERVAL 30 DAY)"
    L, C = I / "labels", I / "crowd"
    ae, re_ = rp(L / "account_label_events.parquet"), rp(L / "record_label_events.parquet")
    be = rp(C / "block_edges.parquet")
    pd.set_option("display.width", 250)
    k = args.k

    con.execute(f"""CREATE OR REPLACE TEMP TABLE heavy AS
        SELECT blocker_did AS did FROM {be} WHERE coalesce(t_created, {T0} - INTERVAL 1 DAY) < {T0}
        GROUP BY 1 HAVING count(*) > {args.heavy}""")

    tox2 = {c: P / f"tox2_{c}.parquet" for c in args.cohorts}
    base = {
        "established": f"""SELECT did, {T0} AS start_t, coalesce(net_followers, 0) AS followers, acc_age_days AS age_days,
                                  {norm_lang('act_top_lang')} AS lang FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d""",
        "new": f"""SELECT did, first_seen AS start_t, NULL::BIGINT AS followers, 0 AS age_days,
                          {norm_lang('new_act_top_lang')} AS lang FROM {rp(P / 'accounts_new.parquet')}
                   WHERE NOT instant_takedown AND first_seen < {FHE}"""}
    who_all, lead_all, queue_all = [], [], []

    for cohort in args.cohorts:
        con.execute(f"CREATE OR REPLACE TEMP TABLE c AS {base[cohort]}")
        # first block of each (subject, blocker) after start, with blocker type
        con.execute(f"""CREATE OR REPLACE TEMP TABLE bl AS
            SELECT e.subject_did AS did, e.blocker_did, min(e.t_created) AS t, h.did IS NOT NULL AS is_heavy
            FROM c JOIN {be} e ON e.subject_did = c.did AND e.t_created > c.start_t AND e.t_created < {FHE}
            LEFT JOIN heavy h ON h.did = e.blocker_did
            GROUP BY e.subject_did, e.blocker_did, h.did""")
        # alert times: k-th ordinary blocker, k-th blocker of any kind
        con.execute(f"""CREATE OR REPLACE TEMP TABLE al AS
            WITH o AS (SELECT did, t, row_number() OVER (PARTITION BY did ORDER BY t, blocker_did) rn FROM bl WHERE NOT is_heavy),
                 a AS (SELECT did, t, row_number() OVER (PARTITION BY did ORDER BY t, blocker_did) rn FROM bl)
            SELECT c.did, o.t AS alert_ord, a.t AS alert_any
            FROM c LEFT JOIN (SELECT did, t FROM o WHERE rn = {k}) o USING (did)
                   LEFT JOIN (SELECT did, t FROM a WHERE rn = {k}) a USING (did)""")
        # first platform action of each kind after start
        aggs = ",\n".join(f"min(e.ts) FILTER ({ACTIONS[a][1]}) AS ts_{a}" for a in TIMING_ACTIONS)
        df = con.execute(f"""
            WITH e AS (SELECT e.did, e.val, e.lvl, e.ts FROM {events_sql(ae, re_)} e
                       JOIN c ON c.did = e.did AND e.ts > c.start_t AND e.ts <= {END}),
                 f AS (SELECT did, {aggs} FROM e GROUP BY did)
            SELECT c.*, al.alert_ord, al.alert_any, f.* EXCLUDE (did)
            FROM c LEFT JOIN al USING (did) LEFT JOIN f USING (did)""").df()
        if tox2[cohort].exists():
            df = df.merge(con.execute(f"SELECT did, tox2_max FROM {rp(tox2[cohort])}").df(), on="did", how="left")
        # heavy-only alert: reached k blockers in total but never k ordinary ones before that point
        df["alert_heavy_only"] = df.alert_any.where(df.alert_ord.isna())
        fhe = pd.Timestamp(args.fh_end, tz="UTC")
        log.info(f"[{cohort}] accounts {len(df):,}; alerts: ordinary {df.alert_ord.notna().sum():,}, "
                 f"any {df.alert_any.notna().sum():,}, heavy-only {df.alert_heavy_only.notna().sum():,}")

        # ---------- 1. who acts first, and lead time
        rows, leads = [], []
        for a in TIMING_ACTIONS:
            ts = df[f"ts_{a}"]
            inwin = ts.notna() & (ts < fhe)
            d = df[inwin]
            for sig in ["alert_ord", "alert_any"]:
                crowd_first = d[sig].notna() & (d[sig] < d[f"ts_{a}"])
                plat_first = d[sig].notna() & ~crowd_first
                rows.append({"action": a, "signal": sig, "actions_in_window": len(d),
                             "crowd_first": int(crowd_first.sum()), "platform_first": int(plat_first.sum()),
                             "platform_only": int(d[sig].isna().sum()),
                             "share_crowd_first": crowd_first.mean(), "share_platform_first": plat_first.mean(),
                             "share_platform_only": d[sig].isna().mean()})
                lt = (d.loc[crowd_first, f"ts_{a}"] - d.loc[crowd_first, sig]).dt.total_seconds() / 86400
                if sig == "alert_ord":
                    leads.append(pd.DataFrame({"action": a, "lead_days": lt.values, "lang": d.loc[crowd_first, "lang"].values}))
                rows[-1].update({"lead_median_days": lt.median(), "lead_q25": lt.quantile(0.25), "lead_q75": lt.quantile(0.75),
                                 "lead_share_over_1d": (lt > 1).mean(), "lead_share_over_7d": (lt > 7).mean()})
        who = pd.DataFrame(rows)
        who.insert(0, "cohort", cohort)
        viz.table(who, R, f"T90_who_first_{cohort}")
        who_all.append(who)
        ld = pd.concat(leads, ignore_index=True)
        ld["cohort"] = cohort
        lead_all.append(ld)
        viz.table(who[["action", "signal", "lead_median_days", "lead_q25", "lead_q75", "lead_share_over_1d", "lead_share_over_7d"]],
                  R, f"T91_lead_time_{cohort}")
        log.info(f"[{cohort}] who first / lead time\n" + who.drop(columns="cohort").round(3).to_string())

        # ---------- 2. what happens after an alert (prospective, 30 days) + baseline
        from lifelines import KaplanMeierFitter
        rows = []
        f, axes = viz.plt.subplots(1, len(MAIN), figsize=(12, 3.6), squeeze=False)
        for ax, a in zip(axes.flat, MAIN):
            for i, (sig, lab, col) in enumerate([("alert_ord", "ordinary-blocker alert", viz.SERIES[0]),
                                                 ("alert_any", "any-blocker alert", viz.SERIES[1]),
                                                 ("alert_heavy_only", "heavy-only alert", viz.SERIES[2])]):
                d = df[df[sig].notna()]
                d = d[d[f"ts_{a}"].isna() | (d[f"ts_{a}"] > d[sig])]      # not yet actioned at the alert
                if len(d) < 50:
                    continue
                stop = d[f"ts_{a}"].where(d[f"ts_{a}"].notna() & (d[f"ts_{a}"] <= d[sig] + pd.Timedelta(days=30)),
                                          d[sig] + pd.Timedelta(days=30))
                ev = d[f"ts_{a}"].notna() & (d[f"ts_{a}"] <= d[sig] + pd.Timedelta(days=30))
                days = ((stop - d[sig]).dt.total_seconds() / 86400).clip(lower=1e-3)
                km = KaplanMeierFitter().fit(days, ev)
                cum = 1 - km.survival_function_.iloc[:, 0]
                ax.step(cum.index, cum.values * 100, where="post", color=col, label=lab)
                rows.append({"action": a, "signal": sig, "alerted_not_yet_actioned": len(d), "events_30d": int(ev.sum()),
                             **{f"cum_{t}d": float(1 - km.predict(t)) for t in [1, 7, 30]}})
            if cohort == "established":       # baseline: accounts never alerted, 30 days from T0
                nb = df[df.alert_any.isna()]
                p0 = (nb[f"ts_{a}"].notna() & (nb[f"ts_{a}"] <= pd.Timestamp(args.t0, tz="UTC") + pd.Timedelta(days=30))).mean()
                rows.append({"action": a, "signal": "never alerted (30 d from T0)", "alerted_not_yet_actioned": len(nb),
                             "events_30d": np.nan, "cum_30d": p0})
                ax.axhline(p0 * 100, color=viz.MUTED, lw=1, ls=":", label="never alerted (30-day rate)")
            ax.set_title(ACTIONS[a][0], fontsize=9)
            ax.set_xlabel("days after alert", fontsize=8)
            ax.set_xlim(0, 30)
            ax.set_yscale("symlog", linthresh=0.1)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}"))
        axes.flat[0].set_ylabel("% of alerted accounts with the action", fontsize=8)
        axes.flat[0].legend(fontsize=7)
        viz.save(f, R, f"F92_after_alert_{cohort}", f"{cohort} accounts. Alert = distinct blocker number {k} after the start "
                 f"(ordinary = blocker with <= {args.heavy:,} earlier blocks). Kaplan-Meier, accounts not yet actioned at the alert.")
        aa = pd.DataFrame(rows)
        viz.table(aa, R, f"T92_after_alert_{cohort}")
        log.info(f"[{cohort}] after alert\n{aa.round(4).to_string()}")

        # ---------- 3. daily review queue ranked by ordinary blockers in the last W days
        W = args.queue_window
        con.register("acts", df[["did"] + [f"ts_{a}" for a in TIMING_ACTIONS]])
        q = con.execute(f"""
            WITH days AS (SELECT unnest(generate_series(CAST(date_trunc('day', {T0}) AS TIMESTAMPTZ) + INTERVAL {W} DAY,
                                                         {FHE} - INTERVAL 1 DAY, INTERVAL 1 DAY)) AS day),
                 sc AS (SELECT d.day, b.did, count(*) AS score FROM days d JOIN bl b
                        ON NOT b.is_heavy AND b.t >= d.day - INTERVAL {W} DAY AND b.t < d.day GROUP BY 1, 2),
                 rk AS (SELECT *, row_number() OVER (PARTITION BY day ORDER BY score DESC, hash(did)) AS rn FROM sc)
            SELECT day, did, score FROM rk WHERE rn <= {max(QUEUE_N) * 3}""").df()
        rows = []
        n_days = q.day.nunique()
        for a in MAIN + ["review"]:
            col = f"ts_{a}"
            m = q.merge(df[["did", col]], on="did", how="left")
            m = m[m[col].isna() | (m[col] > m.day)]                    # not yet actioned on that day
            m["rank_open"] = m.groupby("day").score.rank(method="first", ascending=False)
            total = df[col].notna() & (df[col] > pd.Timestamp(args.t0, tz="UTC") + pd.Timedelta(days=W)) & (df[col] < fhe + pd.Timedelta(days=30))
            for N in QUEUE_N:
                s = m[m.rank_open <= N].sort_values("day").drop_duplicates("did")   # first time each account enters the queue
                hit = s[col].notna() & (s[col] <= s.day + pd.Timedelta(days=30))
                lead = (s.loc[hit, col] - s.loc[hit, "day"]).dt.total_seconds() / 86400
                # random queue: same number of picks per day from all open cohort accounts
                rate = df[col].between(pd.Timestamp(args.t0, tz="UTC") + pd.Timedelta(days=W), fhe + pd.Timedelta(days=30)).mean()
                rows.append({"cohort": cohort, "action": a, "queue_size_per_day": N, "days": n_days,
                             "accounts_flagged": len(s), "actioned_within_30d": int(hit.sum()), "precision": hit.mean(),
                             "random_precision_approx": rate * 30 / max((fhe - pd.Timestamp(args.t0, tz='UTC')).days + 30, 1),
                             "median_days_before_action": lead.median(),
                             "share_of_all_actions_found_early": hit.sum() / max(total.sum(), 1)})
        qt = pd.DataFrame(rows)
        qt["lift_vs_random"] = qt.precision / qt.random_precision_approx
        viz.table(qt, R, f"T93_queue_{cohort}")
        queue_all.append(qt)
        log.info(f"[{cohort}] review queue\n{qt.round(4).to_string()}")

        # ---------- 4. by language
        top = df.lang.value_counts().index[:10].tolist()
        rows = []
        for a in ["abuse", "takedown"]:
            ts = df[f"ts_{a}"]
            d = df[ts.notna() & (ts < fhe)]
            for lg in top + ["ALL"]:
                x = d if lg == "ALL" else d[d.lang == lg]
                cf = x.alert_ord.notna() & (x.alert_ord < x[f"ts_{a}"])
                lt = (x.loc[cf, f"ts_{a}"] - x.loc[cf, "alert_ord"]).dt.total_seconds() / 86400
                rows.append({"action": a, "lang": lg, "actions": len(x), "crowd_first": int(cf.sum()),
                             "share_crowd_first": cf.mean() if len(x) else np.nan, "lead_median_days": lt.median()})
        lg = pd.DataFrame(rows)
        viz.table(lg, R, f"T94_lang_timing_{cohort}")
        log.info(f"[{cohort}] by language\n{lg.round(3).to_string()}")

        # ---------- 5. Cox among accounts alerted by ordinary blockers: what makes the platform act faster
        from lifelines import CoxPHFitter
        con.register("alerted", df.loc[df.alert_ord.notna(), ["did", "alert_ord"]])
        hv = con.execute("""SELECT a.did, count(*) FILTER (b.is_heavy AND b.t <= a.alert_ord) AS heavy_at_alert,
                                   count(*) FILTER (NOT b.is_heavy AND b.t <= a.alert_ord + INTERVAL 1 DAY) AS ordinary_by_alert_plus_1d
                            FROM alerted a JOIN bl b ON b.did = a.did GROUP BY 1""").df()
        da = df[df.alert_ord.notna()].merge(hv, on="did", how="left")
        da["days_to_alert"] = (da.alert_ord - da.start_t).dt.total_seconds() / 86400
        for a in ["abuse", "takedown"]:
            col = f"ts_{a}"
            d = da[da[col].isna() | (da[col] > da.alert_ord + pd.Timedelta(days=1))].copy()   # landmark: alert + 1 day
            lm = d.alert_ord + pd.Timedelta(days=1)
            ev = d[col].notna() & (d[col] <= lm + pd.Timedelta(days=30))
            stop = d[col].where(ev, lm + pd.Timedelta(days=30))
            d["days"], d["event"] = ((stop - lm).dt.total_seconds() / 86400).clip(lower=1e-3), ev.astype(int)
            covs = ["ordinary_by_alert_plus_1d", "heavy_at_alert", "days_to_alert"]
            if cohort == "established":
                covs += ["followers", "age_days"]
            if "tox2_max" in d:
                d["tox2_max"] = d.tox2_max.fillna(d.tox2_max.median())
                covs.append("tox2_max")
            for lgx in [x for x in top[:5] if x != "en"]:
                d[f"lang_{lgx}"] = (d.lang == lgx).astype(int)
                covs.append(f"lang_{lgx}")
            d = d[["days", "event"] + covs].replace([np.inf, -np.inf], np.nan).dropna()
            if d.event.sum() < 30:
                log.warning(f"[{cohort}] Cox {a}: only {int(d.event.sum())} events, skipped")
                continue
            if len(d) > args.cox_rows:
                d = d.sample(args.cox_rows, random_state=42)
            keep = [c for c in covs if d[c].std() > 0]
            for c_ in keep:
                if d[c_].nunique() > 2:
                    if (d[c_] >= 0).all() and d[c_].max() > 20:
                        d[c_] = np.log1p(d[c_])
                    d[c_] = (d[c_] - d[c_].mean()) / d[c_].std()
            cph = CoxPHFitter(penalizer=0.01).fit(d[["days", "event"] + keep], "days", "event")
            s = cph.summary.reset_index()[["covariate", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
            s.columns = ["covariate", "hazard_ratio", "hr_low", "hr_high", "p"]
            s["rows"], s["events"], s["concordance"] = len(d), int(d.event.sum()), cph.concordance_index_
            viz.table(s, R, f"T95_cox_{cohort}_{a}")
            log.info(f"[{cohort}] Cox {a}: C={cph.concordance_index_:.3f}\n{s.round(3).to_string()}")
            s = s.sort_values("hazard_ratio")
            f, ax = viz.fig(6.8, 0.34 * len(s) + 1.5)
            y = np.arange(len(s))
            ax.errorbar(s.hazard_ratio, y, xerr=[s.hazard_ratio - s.hr_low, s.hr_high - s.hazard_ratio], fmt="none",
                        ecolor=viz.INK2, elinewidth=1)
            ax.scatter(s.hazard_ratio, y, color=[viz.SERIES[1] if h > 1 else viz.SERIES[0] for h in s.hazard_ratio], s=34, zorder=3)
            ax.axvline(1, color=viz.MUTED, lw=1, ls=":")
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}" if round(v * 10) % 2 == 0 else ""))
            ax.set_yticks(y, s.covariate, fontsize=8)
            ax.set_xlabel(f"hazard ratio for {ACTIONS[a][0]} (> 1 = sooner)")
            ax.set_title(f"What speeds up the platform after an alert ({cohort})", fontsize=10)
            ax.grid(axis="y", visible=False)
            viz.save(f, R, f"F95_cox_{cohort}_{a}", f"Cox model from alert + 1 day, 30-day follow-up; {len(d):,} accounts, "
                     f"{int(d.event.sum()):,} events, C-index {cph.concordance_index_:.2f}. Continuous covariates per SD; "
                     "language vs English and all others.")

    # ---------- figures across cohorts
    who = pd.concat(who_all)
    w = who[who.signal == "alert_ord"]
    f, axes = viz.plt.subplots(1, len(args.cohorts), figsize=(6.6 * len(args.cohorts), 3.6), squeeze=False)
    for ax, cohort in zip(axes.flat, args.cohorts):
        d = w[w.cohort == cohort].set_index("action").reindex(TIMING_ACTIONS[::-1])
        y = np.arange(len(d))
        left = np.zeros(len(d))
        for col, lab, colr in [("share_crowd_first", "crowd first", viz.SERIES[0]),
                               ("share_platform_first", "platform first", viz.SERIES[1]),
                               ("share_platform_only", "platform only (no crowd alert)", viz.MUTED)]:
            v = d[col].fillna(0).values * 100
            ax.barh(y, v, left=left, color=colr, height=0.62, label=lab)
            for yy, l0, vv in zip(y, left, v):
                if vv >= 7:
                    ax.text(l0 + vv / 2, yy, f"{vv:.0f}%", ha="center", va="center", fontsize=7, color="white")
            left += v
        ax.set_yticks(y, [f"{ACTIONS[a][0]} (n={int(n):,})" for a, n in zip(d.index, d.actions_in_window.fillna(0))], fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of platform actions")
        ax.set_title(f"Who acts first - {cohort}", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes.flat[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    viz.save(f, R, "F90_who_first", f"Platform actions between the start and {args.fh_end}. Crowd alert = ordinary blocker number {k}.")

    ld = pd.concat(lead_all)
    f, axes = viz.plt.subplots(1, len(args.cohorts), figsize=(5.8 * len(args.cohorts), 3.4), sharey=True, squeeze=False)
    for ax, cohort in zip(axes.flat, args.cohorts):
        for a, colr in zip(MAIN, viz.SERIES):
            v = ld[(ld.cohort == cohort) & (ld.action == a)].lead_days
            if len(v) >= 20:
                viz.ecdf(ax, v.clip(lower=0.01), colr, ACTIONS[a][0], logx=True)
        ax.set_xlabel("days from crowd alert to platform action (log)")
        ax.set_title(f"How much earlier the crowd is - {cohort}", fontsize=10)
        ax.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.legend(fontsize=7, loc="lower right")
    viz.save(f, R, "F91_lead_time", "Only actions where the ordinary-blocker alert came first.")

    qt = pd.concat(queue_all)
    f, (a1, a2) = viz.plt.subplots(1, 2, figsize=(11, 3.6))
    for (cohort, a), colr, ls in [((c, a), viz.SERIES[i], "-" if c == "established" else "--")
                                  for c in args.cohorts for i, a in enumerate(MAIN)]:
        d = qt[(qt.cohort == cohort) & (qt.action == a)]
        a1.plot(d.queue_size_per_day, d.precision * 100, marker="o", color=colr, ls=ls, label=f"{ACTIONS[a][0]} ({cohort})")
        a2.plot(d.queue_size_per_day, d.share_of_all_actions_found_early * 100, marker="o", color=colr, ls=ls)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel("accounts reviewed per day")
    a1.set_ylabel("% of reviewed accounts actioned within 30 days")
    a1.set_title("Precision of a crowd-ranked review queue", fontsize=10)
    a2.set_ylabel("% of these actions found early")
    a2.set_title("Share of platform actions found early", fontsize=10)
    a1.set_ylim(bottom=0)
    a2.legend(*a1.get_legend_handles_labels(), fontsize=6.5, loc="upper left", ncol=1)
    viz.save(f, R, "F93_queue", f"Each day, accounts ranked by ordinary blockers in the last {args.queue_window} days, "
             "not yet actioned; each account counted the first time it enters the queue.")
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
