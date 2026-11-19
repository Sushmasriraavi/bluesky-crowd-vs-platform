#!/usr/bin/env python3
"""R1 - first look: does the crowd see what the platform sees, and earlier? (results/R1/)

Population for takedown questions: NEW takedowns (not active on 1 April) with takedown time before
the end of firehose coverage (1 June), so every block before the takedown is observable.

  T10_takedown_crowd_status.csv  crowd signal present before takedown (none / list only / 1-4 / 5-19 / 20+ blockers)
                                 + Sokoto et al. replication: share with no prior block record
  F10_takedown_crowd_status      same as bar chart
  F11_lead_time                  crowd onset (1st and 5th blocker, inside window) -> takedown, hours (ECDF)
  F12_crowd_vs_review            who flags first: crowd (1st blocker) or Bluesky (needs-review)?
  F13_conversion                 early-warning curve: takedown within 30 days by blockers in first 24h
  F14_time_to_takedown_by_signal account age at takedown: crowd-flagged vs no crowd signal ("too fast for the crowd?")
  T15_crowd_only.csv             heavily blocked but never taken down: what does the platform say about them?
  F16_blockers_ccdf              distribution of blockers per account, taken down vs not
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--fh-end", default="2026-06-01")
    ap.add_argument("--label-end", default="2026-07-01")
    ap.add_argument("--signals", default="crowd", help="crowd (snapshot+firehose) or crowd_snapshot")
    ap.add_argument("--outcomes", default="labels", help="labels (Apr-Jul 2026) or labels_hist (2023-2026)")
    ap.add_argument("--out", default="R1", help="results subfolder")
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, R = cfg["paths"]["interim"], cfg["paths"]["results"] / args.out
    log = get_logger(args.out, cfg["paths"]["results"] / "logs" / f"{args.out}.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    q = lambda s: con.execute(s).df()
    S = f"TIMESTAMPTZ '{args.start} 00:00:00+00'"
    E = f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"

    C = I / args.signals
    con.execute(f"CREATE TEMP VIEW o AS SELECT * FROM {rp(I / args.outcomes / 'account_outcomes.parquet')}")
    con.execute(f"CREATE TEMP VIEW s AS SELECT * FROM {rp(C / 'subject_signals.parquet')}")
    has_activity = (C / "account_activity.parquet").exists()
    if has_activity:
        con.execute(f"CREATE TEMP VIEW a AS SELECT * FROM {rp(C / 'account_activity.parquet')}")
    else:  # snapshot-only: no firehose activity; first seen comes from snapshot post history only
        con.execute("CREATE TEMP VIEW a AS SELECT CAST(NULL AS VARCHAR) AS did, CAST(NULL AS TIMESTAMPTZ) AS first_t WHERE false")
    log.info(f"signals={args.signals} outcomes={args.outcomes} window {args.start}..{args.fh_end} activity={has_activity}")
    con.execute(f"CREATE TEMP VIEW ph AS SELECT did, first_post_at FROM {rp(I / 'snapshot' / 'post_history.parquet')}")
    con.execute(f"CREATE TEMP VIEW be AS SELECT * FROM {rp(C / 'block_edges.parquet')}")

    # takedown cohort with crowd state at takedown time
    con.execute(f"""CREATE TEMP TABLE td AS
        WITH base AS (
          SELECT o.did, o.takedown_ts, o.needs_review_ts, o.needs_review_before_takedown,
                 s.t_k1, s.t_k5, s.t_list1, s.n_blockers,
                 s.t_list1 < o.takedown_ts AS listed_before,
                 least(a.first_t, ph.first_post_at) AS first_seen
          FROM o LEFT JOIN s USING (did) LEFT JOIN a USING (did) LEFT JOIN ph USING (did)
          WHERE o.takedown_ts IS NOT NULL AND NOT o.takedown_active_at_start AND o.takedown_ts < {E}),
        bb AS (
          SELECT b.did, count(DISTINCT e.blocker_did) AS n
          FROM base b JOIN be e ON e.subject_did = b.did
           AND coalesce(e.t_created, TIMESTAMPTZ '2000-01-01 00:00:00+00') < b.takedown_ts
          GROUP BY b.did)
        SELECT base.*, coalesce(bb.n, 0) AS blockers_before FROM base LEFT JOIN bb USING (did)""")
    n_td = con.execute("SELECT count(*) FROM td").fetchone()[0]
    log.info(f"takedown cohort (new, before {args.fh_end}): {n_td:,}")

    # ---- T10 / F10 crowd status before takedown
    st = q("""SELECT CASE
                 WHEN blockers_before >= 20 THEN '20+ blockers'
                 WHEN blockers_before >= 5 THEN '5-19 blockers'
                 WHEN blockers_before >= 1 THEN '1-4 blockers'
                 WHEN coalesce(listed_before, false) THEN 'on a mod list only'
                 WHEN first_seen IS NULL THEN 'no crowd signal, no activity record'
                 ELSE 'no crowd signal, seen active' END AS status, count(*) AS accounts
              FROM td GROUP BY 1""")
    order = ["20+ blockers", "5-19 blockers", "1-4 blockers", "on a mod list only",
             "no crowd signal, seen active", "no crowd signal, no activity record"]
    st = st.set_index("status").reindex(order).fillna(0).reset_index()
    st["share"] = st.accounts / st.accounts.sum()
    no_block = con.execute("SELECT avg(CASE WHEN blockers_before = 0 THEN 1.0 ELSE 0 END) FROM td").fetchone()[0]
    st.loc[len(st)] = ["REPLICATION: no prior block record (Sokoto et al. 2026 report 38.5%)", None, no_block]
    viz.table(st, R, "T10_takedown_crowd_status")
    p = st.iloc[:-1]
    f, ax = viz.fig(7.5, 3.4)
    colors = [viz.SERIES[0]] * 3 + [viz.SERIES[2]] + [viz.SERIES[1]] * 2
    ax.barh(p.status[::-1], p.share[::-1] * 100, color=colors[::-1], height=0.6)
    for y, (v, n) in enumerate(zip(p.share[::-1], p.accounts[::-1])):
        ax.text(v * 100 + 0.5, y, f"{v*100:.1f}%  ({int(n):,})", va="center", fontsize=9, color=viz.INK2)
    ax.set_xlabel("share of new takedowns (%)")
    ax.set_title(f"Crowd signal before Bluesky's takedown (n={n_td:,})")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(p.share) * 100 * 1.35 + 1)
    viz.save(f, R, "F10_takedown_crowd_status",
             f"Blue: blocked by users before takedown. Orange: no crowd signal. No prior block record: {no_block*100:.1f}% (Sokoto et al.: 38.5%).")

    # ---- F11 lead time (onset inside the window only, so the clock start is observed)
    lt = q(f"""SELECT epoch(takedown_ts - t_k1)/3600.0 AS k1_hours,
                      CASE WHEN t_k5 >= {S} AND t_k5 < takedown_ts THEN epoch(takedown_ts - t_k5)/3600.0 END AS k5_hours
               FROM td WHERE t_k1 >= {S} AND t_k1 < takedown_ts""")
    viz.table(lt.describe(percentiles=[.1, .25, .5, .75, .9]).reset_index(), R, "F11_lead_time_quantiles")
    f, ax = viz.fig(6.8, 3.6)
    viz.ecdf(ax, lt.k1_hours.clip(lower=1 / 60), viz.SERIES[0], "from 1st blocker", logx=True)
    viz.ecdf(ax, lt.k5_hours.dropna().clip(lower=1 / 60), viz.SERIES[1], "from 5th blocker", logx=True)
    for x, lab in [(24, "1 day"), (168, "1 week")]:
        ax.axvline(x, color=viz.MUTED, lw=0.8, ls=":")
        ax.text(x, 0.02, f" {lab}", fontsize=8, color=viz.INK2)
    ax.set_xlabel("hours from crowd onset to takedown (log scale)")
    ax.set_title("How long after the crowd starts blocking does Bluesky act?")
    ax.legend(loc="upper left")
    viz.save(f, R, "F11_lead_time", f"Accounts whose first block happened on/after {args.start} and before the takedown.")

    # ---- F12 crowd vs needs-review: who is first?
    cr = q(f"""SELECT epoch(needs_review_ts - t_k1)/3600.0 AS h FROM td
               WHERE needs_review_before_takedown AND t_k1 >= {S} AND t_k1 < takedown_ts""")
    bins = [-np.inf, -168, -24, 0, 24, 168, np.inf]
    labels = ["review >7d earlier", "review 1-7d earlier", "review <1d earlier",
              "crowd <1d earlier", "crowd 1-7d earlier", "crowd >7d earlier"]
    cr["bucket"] = pd.cut(cr.h, bins=bins, labels=labels, right=False)
    cb = cr.bucket.value_counts().reindex(labels).fillna(0).rename_axis("who_first").reset_index(name="accounts")
    cb["share"] = cb.accounts / max(cb.accounts.sum(), 1)
    viz.table(cb, R, "F12_crowd_vs_review")
    f, ax = viz.fig(7.5, 3.4)
    ax.bar(cb.who_first, cb.share * 100, color=[viz.SERIES[1]] * 3 + [viz.SERIES[0]] * 3, width=0.65)
    for i, (v, n) in enumerate(zip(cb.share, cb.accounts)):
        ax.text(i, v * 100 + 0.5, f"{v*100:.0f}%", ha="center", fontsize=9, color=viz.INK2)
    ax.set_ylabel("share of accounts (%)")
    ax.set_title(f"Who flags first: Bluesky's needs-review or the crowd's first block? (n={len(cr):,})")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", visible=False)
    viz.save(f, R, "F12_crowd_vs_review", "Taken-down accounts with both signals inside the window. Blue: crowd earlier. Orange: Bluesky earlier.")

    # ---- F13 early-warning conversion curve
    cv = q(f"""
        WITH c AS (
          SELECT s.did, s.vel_1d AS b24, o.takedown_ts, coalesce(o.takedown_active_at_start, false) AS active_start, s.t_k1
          FROM s LEFT JOIN o USING (did)
          WHERE s.t_k1 >= {S} AND s.t_k1 < {E})
        SELECT CASE WHEN b24 >= 20 THEN '20+' WHEN b24 >= 5 THEN '5-19' WHEN b24 >= 2 THEN '2-4' ELSE '1' END AS blockers_24h,
               count(*) AS accounts,
               count(*) FILTER (takedown_ts > t_k1 + INTERVAL 1 DAY AND takedown_ts <= t_k1 + INTERVAL 31 DAY) AS taken_down_30d
        FROM c WHERE NOT active_start
          AND NOT (takedown_ts IS NOT NULL AND takedown_ts <= t_k1 + INTERVAL 1 DAY)
        GROUP BY 1""")
    base = pd.DataFrame(columns=["blockers_24h", "accounts", "taken_down_30d"]) if not has_activity else q(f"""
        SELECT 'no blocks (active accounts)' AS blockers_24h, count(*) AS accounts,
               count(*) FILTER (o.takedown_ts > a.first_t + INTERVAL 1 DAY AND o.takedown_ts <= a.first_t + INTERVAL 31 DAY) AS taken_down_30d
        FROM a LEFT JOIN o USING (did) LEFT JOIN s USING (did)
        WHERE (s.n_blockers IS NULL OR s.n_blockers = 0) AND NOT coalesce(o.takedown_active_at_start, false)
          AND a.first_t < {E}
          AND NOT (o.takedown_ts IS NOT NULL AND o.takedown_ts <= a.first_t + INTERVAL 1 DAY)""")
    cats = (["no blocks (active accounts)"] if has_activity else []) + ["1", "2-4", "5-19", "20+"]
    cv = pd.concat([base, cv]).set_index("blockers_24h").reindex(cats)
    cv = cv.apply(pd.to_numeric, errors="coerce").fillna(0).reset_index()
    cv[["accounts", "taken_down_30d"]] = cv[["accounts", "taken_down_30d"]].astype(float)
    cv["rate"] = cv.taken_down_30d / cv.accounts.replace(0, np.nan)
    ci = [wilson(k, n) for k, n in zip(cv.taken_down_30d, cv.accounts)]
    cv["ci_low"], cv["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    viz.table(cv, R, "F13_conversion")
    f, ax = viz.fig(6.8, 3.6)
    x = np.arange(len(cv))
    ax.bar(x, cv.rate * 100, color=([viz.MUTED] if has_activity else []) + [viz.SERIES[0]] * 4, width=0.6)
    ax.errorbar(x, cv.rate * 100, yerr=np.clip([(cv.rate - cv.ci_low) * 100, (cv.ci_high - cv.rate) * 100], 0, None),
                fmt="none", ecolor=viz.INK2, elinewidth=1, capsize=3)
    for i, (r, hi, n) in enumerate(zip(cv.rate, cv.ci_high, cv.accounts)):
        r = 0 if r != r else r
        top = max(r, hi if hi == hi else r) * 100
        ax.text(i, top + 1.5, f"{r*100:.1f}%\nn={int(n):,}", ha="center", va="bottom", fontsize=8, color=viz.INK2)
    ax.set_ylim(0, max(100 * cv.ci_high.fillna(0).max() * 1.25, 5))
    ax.set_xticks(x, cv.blockers_24h)
    ax.set_xlabel("distinct blockers in the first 24 h after the first block")
    ax.set_ylabel("taken down within next 30 days (%)")
    ax.set_title("Early warning: more blockers in day one, higher takedown risk")
    ax.grid(axis="x", visible=False)
    viz.save(f, R, "F13_conversion", "Outcome counted from 24 h after first block (decision time) to 31 days. 95% Wilson intervals.")

    # ---- F14 account age at takedown: crowd-flagged vs no signal
    ag = q("""SELECT epoch(takedown_ts - first_seen)/3600.0 AS hours,
                     CASE WHEN blockers_before > 0 OR coalesce(listed_before, false) THEN 'crowd signal before takedown'
                          ELSE 'no crowd signal' END AS grp
              FROM td WHERE first_seen IS NOT NULL AND first_seen <= takedown_ts""")
    viz.table(ag.groupby("grp").hours.describe(percentiles=[.1, .25, .5, .75, .9]).reset_index(), R,
              "F14_time_to_takedown_by_signal_quantiles")
    f, ax = viz.fig(6.8, 3.6)
    for i, g in enumerate(["crowd signal before takedown", "no crowd signal"]):
        viz.ecdf(ax, ag[ag.grp == g].hours.clip(lower=1 / 60), viz.SERIES[i], g, logx=True)
    for xv, lab in [(1, "1 h"), (24, "1 day"), (168, "1 week")]:
        ax.axvline(xv, color=viz.MUTED, lw=0.8, ls=":")
        ax.text(xv, 0.02, f" {lab}", fontsize=8, color=viz.INK2)
    ax.set_xlabel("hours from first seen activity to takedown (log scale)")
    ax.set_title("Are platform-only takedowns just too fast for the crowd?")
    ax.legend(loc="upper left")
    viz.save(f, R, "F14_time_to_takedown_by_signal", "First seen = earliest of first firehose event and first snapshot post." if has_activity else "First seen = first post in the snapshot (only accounts whose posts survive in the snapshot).")

    # ---- T15 crowd-only: heavily blocked in window, never taken down
    rl = rp(I / args.outcomes / "record_label_events.parquet")
    co = q(f"""
        WITH g AS (
          SELECT s.did,
                 CASE WHEN o.takedown_ts IS NOT NULL THEN 'blocked 5+ and taken down' ELSE 'blocked 5+, NOT taken down (crowd-only)' END AS grp,
                 o.first_spam_ts IS NOT NULL OR o.first_rude_ts IS NOT NULL OR o.first_intolerant_ts IS NOT NULL
                   OR o.first_threat_ts IS NOT NULL OR o.first_impersonation_ts IS NOT NULL OR o.first_misleading_ts IS NOT NULL AS acct_reason_label,
                 o.needs_review_ts IS NOT NULL AS flagged_needs_review
          FROM s LEFT JOIN o USING (did)
          WHERE s.n_blockers_win >= 5 AND NOT coalesce(o.takedown_active_at_start, false)),
        r AS (SELECT did, count(*) FILTER (NOT neg) n_post_labels FROM {rl} WHERE ts >= {S} GROUP BY did)
        SELECT grp, count(*) accounts,
               avg(CASE WHEN acct_reason_label THEN 1.0 ELSE 0 END) share_account_reason_label,
               avg(CASE WHEN flagged_needs_review THEN 1.0 ELSE 0 END) share_flagged_needs_review,
               avg(CASE WHEN coalesce(r.n_post_labels, 0) > 0 THEN 1.0 ELSE 0 END) share_with_post_labels,
               avg(CASE WHEN acct_reason_label OR flagged_needs_review OR coalesce(r.n_post_labels, 0) > 0 THEN 0 ELSE 1.0 END) share_no_platform_signal_at_all
        FROM g LEFT JOIN r USING (did) GROUP BY grp""")
    viz.table(co, R, "T15_crowd_only")

    # ---- F17 trends by takedown month: is the platform catching up with the crowd?
    tr = q(f"""
        SELECT date_trunc('month', takedown_ts) AS month, count(*) AS takedowns,
               avg(CASE WHEN blockers_before > 0 THEN 1.0 ELSE 0 END) AS share_blocked_before,
               count(*) FILTER (needs_review_before_takedown) AS with_review_flag,
               avg(CASE WHEN needs_review_ts > t_k1 THEN 1.0 ELSE 0 END)
                 FILTER (needs_review_before_takedown AND t_k1 >= {S} AND t_k1 < takedown_ts) AS share_crowd_first,
               median(epoch(takedown_ts - t_k1) / 3600.0) FILTER (t_k1 >= {S} AND t_k1 < takedown_ts) AS median_lead_hours
        FROM td GROUP BY 1 ORDER BY 1""")
    tr = tr[tr.takedowns >= 500]
    viz.table(tr, R, "F17_trends")
    f, ax = viz.fig(7.5, 3.6)
    ax.plot(tr.month, tr.share_blocked_before * 100, color=viz.SERIES[0], marker="o", ms=3,
            label="takedowns with a prior block")
    m = tr.with_review_flag >= 500
    ax.plot(tr.month[m], tr.share_crowd_first[m] * 100, color=viz.SERIES[1], marker="o", ms=3,
            label="crowd before needs-review (when both exist)")
    viz.dates(ax)
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of takedowns that month (%)")
    ax.set_title("Crowd vs platform over time")
    ax.legend(loc="lower left")
    viz.save(f, R, "F17_trends", "Months with at least 500 takedowns; crowd-first line only where at least 500 takedowns had a needs-review flag.")

    # ---- F16 blockers per account CCDF
    cc = q("""SELECT s.n_blockers, coalesce(o.takedown_ts IS NOT NULL OR o.takedown_active_at_start, false) AS taken_down
              FROM s LEFT JOIN o USING (did) WHERE s.n_blockers > 0""")
    f, ax = viz.fig(6.5, 3.6)
    out = []
    for i, (g, lab) in enumerate([(True, "taken down (any time)"), (False, "never taken down")]):
        v = np.sort(cc[cc.taken_down == g].n_blockers.values)
        if len(v) == 0:
            continue
        uniq = np.unique(v)
        ccdf = 1 - np.searchsorted(v, uniq, side="left") / len(v)
        ax.plot(uniq, ccdf, color=viz.SERIES[i], label=f"{lab} (n={len(v):,})")
        out.append(pd.DataFrame({"group": lab, "n_blockers": uniq, "share_at_least": ccdf}))
    viz.table(pd.concat(out) if out else pd.DataFrame(), R, "F16_blockers_ccdf")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("distinct blockers (log)"); ax.set_ylabel("share with at least x blockers (log)")
    ax.set_title("Blockers per account")
    ax.legend(loc="lower left")
    viz.save(f, R, "F16_blockers_ccdf")
    log.info(f"R1 done -> {R}")


if __name__ == "__main__":
    main()
