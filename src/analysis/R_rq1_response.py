#!/usr/bin/env python3
"""RQ1 (final): when the crowd blocks an account, what does the platform do - and on which harms do they agree?

Crowd signal: distinct new blockers in (index, index + 30 d]; block burst B = 5+ (as in step 08).
Platform side: every action of the official labeler (src/bskymod/outcomes.py) in the same 30 days,
and up to the end of the label data (2026-07-01) as a longer follow-up.

Outputs (results/RQ1/response/)
  T45_ladder_<cohort>          dose-response: share of accounts with each platform action, by number of new blockers
  F46_ladder                   the same as small multiples (one panel per action, established vs new)
  T46_lift                     per cohort x action: P(action | burst), P(action | no burst), lift with 95% CI,
                               and P(burst | action) = share of the platform's actions that the crowd had flagged
  F47_lift                     lift (left) and share of platform actions flagged by the crowd (right)
  T48_lift_by_followers        established: lift within follower bands (is it just visibility?)
  F48_lift_by_followers
  T49_crowd_flagged_outcomes   what happened to blocked accounts: strongest platform action they received
  F49_crowd_flagged_outcomes   (burst vs no burst)
  T49t_tox_by_action           mean toxicity of accounts with each action, burst vs no burst (if step 10 ran)
  T4A_alignment_by_outcome     from step 08 runs (tags main_tox / out_<outcome>): SHAP rank correlation and
  F4A_alignment_by_outcome     cross-prediction AUC between the block model and each platform-action model
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str
from bskymod.outcomes import ACTIONS, ORDER, flags_sql

BINS = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 19, "5-19"), (20, 99, "20-99"), (100, 10**12, "100+")]
FBANDS = [(0, 9, "0-9"), (10, 99, "10-99"), (100, 999, "100-999"), (1000, 9999, "1k-10k"), (10000, 10**12, "10k+")]
# strongest action first (for the "what happened to them" ladder)
SEVERITY = ["takedown", "suspend", "abuse", "post_removed", "adult", "review"]
TAG_FOR = {"takedown": "main_tox", "review": "out_review", "abuse": "out_abuse", "adult": "out_adult"}
COH_COL = {"established": viz.SERIES[0], "new": viz.SERIES[1]}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def lift_row(y, b):
    """risk ratio P(y|b) / P(y|not b) with a 95% CI on the log scale (0.5 added to empty cells)."""
    a, n1 = int((y & b).sum()), int(b.sum())
    c, n0 = int((y & ~b).sum()), int((~b).sum())
    if a + c == 0 or not n1 or not n0:
        return {"accounts_burst": n1, "accounts_no_burst": n0, "actioned_burst": a, "actioned_no_burst": c,
                "p_action_given_burst": a / n1 if n1 else np.nan, "p_action_given_no_burst": c / n0 if n0 else np.nan,
                "lift": np.nan, "lift_low": np.nan, "lift_high": np.nan, "share_of_actions_flagged_by_crowd": np.nan}
    aa, cc = (a, c) if a and c else (a + 0.5, c + 0.5)
    rr = (aa / n1) / (cc / n0) if n1 and n0 else np.nan
    se = np.sqrt(1 / aa - 1 / n1 + 1 / cc - 1 / n0) if n1 and n0 else np.nan
    return {"accounts_burst": n1, "accounts_no_burst": n0, "actioned_burst": a, "actioned_no_burst": c,
            "p_action_given_burst": a / n1 if n1 else np.nan, "p_action_given_no_burst": c / n0 if n0 else np.nan,
            "lift": rr, "lift_low": rr * np.exp(-1.96 * se), "lift_high": rr * np.exp(1.96 * se),
            "share_of_actions_flagged_by_crowd": a / (a + c) if a + c else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--t0", default="2026-04-09")
    ap.add_argument("--fh-end", default="2026-06-01")
    ap.add_argument("--labels-end", default="2026-07-01")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cohorts", nargs="*", default=["established", "new"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    RQ = cfg["paths"]["results"] / "RQ1"
    R = RQ / "response"
    log = get_logger("RQ1resp", cfg["paths"]["results"] / "logs" / "R_rq1_response.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    H = "INTERVAL 30 DAY"
    T0 = f"TIMESTAMPTZ '{args.t0} 00:00:00+00'"
    FHE = f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    LEND = f"TIMESTAMPTZ '{args.labels_end} 00:00:00+00'"
    L = I / "labels"
    ae, re_ = rp(L / "account_label_events.parquet"), rp(L / "record_label_events.parquet")
    be = rp(I / "crowd" / "block_edges.parquet")
    base = {"established": f"SELECT did, {T0} AS index_t, coalesce(net_followers, 0) AS followers "
                           f"FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d",
            "new": f"SELECT did, index_t, NULL::BIGINT AS followers FROM {rp(P / 'accounts_new.parquet')} "
                   f"WHERE NOT instant_takedown AND index_t + {H} <= {FHE}"}

    ladders, lifts, fol, flagged, toxr = [], [], [], [], []
    for cohort in args.cohorts:
        con.execute(f"CREATE OR REPLACE TEMP TABLE c AS {base[cohort]}")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE b AS
            SELECT c.did, count(DISTINCT e.blocker_did) AS blockers FROM c JOIN {be} e
            ON e.subject_did = c.did AND e.t_created > c.index_t AND e.t_created <= c.index_t + {H} GROUP BY 1""")
        f30 = con.execute(flags_sql("SELECT did, index_t FROM c", ae, re_, f"c.index_t + {H}")).df()
        fend = con.execute(flags_sql("SELECT did, index_t FROM c", ae, re_, LEND)).df()
        df = con.execute("SELECT c.did, c.followers, coalesce(b.blockers, 0) AS blockers FROM c LEFT JOIN b USING (did)").df()
        df = df.merge(f30, on="did").merge(fend, on="did", suffixes=("", "_end"))
        df["burst"] = df.blockers >= args.k
        log.info(f"[{cohort}] accounts {len(df):,}, burst {int(df.burst.sum()):,}; "
                 + ", ".join(f"{a} {int(df['act_' + a].sum()):,}" for a in ORDER))

        # 1. dose-response ladder
        for lo, hi, lab in BINS:
            m = df.blockers.between(lo, hi)
            r = {"cohort": cohort, "new_blockers_30d": lab, "accounts": int(m.sum())}
            for a in ORDER:
                k = int(df.loc[m, f"act_{a}"].sum())
                r[a], (r[f"{a}_low"], r[f"{a}_high"]) = (k / m.sum() if m.sum() else np.nan), wilson(k, int(m.sum()))
            ladders.append(r)

        # 2. lift + share of platform actions flagged by the crowd
        for horizon, suf in [("30 days", ""), ("to labels end", "_end")]:
            for a in ORDER:
                lifts.append({"cohort": cohort, "horizon": horizon, "action": a, "description": ACTIONS[a][0],
                              **lift_row(df[f"act_{a}{suf}"], df.burst)})

        # 3. is it just visibility? lift inside follower bands (established only)
        if cohort == "established":
            for lo, hi, lab in FBANDS:
                d = df[df.followers.between(lo, hi)]
                for a in ORDER:
                    fol.append({"followers": lab, "action": a, "accounts": len(d), **lift_row(d[f"act_{a}"], d.burst)})

        # 4. strongest action each account received (mutually exclusive)
        for horizon, suf in [("30 days", ""), ("to labels end", "_end")]:
            strongest = np.select([df[f"act_{a}{suf}"] for a in SEVERITY], SEVERITY, "none")
            for grp, m in [("burst (5+ blockers)", df.burst), ("no burst", ~df.burst)]:
                vc = pd.Series(strongest[m.values]).value_counts(normalize=True)
                for a in SEVERITY + ["none"]:
                    flagged.append({"cohort": cohort, "horizon": horizon, "group": grp, "strongest_action": a,
                                    "share": float(vc.get(a, 0.0)), "accounts": int(m.sum())})

        # 5. toxicity of accounts with each action (validation: do abuse labels track toxic posts?)
        tox = P / f"tox_{cohort}.parquet"
        if tox.exists():
            t = df.merge(con.execute(f"SELECT did, tox_mean, tox_max FROM {rp(tox)}").df(), on="did")
            for a in ORDER + ["none"]:
                ma = ~t[[f"act_{x}" for x in ORDER]].any(axis=1) if a == "none" else t[f"act_{a}"]
                for grp, m in [("burst", t.burst), ("no burst", ~t.burst)]:
                    v = t.loc[ma & m, "tox_mean"]
                    toxr.append({"cohort": cohort, "action": a, "group": grp, "accounts_scored": len(v),
                                 "tox_mean": v.mean(), "tox_ci": 1.96 * v.std() / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                                 "share_max_post_toxic": (t.loc[ma & m, "tox_max"] > 0.5).mean() if len(v) else np.nan})

    lad, lif = pd.DataFrame(ladders), pd.DataFrame(lifts)
    for cohort in args.cohorts:
        viz.table(lad[lad.cohort == cohort], R, f"T45_ladder_{cohort}")
    viz.table(lif, R, "T46_lift")
    pd.set_option("display.width", 250)
    log.info("lift (30 days)\n" + lif[lif.horizon == "30 days"][["cohort", "action", "p_action_given_burst",
             "p_action_given_no_burst", "lift", "lift_low", "lift_high", "share_of_actions_flagged_by_crowd"]].to_string())

    # F46 ladder small multiples
    labs = [b[2] for b in BINS]
    f, axes = viz.plt.subplots(2, 4, figsize=(12, 6), squeeze=False)
    for ax, a in zip(axes.flat, ORDER):
        for cohort in args.cohorts:
            d = lad[lad.cohort == cohort].set_index("new_blockers_30d").reindex(labs)
            x = np.arange(len(labs))
            y = (d[a] * 100).where(d[a] > 0)          # zeros cannot be drawn on a log axis
            ax.plot(x, y, marker="o", ms=4, color=COH_COL[cohort], label=f"{cohort} accounts")
            ax.fill_between(x, (d[f"{a}_low"] * 100).where(d[a] > 0), (d[f"{a}_high"] * 100).where(d[a] > 0),
                            color=COH_COL[cohort], alpha=0.15, lw=0)
        if (lad[a] > 0).any():                       # an action with no cases at all cannot use a log axis
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(viz.plt.NullFormatter())
        else:
            ax.text(0.5, 0.5, "no cases", transform=ax.transAxes, ha="center", color=viz.MUTED)
        ax.set_xticks(np.arange(len(labs)), labs, fontsize=8)
        ax.set_title(ACTIONS[a][0].replace(" (", "\n("), fontsize=9)
        ax.tick_params(labelsize=8)
    axes.flat[-1].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of accounts with this action (log)", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("new blockers in 30 days", fontsize=8)
    axes.flat[0].legend(fontsize=8)
    viz.save(f, R, "F46_ladder", "Each panel: share of accounts that got this platform action within the same 30 days, "
             "by how many new blockers they got. Shaded: 95% CI.")

    # F47 lift + share flagged
    l30 = lif[lif.horizon == "30 days"]
    order = ORDER[::-1]
    f, (a1, a2) = viz.plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for i, cohort in enumerate(args.cohorts):
        d = l30[l30.cohort == cohort].set_index("action").reindex(order)
        y = np.arange(len(order)) + (0.15 if i == 0 else -0.15)
        a1.errorbar(d.lift, y, xerr=[(d.lift - d.lift_low).clip(lower=0), (d.lift_high - d.lift).clip(lower=0)], fmt="o",
                    color=COH_COL[cohort], ms=6, capsize=2, label=f"{cohort} accounts")
        a2.barh(y, d.share_of_actions_flagged_by_crowd * 100, height=0.3, color=COH_COL[cohort])
    a1.set_xscale("log")
    a1.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}"))
    a1.axvline(1, color=viz.MUTED, lw=1, ls=":")
    a1.set_yticks(np.arange(len(order)), [ACTIONS[a][0] for a in order], fontsize=8)
    a1.set_xlabel("lift: P(action | burst) / P(action | no burst), log scale")
    a1.set_title("Lift of each platform action after a block burst", fontsize=10)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    a2.set_xlabel("% of these platform actions where the crowd had a burst")
    a2.set_title("Share of each action the crowd had flagged", fontsize=10)
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    viz.save(f, R, "F47_lift", f"Burst = {args.k}+ new blockers; action and burst within the same 30 days. Whiskers: 95% CI.")

    # F48 lift by follower band
    if fol:
        fo = pd.DataFrame(fol)
        viz.table(fo, R, "T48_lift_by_followers")
        bl = [b[2] for b in FBANDS]
        f, ax = viz.fig(7.2, 3.8)
        for i, a in enumerate(["abuse", "review", "takedown"]):
            d = fo[fo.action == a].set_index("followers").reindex(bl)
            ax.errorbar(np.arange(len(bl)) + (i - 1) * 0.08, d.lift,
                        yerr=[(d.lift - d.lift_low).clip(lower=0), (d.lift_high - d.lift).clip(lower=0)],
                        marker="o", ms=5, capsize=2, color=viz.SERIES[i], label=ACTIONS[a][0])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.axhline(1, color=viz.MUTED, lw=1, ls=":")
        ax.set_xticks(np.arange(len(bl)), bl)
        ax.set_xlabel("followers at T0")
        ax.set_ylabel("lift (log scale)")
        ax.set_title("Lift within follower bands - established accounts")
        ax.legend(fontsize=8)
        viz.save(f, R, "F48_lift_by_followers", "If lift stays high inside each band, the crowd signal is not just visibility.")

    # F49 strongest action of crowd-flagged accounts
    fl = pd.DataFrame(flagged)
    viz.table(fl, R, "T49_crowd_flagged_outcomes")
    cats = SEVERITY + ["none"]
    f, axes = viz.plt.subplots(1, len(args.cohorts), figsize=(5.6 * len(args.cohorts), 3.8), sharey=True, squeeze=False)
    for ax, cohort in zip(axes.flat, args.cohorts):
        d = fl[(fl.cohort == cohort) & (fl.horizon == "30 days")]
        y = np.arange(len(cats))[::-1]
        for i, grp in enumerate(["burst (5+ blockers)", "no burst"]):
            v = d[d.group == grp].set_index("strongest_action").reindex(cats).share * 100
            ax.barh(y + (0.18 if i == 0 else -0.18), v, height=0.36, color=viz.SERIES[0] if i == 0 else viz.MUTED, label=grp)
            if i == 0:
                for yy, vv in zip(y, v):
                    ax.text(vv, yy + 0.18, f" {vv:.1f}%", va="center", fontsize=7, color=viz.INK2)
        ax.set_yticks(y, [ACTIONS[c][0] if c in ACTIONS else "no platform action" for c in cats], fontsize=8)
        ax.set_xscale("symlog", linthresh=1)
        ax.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("% of accounts (strongest action within 30 days)")
        ax.set_title(f"What the platform did - {cohort}")
        ax.grid(axis="y", visible=False)
    axes.flat[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    viz.save(f, R, "F49_crowd_flagged_outcomes", "Each account counted once, at its strongest action (top = strongest).")

    if toxr:
        viz.table(pd.DataFrame(toxr), R, "T49t_tox_by_action")

    # alignment between the block model and each platform-action model (step 08 runs)
    rows = []
    for out, tag in TAG_FOR.items():
        fa, fo_ = RQ / tag / "T41_alignment.csv", RQ / tag / "F41_outcome_overlap.csv"
        if not fa.exists():
            log.warning(f"no step-08 run for {out} ({RQ / tag}) - run 08 with --platform-outcome {out} --tag {tag}")
            continue
        d = pd.read_csv(fa).merge(pd.read_csv(fo_)[["cohort", "T", "B_and_T", "P(T|B)", "P(T|not B)"]], on="cohort")
        mt = pd.read_csv(RQ / tag / "T40_model_metrics.csv").groupby(["cohort", "model"]).roc_auc.mean().unstack()
        d = d.merge(mt.rename(columns={"B": "auc_B_model", "T": "auc_platform_model"}).reset_index(), on="cohort")
        d.insert(0, "platform_outcome", out)
        rows.append(d)
    if rows:
        al = pd.concat(rows, ignore_index=True)
        viz.table(al, R, "T4A_alignment_by_outcome")
        log.info("alignment by outcome\n" + al[["platform_outcome", "cohort", "T", "spearman_rho_mean", "top10_jaccard_mean",
                 "auc_B_model_predicting_T", "auc_T_model_predicting_B", "auc_platform_model"]].to_string())
        outs = [o for o in TAG_FOR if o in set(al.platform_outcome)][::-1]
        f, axes = viz.plt.subplots(1, 2, figsize=(10, 0.55 * len(outs) + 1.8), sharey=True)
        for ax, (col, lo, hi, xl) in zip(axes, [
                ("spearman_rho_mean", "rho_seed_min", "rho_seed_max", "rank correlation of SHAP importances (block model vs platform model)"),
                ("auc_B_model_predicting_T", "auc_B_predicting_T_min", "auc_B_predicting_T_max", "AUC: block-burst model finding this platform action")]):
            for i, cohort in enumerate(args.cohorts):
                d = al[al.cohort == cohort].set_index("platform_outcome").reindex(outs)
                y = np.arange(len(outs)) + (0.15 if i == 0 else -0.15)
                ax.errorbar(d[col], y, xerr=[(d[col] - d[lo]).clip(lower=0), (d[hi] - d[col]).clip(lower=0)], fmt="o",
                            color=COH_COL[cohort], ms=6, capsize=2, label=f"{cohort} accounts")
            ax.set_yticks(np.arange(len(outs)), [ACTIONS[o][0] for o in outs], fontsize=8)
            ax.set_xlabel(xl, fontsize=8)
            ax.grid(axis="y", visible=False)
        axes[1].axvline(0.5, color=viz.MUTED, lw=1, ls=":")
        axes[0].set_title("Does the crowd react to the same behaviour as each platform action?")
        axes[0].legend(fontsize=8, loc="lower left")
        viz.save(f, R, "F4A_alignment_by_outcome", "Same features and 30-day horizon for every model; points: mean over 5 seeds, bars: range.")
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
