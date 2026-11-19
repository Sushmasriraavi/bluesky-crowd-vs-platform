#!/usr/bin/env python3
"""RQ1 final figures: HOW crowd and platform differ (results/RQ1/final/).

  F44_dependence_<cohort>     for the features where the two models differ most (plus toxicity): how the feature
                              value changes block-burst risk (blue) and takedown risk (orange). y = mean SHAP value
                              (log-odds contribution); x = feature value in quantile bins.
  T44_group_profiles_<cohort> the 4 groups (blocked+taken down / crowd only / platform only / neither, 30-day windows,
                              5+ blockers): median and mean of key behaviours, and standardized difference vs "neither"
  F45_group_profiles_<cohort> dot plot of those standardized differences (which behaviours set each group apart)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str

GROUPS = ["1 blocked + taken down", "2 blocked only (crowd)", "3 taken down only (platform)", "4 neither"]
FEATS = {
    "established": ["acc_age_days", "net_followers", "net_followers_gained_30d", "net_following", "net_follows_made_30d",
                    "net_log_follower_ratio", "act_posts_90d", "act_reply_ratio", "act_reply_to_nonfollowed_ratio",
                    "act_link_ratio", "act_duplicate_ratio", "act_mention_ratio", "acc_days_since_last_commit",
                    "tox_mean", "tox_insult_mean", "tox_identity_attack_mean"],
    "new": ["new_act_ops", "new_act_active_hours", "new_act_follows_made", "new_act_likes_made", "new_act_posts",
            "new_act_reply_ratio", "new_act_link_ratio", "new_act_duplicate_ratio", "new_act_has_avatar",
            "new_act_has_bio", "new_act_profile_edits", "new_rcv_likes", "new_rcv_replies", "tox_mean", "tox_insult_mean"],
}


def dependence(samp, imp, cohort, R, n_panels=8):
    num = [f for f in imp.feature if f"shapB__{f}" in samp and pd.api.types.is_numeric_dtype(samp[f])
           and samp[f].nunique() >= 3]
    imp = imp[imp.feature.isin(num)].copy()
    imp["gap"] = (imp.share_B - imp.share_T).abs()
    pick = list(imp[imp[["share_B", "share_T"]].max(axis=1) >= 0.01].nlargest(n_panels - 1, "gap").feature)
    for t in ["tox_mean", "tox_insult_mean"]:
        if t in num and t not in pick and len(pick) < n_panels:
            pick.append(t)
    rows = []
    ncol = 4
    nrow = int(np.ceil(len(pick) / ncol))
    f, axes = viz.plt.subplots(nrow, ncol, figsize=(12, 2.9 * nrow + 0.6), squeeze=False)
    for ax, feat in zip(axes.flat, pick):
        d = samp[[feat, f"shapB__{feat}", f"shapT__{feat}"]].dropna()
        x = d[feat].astype(float)
        try:
            bins = pd.qcut(x.rank(method="first"), q=min(20, x.nunique()), labels=False)
        except ValueError:
            bins = pd.Series(0, index=x.index)
        g = d.assign(b=bins.values).groupby("b").agg(x=(feat, "median"), B=(f"shapB__{feat}", "mean"),
                                                      T=(f"shapT__{feat}", "mean"), n=(feat, "size"))
        g["feature"] = feat
        rows.append(g.reset_index(drop=True))
        ax.plot(g.x, g.B, color=viz.SERIES[0], marker="o", ms=3, label="block burst (crowd)")
        ax.plot(g.x, g["T"], color=viz.SERIES[1], marker="o", ms=3, label="takedown (platform)")
        ax.axhline(0, color=viz.MUTED, lw=0.8, ls=":")
        if (x > 0).all() and x.max() / max(x.min(), 1e-9) > 100:
            ax.set_xscale("log")
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(pick):]:
        ax.set_visible(False)
    axes.flat[0].set_ylabel("effect on risk (mean SHAP, log-odds)", fontsize=8)
    axes.flat[0].legend(fontsize=7, loc="best")
    viz.save(f, R, f"F44_dependence_{cohort}",
             f"{cohort} accounts: features where the two models differ most. Lines going opposite ways = crowd and platform react oppositely.")
    viz.table(pd.concat(rows), R, f"F44_dependence_{cohort}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--tag", default="main_tox", help="RQ1 run whose SHAP sample is used")
    ap.add_argument("--t0", default="2026-04-09")
    ap.add_argument("--fh-end", default="2026-06-01")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cohorts", nargs="*", default=["established", "new"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    RQ = cfg["paths"]["results"] / "RQ1"
    R = RQ / "final"
    log = get_logger("RQ1final", cfg["paths"]["results"] / "logs" / "R_rq1_final.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    H, T0, FHE = "INTERVAL 30 DAY", f"TIMESTAMPTZ '{args.t0} 00:00:00+00'", f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    be = rp(I / "crowd" / "block_edges.parquet")
    base = {"established": f"SELECT *, {T0} AS index_t FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d",
            "new": f"SELECT * FROM {rp(P / 'accounts_new.parquet')} WHERE NOT instant_takedown AND index_t + {H} <= {FHE}"}

    for cohort in args.cohorts:
        sp, ip = RQ / args.tag / f"shap_sample_{cohort}.parquet", RQ / args.tag / f"T42_shap_{cohort}.csv"
        if sp.exists() and ip.exists():
            dependence(pd.read_parquet(sp), pd.read_csv(ip), cohort, R)
            log.info(f"[{cohort}] dependence plots done")
        else:
            log.warning(f"[{cohort}] no SHAP sample in {RQ / args.tag} - run 08 with --tag {args.tag} first")

        tox = P / f"tox_{cohort}.parquet"
        tj = f"LEFT JOIN {rp(tox)} tx USING (did)" if tox.exists() else ""
        tc = ", tx.tox_mean, tx.tox_insult_mean, tx.tox_identity_attack_mean" if tox.exists() else ""
        cols = [c for c in FEATS[cohort] if not c.startswith("tox_")]
        df = con.execute(f"""
            WITH a AS ({base[cohort]}),
                 b AS (SELECT a.did, count(DISTINCT e.blocker_did) AS n FROM a JOIN {be} e
                       ON e.subject_did = a.did AND e.t_created > a.index_t AND e.t_created <= a.index_t + {H} GROUP BY 1)
            SELECT {', '.join('a.' + c for c in cols)}, coalesce(b.n, 0) >= {args.k} AS blocked,
                   coalesce(a.y_takedown_ts <= a.index_t + {H}, false) AS taken_down {tc}
            FROM a LEFT JOIN b USING (did) {tj}""").df()
        df["group"] = np.select([df.blocked & df.taken_down, df.blocked & ~df.taken_down, ~df.blocked & df.taken_down],
                                GROUPS[:3], GROUPS[3])
        feats = [c for c in FEATS[cohort] if c in df.columns]
        for c in feats:
            if df[c].dtype == bool or str(df[c].dtype) == "boolean":
                df[c] = df[c].astype(float)
        rows = []
        ref = df[df.group == GROUPS[3]]
        for c in feats:
            v = pd.to_numeric(df[c], errors="coerce")
            logit = (v.dropna() >= 0).all() and v.max() > 20          # heavy-tailed counts: compare on log1p scale
            tv = np.log1p(v) if logit else v
            r0 = tv[df.group == GROUPS[3]].dropna()
            for g in GROUPS:
                x = tv[df.group == g].dropna()
                raw = v[df.group == g].dropna()
                pooled = np.sqrt((x.var() + r0.var()) / 2) if len(x) > 1 else np.nan
                rows.append({"feature": c, "group": g, "accounts_with_value": len(raw), "median": raw.median(),
                             "mean": raw.mean(), "smd_vs_neither": (x.mean() - r0.mean()) / pooled if pooled else np.nan,
                             "scale": "log1p" if logit else "raw"})
        t = pd.DataFrame(rows)
        viz.table(t, R, f"T44_group_profiles_{cohort}")
        sizes = df.group.value_counts().reindex(GROUPS).fillna(0).astype(int)
        log.info(f"[{cohort}] group sizes: {sizes.to_dict()}")

        piv = t[t.group != GROUPS[3]].pivot(index="feature", columns="group", values="smd_vs_neither").reindex(feats)
        f, ax = viz.fig(7.8, 0.33 * len(feats) + 1.6)
        y = np.arange(len(feats))[::-1]
        for i, g in enumerate(GROUPS[:3]):
            ax.scatter(piv[g], y + (0.18 - 0.18 * i), color=viz.SERIES[i], s=30, label=f"{g} (n={sizes[g]:,})", zorder=3)
        ax.axvline(0, color=viz.MUTED, lw=1, ls=":")
        ax.set_yticks(y, feats, fontsize=8)
        ax.set_xlabel("standardized difference vs accounts that were neither blocked nor taken down")
        ax.set_title(f"What sets each group apart - {cohort} accounts")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09 - 0.6 / len(feats)), ncol=3, fontsize=7)
        viz.save(f, R, f"F45_group_profiles_{cohort}",
                 "Right of 0 = higher than ordinary accounts. Counts compared on log scale. Groups: 5+ new blockers / takedown within 30 days.")
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
