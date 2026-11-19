#!/usr/bin/env python3
"""Toxicity across languages with the multilingual scores of step 11 (results/TOX/multi/).

  T61_tox_validation_<cohort>   per language: accounts scored, mean toxicity (old Detoxify vs new), share of accounts
                                with a toxic post, and how well each score separates accounts that got a platform
                                abuse label (AUC) -> which score agrees better with the platform, per language
  T62_tox_groups_lang_<cohort>  per language x 4 crowd/platform groups (5+ blockers / takedown, 30 days):
                                mean toxicity and share of accounts with a toxic post (95% CI)
  F62_tox_groups_lang_<cohort>  crowd-only vs platform-only vs ordinary accounts, per language
  F63_tox_validation_<cohort>   AUC for abuse labels per language: Detoxify vs new model
Language = main language of the account's scored posts (post lang tags).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str
from bskymod.outcomes import flags_sql

GROUPS = ["1 blocked + taken down", "2 blocked only (crowd)", "3 taken down only (platform)", "4 neither"]


def ci(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--t0", default="2026-04-09")
    ap.add_argument("--fh-end", default="2026-06-01")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--cohorts", nargs="*", default=["established", "new"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    R = cfg["paths"]["results"] / "TOX" / "multi"
    log = get_logger("TOXm", cfg["paths"]["results"] / "logs" / "R_tox_multi.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    H = "INTERVAL 30 DAY"
    T0 = f"TIMESTAMPTZ '{args.t0} 00:00:00+00'"
    FHE = f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    L = I / "labels"
    ae, re_ = rp(L / "account_label_events.parquet"), rp(L / "record_label_events.parquet")
    be = rp(I / "crowd" / "block_edges.parquet")
    base = {"established": f"SELECT did, {T0} AS index_t FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d",
            "new": f"SELECT did, index_t FROM {rp(P / 'accounts_new.parquet')} "
                   f"WHERE NOT instant_takedown AND index_t + {H} <= {FHE}"}
    pd.set_option("display.width", 250)

    for cohort in args.cohorts:
        t2 = P / f"tox2_{cohort}.parquet"
        if not t2.exists():
            log.warning(f"[{cohort}] {t2} missing - run step 11 first")
            continue
        con.execute(f"CREATE OR REPLACE TEMP TABLE c AS {base[cohort]}")
        df = con.execute(f"""
            WITH b AS (SELECT c.did, count(DISTINCT e.blocker_did) AS n FROM c JOIN {be} e
                       ON e.subject_did = c.did AND e.t_created > c.index_t AND e.t_created <= c.index_t + {H} GROUP BY 1)
            SELECT c.did, coalesce(b.n, 0) >= {args.k} AS burst, t2.*  EXCLUDE (did), t1.tox_mean, t1.tox_max
            FROM c JOIN {rp(t2)} t2 USING (did) LEFT JOIN b USING (did)
            LEFT JOIN {rp(P / f'tox_{cohort}.parquet')} t1 USING (did)""").df()
        fl = con.execute(flags_sql("SELECT did, index_t FROM c", ae, re_, f"c.index_t + {H}",
                                   ["abuse", "takedown"])).df()
        df = df.merge(fl, on="did", how="left").fillna({"act_abuse": False, "act_takedown": False})
        df["group"] = np.select([df.burst & df.act_takedown, df.burst & ~df.act_takedown, ~df.burst & df.act_takedown],
                                GROUPS[:3], GROUPS[3])
        df["toxic_post"] = df.tox2_max > 0.5
        df["toxic_post_old"] = df.tox_max > 0.5
        top = df.tox2_lang.value_counts().index[:args.top].tolist()
        df["lang"] = np.where(df.tox2_lang.isin(top), df.tox2_lang, "other")
        langs = top + ["other"]

        # validation per language
        rows = []
        for lg in langs + ["ALL"]:
            d = df if lg == "ALL" else df[df.lang == lg]
            r = {"lang": lg, "accounts_scored": len(d), "abuse_labelled": int(d.act_abuse.sum()),
                 "mean_tox_old_detoxify": d.tox_mean.mean(), "mean_tox_new": d.tox2_mean.mean(),
                 "share_toxic_post_old": d.toxic_post_old.mean(), "share_toxic_post_new": d.toxic_post.mean(),
                 "lang_supported_share": d.tox2_supported_share.mean()}
            for name, col in [("old_detoxify", "tox_max"), ("new", "tox2_max")]:
                v = d[[col, "act_abuse"]].dropna()
                r[f"auc_abuse_{name}"] = roc_auc_score(v.act_abuse, v[col]) if v.act_abuse.sum() >= 10 and (~v.act_abuse).sum() else np.nan
            rows.append(r)
        val = pd.DataFrame(rows)
        viz.table(val, R, f"T61_tox_validation_{cohort}")
        log.info(f"[{cohort}] validation\n{val.round(4).to_string()}")

        # groups per language
        rows = []
        for lg in langs + ["ALL"]:
            d = df if lg == "ALL" else df[df.lang == lg]
            for g in GROUPS:
                x = d[d.group == g]
                n = len(x)
                p = x.toxic_post.mean() if n else np.nan
                rows.append({"lang": lg, "group": g, "accounts": n, "mean_tox": x.tox2_mean.mean(),
                             "mean_tox_ci": 1.96 * x.tox2_mean.std() / np.sqrt(n) if n > 1 else np.nan,
                             "share_toxic_post": p, "share_toxic_post_ci": ci(p, n)})
        gt = pd.DataFrame(rows)
        viz.table(gt, R, f"T62_tox_groups_lang_{cohort}")
        log.info(f"[{cohort}] groups by language\n" + gt.pivot(index="lang", columns="group", values="share_toxic_post")
                 .reindex(langs + ["ALL"]).round(4).to_string())

        # F62: crowd-only vs platform-only vs ordinary per language
        order = (langs + ["ALL"])[::-1]
        f, ax = viz.fig(7.8, 0.36 * len(order) + 1.6)
        y = np.arange(len(order))
        for i, (g, col) in enumerate([(GROUPS[1], viz.SERIES[0]), (GROUPS[2], viz.SERIES[1]), (GROUPS[3], viz.MUTED)]):
            v = gt[gt.group == g].set_index("lang").reindex(order)
            ok = v.accounts >= 20
            ax.errorbar((v.share_toxic_post * 100).where(ok), y + (0.2 - 0.2 * i), xerr=(v.share_toxic_post_ci * 100).where(ok),
                        fmt="o", ms=5, capsize=2, color=col, label=g)
        ax.set_yticks(y, order, fontsize=8)
        ax.set_xlabel("% of accounts with at least one toxic post (new multilingual model, score > 0.5)")
        ax.set_title(f"Toxicity by group and language - {cohort}")
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1 - 0.5 / len(order)), ncol=3)
        viz.save(f, R, f"F62_tox_groups_lang_{cohort}", "Groups with fewer than 20 accounts not shown. Whiskers: 95% CI.")

        # F63: does the new model agree better with platform abuse labels?
        v = val[val.auc_abuse_new.notna()].set_index("lang")
        if len(v):
            order = v.index.tolist()[::-1]
            f, ax = viz.fig(6.8, 0.36 * len(order) + 1.6)
            y = np.arange(len(order))
            ax.barh(y + 0.18, v.reindex(order).auc_abuse_old_detoxify, height=0.36, color=viz.MUTED, label="Detoxify (old)")
            ax.barh(y - 0.18, v.reindex(order).auc_abuse_new, height=0.36, color=viz.SERIES[0], label="multilingual (new)")
            ax.axvline(0.5, color=viz.INK2, lw=1, ls=":")
            ax.set_xlim(0.4, 1)
            ax.set_yticks(y, [f"{lg} ({int(v.loc[lg, 'abuse_labelled'])})" for lg in order], fontsize=8)
            ax.set_xlabel("AUC: most toxic post score separating abuse-labelled accounts")
            ax.set_title(f"Toxicity scores vs platform abuse labels - {cohort}")
            ax.grid(axis="y", visible=False)
            ax.legend(fontsize=8, loc="lower right")
            viz.save(f, R, f"F63_tox_validation_{cohort}", "Only languages with 10+ abuse-labelled accounts. In brackets: abuse-labelled accounts.")
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
