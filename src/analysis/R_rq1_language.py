#!/usr/bin/env python3
"""RQ1 by language: do crowd and platform treat accounts differently depending on the language they post in?

Language = an account's most-used post language (act_top_lang / new_act_top_lang), normalised to the base code
(en-US -> en). Top --top languages are kept; the rest = "other"; accounts without a post language = "none".
Full cohorts (not the model sample), same 30-day windows as RQ1 (burst = 5+ new blockers).

Outputs (results/RQ1/language/)
  T70_lang_rates_<cohort>     per language: accounts, block-burst rate, rate of every platform action (95% CI),
                              mean toxicity, share added to a moderation list in the 30 days
  T71_lang_lift_<cohort>      per language x action: P(action | burst), P(action | no burst), lift (CI),
                              share of the platform's actions the crowd had flagged  -> does agreement differ by language?
  T72_lang_blockers           established: who blocks each language - share of blocks from blockers posting in
                              the same language (in-community conflict vs outside reaction)
  T73_lang_shap               mean SHAP effect of each language in the block model and each platform-action model
                              (adjusted for all other features; from step 08 SHAP samples)
  F70_lang_rates_<cohort>     block-burst rate vs takedown / abuse-label rate per language
  F71_lang_lift_<cohort>      lift and crowd coverage of abuse labels and takedowns per language
  F72_lang_shap               adjusted effect of language: block model vs takedown and abuse models
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str
from bskymod.outcomes import ACTIONS, ORDER, flags_sql

TAGS = {"takedown": "main_tox", "review": "out_review", "abuse": "out_abuse", "adult": "out_adult"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def lift(y, b):
    a, n1, c, n0 = int((y & b).sum()), int(b.sum()), int((y & ~b).sum()), int((~b).sum())
    out = {"accounts_burst": n1, "actioned_burst": a, "actioned_no_burst": c,
           "p_action_given_burst": a / n1 if n1 else np.nan, "p_action_given_no_burst": c / n0 if n0 else np.nan,
           "share_of_actions_flagged_by_crowd": a / (a + c) if a + c else np.nan,
           "lift": np.nan, "lift_low": np.nan, "lift_high": np.nan}
    if n1 and n0 and a + c and a >= 3:          # too few events -> no ratio
        aa, cc = (a, c) if c else (a + 0.5, c + 0.5)
        rr = (aa / n1) / (cc / n0)
        se = np.sqrt(1 / aa - 1 / n1 + 1 / cc - 1 / n0)
        out.update(lift=rr, lift_low=rr * np.exp(-1.96 * se), lift_high=rr * np.exp(1.96 * se))
    return out


def norm_lang(col):
    return f"nullif(lower(split_part(replace(CAST({col} AS VARCHAR), '_', '-'), '-', 1)), '')"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--t0", default="2026-04-09")
    ap.add_argument("--fh-end", default="2026-06-01")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--cohorts", nargs="*", default=["established", "new"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    RQ = cfg["paths"]["results"] / "RQ1"
    R = RQ / "language"
    log = get_logger("RQ1lang", cfg["paths"]["results"] / "logs" / "R_rq1_language.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    H = "INTERVAL 30 DAY"
    T0 = f"TIMESTAMPTZ '{args.t0} 00:00:00+00'"
    FHE = f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    L = I / "labels"
    ae, re_ = rp(L / "account_label_events.parquet"), rp(L / "record_label_events.parquet")
    C = I / "crowd"
    be, li, la = rp(C / "block_edges.parquet"), rp(C / "listitem_edges.parquet"), rp(C / "lists_all.parquet")
    base = {"established": f"SELECT did, {T0} AS index_t, {norm_lang('act_top_lang')} AS lang "
                           f"FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d",
            "new": f"SELECT did, index_t, {norm_lang('new_act_top_lang')} AS lang FROM {rp(P / 'accounts_new.parquet')} "
                   f"WHERE NOT instant_takedown AND index_t + {H} <= {FHE}"}
    pd.set_option("display.width", 250)
    cohort_langs = {}

    for cohort in args.cohorts:
        con.execute(f"CREATE OR REPLACE TEMP TABLE c AS {base[cohort]}")
        top = [r[0] for r in con.execute(f"""SELECT lang FROM c WHERE lang IS NOT NULL GROUP BY 1
                                              ORDER BY count(*) DESC LIMIT {args.top}""").fetchall()]
        langs = top + ["other", "none"]
        cohort_langs[cohort] = langs
        con.execute(f"""CREATE OR REPLACE TEMP TABLE c2 AS SELECT did, index_t,
                        CASE WHEN lang IS NULL THEN 'none' WHEN lang IN ({', '.join(repr(x) for x in top)}) THEN lang
                             ELSE 'other' END AS lang FROM c""")
        df = con.execute(f"""
            WITH b AS (SELECT c2.did, count(DISTINCT e.blocker_did) AS n FROM c2 JOIN {be} e
                       ON e.subject_did = c2.did AND e.t_created > c2.index_t AND e.t_created <= c2.index_t + {H} GROUP BY 1),
                 m AS (SELECT DISTINCT c2.did FROM c2 JOIN {li} i ON i.subject_did = c2.did
                       AND i.t_added > c2.index_t AND i.t_added <= c2.index_t + {H}
                       JOIN {la} l ON l.list_uri = i.list_uri AND l.purpose LIKE '%modlist')
            SELECT c2.did, c2.lang, coalesce(b.n, 0) AS blockers, m.did IS NOT NULL AS added_to_modlist
            FROM c2 LEFT JOIN b USING (did) LEFT JOIN m USING (did)""").df()
        df = df.merge(con.execute(flags_sql("SELECT did, index_t FROM c2", ae, re_, f"c.index_t + {H}")).df(), on="did")
        tox = P / f"tox_{cohort}.parquet"
        if tox.exists():
            df = df.merge(con.execute(f"SELECT did, tox_mean FROM {rp(tox)}").df(), on="did", how="left")
        df["burst"] = df.blockers >= args.k

        rates, lifts = [], []
        for lg in langs + ["ALL"]:
            d = df if lg == "ALL" else df[df.lang == lg]
            n = len(d)
            r = {"lang": lg, "accounts": n, "share_of_cohort": n / len(df)}
            for name, col in [("block_burst", "burst"), ("added_to_modlist", "added_to_modlist")] + \
                             [(a, f"act_{a}") for a in ORDER]:
                k = int(d[col].sum())
                r[name], r[f"{name}_n"] = (k / n if n else np.nan), k
                r[f"{name}_low"], r[f"{name}_high"] = wilson(k, n)
            r["tox_mean"] = d.tox_mean.mean() if "tox_mean" in d else np.nan
            rates.append(r)
            for a in ORDER:
                lifts.append({"lang": lg, "action": a, **lift(d[f"act_{a}"], d.burst)})
        rt, lt = pd.DataFrame(rates), pd.DataFrame(lifts)
        viz.table(rt, R, f"T70_lang_rates_{cohort}")
        viz.table(lt, R, f"T71_lang_lift_{cohort}")
        log.info(f"[{cohort}] rates\n" + rt[["lang", "accounts", "block_burst", "takedown", "abuse", "adult", "review",
                                              "suspend", "added_to_modlist", "tox_mean"]].round(5).to_string())
        log.info(f"[{cohort}] lift (abuse / takedown)\n" + lt[lt.action.isin(["abuse", "takedown"])][
            ["lang", "action", "actioned_burst", "actioned_no_burst", "lift", "share_of_actions_flagged_by_crowd"]].round(3).to_string())

        # F70: block-burst rate vs takedown and abuse-label rate per language
        d = rt[rt.lang != "ALL"].set_index("lang").reindex(langs)
        y = np.arange(len(langs))[::-1]
        f, axes = viz.plt.subplots(1, 3, figsize=(12, 0.32 * len(langs) + 1.8), sharey=True)
        for ax, (col, name, colr) in zip(axes, [("block_burst", "block burst (5+ new blockers)", viz.SERIES[0]),
                                                ("takedown", "account taken down", viz.SERIES[1]),
                                                ("abuse", "abuse label (rude/intolerant/threat)", viz.SERIES[2])]):
            ax.barh(y, d[col] * 100, color=colr, height=0.65,
                    xerr=[(d[col] - d[f"{col}_low"]).clip(lower=0) * 100, (d[f"{col}_high"] - d[col]).clip(lower=0) * 100],
                    error_kw={"elinewidth": 0.8, "ecolor": viz.INK2})
            allv = rt.loc[rt.lang == "ALL", col].iloc[0] * 100
            ax.axvline(allv, color=viz.MUTED, lw=1, ls=":")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("% of accounts in 30 days", fontsize=8)
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(y, [f"{lg} (n={int(n):,})" for lg, n in zip(langs, d.accounts)], fontsize=8)
        viz.save(f, R, f"F70_lang_rates_{cohort}", f"{cohort} accounts by main posting language. Dotted line: all accounts. Whiskers: 95% CI.")

        # F71: lift + crowd coverage for abuse labels and takedowns per language
        f, (a1, a2) = viz.plt.subplots(1, 2, figsize=(11, 0.32 * len(langs) + 1.8), sharey=True)
        for i, a in enumerate(["abuse", "takedown"]):
            v = lt[lt.action == a].set_index("lang").reindex(langs)
            yy = y + (0.15 if i == 0 else -0.15)
            col = viz.SERIES[2] if a == "abuse" else viz.SERIES[1]
            a1.errorbar(v.lift, yy, xerr=[(v.lift - v.lift_low).clip(lower=0), (v.lift_high - v.lift).clip(lower=0)],
                        fmt="o", ms=5, capsize=2, color=col, label=ACTIONS[a][0])
            a2.barh(yy, v.share_of_actions_flagged_by_crowd * 100, height=0.3, color=col)
        a1.set_xscale("log")
        a1.xaxis.set_major_formatter(viz.plt.FuncFormatter(lambda v, _: f"{v:g}"))
        a1.axvline(1, color=viz.MUTED, lw=1, ls=":")
        a1.set_yticks(y, langs, fontsize=8)
        a1.set_xlabel("lift after a block burst (log)")
        a1.set_title("Crowd-platform agreement by language", fontsize=10)
        a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        a2.set_xlabel("% of these actions where the crowd had a burst")
        a2.set_title("Share the crowd had flagged", fontsize=10)
        for ax in (a1, a2):
            ax.grid(axis="y", visible=False)
        viz.save(f, R, f"F71_lang_lift_{cohort}", "Lift shown only where the burst group had 3+ actioned accounts. Whiskers: 95% CI.")

    # who blocks each language (established): same-language blockers vs others
    if "established" in args.cohorts:
        langs = cohort_langs["established"]
        con.execute(f"""CREATE OR REPLACE TEMP TABLE al AS SELECT did, {norm_lang('act_top_lang')} AS lang
                        FROM {rp(P / 'accounts_T0.parquet')} WHERE act_top_lang IS NOT NULL""")
        # c2 was overwritten by the last cohort; rebuild it for established
        con.execute(f"CREATE OR REPLACE TEMP TABLE c AS {base['established']}")
        top = [x for x in langs if x not in ("other", "none")]
        con.execute(f"""CREATE OR REPLACE TEMP TABLE c2 AS SELECT did, index_t,
                        CASE WHEN lang IS NULL THEN 'none' WHEN lang IN ({', '.join(repr(x) for x in top)}) THEN lang
                             ELSE 'other' END AS lang FROM c""")
        bl = con.execute(f"""
            SELECT s.lang AS lang, count(*) AS blocks,
                   count(*) FILTER (bk.lang IS NOT NULL) AS blocks_from_known_lang,
                   count(*) FILTER (bk.lang = s.lang) AS blocks_same_lang,
                   count(*) FILTER (bk.lang = 'en') AS blocks_from_en
            FROM c2 s JOIN {be} e ON e.subject_did = s.did AND e.t_created > s.index_t AND e.t_created <= s.index_t + {H}
            LEFT JOIN al bk ON bk.did = e.blocker_did
            GROUP BY 1""").df()
        bl["share_same_lang"] = bl.blocks_same_lang / bl.blocks_from_known_lang
        bl["share_from_en"] = bl.blocks_from_en / bl.blocks_from_known_lang
        bl = bl.set_index("lang").reindex(langs).reset_index()
        viz.table(bl, R, "T72_lang_blockers")
        log.info("who blocks each language\n" + bl.round(3).to_string())

    # adjusted effect of language (SHAP) in each model
    rows = []
    for cohort, col in [("established", "act_top_lang"), ("new", "new_act_top_lang")]:
        for out, tag in TAGS.items():
            sp = RQ / tag / f"shap_sample_{cohort}.parquet"
            if not sp.exists():
                continue
            s = pd.read_parquet(sp, columns=[col, f"shapB__{col}", f"shapT__{col}"])
            g = s.groupby(col).agg(sample_rows=(col, "size"), effect_block=(f"shapB__{col}", "mean"),
                                   effect_platform=(f"shapT__{col}", "mean")).reset_index().rename(columns={col: "lang"})
            g.insert(0, "platform_outcome", out)
            g.insert(0, "cohort", cohort)
            rows.append(g)
    if rows:
        sh = pd.concat(rows, ignore_index=True)
        viz.table(sh, R, "T73_lang_shap")
        e = sh[sh.cohort == "established"]
        if len(e):
            order = e[e.platform_outcome == "takedown"].sort_values("sample_rows", ascending=False).lang.tolist()
            f, ax = viz.fig(7.5, 0.33 * len(order) + 1.6)
            y = np.arange(len(order))[::-1]
            blk = e[e.platform_outcome == "takedown"].set_index("lang").reindex(order).effect_block
            ax.scatter(blk, y, color=viz.SERIES[0], s=34, label="block-burst model (crowd)", zorder=3)
            for i, (out, colr) in enumerate([("takedown", viz.SERIES[1]), ("abuse", viz.SERIES[2])]):
                v = e[e.platform_outcome == out].set_index("lang").reindex(order).effect_platform
                ax.scatter(v, y + (0.18 if i == 0 else -0.18), color=colr, s=34, marker="D", label=f"{ACTIONS[out][0]} model", zorder=3)
            ax.axvline(0, color=viz.MUTED, lw=1, ls=":")
            ax.set_yticks(y, order, fontsize=8)
            ax.set_xlabel("mean SHAP effect of the language (log-odds), adjusted for all other features")
            ax.set_title("Effect of posting language - established accounts")
            ax.grid(axis="y", visible=False)
            ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.5 / len(order)), ncol=3)
            viz.save(f, R, "F72_lang_shap", "Right of 0 = this language raises the risk compared with an average account. "
                     "Language groups as used in the models (top 10 + other).")
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
