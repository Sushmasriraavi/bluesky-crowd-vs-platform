#!/usr/bin/env python3
"""RQ3 - blind spots: what does each system miss? (results/RQ3/)

Established cohort, same 30-day window as RQ1/RQ2 (T0 = 9 April 2026). Crowd signal = ordinary-blocker burst
(5+ blockers who had made <= --heavy blocks before T0). Platform sanction = abuse label, post removal,
suspension or takedown (review flags are triage and reported apart) (adult-content labels are content warnings, reported separately).

An INDEPENDENT judge: an XGBoost model trained to recognise accounts that got a platform abuse label, from behaviour,
multilingual toxicity (step 11) only - no language, block or list data - scored out-of-fold (5 folds), so every
account is judged by a model that never saw it. "Abuse-like" = score at or above the median score of the accounts
that really got an abuse label (the model's 50%-recall point).

1. Crowd-only blind spot: accounts with an ordinary-blocker burst but no sanction in 30 days. Mutually exclusive
   breakdown: sanctioned later (day 31 .. 1 July) / adult-content label only / abuse-like / toxic post but not
   abuse-like / no harm indicator. The same breakdown for ordinary accounts gives the baseline, so the EXCESS over
   ordinary accounts estimates missed harm.
2. Platform-only blind spot: takedowns without an ordinary-blocker burst: spam-like (young, almost no followers) or
   not, abuse-like or not, and whether anyone blocked them at all.
3. Double blind spot by language: among abuse-like accounts, the share the platform sanctioned, the crowd flagged,
   both, or neither - per language.

         4. Validation: do abuse-like crowd-only accounts get sanctioned later, or blocked again, more often?
         5. Visibility: among abuse-like accounts, who acts (platform / crowd) by account size?
         6. Language-fair harm: the most toxic 2% of accounts WITHIN each language - who acts on them?
Outputs: T100_groups, T101_crowd_only_breakdown, F101_crowd_only_breakdown, T102_platform_only, F102_platform_only,
         T103_language_blindspots, F103_language_blindspots, T104_judge_model, F104_judge_scores
"""
from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str
from bskymod.outcomes import flags_sql

warnings.filterwarnings("ignore", category=DeprecationWarning)
SANCTIONS = ["abuse", "post_removed", "suspend", "takedown"]   # enforcement; review = triage, reported apart
GROUPS = ["crowd + platform", "crowd only", "platform only", "neither"]


def load_prepare():
    p = Path(__file__).with_name("08_rq1_models.py")
    spec = importlib.util.spec_from_file_location("rq1models", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.prepare


def norm_lang(col):
    return f"nullif(lower(split_part(replace(CAST({col} AS VARCHAR), '_', '-'), '-', 1)), '')"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--t0", default="2026-04-09")
    ap.add_argument("--labels-end", default="2026-07-01")
    ap.add_argument("--fh-end", default="2026-06-01", help="end of block data (for the repeat-burst window)")
    ap.add_argument("--top-toxic", type=float, default=0.02, help="most toxic share within each language")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--heavy", type=int, default=1000)
    ap.add_argument("--sample-other", type=int, default=1_500_000, help="random ordinary accounts used for the judge")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    R = cfg["paths"]["results"] / "RQ3"
    log = get_logger("14", cfg["paths"]["results"] / "logs" / "14_rq3.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    T0 = f"TIMESTAMPTZ '{args.t0} 00:00:00+00'"
    LEND = f"TIMESTAMPTZ '{args.labels_end} 00:00:00+00'"
    FHE = f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    H = "INTERVAL 30 DAY"
    L, C = I / "labels", I / "crowd"
    ae, re_ = rp(L / "account_label_events.parquet"), rp(L / "record_label_events.parquet")
    be = rp(C / "block_edges.parquet")
    pd.set_option("display.width", 250)

    # ---------- data
    con.execute(f"""CREATE OR REPLACE TEMP TABLE heavy AS
        SELECT blocker_did AS did FROM {be} WHERE coalesce(t_created, {T0} - INTERVAL 1 DAY) < {T0}
        GROUP BY 1 HAVING count(*) > {args.heavy}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
        SELECT did, {T0} AS index_t FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d""")
    blk = con.execute(f"""
        SELECT c.did, count(DISTINCT e.blocker_did) FILTER (e.t_created <= c.index_t + {H}) AS n_all,
               count(DISTINCT e.blocker_did) FILTER (h.did IS NULL AND e.t_created <= c.index_t + {H}) AS n_ord,
               count(DISTINCT e.blocker_did) FILTER (h.did IS NULL AND e.t_created > c.index_t + {H}) AS n_ord_later
        FROM c JOIN {be} e ON e.subject_did = c.did AND e.t_created > c.index_t AND e.t_created < {FHE}
        LEFT JOIN heavy h ON h.did = e.blocker_did GROUP BY 1""").df()
    acts = SANCTIONS + ["adult", "review"]
    ren = lambda d: d.rename(columns=lambda c: c.replace("act_", "fl_", 1) if c.startswith("act_") else c)
    f30 = ren(con.execute(flags_sql("SELECT did, index_t FROM c", ae, re_, f"c.index_t + {H}", acts)).df())
    fend = ren(con.execute(flags_sql("SELECT did, index_t FROM c", ae, re_, LEND, acts)).df())
    t2 = P / "tox2_established.parquet"
    tox = con.execute(f"SELECT did, tox2_mean, tox2_max, tox2_share_toxic, tox2_n_scored FROM {rp(t2)}").df() if t2.exists() else None
    acc = con.execute(f"SELECT * FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d").df()
    df = acc.merge(blk, on="did", how="left").merge(f30, on="did").merge(fend, on="did", suffixes=("", "_end"))
    if tox is not None:
        df = df.merge(tox, on="did", how="left")
    df[["n_all", "n_ord", "n_ord_later"]] = df[["n_all", "n_ord", "n_ord_later"]].fillna(0)
    df["lang"] = df.act_top_lang.astype("string").str.replace("_", "-").str.split("-").str[0].str.lower()
    df["burst"] = df.n_ord >= args.k
    df["sanction"] = df[[f"fl_{a}" for a in SANCTIONS]].any(axis=1)
    df["sanction_end"] = df[[f"fl_{a}_end" for a in SANCTIONS]].any(axis=1)
    df["group"] = np.select([df.burst & df.sanction, df.burst & ~df.sanction, ~df.burst & df.sanction],
                            GROUPS[:3], GROUPS[3])
    log.info(f"{len(df):,} accounts; groups: {df.group.value_counts().to_dict()}")

    # ---------- independent judge: abuse-label model without crowd features, out-of-fold
    import xgboost as xgb
    prepare = load_prepare()
    y = df.fl_abuse.astype(int)
    rng = np.random.default_rng(42)
    special = (y == 1) | df.burst | df.fl_takedown
    others = np.flatnonzero(~special.values)
    pick = rng.choice(others, min(args.sample_other, len(others)), replace=False)
    idx = np.concatenate([np.flatnonzero(special.values), pick])
    w_other = len(others) / len(pick)
    s = df.iloc[idx].copy()
    s["w"] = np.where(special.values[idx], 1.0, w_other)        # weight back to the population
    drop = [c for c in s.columns if c in ("n_all", "n_ord", "n_ord_later", "tox2_n_scored", "burst", "sanction", "sanction_end", "group", "lang", "w") or c.startswith("fl_")]
    X = prepare(s.drop(columns=drop), log)
    # language-free judge: the platform's abuse labels are mostly on English accounts, so a judge that sees the language would
    # inherit that bias and could not be used to compare languages
    X = X[[c for c in X.columns if not c.startswith(("crowd_", "net_blocks", "net_modlists", "net_curatelists"))
           and c not in ("act_top_lang", "act_n_langs")]]
    log.info(f"judge features ({X.shape[1]}): {list(X.columns)}")
    ys = y.iloc[idx].values
    oof = np.zeros(len(s))
    aucs = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, ys)):
        pos = ys[tr].sum()
        m = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=5, tree_method="hist", enable_categorical=True, eval_metric="aucpr",
                              scale_pos_weight=(len(tr) - pos) / pos, early_stopping_rounds=50, random_state=fold,
                              device="cuda" if args.gpu else "cpu", n_jobs=args.threads)
        cut = int(len(tr) * 0.9)
        perm = rng.permutation(tr)
        m.fit(X.iloc[perm[:cut]], ys[perm[:cut]], eval_set=[(X.iloc[perm[cut:]], ys[perm[cut:]])], verbose=False)
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(ys[te], oof[te], sample_weight=s.w.values[te]))
        log.info(f"judge fold {fold}: AUC {aucs[-1]:.3f}")
    s["score"] = oof
    thr = float(np.median(s.loc[s.fl_abuse, "score"]))
    s["abuse_like"] = s.score >= thr
    jm = pd.DataFrame([{"accounts_in_sample": len(s), "abuse_labelled": int(ys.sum()), "auc_mean": np.mean(aucs),
                        "auc_min": np.min(aucs), "pr_auc": average_precision_score(ys, oof, sample_weight=s.w),
                        "threshold_median_abuse_score": thr, "weight_other": w_other}])
    viz.table(jm, R, "T104_judge_model")
    log.info(f"judge model\n{jm.round(4).to_string()}")
    wsum = lambda d: d.w.sum()

    # F104 score distribution by group
    f, ax = viz.fig(7.2, 3.6)
    for g, colr, lab in [("abuse", viz.SERIES[1], "got an abuse label"), ("crowd only", viz.SERIES[0], "crowd only (no sanction)"),
                         ("neither", viz.MUTED, "neither")]:
        d = s[s.fl_abuse] if g == "abuse" else s[(s.group == g) & ~s.fl_abuse]
        v = np.sort(d.score.values)
        wv = d.w.values[np.argsort(d.score.values)]
        ax.step(v, np.cumsum(wv) / wv.sum(), where="post", color=colr, label=f"{lab} (n={len(d):,})")
    ax.axvline(thr, color=viz.INK2, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("judge score (resemblance to abuse-labelled accounts, out-of-fold)")
    ax.set_ylabel("share of accounts (cumulative)")
    ax.set_title("How abuse-like are the accounts the crowd flags but the platform leaves?", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    viz.save(f, R, "F104_judge_scores", "Dotted line: the score reached by half of the accounts that really got an abuse label "
             "(the 'abuse-like' threshold). Judge uses behaviour and multilingual toxicity, never language or blocks.")

    # ---------- groups overview
    rows = []
    for g in GROUPS:
        d = s[s.group == g]
        rows.append({"group": g, "accounts": round(wsum(d)), "abuse_like_share": (d.w * d.abuse_like).sum() / max(wsum(d), 1),
                     "toxic_post_share": (d.w * (d.tox2_max > 0.5)).sum() / max(wsum(d), 1),
                     "median_followers": d.net_followers.median(), "median_age_days": d.acc_age_days.median()})
    viz.table(pd.DataFrame(rows), R, "T100_groups")
    log.info(f"groups\n{pd.DataFrame(rows).round(4).to_string()}")

    # ---------- 1. crowd-only breakdown vs ordinary accounts
    CATS = ["sanctioned later (day 31 - 1 July)", "reviewed by platform, not sanctioned", "adult-content label only",
            "abuse-like (judge)", "toxic post, not abuse-like", "no harm indicator"]

    def breakdown(d):
        cat = np.select([d.sanction_end & ~d.sanction, d.fl_review, d.fl_adult, d.abuse_like, d.tox2_max > 0.5], CATS[:5], CATS[5])
        return pd.Series(d.w.values, index=cat).groupby(level=0).sum().reindex(CATS).fillna(0) / max(wsum(d), 1)

    co, ne = s[s.group == "crowd only"], s[s.group == "neither"]
    bd = pd.DataFrame({"crowd_only_share": breakdown(co), "ordinary_share": breakdown(ne)})
    bd["crowd_only_accounts"] = (bd.crowd_only_share * wsum(co)).round()
    bd["excess_over_ordinary"] = bd.crowd_only_share - bd.ordinary_share
    bd["excess_accounts"] = (bd.excess_over_ordinary * wsum(co)).round()
    bd = bd.reset_index().rename(columns={"index": "category"})
    viz.table(bd, R, "T101_crowd_only_breakdown")
    log.info(f"crowd-only breakdown ({round(wsum(co)):,} accounts)\n{bd.round(4).to_string()}")
    f, ax = viz.fig(7.6, 3.4)
    y = np.arange(len(CATS))[::-1]
    ax.barh(y + 0.18, bd.crowd_only_share * 100, height=0.36, color=viz.SERIES[0], label="crowd only (burst, no sanction)")
    ax.barh(y - 0.18, bd.ordinary_share * 100, height=0.36, color=viz.MUTED, label="ordinary accounts")
    for yy, v in zip(y, bd.crowd_only_share * 100):
        ax.text(v, yy + 0.18, f" {v:.1f}%", va="center", fontsize=7, color=viz.INK2)
    ax.set_yticks(y, CATS, fontsize=8)
    ax.set_xlabel("% of accounts (each account counted once, top category first)")
    ax.set_title("What are the accounts the crowd flags but the platform leaves?", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="center right")
    viz.save(f, R, "F101_crowd_only_breakdown", "Established accounts, 30 days from 9 April 2026. Excess over ordinary "
             "accounts = estimated missed harm.")

    # ---------- 2. platform-only takedowns
    po = s[(s.group == "platform only") & s.fl_takedown]
    po_rows = []
    spam = (po.acc_age_days <= 30) & (po.net_followers <= 10)
    for name, m in [("all platform-only takedowns", pd.Series(True, index=po.index)), ("spam-like (<=30 days, <=10 followers)", spam),
                    ("not spam-like", ~spam)]:
        d = po[m]
        po_rows.append({"subset": name, "accounts": len(d), "share": len(d) / max(len(po), 1),
                        "no_blocker_at_all": (d.n_all == 0).mean(), "1_to_4_ordinary_blockers": d.n_ord.between(1, 4).mean(),
                        "abuse_like": d.abuse_like.mean(), "toxic_post": (d.tox2_max > 0.5).mean(),
                        "median_followers": d.net_followers.median(), "median_age_days": d.acc_age_days.median(),
                        "median_posts_90d": d.act_posts_90d.median(),
                        "median_reply_to_nonfollowed": d.act_reply_to_nonfollowed_ratio.median()})
    pt = pd.DataFrame(po_rows)
    viz.table(pt, R, "T102_platform_only")
    log.info(f"platform-only takedowns\n{pt.round(3).to_string()}")
    f, ax = viz.fig(7.2, 2.8)
    cols = ["no_blocker_at_all", "1_to_4_ordinary_blockers", "abuse_like", "toxic_post"]
    labs = ["no blocker at all", "1-4 ordinary blockers", "abuse-like (judge)", "has a toxic post"]
    y = np.arange(len(cols))[::-1]
    for i, (subset, colr) in enumerate([(pt.subset[1], viz.SERIES[1]), (pt.subset[2], viz.SERIES[0])]):
        v = pt.set_index("subset").loc[subset, cols].astype(float).values * 100
        n = int(pt.set_index("subset").loc[subset, "accounts"])
        ax.barh(y + (0.18 if i == 0 else -0.18), v, height=0.36, color=colr, label=f"{subset} (n={n:,})")
    ax.set_yticks(y, labs, fontsize=8)
    ax.set_xlabel("% of platform-only takedowns")
    ax.set_title("What does the platform remove that the crowd never flags?", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="lower right")
    viz.save(f, R, "F102_platform_only", "Established accounts taken down within 30 days without an ordinary-blocker burst.")

    # ---------- 3. double blind spot by language (among abuse-like accounts)
    al = s[s.abuse_like].copy()
    top = al.groupby("lang").w.sum().sort_values(ascending=False).index[:10].tolist()
    rows = []
    for lg in top + ["ALL"]:
        d = al if lg == "ALL" else al[al.lang == lg]
        W = max(wsum(d), 1)
        both = d.burst & d.sanction
        rows.append({"lang": lg, "abuse_like_accounts": round(wsum(d)),
                     "share_of_language_accounts_abuse_like": wsum(d) / max(wsum(s if lg == "ALL" else s[s.lang == lg]), 1),
                     "platform_sanctioned": (d.w * d.sanction).sum() / W, "crowd_flagged": (d.w * d.burst).sum() / W,
                     "both": (d.w * both).sum() / W, "platform_only": (d.w * (d.sanction & ~d.burst)).sum() / W,
                     "crowd_only": (d.w * (d.burst & ~d.sanction)).sum() / W,
                     "neither_missed_by_both": (d.w * (~d.burst & ~d.sanction)).sum() / W})
    lt = pd.DataFrame(rows)
    viz.table(lt, R, "T103_language_blindspots")
    log.info(f"double blind spot by language (abuse-like accounts)\n{lt.round(3).to_string()}")
    order = (top + ["ALL"])[::-1]
    d = lt.set_index("lang").reindex(order)
    f, ax = viz.fig(7.8, 0.36 * len(order) + 1.6)
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for col, lab, colr in [("both", "both", viz.SERIES[2]), ("platform_only", "platform only", viz.SERIES[1]),
                           ("crowd_only", "crowd only", viz.SERIES[0]), ("neither_missed_by_both", "missed by both", viz.MUTED)]:
        v = d[col].values * 100
        ax.barh(y, v, left=left, color=colr, height=0.62, label=lab)
        left += v
    ax.set_yticks(y, [f"{lg} ({int(n):,})" for lg, n in zip(order, d.abuse_like_accounts)], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of abuse-like accounts (judge) in each language")
    ax.set_title("Who catches abuse-like accounts, by language?", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.4 / len(order)), ncol=4)
    viz.save(f, R, "F103_language_blindspots", "Abuse-like = judge score at or above the median of really abuse-labelled "
             "accounts. Sanction and crowd burst within 30 days of 9 April 2026.")

    # ---------- 4. validation of the judge: are abuse-like crowd-only accounts caught later / blocked again?
    co_all = s[s.group == "crowd only"].copy()
    co_all["score_q"] = pd.qcut(co_all.score.rank(method="first"), 5, labels=["Q1 (least)", "Q2", "Q3", "Q4", "Q5 (most)"])
    rows = []
    for name, m in [("abuse-like", co_all.abuse_like), ("not abuse-like", ~co_all.abuse_like)] + \
                   [(f"judge score {q}", co_all.score_q == q) for q in co_all.score_q.cat.categories]:
        d = co_all[m]
        rows.append({"crowd_only_subset": name, "accounts": round(wsum(d)),
                     "sanctioned_day31_to_1july": (d.w * (d.sanction_end & ~d.sanction)).sum() / max(wsum(d), 1),
                     "abuse_label_later": (d.w * (d.fl_abuse_end & ~d.fl_abuse)).sum() / max(wsum(d), 1),
                     "repeat_burst_day31_52": (d.w * (d.n_ord_later >= args.k)).sum() / max(wsum(d), 1)})
    vt = pd.DataFrame(rows)
    nb = s[s.group == "neither"]
    vt.loc[len(vt)] = {"crowd_only_subset": "ordinary accounts (reference)", "accounts": round(wsum(nb)),
                       "sanctioned_day31_to_1july": (nb.w * (nb.sanction_end & ~nb.sanction)).sum() / wsum(nb),
                       "abuse_label_later": (nb.w * (nb.fl_abuse_end & ~nb.fl_abuse)).sum() / wsum(nb),
                       "repeat_burst_day31_52": (nb.w * (nb.n_ord_later >= args.k)).sum() / wsum(nb)}
    viz.table(vt, R, "T105_validation")
    log.info(f"validation: what happens next to crowd-only accounts\n{vt.round(4).to_string()}")
    f, (a1, a2) = viz.plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    q = vt[vt.crowd_only_subset.str.startswith("judge score")]
    y = np.arange(len(q))
    labs = [x.replace("judge score ", "") for x in q.crowd_only_subset]
    a1.barh(y, q.abuse_label_later * 100, color=viz.SERIES[1], height=0.6)
    a2.barh(y, q.repeat_burst_day31_52 * 100, color=viz.SERIES[0], height=0.6)
    ref = vt.iloc[-1]
    a1.axvline(ref.abuse_label_later * 100, color=viz.MUTED, ls=":", lw=1)
    a2.axvline(ref.repeat_burst_day31_52 * 100, color=viz.MUTED, ls=":", lw=1)
    a1.set_yticks(y, labs, fontsize=8)
    a1.set_xlabel("% with an abuse label later (day 31 - 1 July)")
    a2.set_xlabel(f"% blocked again by {args.k}+ ordinary users (day 31 - 52)")
    a1.set_title("Caught later?", fontsize=10)
    a2.set_title("Still causing trouble?", fontsize=10)
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    viz.save(f, R, "F105_validation", "Crowd-only accounts (ordinary-blocker burst, no sanction in 30 days), by judge score "
             "quintile. Dotted line: ordinary accounts.")

    # ---------- 5. visibility: are popular abuse-like accounts sanctioned less?
    FB = [(0, 99, "<100"), (100, 999, "100-999"), (1000, 9999, "1k-10k"), (10000, 10**12, "10k+")]
    al_all = s[s.abuse_like]
    rows = []
    for lo, hi, lab in FB:
        d = al_all[al_all.net_followers.between(lo, hi)]
        if wsum(d) < 20:          # too few accounts in this band to report a share
            rows.append({"followers": lab, "abuse_like_accounts": round(wsum(d)), "platform_sanctioned_30d": np.nan,
                         "abuse_label_30d": np.nan, "crowd_burst_30d": np.nan})
            continue
        rows.append({"followers": lab, "abuse_like_accounts": round(wsum(d)),
                     "platform_sanctioned_30d": (d.w * d.sanction).sum() / max(wsum(d), 1),
                     "abuse_label_30d": (d.w * d.fl_abuse).sum() / max(wsum(d), 1),
                     "crowd_burst_30d": (d.w * d.burst).sum() / max(wsum(d), 1)})
    vb = pd.DataFrame(rows)
    viz.table(vb, R, "T106_visibility")
    log.info(f"abuse-like accounts by followers\n{vb.round(4).to_string()}")
    f, ax = viz.fig(6.6, 3.4)
    x = np.arange(len(vb))
    for col, lab, colr in [("platform_sanctioned_30d", "platform sanction", viz.SERIES[1]),
                           ("crowd_burst_30d", "crowd burst (ordinary blockers)", viz.SERIES[0])]:
        ax.plot(x, vb[col] * 100, marker="o", color=colr, label=lab)
    ax.set_xticks(x, [f"{l}\n(n={n:,})" for l, n in zip(vb.followers, vb.abuse_like_accounts)], fontsize=8)
    ax.set_xlabel("followers at 9 April")
    ax.set_ylabel("% of abuse-like accounts")
    ax.set_title("Who acts on abuse-like accounts, by account size?", fontsize=10)
    ax.legend(fontsize=8)
    viz.save(f, R, "F106_visibility", "Abuse-like accounts (language-free judge), 30 days from 9 April 2026.")

    # ---------- 6. language-fair harm: the most toxic accounts within each language
    tq = s[(s.tox2_n_scored >= 3) & s.lang.notna()].copy()
    topl = tq.groupby("lang").w.sum().sort_values(ascending=False).index[:12].tolist()
    tq = tq[tq.lang.isin(topl)]
    tq["cut"] = tq.groupby("lang").tox2_mean.transform(lambda v: v.quantile(1 - args.top_toxic))
    mt = tq[tq.tox2_mean >= tq.cut]
    rows = []
    for lg in topl + ["ALL"]:
        d = mt if lg == "ALL" else mt[mt.lang == lg]
        W = max(wsum(d), 1)
        rows.append({"lang": lg, "most_toxic_accounts": round(wsum(d)),
                     "platform_sanctioned": (d.w * d.sanction).sum() / W, "abuse_label": (d.w * d.fl_abuse).sum() / W,
                     "crowd_flagged": (d.w * d.burst).sum() / W,
                     "both": (d.w * (d.burst & d.sanction)).sum() / W,
                     "platform_only": (d.w * (d.sanction & ~d.burst)).sum() / W,
                     "crowd_only": (d.w * (d.burst & ~d.sanction)).sum() / W,
                     "missed_by_both": (d.w * (~d.burst & ~d.sanction)).sum() / W})
    lf = pd.DataFrame(rows)
    viz.table(lf, R, "T107_language_toxic")
    log.info(f"most toxic {args.top_toxic:.0%} within each language: who acts?\n{lf.round(3).to_string()}")
    order = (topl + ["ALL"])[::-1]
    d = lf.set_index("lang").reindex(order)
    f, ax = viz.fig(7.8, 0.36 * len(order) + 1.6)
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for col, lab, colr in [("both", "both", viz.SERIES[2]), ("platform_only", "platform only", viz.SERIES[1]),
                           ("crowd_only", "crowd only", viz.SERIES[0]), ("missed_by_both", "missed by both", viz.MUTED)]:
        v = d[col].values * 100
        ax.barh(y, v, left=left, color=colr, height=0.62, label=lab)
        left += v
    ax.set_yticks(y, [f"{lg} ({int(n):,})" for lg, n in zip(order, d.most_toxic_accounts)], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of the most toxic {args.top_toxic:.0%} of accounts in each language")
    ax.set_title("The most toxic accounts in each language: who acts?", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.4 / len(order)), ncol=4)
    viz.save(f, R, "F107_language_toxic", "Language-fair: accounts compared only with others posting in the same language "
             "(multilingual toxicity, 3+ scored posts). 30 days from 9 April 2026.")
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
