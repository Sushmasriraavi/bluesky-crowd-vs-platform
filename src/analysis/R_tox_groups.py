#!/usr/bin/env python3
"""Toxicity in the 4 crowd x platform groups (results/TOX/).

Groups (same 30-day windows as RQ1, per cohort):
  1 blocked + taken down        crowd and platform agree
  2 blocked, not taken down     crowd-only
  3 not blocked, taken down     platform-only
  4 neither                     baseline
"Blocked" = >= k distinct new blockers in (index, index + 30 d]; main k=5, sensitivity k=1.
Toxicity = Detoxify multilingual on up to 5 posts per account (step 10): established = posts before T0,
new = posts in the first 24 h. Also reported for English posts only (Detoxify is strongest in English).

Outputs
  T60_tox_groups_<cohort>.csv   per group: accounts, accounts with scored posts, mean (95% CI) of toxicity,
                                share of toxic posts, insult, threat, identity attack, sexual; for k=5/k=1, all/English
  F60_tox_groups_<cohort>       4 panels (toxicity, insult, identity attack, threat), one bar per group with CI
  F61_tox_ecdf_<cohort>         distribution of each account's most toxic post, per group
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str

GROUPS = ["1 blocked + taken down", "2 blocked only (crowd)", "3 taken down only (platform)", "4 neither"]
GCOL = dict(zip(GROUPS, viz.SERIES + [viz.MUTED]))
METRICS = [("tox_mean", "toxicity"), ("tox_insult_mean", "insult"), ("tox_identity_attack_mean", "identity attack"),
           ("tox_threat_mean", "threat"), ("tox_share_toxic", "share of toxic posts"), ("tox_sexual_mean", "sexual")]


def group(b, t):
    return np.select([b & t, b & ~t, ~b & t], GROUPS[:3], GROUPS[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--t0", default="2026-04-09")
    ap.add_argument("--fh-end", default="2026-06-01")
    ap.add_argument("--horizon-days", type=int, default=30)
    ap.add_argument("--cohorts", nargs="*", default=["established", "new"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P, R = cfg["paths"]["interim"], cfg["paths"]["processed"], cfg["paths"]["results"] / "TOX"
    log = get_logger("TOX", cfg["paths"]["results"] / "logs" / "R_tox_groups.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    H, T0, FHE = f"INTERVAL {args.horizon_days} DAY", f"TIMESTAMPTZ '{args.t0} 00:00:00+00'", f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    be = rp(I / "crowd" / "block_edges.parquet")
    lang_src = {"established": rp(I / "features" / "posts90.parquet"),
                "new": rp(I / "firehose" / "posts" / "*" / "*.parquet")}
    base = {"established": f"SELECT did, {T0} AS index_t, y_takedown_ts FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d",
            "new": f"SELECT did, index_t, y_takedown_ts FROM {rp(P / 'accounts_new.parquet')} WHERE NOT instant_takedown AND index_t + {H} <= {FHE}"}

    for cohort in args.cohorts:
        tox = P / f"tox_{cohort}.parquet"
        if not tox.exists():
            log.warning(f"[{cohort}] {tox} missing - run step 10 first")
            continue
        # English-only per-account toxicity
        con.execute(f"""CREATE OR REPLACE TEMP TABLE tox_en AS
            SELECT t.did, avg(t.p_toxicity) AS tox_mean, avg((t.p_toxicity > 0.5)::INT) AS tox_share_toxic,
                   avg(t.p_insult) AS tox_insult_mean, avg(t.p_threat) AS tox_threat_mean,
                   avg(t.p_identity_attack) AS tox_identity_attack_mean, avg(t.p_sexual_explicit) AS tox_sexual_mean
            FROM {rp(I / 'features' / f'tox_posts_{cohort}.parquet')} t
            JOIN (SELECT did, rkey, any_value(lang) AS lang FROM {lang_src[cohort]} GROUP BY ALL) l USING (did, rkey)
            WHERE l.lang LIKE 'en%' GROUP BY t.did""")
        df = con.execute(f"""
            WITH a AS ({base[cohort]}),
                 b AS (SELECT a.did, count(DISTINCT e.blocker_did) AS n FROM a JOIN {be} e
                       ON e.subject_did = a.did AND e.t_created > a.index_t AND e.t_created <= a.index_t + {H} GROUP BY 1)
            SELECT a.did, coalesce(b.n, 0) AS blockers_30d,
                   coalesce(a.y_takedown_ts <= a.index_t + {H}, false) AS taken_down_30d
            FROM a LEFT JOIN b USING (did)""").df()
        all_t = con.execute(f"SELECT * FROM {rp(tox)}").df()
        en_t = con.execute("SELECT * FROM tox_en").df()
        rows = []
        for k in [5, 1]:
            df["group"] = group(df.blockers_30d >= k, df.taken_down_30d)
            sizes = df.group.value_counts()
            for lang_name, tt in [("all languages", all_t), ("English only", en_t)]:
                m = df.merge(tt, on="did", how="inner")
                for g in GROUPS:
                    d = m[m.group == g]
                    r = {"k": k, "posts": lang_name, "group": g, "accounts": int(sizes.get(g, 0)), "accounts_scored": len(d)}
                    for col, name in METRICS:
                        v = d[col].dropna()
                        r[name] = v.mean() if len(v) else np.nan
                        r[f"{name}_ci"] = 1.96 * v.std() / np.sqrt(len(v)) if len(v) > 1 else np.nan
                    rows.append(r)
        t = pd.DataFrame(rows)
        viz.table(t, R, f"T60_tox_groups_{cohort}")
        log.info(f"[{cohort}]\n{t[['k', 'posts', 'group', 'accounts', 'accounts_scored', 'toxicity', 'insult', 'identity attack']].to_string()}")

        main_t = t[(t.k == 5) & (t.posts == "all languages")].set_index("group").reindex(GROUPS)
        f, axes = viz.plt.subplots(1, 4, figsize=(11, 3.6), sharey=False)
        for ax, (col, name) in zip(axes, METRICS[:4]):
            ax.bar(range(4), main_t[name], yerr=main_t[f"{name}_ci"], color=[GCOL[g] for g in GROUPS], width=0.65,
                   error_kw={"elinewidth": 1, "ecolor": viz.INK2, "capsize": 2})
            ax.set_xticks(range(4), ["1", "2", "3", "4"])
            ax.set_title(name, fontsize=10)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("mean score (Detoxify)")
        handles = [viz.plt.Rectangle((0, 0), 1, 1, color=GCOL[g]) for g in GROUPS]
        f.legend(handles, GROUPS, loc="upper center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
        f.suptitle("")
        viz.save(f, R, f"F60_tox_groups_{cohort}",
                 f"{cohort} accounts; blocked = 5+ new blockers and taken down = takedown, both within 30 days; up to 5 posts per account; 95% CI.")

        m = df.assign(group=group(df.blockers_30d >= 5, df.taken_down_30d)).merge(all_t, on="did")
        f, ax = viz.fig(6.8, 3.6)
        for g in GROUPS:
            viz.ecdf(ax, m[m.group == g].tox_max, GCOL[g], g)
        ax.set_xlabel("toxicity of the account's most toxic post")
        ax.set_title(f"Most toxic post per account - {cohort}")
        ax.legend(loc="lower right", fontsize=8)
        viz.save(f, R, f"F61_tox_ecdf_{cohort}")
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
