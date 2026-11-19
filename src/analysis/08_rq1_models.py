#!/usr/bin/env python3
"""Step 08 / RQ1 - do community blocking and official takedown react to the same account behaviour?

For each cohort we train two XGBoost models on the SAME behaviour features (crowd signals and platform labels are
never inputs) with two prospective outcomes over the SAME 30-day horizon after the index time:
  B (block burst) : >= k distinct new blockers in (index, index + 30 d]   (snapshot + firehose block records)
  T (takedown)    : official !takedown label in (index, index + 30 d]
Cohorts
  established : accounts in the snapshot, active in the 90 days before T0 (index = T0 = 2026-04-09)
  new         : accounts without snapshot records, first seen in the firehose (index = first seen + 24 h),
                not taken down in their first 24 h, index + 30 d inside firehose coverage

Alignment measures (per cohort)
  - Spearman rank correlation of mean |SHAP| between the two models (95% bootstrap CI over test rows)
  - overlap of the top-10 features (Jaccard), and sign agreement (does a feature push both outcomes the same way?)
  - cross-prediction: AUC of the B model's score for predicting T, and vice versa
  - outcome overlap: how many accounts are B and T at the same time

Outputs in results/RQ1/: T40_model_metrics.csv, T41_alignment.csv, T42_shap_<cohort>.csv,
  F40_shap_<cohort>.png/.pdf (top features, both models side by side), F41_outcome_overlap.csv, models/*.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from bskymod import viz
from bskymod.common import connect, get_logger, load_config, sql_str
from bskymod.outcomes import ACTIONS, MODEL_OUTCOMES, flags_sql

EXCLUDE_PREFIX = ("crowd_", "y_", "new_crowd_")
EXCLUDE_COLS = {"did", "acc_first_seen", "first_seen", "index_t", "instant_takedown", "instant_takedown_ts",
                "needs_review_in_window_ts", "active_90d",
                # enforcement history = platform signals, not behaviour (see decisions D6)
                "acc_prior_takedowns", "acc_prior_needs_review", "acc_prior_account_labels",
                "acc_post_labels_90d", "acc_post_takedowns_before"}


def prepare(df: pd.DataFrame, log):
    cols = [c for c in df.columns if c not in EXCLUDE_COLS and not c.startswith(EXCLUDE_PREFIX)]
    X = df[cols].copy()
    for c in X.columns:
        if c == "acc_pds":
            continue
        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]):
            top = X[c].value_counts().index[:10]
            X[c] = pd.Categorical(X[c].where(X[c].isin(top), "other").astype(object))
        elif pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("float")
    if "acc_pds" in X:
        top = X["acc_pds"].value_counts().index[:10]
        X["acc_pds"] = pd.Categorical(X["acc_pds"].where(X["acc_pds"].isin(top), -1).astype("int64"))
    log.info(f"{len(cols)} features: {cols}")
    return X


def fit(X, y, args, seed):
    import xgboost as xgb
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.125, stratify=ytr, random_state=seed)
    pos = max(ytr.sum(), 1)
    m = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, tree_method="hist", enable_categorical=True, eval_metric="aucpr",
        scale_pos_weight=(len(ytr) - pos) / pos, early_stopping_rounds=50, random_state=seed,
        device="cuda" if args.gpu else "cpu", n_jobs=args.threads)
    try:
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    except Exception as e:  # GPU build missing or no CUDA -> train on CPU instead
        if not args.gpu:
            raise
        print(f"GPU training failed ({type(e).__name__}: {str(e)[:120]}); falling back to CPU")
        args.gpu = False
        m.set_params(device="cpu")
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, Xte, yte


def shap_values(model, X):
    import xgboost as xgb
    d = xgb.DMatrix(X, enable_categorical=True)
    contrib = model.get_booster().predict(d, pred_contribs=True)   # last column = bias
    return contrib[:, :-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--t0", default="2026-04-09")
    ap.add_argument("--fh-end", default="2026-06-01")
    ap.add_argument("--horizon-days", type=int, default=30)
    ap.add_argument("--k", type=int, default=5, help="blockers needed for a block burst")
    ap.add_argument("--max-train", type=int, default=3_000_000, help="subsample very large cohorts (keeps all positives)")
    ap.add_argument("--shap-rows", type=int, default=200_000)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--cohorts", nargs="*", default=["established", "new"])
    ap.add_argument("--seeds", type=int, default=1, help="retrain on N different random splits (CI over seeds)")
    ap.add_argument("--tag", default="", help="write results to results/RQ1/<tag>")
    ap.add_argument("--b-until-takedown", action="store_true",
                    help="count blocks only until the takedown (taken-down accounts stop receiving blocks)")
    ap.add_argument("--b-before-index", action="store_true",
                    help="block burst measured BEFORE the index (established: 30 days before T0; new: first 24 h)")
    ap.add_argument("--exclude-wave", action="store_true", help="drop accounts taken down in the 2026-05-18..21 mass wave")
    ap.add_argument("--with-toxicity", action="store_true",
                    help="add Detoxify features (step 10); accounts without scored posts get missing values")
    ap.add_argument("--platform-outcome", default="takedown", choices=MODEL_OUTCOMES,
                    help="platform side of the comparison: takedown (default), review, abuse label, adult label")
    args = ap.parse_args()
    PNAME = ACTIONS[args.platform_outcome][0]
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    R = cfg["paths"]["results"] / "RQ1" / args.tag if args.tag else cfg["paths"]["results"] / "RQ1"
    (R / "models").mkdir(parents=True, exist_ok=True)
    log = get_logger("08", cfg["paths"]["results"] / "logs" / f"08_rq1{'_' + args.tag if args.tag else ''}.log")
    log.info(f"settings: {vars(args)}")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    H = f"INTERVAL {args.horizon_days} DAY"
    FHE = f"TIMESTAMPTZ '{args.fh_end} 00:00:00+00'"
    T0 = f"TIMESTAMPTZ '{args.t0} 00:00:00+00'"
    be = rp(I / "crowd" / "block_edges.parquet")
    rng = np.random.default_rng(cfg.get("seed", 42))

    def b_window(a):
        if args.b_before_index:   # crowd reaction already visible at the index time (new accounts: their first 24 h)
            return f"e.t_created >= {a}.index_t - {H} AND e.t_created < {a}.index_t"
        end = f"{a}.index_t + {H}"
        if args.b_until_takedown:
            end = f"least({end}, coalesce({a}.y_takedown_ts, {end}))"
        return f"e.t_created > {a}.index_t AND e.t_created <= {end}"

    def has_tox(cohort):
        return args.with_toxicity and (P / f"tox_{cohort}.parquet").exists()

    def tox_cols(cohort):
        if not has_tox(cohort):
            if args.with_toxicity:
                log.warning(f"[{cohort}] tox_{cohort}.parquet missing - this cohort runs WITHOUT toxicity")
            return ""
        return ", tx.tox_mean, tx.tox_max, tx.tox_share_toxic, tx.tox_insult_mean, tx.tox_threat_mean, " \
               "tx.tox_identity_attack_mean, tx.tox_sexual_mean"

    def tox_join(cohort):
        if not has_tox(cohort):
            return ""
        return f"LEFT JOIN {rp(P / f'tox_{cohort}.parquet')} tx USING (did)"

    queries = {
        "established": f"""
            WITH a AS (SELECT *, {T0} AS index_t FROM {rp(P / 'accounts_T0.parquet')} WHERE active_90d),
                 b AS (SELECT a.did, count(DISTINCT e.blocker_did) AS n FROM a
                       JOIN {be} e ON e.subject_did = a.did AND {b_window('a')}
                       GROUP BY 1)
            SELECT a.*, coalesce(b.n, 0) AS y_blockers_h,
                   coalesce(a.y_takedown_ts <= a.index_t + {H}, false) AS y_T {tox_cols('established')}
            FROM a LEFT JOIN b USING (did) {tox_join('established')}""",
        "new": f"""
            WITH a AS (SELECT * FROM {rp(P / 'accounts_new.parquet')}
                       WHERE NOT instant_takedown AND index_t + {H} <= {FHE}),
                 b AS (SELECT a.did, count(DISTINCT e.blocker_did) AS n FROM a
                       JOIN {be} e ON e.subject_did = a.did AND {b_window('a')}
                       GROUP BY 1)
            SELECT a.*, coalesce(b.n, 0) AS y_blockers_h,
                   coalesce(a.y_takedown_ts <= a.index_t + {H}, false) AS y_T {tox_cols('new')}
            FROM a LEFT JOIN b USING (did) {tox_join('new')}""",
    }

    metrics, align, overlap, align_seeds = [], [], [], []
    for cohort in args.cohorts:
        df = con.execute(queries[cohort]).df()
        if args.exclude_wave:
            w = df.y_takedown_ts.between(pd.Timestamp("2026-05-18", tz="UTC"), pd.Timestamp("2026-05-22", tz="UTC"), inclusive="left")
            log.info(f"[{cohort}] excluding {int(w.sum()):,} accounts taken down in the 18-21 May wave")
            df = df[~w.fillna(False)]
        df["y_B"] = df.y_blockers_h >= args.k
        if args.platform_outcome != "takedown":   # y_T = the chosen platform action within the same horizon
            L = I / "labels"
            con.register("cohort_idx", df[["did", "index_t"]])
            fl = con.execute(flags_sql("SELECT did, index_t FROM cohort_idx", rp(L / "account_label_events.parquet"),
                                       rp(L / "record_label_events.parquet"), f"c.index_t + {H}",
                                       [args.platform_outcome])).df()
            df = df.drop(columns="y_T").merge(fl.rename(columns={f"act_{args.platform_outcome}": "y_T"}),
                                              on="did", how="left")
            log.info(f"[{cohort}] platform outcome = {PNAME}")
        df["y_T"] = df.y_T.fillna(False).astype(bool)
        log.info(f"[{cohort}] rows {len(df):,}  B={df.y_B.sum():,}  T={df.y_T.sum():,}  both={(df.y_B & df.y_T).sum():,}")
        overlap.append({"cohort": cohort, "accounts": len(df), "B": int(df.y_B.sum()), "T": int(df.y_T.sum()),
                        "B_and_T": int((df.y_B & df.y_T).sum()),
                        "P(T|B)": df.y_T[df.y_B].mean(), "P(T|not B)": df.y_T[~df.y_B].mean(),
                        "P(B|T)": df.y_B[df.y_T].mean(),
                        "jaccard_B_T": (df.y_B & df.y_T).sum() / max((df.y_B | df.y_T).sum(), 1)})
        # subsample huge cohorts, keeping every positive of either outcome
        neg_weight = 1.0
        if len(df) > args.max_train:
            keep = df.y_B | df.y_T
            neg = df[~keep].sample(args.max_train - int(keep.sum()), random_state=42)
            neg_weight = (~keep).sum() / len(neg)
            df = pd.concat([df[keep], neg])
            log.info(f"[{cohort}] subsampled negatives (weight {neg_weight:.2f}); prevalence below is in the sample")
        X = prepare(df, log)
        if min(df.y_B.sum(), df.y_T.sum()) < 50:
            log.warning(f"[{cohort}] fewer than 50 positives for B or T, skipped")
            continue

        shares_B, shares_T, first = [], [], None
        for seed in range(args.seeds):
            res = {}
            for name in ["B", "T"]:
                y = df[f"y_{name}"].astype(int)
                m, Xte, yte = fit(X, y, args, seed=seed)
                p = m.predict_proba(Xte)[:, 1]
                if seed == 0:
                    m.save_model(str(R / "models" / f"{cohort}_{name}.json"))
                res[name] = {"model": m, "Xte": Xte, "yte": yte, "p": p}
                metrics.append({"cohort": cohort, "model": name, "seed": seed, "test_rows": len(yte),
                                "positives": int(yte.sum()), "prevalence_in_sample": float(yte.mean()),
                                "prevalence_in_population": float(yte.mean() / (yte.mean() + (1 - yte.mean()) * neg_weight)),
                                "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
                                "trees": m.best_iteration + 1})
                log.info(f"[{cohort}] seed {seed} {name}: AUC {metrics[-1]['roc_auc']:.3f}  PR-AUC {metrics[-1]['pr_auc']:.3f}")
            Xc, yT = res["T"]["Xte"], res["T"]["yte"]
            yB = df.loc[Xc.index, "y_B"].astype(int)
            idx = Xc.index if len(Xc) <= args.shap_rows else rng.choice(Xc.index, args.shap_rows, replace=False)
            Xs = Xc.loc[idx]
            sB, sT = shap_values(res["B"]["model"], Xs), shap_values(res["T"]["model"], Xs)
            feats = list(Xs.columns)
            mB, mT = np.abs(sB).mean(0), np.abs(sT).mean(0)
            shares_B.append(mB / mB.sum()); shares_T.append(mT / mT.sum())
            topB = set(np.array(feats)[np.argsort(-mB)[:10]]); topT = set(np.array(feats)[np.argsort(-mT)[:10]])
            align_seeds.append({"cohort": cohort, "seed": seed, "spearman_rho": spearmanr(mB, mT).correlation,
                                "top10_jaccard": len(topB & topT) / len(topB | topT),
                                "auc_B_model_predicting_T": roc_auc_score(yT, res["B"]["model"].predict_proba(Xc)[:, 1]),
                                "auc_T_model_predicting_B": roc_auc_score(yB, res["T"]["p"]) if yB.sum() else np.nan})
            if seed == 0:
                first = (Xs, sB, sT, feats, mB, mT)
                # keep the seed-0 SHAP sample for dependence plots (R_rq1_final.py)
                samp = Xs.copy()
                for c in samp.columns:
                    if str(samp[c].dtype) == "category":
                        samp[c] = samp[c].astype(str)
                sv = pd.DataFrame(np.hstack([sB, sT]), index=Xs.index,
                                  columns=[f"shapB__{f}" for f in feats] + [f"shapT__{f}" for f in feats])
                samp = pd.concat([samp, sv, df.loc[Xs.index, ["y_B", "y_T"]]], axis=1)
                samp.to_parquet(R / f"shap_sample_{cohort}.parquet", index=False)

        # row bootstrap on seed 0 (uncertainty from the data) ...
        Xs, sB, sT, feats, mB, mT = first
        boots = []
        for _ in range(args.boot):
            b = rng.integers(0, len(Xs), len(Xs))
            boots.append(spearmanr(np.abs(sB[b]).mean(0), np.abs(sT[b]).mean(0)).correlation)
        # ... and spread over retrained models (uncertainty from training)
        sd = pd.DataFrame([r for r in align_seeds if r["cohort"] == cohort])

        def direction(s):
            out = []
            for j, f in enumerate(feats):
                v = pd.to_numeric(Xs[f], errors="coerce") if str(Xs[f].dtype) != "category" else Xs[f].cat.codes
                ok = ~pd.isna(v)
                out.append(np.sign(np.corrcoef(v[ok], s[ok.values, j])[0, 1]) if ok.sum() > 10 and np.std(v[ok]) > 0 else 0)
            return np.array(out)
        dB, dT = direction(sB), direction(sT)
        shB, shT = np.mean(shares_B, 0), np.mean(shares_T, 0)
        imp = pd.DataFrame({"feature": feats, "share_B": shB, "share_T": shT,
                            "share_B_sd": np.std(shares_B, 0), "share_T_sd": np.std(shares_T, 0),
                            "rank_B": pd.Series(-shB).rank().values, "rank_T": pd.Series(-shT).rank().values,
                            "direction_B": dB, "direction_T": dT}).sort_values("share_T", ascending=False)
        viz.table(imp, R, f"T42_shap_{cohort}")
        top = imp.assign(m=imp[["share_B", "share_T"]].max(axis=1)).nlargest(15, "m")
        lo_b, hi_b = np.nanpercentile(boots, [2.5, 97.5])
        align.append({"cohort": cohort, "seeds": args.seeds,
                      "spearman_rho_mean": sd.spearman_rho.mean(),
                      "rho_seed_min": sd.spearman_rho.min(), "rho_seed_max": sd.spearman_rho.max(),
                      "rho_rowboot_low": lo_b, "rho_rowboot_high": hi_b,
                      "top10_jaccard_mean": sd.top10_jaccard.mean(),
                      "sign_agreement_top15": float(np.mean((top.direction_B == top.direction_T).values)),
                      "auc_B_model_predicting_T": sd.auc_B_model_predicting_T.mean(),
                      "auc_B_predicting_T_min": sd.auc_B_model_predicting_T.min(),
                      "auc_B_predicting_T_max": sd.auc_B_model_predicting_T.max(),
                      "auc_T_model_predicting_B": sd.auc_T_model_predicting_B.mean(),
                      "toxicity_share_of_importance_B": float(imp.loc[imp.feature.str.startswith("tox_"), "share_B"].sum()),
                      "toxicity_share_of_importance_T": float(imp.loc[imp.feature.str.startswith("tox_"), "share_T"].sum()),
                      "shap_rows": len(Xs)})
        log.info(f"[{cohort}] alignment {align[-1]}")

        f, ax = viz.fig(7.5, 0.32 * len(top) + 1.4)
        y = np.arange(len(top))[::-1]
        ax.barh(y + 0.2, top.share_B * 100, height=0.38, xerr=top.share_B_sd * 100, color=viz.SERIES[0],
                error_kw={"elinewidth": 0.8, "ecolor": viz.INK2}, label="block burst model (crowd)")
        ax.barh(y - 0.2, top.share_T * 100, height=0.38, xerr=top.share_T_sd * 100, color=viz.SERIES[1],
                error_kw={"elinewidth": 0.8, "ecolor": viz.INK2}, label=f"{PNAME} model (platform)")
        ax.set_yticks(y, top.feature)
        ax.set_xlabel("share of total mean |SHAP| (%)")
        ax.set_title(f"What each outcome relies on - {cohort} accounts")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        a0 = align[-1]
        viz.save(f, R, f"F40_shap_{cohort}",
                 f"rho = {a0['spearman_rho_mean']:.2f} (seeds {a0['rho_seed_min']:.2f}-{a0['rho_seed_max']:.2f}, n={args.seeds}); "
                 f"top-10 overlap {a0['top10_jaccard_mean']:.2f}. Bars: mean over seeds, whiskers: SD.")

    viz.table(pd.DataFrame(align_seeds), R, "T41_alignment_by_seed")
    viz.table(pd.DataFrame(metrics), R, "T40_model_metrics")
    viz.table(pd.DataFrame(align), R, "T41_alignment")
    viz.table(pd.DataFrame(overlap), R, "F41_outcome_overlap")
    (R / "run_settings.json").write_text(json.dumps(vars(args), indent=1, default=str))
    log.info(f"done -> {R}")


if __name__ == "__main__":
    main()
