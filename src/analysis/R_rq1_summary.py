#!/usr/bin/env python3
"""Collect RQ1 robustness variants (results/RQ1/<tag>/T41_alignment.csv) into one table and one figure.

  results/RQ1/T43_robustness.csv
  results/RQ1/F43_robustness.png/.pdf   rho of SHAP importances and cross-prediction AUC per variant and cohort
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bskymod import viz
from bskymod.common import load_config

LABELS = {"main": "main (5 seeds)", "main_tox": "main + toxicity (5 seeds)", "b_until_takedown": "blocks counted until takedown",
          "b_before_index": "block burst before index", "k2": "burst = 2+ blockers",
          "k10": "burst = 10+ blockers", "no_wave": "without 18-21 May wave"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    R = cfg["paths"]["results"] / "RQ1"
    rows = []
    for tag in LABELS:
        f = R / tag / "T41_alignment.csv"
        if not f.exists() or f.stat().st_size < 5:
            print(f"skip {tag}: no alignment result (too few positives?)")
            continue
        d = pd.read_csv(f)
        ov = pd.read_csv(R / tag / "F41_outcome_overlap.csv")
        d = d.merge(ov[["cohort", "B", "T", "B_and_T", "P(T|B)", "P(T|not B)"]], on="cohort", how="left")
        d.insert(0, "variant", LABELS[tag])
        rows.append(d)
    if not rows:
        raise SystemExit("no variants found under results/RQ1/<tag>/")
    t = pd.concat(rows, ignore_index=True)
    viz.table(t, R, "T43_robustness")

    f, axes = viz.plt.subplots(1, 2, figsize=(9, 0.45 * t.variant.nunique() + 1.8), sharey=True)
    order = [LABELS[k] for k in LABELS if LABELS[k] in set(t.variant)][::-1]
    for ax, (col, lo, hi, xl) in zip(axes, [("spearman_rho_mean", "rho_seed_min", "rho_seed_max", "rank correlation of SHAP importances"),
                                            ("auc_B_model_predicting_T", "auc_B_predicting_T_min", "auc_B_predicting_T_max", "AUC: block-burst model predicting takedown")]):
        for i, (c, off) in enumerate([("established", 0.15), ("new", -0.15)]):
            d = t[t.cohort == c].set_index("variant").reindex(order)
            y = np.arange(len(order)) + off
            ax.errorbar(d[col], y, xerr=[(d[col] - d[lo]).clip(lower=0), (d[hi] - d[col]).clip(lower=0)], fmt="o",
                        color=viz.SERIES[i], ms=6, elinewidth=1.5, capsize=2, label=f"{c} accounts")
        ax.set_yticks(np.arange(len(order)), order)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[1].axvline(0.5, color=viz.MUTED, lw=1, ls=":")
    axes[0].set_title("RQ1 robustness: do crowd and platform agree?")
    axes[0].legend(loc="lower left")
    viz.save(f, R, "F43_robustness", "Points: mean over retrained models; bars: range over seeds. Dotted line: AUC 0.5 = no better than chance.")
    print(t.to_string())


if __name__ == "__main__":
    main()
