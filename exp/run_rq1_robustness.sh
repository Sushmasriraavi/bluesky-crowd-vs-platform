#!/usr/bin/env bash
# RQ1 robustness - the main comparison with other definitions (summary table/figure at the end).
source "$(dirname "$0")/common.sh"
R=$(( SEEDS < 3 ? SEEDS : 3 ))
step src/analysis/08_rq1_models.py $G --seeds $SEEDS --tag main
step src/analysis/08_rq1_models.py $G --seeds $R --tag b_until_takedown --b-until-takedown
step src/analysis/08_rq1_models.py $G --seeds $R --tag b_before_index --b-before-index
step src/analysis/08_rq1_models.py $G --seeds $R --tag k2 --k 2
step src/analysis/08_rq1_models.py $G --seeds $R --tag k10 --k 10
step src/analysis/08_rq1_models.py $G --seeds $R --tag no_wave --exclude-wave --cohorts new
step src/analysis/R_rq1_summary.py
echo "RQ1 robustness done"
