#!/usr/bin/env bash
# Stage 1 - raw data -> clean tables + feature tables (steps 00-07) and descriptive figures.
# Real data: several hours (full snapshot, 61 firehose days). Demo data: under a minute.
source "$(dirname "$0")/common.sh"
step src/pipeline/00_inventory.py                                   # what is in the raw folders
step src/pipeline/01_firehose_to_parquet.py ${FH_ARGS:-}            # firehose json.gz -> parquet per day
step src/pipeline/02_snapshot_tables.py                             # snapshot -> DID-keyed tables
step src/pipeline/05_snapshot_timing.py                             # crawl time per repo (needed by 06/07)
step src/pipeline/03_label_outcomes.py                              # labels Apr-Jul 2026 -> outcomes
step src/pipeline/03_label_outcomes.py --out-dir labels_hist --start 2023-04-13   # full label history
step src/pipeline/04_crowd_signals.py                               # blocks + mod lists, snapshot + firehose
step src/pipeline/04_crowd_signals.py --source snapshot             # snapshot-only view (history)
step src/pipeline/06_snapshot_features.py                           # established accounts at T0
step src/pipeline/07_new_accounts.py                                # new accounts, first 24 h
step src/analysis/R0_descriptives.py                                # data + platform enforcement overview
step src/analysis/R1_crowd_vs_platform.py                           # crowd vs platform, Apr-Jul 2026
step src/analysis/R1_crowd_vs_platform.py --signals crowd_snapshot --outcomes labels_hist --out S1   # 2023-2026 history
echo "prepare done"
