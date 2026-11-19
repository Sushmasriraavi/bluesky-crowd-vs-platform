#!/usr/bin/env bash
# Whole pipeline in order. Real data: run each stage separately on the server (prepare takes hours).
# Demo: bash exp/demo/run_demo.sh
set -euo pipefail
D="$(dirname "$0")"
bash "$D/run_prepare.sh"
bash "$D/run_toxicity.sh"
bash "$D/run_rq1.sh"
bash "$D/run_rq1_robustness.sh"
bash "$D/run_rq2.sh"
bash "$D/run_rq3.sh"
echo "ALL DONE"
