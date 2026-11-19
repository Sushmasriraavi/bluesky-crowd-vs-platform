#!/usr/bin/env bash
# End-to-end demo on FAKE data (no GPU, no private data). About 2-3 minutes on a laptop.
#   bash exp/demo/run_demo.sh            run every stage on data/demo/raw, write exp/demo/results/
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
export CONFIG=exp/demo/config_demo.yaml GPU="" DEMO="--demo" SEEDS=${SEEDS:-2} PY
export FH_ARGS="--workers 2 --threads 2 --mem 2GB"
rm -rf exp/demo/work exp/demo/results                      # start clean every time
# step 0: fake raw data (same format as the real data). It is committed in data/demo/raw;
# REGEN=1 bash exp/demo/run_demo.sh makes it again from the generator (same seed -> same content).
if [[ "${REGEN:-0}" == 1 || ! -f data/demo/raw/snapshot/repos.parquet ]]; then
  "$PY" src/demo/make_demo_data.py --out data/demo/raw
fi
bash exp/run_all.sh 2>&1 | tee exp/demo/demo_run.log
echo
echo "Demo finished. Figures and tables: exp/demo/results/   Full log: exp/demo/demo_run.log"
