#!/usr/bin/env bash
# Demo on FAKE data (no GPU, no private data). Run everything, or one stage at a time:
#
#   bash exp/demo/run_demo.sh              all stages (about 2 minutes)
#   bash exp/demo/run_demo.sh prepare      stage 1: raw data -> clean tables (steps 00-07)  [starts clean]
#   bash exp/demo/run_demo.sh toxicity     stage 2: toxicity scores (steps 10-11)
#   bash exp/demo/run_demo.sh rq1          RQ1: crowd vs platform models, SHAP, language
#   bash exp/demo/run_demo.sh robustness   RQ1 with other definitions (k=2, k=10, ...)
#   bash exp/demo/run_demo.sh rq2          RQ2: who acts first, lead time, review queue
#   bash exp/demo/run_demo.sh rq3          RQ3: blind spots
#
# Order: prepare -> toxicity -> rq1 / robustness / rq2 / rq3. After each stage the script lists its outputs.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
export CONFIG=exp/demo/config_demo.yaml GPU="" DEMO="--demo" SEEDS=${SEEDS:-2} PY
export FH_ARGS="--workers 2 --threads 2 --mem 2GB"
W=exp/demo/work; R=exp/demo/results; L=exp/demo/logs
STAGE=${1:-all}

need() {   # stop with a clear message if an earlier stage has not been run
  [[ -f "$1" ]] || { echo "Missing $1 -> run 'bash exp/demo/run_demo.sh $2' first."; exit 1; }
}

show() {   # list what a stage produced
  echo; echo "---------------- outputs of stage '$1' ----------------"
  shift
  for p in "$@"; do
    [[ -e "$p" ]] || continue
    if [[ -d "$p" ]]; then
      local np nc nq; np=$(find "$p" -name '*.png' | wc -l); nc=$(find "$p" -name '*.csv' | wc -l); nq=$(find "$p" -name '*.parquet' | wc -l)
      if (( np + nc > 0 )); then echo "$p/  ($np figures, $nc tables)"; else echo "$p/  ($nq parquet files)"; fi
      find "$p" -name '*.png' | sort | sed 's/^/   /'
    else
      echo "$p"
    fi
  done
}

run_stage() {
  local s=$1
  mkdir -p "$L"
  case $s in
    prepare)
      rm -rf "$W" "$R"                                       # stage 1 always starts clean
      # fake raw data is committed in data/demo/raw; REGEN=1 rebuilds it (same seed -> same content)
      if [[ "${REGEN:-0}" == 1 || ! -f data/demo/raw/snapshot/repos.parquet ]]; then
        "$PY" src/demo/make_demo_data.py --out data/demo/raw
      fi
      bash exp/run_prepare.sh 2>&1 | tee "$L/prepare.log"
      show prepare $W/interim/firehose $W/interim/snapshot $W/interim/labels $W/interim/crowd \
                   $W/processed/accounts_T0.parquet $W/processed/accounts_new.parquet \
                   $R/inventory $R/S0 $R/R0 $R/R1 $R/S1 ;;
    toxicity)
      need $W/processed/accounts_new.parquet prepare
      bash exp/run_toxicity.sh 2>&1 | tee "$L/toxicity.log"
      show toxicity $W/processed/tox_established.parquet $W/processed/tox_new.parquet \
                    $W/processed/tox2_established.parquet $W/processed/tox2_new.parquet $R/TOX ;;
    rq1)
      need $W/processed/tox2_new.parquet toxicity
      bash exp/run_rq1.sh 2>&1 | tee "$L/rq1.log"
      show rq1 $R/RQ1/probe $R/RQ1/main_tox $R/RQ1/out_review $R/RQ1/out_abuse $R/RQ1/out_adult \
               $R/RQ1/final $R/RQ1/response $R/RQ1/language $R/RQ1/blockers ;;
    robustness)
      need $W/processed/tox2_new.parquet toxicity
      bash exp/run_rq1_robustness.sh 2>&1 | tee "$L/robustness.log"
      show robustness $R/RQ1/main $R/RQ1/b_until_takedown $R/RQ1/b_before_index $R/RQ1/k2 $R/RQ1/k10 \
                      $R/RQ1/no_wave $R/RQ1/F43_robustness.png ;;
    rq2)
      need $W/processed/tox2_new.parquet toxicity
      bash exp/run_rq2.sh 2>&1 | tee "$L/rq2.log"
      show rq2 $R/RQ2 ;;
    rq3)
      need $W/processed/tox2_new.parquet toxicity
      bash exp/run_rq3.sh 2>&1 | tee "$L/rq3.log"
      show rq3 $R/RQ3 ;;
    *)
      echo "unknown stage '$s' (use: all prepare toxicity rq1 robustness rq2 rq3)"; exit 1 ;;
  esac
  if grep -q "Traceback" "$L/$s.log"; then echo "!! stage $s had an error, see $L/$s.log"; exit 1; fi
  echo; echo "stage '$s' finished OK (log: $L/$s.log)"
}

if [[ "$STAGE" == all ]]; then
  for s in prepare toxicity rq1 robustness rq2 rq3; do run_stage $s; done
  echo; echo "Demo finished. Figures and tables: $R/   Logs: $L/"
else
  run_stage "$STAGE"
fi
