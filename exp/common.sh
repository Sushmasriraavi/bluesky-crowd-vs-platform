# Shared settings for every runner. Sourced, not run.
# Environment variables you can set before a runner:
#   CONFIG   config file (default config.yaml = real data; exp/demo/config_demo.yaml = fake demo data)
#   GPU      "--gpu" (default) to train XGBoost on GPU, or "" for CPU
#   DEMO     "" (default) = real toxicity models on GPUs, "--demo" = word-list scorer, no GPU needed
#   SEEDS    number of retrained models per RQ1 run (default 5)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # always run from the repo root
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
PY=${PY:-python3}
C="--config ${CONFIG:-config.yaml}"
G=${GPU---gpu}
DEMO=${DEMO:-}
SEEDS=${SEEDS:-5}
step() { echo; echo "================ $* ================"; "$PY" "$@" $C; }
