#!/usr/bin/env bash
# One-time setup on the college server.
# Usage: bash setup_server.sh /path/to/folder/with/your/downloads
# The download folder should contain snapshot/, firehose/, labelers/ (any of them).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-}"
if [[ -z "$SRC" || ! -d "$SRC" ]]; then
  echo "Give the folder that holds your downloaded data, e.g.:"
  echo "  bash setup_server.sh /path/to/your/downloads"
  exit 1
fi
SRC="$(cd "$SRC" && pwd)"
source "$ROOT/env.sh"
echo "--- disk space (project disk must have free space) ---"
df -h "$ROOT" "$HOME" | sed 's/^/  /'

mkdir -p "$ROOT"/data/{raw,interim/duckdb_tmp,processed}
mkdir -p "$ROOT"/results/{inventory,logs}
for rq in rq1 rq2 rq3; do mkdir -p "$ROOT/results/$rq"/{tables,figures,models}; done
mkdir -p "$ROOT"/docs

# Link raw data (read-only use). Nothing is copied or moved.
for d in snapshot firehose labelers; do
  if [[ -e "$SRC/$d" ]]; then
    ln -sfn "$SRC/$d" "$ROOT/data/raw/$d"; echo "linked $d -> $SRC/$d"
  else
    echo "WARNING: $SRC/$d not found (fix the path or link it by hand)"
  fi
done

# Optional extra folders (e.g. leo_v2 with the re-sent files): bash setup_server.sh RAW EXTRA1 EXTRA2
shift || true
for X in "$@"; do
  [[ -d "$X" ]] || { echo "WARNING: $X not a folder"; continue; }
  X="$(cd "$X" && pwd)"; ln -sfn "$X" "$ROOT/data/raw/extra_$(basename "$X")"; echo "linked extra_$(basename "$X") -> $X"
done

# Python env
if [[ ! -d "$ROOT/.venv" ]]; then python3 -m venv "$ROOT/.venv"; fi
source "$ROOT/env.sh"
pip install --upgrade pip -q
pip install -q -r "$ROOT/requirements.txt"

echo "--- machine ---"
nproc; free -g | head -2; df -h "$ROOT" | tail -1
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no nvidia-smi"
python -c "import duckdb, xgboost, lifelines; print('python packages OK')"
cd "$ROOT" && git init -q 2>/dev/null && echo "git repo ready" || true
echo "Done. Next: source env.sh && bash exp/run_prepare.sh"
