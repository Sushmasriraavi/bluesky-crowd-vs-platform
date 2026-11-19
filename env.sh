# Source this every time you log in:  source env.sh
# Keeps ALL caches/temp files on /data, because the server's / (and your home) is full.
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TMPDIR="$P/.cache/tmp"
export PIP_CACHE_DIR="$P/.cache/pip"
export XDG_CACHE_HOME="$P/.cache"
export MPLCONFIGDIR="$P/.cache/matplotlib"
export HF_HOME="$P/.cache/huggingface"
export TORCH_HOME="$P/.cache/torch"
export PYTHONPATH="$P/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$HF_HOME" "$TORCH_HOME"
if [ -f "$P/.venv/bin/activate" ]; then source "$P/.venv/bin/activate"; fi
