#!/usr/bin/env bash
# Stage 2 - toxicity of sampled posts. Real: 4 GPUs, a few hours. Demo (DEMO=--demo): word-list scorer, seconds.
source "$(dirname "$0")/common.sh"
step src/pipeline/10_toxicity.py $DEMO           # Detoxify multilingual, 5 posts per account
step src/pipeline/11_toxicity_multi.py $DEMO     # TextDetox XLM-R v2 + Korean kor_unsmile, same posts
step src/analysis/R_tox_groups.py                # toxicity of blocked-only / removed-only / both / neither
step src/analysis/R_tox_multi.py                 # same with the multilingual model, per language
echo "toxicity done"
