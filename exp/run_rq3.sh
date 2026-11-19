#!/usr/bin/env bash
# RQ3 - blind spots: what the crowd flags that the platform misses, and the reverse.
source "$(dirname "$0")/common.sh"
step src/analysis/14_rq3_blindspots.py $G
echo "RQ3 done"
