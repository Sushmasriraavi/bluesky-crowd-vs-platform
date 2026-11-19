#!/usr/bin/env bash
# RQ2 - who acts first, lead time, early-warning queue.
source "$(dirname "$0")/common.sh"
step src/analysis/12_rq2_timing.py      # crowd alert vs platform action: order, lead time, Kaplan-Meier, Cox
step src/analysis/13_rq2_extra.py       # review queue test, follower bands, event study around takedown
echo "RQ2 done"
