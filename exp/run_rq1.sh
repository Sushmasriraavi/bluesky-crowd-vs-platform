#!/usr/bin/env bash
# RQ1 - do the crowd (block bursts) and the platform (labeler actions) react to the same behaviour?
source "$(dirname "$0")/common.sh"
step src/analysis/D1_platform_actions_probe.py                                          # what actions exist
step src/analysis/08_rq1_models.py $G --seeds $SEEDS --tag main_tox --with-toxicity    # B vs T models + SHAP
for o in review abuse adult; do
  step src/analysis/08_rq1_models.py $G --seeds $SEEDS --with-toxicity --platform-outcome $o --tag out_$o
done
step src/analysis/R_rq1_final.py --tag main_tox     # four-group comparison + SHAP figures
step src/analysis/R_rq1_response.py                 # dose-response ladder, lift per action, follower bands
step src/analysis/R_rq1_language.py                 # blocking vs removal by language
step src/analysis/R_rq1_blockers.py                 # who blocks: ordinary vs heavy blockers, same language
echo "RQ1 done"
