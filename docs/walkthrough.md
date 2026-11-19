# Code review walkthrough (order of presentation)

Total about 25 minutes: demo 5-10 min, block-by-block walkthrough 10-15 min.

## Part A - live demo (clone and run)

    git clone <repo url> && cd bluesky-crowd-vs-platform
    python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

Run the demo one stage at a time. After each stage the script prints the files it produced; open one
input and one output per stage:

1. `bash exp/demo/run_demo.sh prepare`
   Input: `data/demo/raw/` - one firehose line (`zcat data/demo/raw/firehose/2026/04/10/*.gz | head -1`)
   and the snapshot parquet files (same format as the real data).
   Output: `exp/demo/work/interim/firehose/posts/...` (one row per post) and
   `exp/demo/work/processed/accounts_T0.parquet`, `accounts_new.parquet` (one row per account:
   features act_/net_/acc_/new_act_, crowd_ columns, y_ outcomes).
2. `bash exp/demo/run_demo.sh toxicity`
   Output: `processed/tox_new.parquet` (tox_mean, tox_max per account), `results/TOX/`.
3. `bash exp/demo/run_demo.sh rq1`
   Output: `results/RQ1/main_tox/F40_shap_new.png` - toxic replies drive block bursts, posting volume and
   links drive takedowns (the planted pattern comes back).
4. `bash exp/demo/run_demo.sh rq2`
   Output: `results/RQ2/F90_who_first.png` - abuse labels are mostly flagged by the crowd first.
5. `bash exp/demo/run_demo.sh rq3`
   Output: `results/RQ3/F104_judge_scores.png`, `F105_validation.png`.
(`robustness` is optional; skip it if time is short.)

Quick look at any table:  `python3 -c "import pandas as pd; print(pd.read_parquet('exp/demo/work/processed/accounts_new.parquet').head())"`

## Part B - walkthrough (docs/architecture.png, block by block)

1. Shared library `src/bskymod`: config + DuckDB connection in UTC (common.py), the definition of every
   platform action as one SQL condition (outcomes.py), figure helper that always saves png + pdf + csv (viz.py).
2. Input: why three sources (snapshot = network before, firehose = live events with exact times,
   labeler = what the platform did).
3. Data preparation:
   - 01: DuckDB read_json over each day file, unnest commit ops, split into five tables; truncated gzip is salvaged.
   - 02: snapshot records are JSON; repo_id -> DID join; one table per record type.
   - 05: at_rev is a TID; first 53 bits = microseconds -> crawl time per account.
   - 03: account labels (uri = did) vs post labels (uri = at://...); negation = reversal.
   - 04: a block seen in both snapshot and firehose is the same record (author + rkey); deletes = unblocks.
   - 06/07: index time per account; features only before it, outcomes only after it (leakage rule).
4. Toxicity: chunked, resumable, one worker per GPU; model choice per post language (XLM-R v2, Korean model).
   In the demo only the model call is replaced (`--demo`).
5. Analysis:
   - RQ1 (08): two XGBoost models on the same features, held-out 20% test set, SHAP from XGBoost
     (pred_contribs), Spearman correlation of importances, several seeds for uncertainty.
   - RQ2 (12, 13): crowd alert = 5th ordinary blocker; compare with the first platform action; Kaplan-Meier
     and Cox (lifelines); queue test = rank accounts by crowd signal vs random.
   - RQ3 (14): judge model trained on platform outcomes without language features; scores crowd-only
     accounts; validated on labels that came later.
6. Experiments `exp/`: one runner per stage; CONFIG / GPU / DEMO / SEEDS switches; the demo uses the same runners.

Likely questions
- Why DuckDB? The snapshot and firehose are far larger than memory; DuckDB runs SQL directly on parquet / json.gz and spills to disk.
- Why not use blocking features in RQ1? They are the label of model B (leakage) and would make the comparison unfair.
- How is leakage avoided? Every feature has a time filter `< index_t`; outcomes use `> index_t`.
- Why fake data? The real data is private; the fake data has the same schema so every line of code runs.
