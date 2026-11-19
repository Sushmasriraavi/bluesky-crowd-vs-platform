# Code review walkthrough (order of presentation)

Total about 25 minutes: demo 5-10 min, block-by-block walkthrough 10-15 min.

## Part A - live demo (clone and run)

    git clone <repo url> && cd bluesky-crowd-vs-platform
    python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
    bash exp/demo/run_demo.sh

While it runs (about 2 minutes), show for each stage its input and output:

1. Fake raw data: `data/demo/raw/` - open one firehose line (`zcat data/demo/raw/firehose/2026/04/10/*.gz | head -1`)
   and one snapshot record. Same format as the real data.
2. Step 01 output: `exp/demo/work/interim/firehose/posts/day=2026-04-10/part.parquet` (one row per post).
3. Steps 03-07 output: `exp/demo/work/processed/accounts_T0.parquet` and `accounts_new.parquet`
   (one row per account: feature columns act_/net_/acc_/new_act_, crowd_ columns, y_ outcomes).
4. Step 10/11 output: `processed/tox_new.parquet` (tox_mean, tox_max per account).
5. Analysis outputs: `exp/demo/results/RQ1/main_tox/F40_shap_new.png`, `RQ2/F90_who_first.png`,
   `RQ3/F104_judge_scores.png`. Point out that the planted pattern comes back (toxic replies drive block
   bursts; posting volume and links drive takedowns; abuse labels are mostly flagged by the crowd first).

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
   - RQ1 (08): two XGBoost models on the same features, out-of-fold predictions, SHAP from XGBoost
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
