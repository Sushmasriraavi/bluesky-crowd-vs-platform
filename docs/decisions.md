# Decision log

Write one entry per decision: date, what, why, what it affects.

2026-10-28  D1  Blocking features removed from the RQ1 classifiers. They are the label of model B, so including them leaks the outcome and makes the SHAP comparison unfair.
2026-10-28  D2  Calendar design for RQ1: features from April, outcomes May to 2026-07-01. Move feature window earlier when Jan-Mar firehose arrives.
2026-10-28  D3  Following count and posts taken from firehose, not snapshot (snapshot lacks records authored by taken-down accounts, per the data provider).
2026-10-28  D4  Self-deactivation/deletion treated as a competing risk in RQ2, not as censoring.
2026-10-28  D5  Block onset defined at k=5 distinct blockers (direct blocks + blocklist subscribers); k=1 and k=20 as sensitivity.
2026-10-28  D6  Enforcement-type official labels never used as features for the takedown model.
2026-10-31  D7  Data facts from inventory: firehose = 61 daily json.gz (2026-04-01..05-31), commit events only (no account/identity events) -> takedown time comes from official !takedown labels (ts), not firehose status.
2026-10-31  D8  2026-05-17 file is truncated (gzip EOF, ~half day). Readable part salvaged, day flagged partial; repaired file requested.
2026-10-31  D9  Use leo_v2/snapshot/app.bsky.graph.list.parquet (raw snapshot/list.parquet unreadable by DuckDB). profile.parquet read through pyarrow (arrow.json extension type).
2026-10-31  D10 Snapshot crawl ends ~2026-04-08 (max block createdAt) -> snapshot = state before/at start of the firehose window.
2026-10-31  D11 Official labeler "needs-review" (auto-expiring account flag) added as the platform's own early signal (RQ2 b2).
2026-10-31  D12 All timestamps handled in UTC (DuckDB TimeZone=UTC). Event time = relay "time"; record createdAt kept separately (client-set).
2026-10-31  D13 Firehose collection gaps (UTC): 2026-04-16 07:00-10:00; 2026-05-17 15:28-24:00 (truncated file, salvaged part kept); 2026-05-18 23:00 - 2026-05-19 01:00. Rates/counts over windows touching these are flagged or exposure-adjusted.
2026-10-31  D14 Outcome = first account-level !takedown label (official labeler) in 2026-04-01..2026-07-01. Accounts already taken down at window start are excluded from "new takedown" analyses. Negations tracked (reversal), prior takedowns counted (repeat offenders). Suspensions (!suspend) kept as a separate, milder outcome.
2026-10-31  D15 Labels from full_labeler.parquet (history since 2023); official_labeler.parquet used as consistency check for the window.
2026-10-31  D16 Crowd signal = distinct blockers (direct block records) + mod-list membership; snapshot and firehose records merged on (author, rkey); deletes = unblock/removal. Record time: firehose relay time if seen live, else plausible client createdAt, else "before window".
2026-10-31  D17 Takedown cohort for crowd questions = new takedowns before 2026-06-01 (end of firehose), so all blocks before the takedown are observable.
2026-10-31  D18 Early-warning test (F13) uses decision time = 24 h after first block; outcome window (24 h, 31 d]; accounts taken down in the first 24 h are excluded (not predictable at decision time).
2026-10-31  D19 Main analysis is snapshot-first (full snapshot + full labeler, history since 2023 and prospective crawl -> 2026-07-01). Firehose Apr-May results (R1) are kept as validation; Jan-Mar firehose merged when available.
2026-10-31  D20 Crawl time per account = decoded at_rev TID (time of last commit before crawl, lower bound). Observation end for snapshot blocks set from step 05 output.
2026-11-01  D21 Two populations: established accounts (in snapshot at T0=2026-04-09, step 06) and new accounts (no snapshot records, first seen in firehose, observed first 24 h, step 07). Instant takedowns (<24 h) reported separately.
2026-11-01  D22 RQ1 outcomes are prospective and symmetric: block burst (>=5 new distinct blockers) and takedown, both within 30 days after the index time; same behaviour features; enforcement history excluded from inputs.
2026-11-01  D23 Outcome follow-up limited by labeler end (2026-07-01); newer labeler export requested from the data provider.
2026-11-01  D24 RQ1 uncertainty = spread over retrained models (5 seeds main, 3 per variant) plus row bootstrap; robustness variants: blocks until takedown, burst before index, k=2, k=10, without 18-21 May wave (new cohort only; the established 30-day window ends 9 May).
2026-11-01  D25 RQ2 landmark design: clock starts 24 h after the first block (history, established) or at first seen + 24 h (new); accounts taken down before the landmark excluded; censoring at end of label data (2026-07-01). Self-deletion not observed -> treated as censoring (limitation). History Cox uses crowd covariates only (own records of pre-crawl takedowns are missing).
2026-11-01  D26 Toxicity = Detoxify 'multilingual' on up to 5 posts per account (established: most recent before T0; new: first 24 h), one model per GPU. 4-group comparison (blocked x taken down, 30-day windows as RQ1), k=5 main and k=1 sensitivity, all languages and English-only.
2026-11-01  D27 RQ1 final: SHAP dependence curves (seed-0 SHAP sample, quantile bins) for the features with the largest crowd-vs-platform importance gap plus toxicity; 4-group behaviour profiles with standardized differences vs "neither" (counts on log1p scale).
