# data/

The real data is private and is NOT in this repository. Only the fake demo data is (`demo/raw/`).

Real layout (on the server; `data/raw/` holds read-only links to the downloads, made by setup_server.sh):

    raw/
      snapshot/            repos.parquet (id, pds, did, ...) and one parquet per record type:
                           block, follow, list, listitem, listblock, profile, post
                           columns: repo_id, collection, rkey, at_rev, content (record JSON)
      firehose/YYYY/MM/DD/YYYY-MM-DD.json.gz
                           one event per line: did, time, received, seq, commit{rev, ops[...]}
      labelers/            full_labeler.parquet (official labeler, 2023 - 2026-07-01),
                           official_labeler.parquet (2026-04-01 onwards)
    interim/               cleaned tables written by steps 01-07 (not in git)
    processed/             one row per account (accounts_T0, accounts_new, tox_*) (not in git)


    demo/raw/              fake data in the same layout, made by src/demo/make_demo_data.py (seed 42)
