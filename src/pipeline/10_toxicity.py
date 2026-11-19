#!/usr/bin/env python3
"""Step 10 - toxicity of posts with Detoxify (multilingual, GPU), aggregated per account.

Inputs
  established: data/interim/features/text_sample.parquet   (<= 20 most recent posts before T0 per account, step 06)
  new:         firehose posts in each new account's first 24 h (step 07 cohort)
Per account we score up to --per-account posts and keep: mean / max toxicity, share of toxic posts (> 0.5),
and mean insult / threat / identity-attack scores.

Outputs
  data/interim/features/tox_posts_<cohort>.parquet     one row per scored post (did, rkey, scores)
  data/processed/tox_<cohort>.parquet                  one row per account (tox_* columns)
Runs on all GPUs in parallel (one worker per GPU) and can resume: finished chunks are skipped.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from bskymod.common import connect, get_logger, load_config, sql_str

LABELS = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"]


_MODEL = None


def _init_worker(gpu_queue):
    """Each worker process claims one GPU and loads the model once."""
    global _MODEL
    gpu = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from detoxify import Detoxify
    _MODEL = Detoxify("multilingual", device="cuda" if torch.cuda.is_available() else "cpu")
    _MODEL._gpu = gpu


def score_chunk(path_in: str, path_out: str, batch: int, max_chars: int) -> str:
    import torch
    df = pd.read_parquet(path_in)
    texts = df.text.fillna("").str.slice(0, max_chars).tolist()
    out = {k: [] for k in LABELS}
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            part = texts[i:i + batch]
            r = _MODEL.predict(part)
            for k in LABELS:
                out[k].extend(r.get(k, [float("nan")] * len(part)))
    res = df[["did", "rkey"]].copy()
    for k in LABELS:
        res[f"p_{k}"] = out[k]
    tmp = path_out + ".tmp"
    res.to_parquet(tmp, index=False)
    os.replace(tmp, path_out)
    return f"{Path(path_in).name}: {len(df):,} posts on GPU {getattr(_MODEL, '_gpu', '?')}"


def score_chunk_demo(path_in: str, path_out: str) -> str:
    """--demo: same input/output files as score_chunk, but a word-list score instead of Detoxify (no GPU)."""
    from bskymod.demo_scorer import word_scores
    df = pd.read_parquet(path_in)
    p = word_scores(df.text.fillna("").tolist())
    res = df[["did", "rkey"]].copy()
    for k in LABELS:
        res[f"p_{k}"] = p if k in ("toxicity", "insult") else p * 0.3
    res.to_parquet(path_out, index=False)
    return f"{Path(path_in).name}: {len(df):,} posts (demo word-list scorer)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--cohorts", nargs="*", default=["established", "new"])
    ap.add_argument("--per-account", type=int, default=5, help="posts scored per account (most recent first)")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=500_000, help="posts per work chunk")
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="score only N posts (for a quick test)")
    ap.add_argument("--demo", action="store_true", help="no GPU: word-list scorer instead of Detoxify (demo data only)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    F = I / "features"
    log = get_logger("10", cfg["paths"]["results"] / "logs" / "10_toxicity.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    lim = f"LIMIT {args.limit}" if args.limit else ""

    sources = {
        "established": f"""
            SELECT did, rkey, text FROM {rp(F / 'text_sample.parquet')}
            QUALIFY row_number() OVER (PARTITION BY did ORDER BY created_at DESC) <= {args.per_account} {lim}""",
        "new": f"""
            SELECT p.did, p.rkey, p.text
            FROM {rp(P / 'accounts_new.parquet')} a
            JOIN {rp(I / 'firehose' / 'posts' / '*' / '*.parquet')} p
              ON p.did = a.did AND p.t >= a.first_seen AND p.t < a.index_t
            WHERE length(coalesce(p.text, '')) >= 3
            QUALIFY row_number() OVER (PARTITION BY p.did ORDER BY p.t) <= {args.per_account} {lim}""",
    }
    for cohort in args.cohorts:
        work = F / f"tox_work_{cohort}"
        work.mkdir(parents=True, exist_ok=True)
        if not any(work.glob("in_*.parquet")):
            t0 = time.time()
            con.execute(f"""COPY (SELECT *, row_number() OVER (ORDER BY did, rkey) - 1 AS rn FROM ({sources[cohort]}))
                            TO '{sql_str(str(work / 'all.parquet'))}' (FORMAT parquet)""")
            n = con.execute(f"SELECT count(*) FROM {rp(work / 'all.parquet')}").fetchone()[0]
            i = -1
            for i, start in enumerate(range(0, n, args.chunk)):   # fixed row numbers: every post in exactly one chunk
                con.execute(f"""COPY (SELECT did, rkey, text FROM {rp(work / 'all.parquet')}
                                      WHERE rn >= {start} AND rn < {start + args.chunk})
                                TO '{sql_str(str(work / f'in_{i:05d}.parquet'))}' (FORMAT parquet)""")
            (work / "all.parquet").unlink()
            log.info(f"[{cohort}] {n:,} posts to score in {i + 1} chunks ({time.time()-t0:.0f}s)")
        if not any(work.glob("in_*.parquet")):
            log.warning(f"[{cohort}] no posts to score - skipped")
            continue
        todo = [p for p in sorted(work.glob("in_*.parquet")) if not (work / p.name.replace("in_", "out_")).exists()]
        log.info(f"[{cohort}] {len(todo)} chunks left")
        t0 = time.time()
        if args.demo:
            for p in todo:
                log.info(f"[{cohort}] {score_chunk_demo(str(p), str(work / p.name.replace('in_', 'out_')))}")
        else:
            ctx = mp.get_context("spawn")
            q = ctx.Manager().Queue()
            for g in range(args.gpus):
                q.put(g)
            with ProcessPoolExecutor(args.gpus, mp_context=ctx, initializer=_init_worker, initargs=(q,)) as ex:
                futs = [ex.submit(score_chunk, str(p), str(work / p.name.replace("in_", "out_")),
                                  args.batch, args.max_chars) for p in todo]
                for k, f in enumerate(futs):
                    log.info(f"[{cohort}] {k + 1}/{len(futs)} {f.result()} ({time.time()-t0:.0f}s)")
        posts = F / f"tox_posts_{cohort}.parquet"
        con.execute(f"COPY (SELECT * FROM {rp(work / 'out_*.parquet')}) TO '{sql_str(str(posts))}' (FORMAT parquet, COMPRESSION zstd)")
        out = P / f"tox_{cohort}.parquet"
        con.execute(f"""COPY (
            SELECT did, count(*) AS tox_n_scored,
                   avg(p_toxicity) AS tox_mean, max(p_toxicity) AS tox_max,
                   avg((p_toxicity > 0.5)::INT) AS tox_share_toxic,
                   avg(p_insult) AS tox_insult_mean, avg(p_threat) AS tox_threat_mean,
                   avg(p_identity_attack) AS tox_identity_attack_mean, avg(p_sexual_explicit) AS tox_sexual_mean
            FROM {rp(posts)} GROUP BY did) TO '{sql_str(str(out))}' (FORMAT parquet, COMPRESSION zstd)""")
        n_acc, mean_tox, share = con.execute(f"SELECT count(*), avg(tox_mean), avg(tox_share_toxic) FROM {rp(out)}").fetchone()
        log.info(f"[{cohort}] -> {out}: {n_acc:,} accounts, mean toxicity {mean_tox:.3f}, mean share toxic posts {share:.3f}")


if __name__ == "__main__":
    main()
