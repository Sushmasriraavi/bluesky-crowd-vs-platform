#!/usr/bin/env python3
"""Step 11 - multilingual toxicity (re-scores the posts of step 10 with models that cover more languages).

Why: Detoxify-multilingual only supports en/fr/es/it/pt/tr/ru and gives near-zero scores for ja/ko/zh/de posts,
so toxicity could not be compared across languages (RQ1 language analysis).

Models (chosen per post language = the post's own lang tag)
  textdetox/xlmr-large-toxicity-classifier-v2   XLM-R large, TextDetox 2025; trained on en ru uk de es ar am hi zh it
                                                fr he ja tt (+ hinglish). Used for every post except Korean.
  smilegate-ai/kor_unsmile                      Korean hate/abuse (KcBERT, multi-label); toxicity = 1 - P(clean).
Posts in other languages (pt, tr, ko-less, th, nl, pl, ...) still get the XLM-R score but are flagged
tox2_lang_supported = false, so results can be shown with and without them.

Outputs
  data/interim/features/tox2_posts_<cohort>.parquet   did, rkey, lang, model, p_tox2, lang_supported
  data/processed/tox2_<cohort>.parquet                per account: tox2_mean, tox2_max, tox2_share_toxic,
                                                      tox2_n_scored, tox2_lang (main language of the scored posts),
                                                      tox2_supported_share
GPU workers (one per GPU), fp16, resumable chunks - same pattern as step 10.
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

XLMR = "textdetox/xlmr-large-toxicity-classifier-v2"
KOR = "smilegate-ai/kor_unsmile"
XLMR_LANGS = {"en", "ru", "uk", "de", "es", "ar", "am", "hi", "zh", "it", "fr", "he", "ja", "tt"}

_M = {}


def _init_worker(gpu_queue):
    gpu = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    for key, name in [("xlmr", XLMR), ("kor", KOR)]:
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=dt).to(dev).eval()
        _M[key] = (tok, mod)
    _M["dev"], _M["gpu"] = dev, gpu
    id2 = {int(k): v for k, v in _M["kor"][1].config.id2label.items()}
    clean = [i for i, v in id2.items() if str(v).lower() == "clean"]
    if not clean:
        raise RuntimeError(f"{KOR}: no 'clean' label in {id2}")
    _M["kor_clean"] = clean[0]


def _score(key, texts, batch, max_len):
    import torch
    tok, mod = _M[key]
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(_M["dev"])
            logits = mod(**enc).logits.float()
            if key == "xlmr":
                out.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())          # idx 1 = toxic
            else:
                out.extend((1 - torch.sigmoid(logits)[:, _M["kor_clean"]]).cpu().tolist())
    return out


def score_chunk(path_in: str, path_out: str, batch: int, max_len: int) -> str:
    df = pd.read_parquet(path_in)
    df["text"] = df.text.fillna("").str.slice(0, 1000)
    df["lang"] = df.lang.fillna("")
    df["model"] = df.lang.eq("ko").map({True: "kor_unsmile", False: "xlmr_v2"})
    df["p_tox2"] = float("nan")
    for model, key in [("xlmr_v2", "xlmr"), ("kor_unsmile", "kor")]:
        m = df.model == model
        if m.any():
            df.loc[m, "p_tox2"] = _score(key, df.loc[m, "text"].tolist(), batch, max_len)
    df["lang_supported"] = df.lang.isin(XLMR_LANGS) | df.lang.eq("ko")
    res = df[["did", "rkey", "lang", "model", "p_tox2", "lang_supported"]]
    tmp = path_out + ".tmp"
    res.to_parquet(tmp, index=False)
    os.replace(tmp, path_out)
    return f"{Path(path_in).name}: {len(df):,} posts ({int((df.model == 'kor_unsmile').sum()):,} Korean) on GPU {_M['gpu']}"


def score_chunk_demo(path_in: str, path_out: str) -> str:
    """--demo: same files as score_chunk, word-list score instead of the two GPU models."""
    from bskymod.demo_scorer import word_scores
    df = pd.read_parquet(path_in)
    df["lang"] = df.lang.fillna("")
    df["model"] = df.lang.eq("ko").map({True: "kor_unsmile", False: "xlmr_v2"})
    df["p_tox2"] = word_scores(df.text.fillna("").tolist(), seed=1)
    df["lang_supported"] = df.lang.isin(XLMR_LANGS) | df.lang.eq("ko")
    df[["did", "rkey", "lang", "model", "p_tox2", "lang_supported"]].to_parquet(path_out, index=False)
    return f"{Path(path_in).name}: {len(df):,} posts (demo word-list scorer)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--cohorts", nargs="*", default=["established", "new"])
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=128, help="tokens per post (Bluesky posts are <= 300 characters)")
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--limit", type=int, default=0, help="score only N posts (quick test)")
    ap.add_argument("--demo", action="store_true", help="no GPU: word-list scorer instead of the models (demo data only)")
    ap.add_argument("--skip-english", action="store_true", help="only re-score non-English posts (faster)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    I, P = cfg["paths"]["interim"], cfg["paths"]["processed"]
    F = I / "features"
    log = get_logger("11", cfg["paths"]["results"] / "logs" / "11_toxicity_multi.log")
    con = connect(cfg)
    rp = lambda p: f"read_parquet('{sql_str(str(p))}')"
    norm = "nullif(lower(split_part(replace(CAST(lang AS VARCHAR), '_', '-'), '-', 1)), '')"
    lim = f"LIMIT {args.limit}" if args.limit else ""
    en = "WHERE coalesce(lang, '') <> 'en'" if args.skip_english else ""
    # exactly the posts step 10 scored, with their text and language
    sources = {
        "established": f"""
            SELECT t.did, t.rkey, s.text, l.lang FROM {rp(F / 'tox_posts_established.parquet')} t
            JOIN {rp(F / 'text_sample.parquet')} s USING (did, rkey)
            LEFT JOIN (SELECT did, rkey, any_value({norm}) AS lang FROM {rp(F / 'posts90.parquet')} GROUP BY ALL) l
              USING (did, rkey)""",
        "new": f"""
            SELECT t.did, t.rkey, any_value(p.text) AS text, any_value({norm.replace('lang', 'p.lang')}) AS lang
            FROM {rp(F / 'tox_posts_new.parquet')} t
            JOIN {rp(I / 'firehose' / 'posts' / '*' / '*.parquet')} p USING (did, rkey) GROUP BY ALL""",
    }
    for cohort in args.cohorts:
        if not (F / f"tox_posts_{cohort}.parquet").exists():
            log.warning(f"[{cohort}] run step 10 first (tox_posts_{cohort}.parquet missing)")
            continue
        work = F / f"tox2_work_{cohort}{'_test' if args.limit else ''}"
        work.mkdir(parents=True, exist_ok=True)
        if not any(work.glob("in_*.parquet")):
            t0 = time.time()
            con.execute(f"""COPY (SELECT *, row_number() OVER (ORDER BY did, rkey) - 1 AS rn
                                  FROM (SELECT * FROM ({sources[cohort]}) {en} {lim}))
                            TO '{sql_str(str(work / 'all.parquet'))}' (FORMAT parquet)""")
            n = con.execute(f"SELECT count(*) FROM {rp(work / 'all.parquet')}").fetchone()[0]
            langs = con.execute(f"SELECT coalesce(lang, '?') l, count(*) n FROM {rp(work / 'all.parquet')} "
                                f"GROUP BY 1 ORDER BY 2 DESC LIMIT 15").fetchall()
            i = -1
            for i, start in enumerate(range(0, n, args.chunk)):
                con.execute(f"""COPY (SELECT did, rkey, text, lang FROM {rp(work / 'all.parquet')}
                                      WHERE rn >= {start} AND rn < {start + args.chunk})
                                TO '{sql_str(str(work / f'in_{i:05d}.parquet'))}' (FORMAT parquet)""")
            (work / "all.parquet").unlink()
            log.info(f"[{cohort}] {n:,} posts in {i + 1} chunks ({time.time()-t0:.0f}s); top languages {langs}")
        todo = [p for p in sorted(work.glob("in_*.parquet")) if not (work / p.name.replace("in_", "out_")).exists()]
        log.info(f"[{cohort}] {len(todo)} chunks left")
        if todo and args.demo:
            for p in todo:
                log.info(f"[{cohort}] {score_chunk_demo(str(p), str(work / p.name.replace('in_', 'out_')))}")
        elif todo:
            t0 = time.time()
            ctx = mp.get_context("spawn")
            q = ctx.Manager().Queue()
            for g in range(args.gpus):
                q.put(g)
            with ProcessPoolExecutor(args.gpus, mp_context=ctx, initializer=_init_worker, initargs=(q,)) as ex:
                futs = [ex.submit(score_chunk, str(p), str(work / p.name.replace("in_", "out_")), args.batch, args.max_len)
                        for p in todo]
                for k, f in enumerate(futs):
                    log.info(f"[{cohort}] {k + 1}/{len(futs)} {f.result()} ({time.time()-t0:.0f}s)")
        if args.limit:
            d = con.execute(f"SELECT model, lang, count(*) n, avg(p_tox2) mean_tox, avg((p_tox2 > 0.5)::INT) share_toxic "
                            f"FROM {rp(work / 'out_*.parquet')} GROUP BY ALL ORDER BY n DESC LIMIT 20").df()
            log.info(f"[{cohort}] TEST ONLY (not saved as final):\n{d.to_string()}")
            continue
        posts = F / f"tox2_posts_{cohort}.parquet"
        con.execute(f"COPY (SELECT * FROM {rp(work / 'out_*.parquet')}) TO '{sql_str(str(posts))}' (FORMAT parquet, COMPRESSION zstd)")
        out = P / f"tox2_{cohort}.parquet"
        con.execute(f"""COPY (
            SELECT did, count(*) AS tox2_n_scored, avg(p_tox2) AS tox2_mean, max(p_tox2) AS tox2_max,
                   avg((p_tox2 > 0.5)::INT) AS tox2_share_toxic, mode(lang) AS tox2_lang,
                   avg(lang_supported::INT) AS tox2_supported_share
            FROM {rp(posts)} GROUP BY did) TO '{sql_str(str(out))}' (FORMAT parquet, COMPRESSION zstd)""")
        d = con.execute(f"""SELECT lang, model, count(*) n, avg(p_tox2) mean_tox, avg((p_tox2 > 0.5)::INT) share_toxic
                            FROM {rp(posts)} GROUP BY ALL ORDER BY n DESC LIMIT 20""").df()
        log.info(f"[{cohort}] -> {out}\n{d.to_string()}")


if __name__ == "__main__":
    main()
