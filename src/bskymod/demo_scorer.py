"""Stand-in toxicity scorer used ONLY with --demo in steps 10 and 11.

The real steps run neural models on GPUs (Detoxify multilingual; TextDetox XLM-R v2; kor_unsmile), which need
torch, several GB of model weights and a GPU. For the code-review demo on a laptop we replace only the model
call by this word-list score; the sampling, chunking, output files and per-account aggregation are the same code.
The word list matches the fake data made by src/demo/make_demo_data.py.
"""
from __future__ import annotations

import numpy as np

TOXIC_WORDS = {
    "idiot", "stupid", "trash", "loser", "pathetic",            # en
    "dumm", "abschaum", "versager", "erbaermlich",              # de ("Idiot" is covered by "idiot")
    "バカ", "クズ", "死ね", "無能", "ゴミ",                       # ja
    "idiota", "burro", "lixo", "otario", "patetico",            # pt
    "дурак", "идиот", "мусор", "урод", "тупой",                 # ru
    "멍청", "쓰레기", "병신", "꺼져", "한심",                     # ko
}


def word_scores(texts: list[str], seed: int = 0) -> np.ndarray:
    """Probability-like score in [0, 1]: 0.05 base + 0.45 per toxic word, plus a little noise."""
    rng = np.random.default_rng(seed)
    hits = np.array([sum(w.lower() in TOXIC_WORDS for w in (t or "").split()) for t in texts], dtype=float)
    return np.clip(0.05 + 0.45 * hits + rng.normal(0, 0.03, len(texts)), 0, 1)
