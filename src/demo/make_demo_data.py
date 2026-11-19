#!/usr/bin/env python3
"""Make a small FAKE Bluesky dataset in exactly the same raw format as the real data, so the full
pipeline can be run end to end on a laptop in a few minutes (code review demo).

Nothing here is real: every account, post, block and label is generated with a fixed random seed.
The real data (snapshot, firehose, official labeler) is private and not in this repo.

Output (same layout and columns as the real downloads):
  <out>/snapshot/repos.parquet                      id, pds, did, tombstoned_at, last_error
  <out>/snapshot/{block,follow,list,listitem,listblock,profile,post}.parquet
                                                    repo_id, collection, rkey, at_rev, content (record JSON)
  <out>/firehose/2026/MM/DD/YYYY-MM-DD.json.gz      one JSON event per line: did, time, received, seq,
                                                    commit{rev, ops[{operation, collection, rkey, record}]}
  <out>/labelers/full_labeler.parquet               seq, ts, src, exp, neg, val, uri, cid, ver
  <out>/labelers/official_labeler.parquet           same, only rows from 2026-04-01

Planted structure (so the analysis has something to find):
  established accounts (in the snapshot)
    ordinary 3000   normal users, they are also the blockers
    troll     200   toxic replies to strangers -> blocked in bursts; abuse labels mostly when English
    spam      200   links + duplicate text -> needs-review then takedown, rarely blocked
    adult      80   adult post labels
    disliked  150   mostly German, blocked a lot by other German users, never labelled (crowd-only)
  history     150   taken down in 2024-2025 (records gone from the snapshot, but blocks on them remain)
  new accounts (first seen in the firehose, not in the snapshot)
    new_spam 400 (half taken down within 24 h), new_troll 300, new_adult 100, new_normal 1500

Usage:  python3 src/demo/make_demo_data.py --out data/demo/raw
"""
from __future__ import annotations

import argparse
import warnings
import gzip
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=DeprecationWarning)   # pandas/numpy timedelta notice, harmless

LABELER = "did:plc:ar7c4by46qjdydhdevvrndac"          # official Bluesky moderation service
CRAWL_END = pd.Timestamp("2026-04-08", tz="UTC")       # snapshot crawl finished here
T0 = pd.Timestamp("2026-04-09", tz="UTC")              # index time for established accounts
FH0, FH1 = pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-06-01", tz="UTC")  # firehose window
LAB_END = pd.Timestamp("2026-07-01", tz="UTC")         # labels available until here
B32 = "234567abcdefghijklmnopqrstuvwxyz"

WORDS = {   # (normal words, toxic words) per language
    "en": ("nice day coffee music cats weekend reading garden football travel".split(), "idiot stupid trash loser pathetic".split()),
    "de": ("schoen Tag Kaffee Musik Katzen Wochenende lesen Garten Fussball Reise".split(), "Idiot dumm Abschaum Versager erbaermlich".split()),
    "ja": ("今日 コーヒー 音楽 猫 週末 読書 庭 旅行 写真 料理".split(), "バカ クズ 死ね 無能 ゴミ".split()),
    "pt": ("bom dia cafe musica gatos fim leitura jardim futebol viagem".split(), "idiota burro lixo otario patetico".split()),
    "ru": ("день кофе музыка кошки выходные чтение сад футбол путешествие".split(), "дурак идиот мусор урод тупой".split()),
    "ko": ("오늘 커피 음악 고양이 주말 독서 정원 축구 여행".split(), "멍청 쓰레기 병신 꺼져 한심".split()),
}
LANGS, LANG_P = ["en", "de", "ja", "pt", "ru", "ko"], [0.60, 0.12, 0.12, 0.08, 0.04, 0.04]


def tid(t: pd.Timestamp, clock: int = 0) -> str:
    """atproto TID: 53-bit microsecond time + 10-bit clock id, base32-sortable, 13 chars."""
    v = (int(t.value // 1000) << 10) | (clock & 1023)
    return "".join(B32[(v >> (5 * i)) & 31] for i in reversed(range(13)))


def iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class World:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.acc = {}                        # did -> dict(kind, lang, established)
        self.snap = defaultdict(list)        # snapshot table -> rows (did, rkey, record)
        self.events = []                     # firehose: (time, did, operation, collection, rkey, record)
        self.labels = []                     # (ts, val, uri, neg, exp)
        self.rk = defaultdict(int)

    # ---------- helpers
    def rkey(self, did: str) -> str:
        self.rk[did] += 1
        return f"3k{self.rk[did]:011d}"

    def text(self, lang: str, toxic: bool, spam: bool = False) -> str:
        normal, bad = WORDS[lang]
        w = list(self.rng.choice(normal, int(self.rng.integers(4, 9))))
        if toxic:
            for _ in range(int(self.rng.integers(1, 3))):
                w.insert(int(self.rng.integers(0, len(w))), str(self.rng.choice(bad)))
        if spam:
            return "BUY cheap followers now, limited offer"
        return " ".join(w)

    def post_record(self, did, t, lang, toxic=False, spam=False, reply_to=None):
        r = {"$type": "app.bsky.feed.post", "text": self.text(lang, toxic, spam), "createdAt": iso(t), "langs": [lang]}
        if reply_to:
            u = f"at://{reply_to}/app.bsky.feed.post/3kparent"
            r["reply"] = {"parent": {"uri": u}, "root": {"uri": u}}
        if spam:
            r["facets"] = [{"index": {"byteStart": 0, "byteEnd": 3},
                            "features": [{"$type": "app.bsky.richtext.facet#link", "uri": "https://spam.example/offer"}]}]
            r["embed"] = {"$type": "app.bsky.embed.external", "external": {"uri": "https://spam.example/offer", "title": "x", "description": ""}}
        return r

    def add(self, kind: str, n: int, established: bool, lang_p=None):
        out = []
        for _ in range(n):
            did = f"did:plc:demo{len(self.acc):06d}{kind[:3]}"
            lang = str(self.rng.choice(LANGS, p=lang_p or LANG_P))
            self.acc[did] = {"kind": kind, "lang": lang, "established": established}
            out.append(did)
        return out

    def label(self, ts, val, uri, neg=False, exp=None):
        if ts < LAB_END:
            self.labels.append((ts, val, uri, neg, exp))

    def block(self, blocker, subject, t, live: bool):
        rec = {"$type": "app.bsky.graph.block", "subject": subject, "createdAt": iso(t)}
        rk = self.rkey(blocker)
        if live:
            self.events.append((t, blocker, "create", "app.bsky.graph.block", rk, rec))
        else:
            self.snap["block"].append((blocker, rk, rec))
        return rk

    def blockers_for(self, lang, n, same_lang_share):
        """Pick n distinct ordinary users; a share of them speak the target's language."""
        same = [d for d in self.ordinary if self.acc[d]["lang"] == lang]
        k = min(int(round(n * same_lang_share)), len(same))
        a = list(self.rng.choice(same, k, replace=False)) if k else []
        chosen = set(a)
        rest = [d for d in self.ordinary if d not in chosen]
        return a + list(self.rng.choice(rest, n - k, replace=False))

    def t_between(self, a, b):
        return a + pd.Timedelta(seconds=float(self.rng.uniform(0, (b - a).total_seconds())))

    # ---------- the world
    def build(self):
        rng = self.rng
        self.ordinary = self.add("ordinary", 3000, True)
        troll = self.add("troll", 200, True)
        spam = self.add("spam", 200, True)
        adult = self.add("adult", 80, True)
        disliked = self.add("disliked", 150, True, lang_p=[0.2, 0.7, 0.05, 0.05, 0.0, 0.0])
        history = self.add("history", 150, False)
        new_spam = self.add("new_spam", 400, False)
        new_troll = self.add("new_troll", 300, False)
        new_adult = self.add("new_adult", 100, False)
        new_normal = self.add("new_normal", 1500, False)
        established = [d for d, a in self.acc.items() if a["established"]]

        # --- snapshot: profiles, posts before the crawl, follows, lists
        for d in established:
            a = self.acc[d]
            self.snap["profile"].append((d, "self", {"$type": "app.bsky.actor.profile", "displayName": "user",
                                                     "avatar": {"ref": "x"} if rng.random() < .8 else None,
                                                     "createdAt": iso(self.t_between(pd.Timestamp("2023-06-01", tz="UTC"), T0 - pd.Timedelta(days=100)))}))
            n = {"ordinary": rng.integers(0, 10), "troll": rng.integers(8, 25), "spam": rng.integers(15, 40),
                 "adult": rng.integers(5, 15), "disliked": rng.integers(5, 20)}[a["kind"]]
            for _ in range(int(n)):
                t = self.t_between(T0 - pd.Timedelta(days=150), CRAWL_END)
                tox = a["kind"] == "troll" and rng.random() < .6 or rng.random() < .05
                reply = str(rng.choice(self.ordinary)) if (a["kind"] == "troll" and rng.random() < .7) or rng.random() < .2 else None
                self.snap["post"].append((d, self.rkey(d), self.post_record(d, t, a["lang"], tox, a["kind"] == "spam", reply)))
        for _ in range(20000):
            f, s = rng.choice(established, 2)
            if f != s:
                self.snap["follow"].append((f, self.rkey(f), {"$type": "app.bsky.graph.follow", "subject": s,
                                                               "createdAt": iso(self.t_between(pd.Timestamp("2024-01-01", tz="UTC"), CRAWL_END))}))
        owner = self.ordinary[5]
        modlist = f"at://{owner}/app.bsky.graph.list/3kmodlist000001"
        self.snap["list"].append((owner, "3kmodlist000001", {"$type": "app.bsky.graph.list", "name": "Trolls to avoid",
                                                             "purpose": "app.bsky.graph.defs#modlist", "createdAt": "2025-11-01T00:00:00.000Z"}))
        self.snap["list"].append((owner, "3kcurate000001", {"$type": "app.bsky.graph.list", "name": "Good follows",
                                                            "purpose": "app.bsky.graph.defs#curatelist", "createdAt": "2025-11-01T00:00:00.000Z"}))
        for s in self.ordinary[100:180]:
            self.snap["listblock"].append((s, self.rkey(s), {"$type": "app.bsky.graph.listblock", "subject": modlist,
                                                             "createdAt": iso(self.t_between(pd.Timestamp("2025-11-02", tz="UTC"), CRAWL_END))}))
        for s in rng.choice(troll, 40, replace=False):
            self.snap["listitem"].append((owner, self.rkey(owner), {"$type": "app.bsky.graph.listitem", "subject": str(s), "list": modlist,
                                                                    "createdAt": iso(self.t_between(pd.Timestamp("2026-01-01", tz="UTC"), CRAWL_END))}))

        # --- background blocks before the crawl (some old blocking on everyone) + 3 heavy blockers
        for _ in range(4000):
            b, s = rng.choice(self.ordinary), rng.choice(established)
            if b != s:
                self.block(b, s, self.t_between(pd.Timestamp("2024-01-01", tz="UTC"), CRAWL_END), live=False)
        for hb in self.ordinary[:3]:
            for s in rng.choice(established, 1100, replace=False):
                if s != hb:
                    self.block(hb, s, self.t_between(pd.Timestamp("2025-01-01", tz="UTC"), CRAWL_END), live=False)

        # --- history: taken down in 2024-2025; 60 % were blocked first (blocks still in the snapshot)
        for d in history:
            td = self.t_between(pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC"))
            if rng.random() < .6:
                for b in self.blockers_for(self.acc[d]["lang"], int(rng.integers(1, 15)), .3):
                    self.block(b, d, td - pd.Timedelta(hours=float(rng.uniform(1, 400))), live=False)
            self.label(td, "!takedown", d)

        # --- firehose activity of established accounts (Apr-May); taken-down accounts stop posting
        stop = {}
        def outcome_after(d, start, p_takedown, p_review, p_abuse, p_adult, delay_days):
            a = self.acc[d]
            t = start + pd.Timedelta(days=float(rng.uniform(*delay_days)))
            if rng.random() < p_review:
                self.label(t - pd.Timedelta(hours=float(rng.uniform(2, 48))), "needs-review", d,
                           exp=iso(t + pd.Timedelta(days=90)))
            if rng.random() < p_abuse:
                self.label(t, str(rng.choice(["rude", "intolerant", "threat"], p=[.6, .3, .1])), f"at://{d}/app.bsky.feed.post/{self.rkey(d)}")
            if rng.random() < p_adult:
                self.label(t, str(rng.choice(["porn", "sexual", "nudity"])), f"at://{d}/app.bsky.feed.post/{self.rkey(d)}")
            if rng.random() < p_takedown / 10:          # a few temporary suspensions
                self.label(t + pd.Timedelta(hours=1), "!suspend", d, exp=iso(t + pd.Timedelta(days=7)))
            if rng.random() < p_takedown:
                td = t + pd.Timedelta(hours=float(rng.uniform(1, 72)))
                self.label(td, "!takedown", d)
                if "spam" in a["kind"] and rng.random() < .1:
                    self.label(td, "spam", d)            # reason labels are rare in the real data too
                if rng.random() < .3:                   # some posts removed before the account
                    self.label(td - pd.Timedelta(hours=6), "!takedown", f"at://{d}/app.bsky.feed.post/{self.rkey(d)}")
                stop[d] = td
                if rng.random() < .05:
                    self.label(td + pd.Timedelta(days=float(rng.uniform(3, 20))), "!takedown", d, neg=True)   # appeal won

        for d in troll:
            en = self.acc[d]["lang"] == "en"
            outcome_after(d, T0, .25 if en else .08, .5, .75 if en else .12, 0, (3, 40))
            onset = self.t_between(T0, T0 + pd.Timedelta(days=25))
            for b in self.blockers_for(self.acc[d]["lang"], int(rng.integers(5, 45)), .7):
                self.block(b, d, onset + pd.Timedelta(hours=float(rng.exponential(36))), live=True)
        for d in spam:
            outcome_after(d, T0, .6, .8, 0, 0, (2, 28))
            for b in self.blockers_for(self.acc[d]["lang"], int(rng.integers(0, 4)), .3):
                self.block(b, d, self.t_between(T0, FH1), live=True)
        for d in adult:
            outcome_after(d, T0, .1, .3, 0, .85, (2, 28))
            for b in self.blockers_for(self.acc[d]["lang"], int(rng.integers(0, 10)), .3):
                self.block(b, d, self.t_between(T0, FH1), live=True)
        for d in disliked:
            onset = self.t_between(T0, T0 + pd.Timedelta(days=25))
            for b in self.blockers_for(self.acc[d]["lang"], int(rng.integers(6, 50)), .75):
                self.block(b, d, onset + pd.Timedelta(hours=float(rng.exponential(48))), live=True)
        for d in self.ordinary:
            if rng.random() < .01:
                outcome_after(d, T0, .5, .5, .3, 0, (2, 50))
            if rng.random() < .05:
                for b in self.blockers_for(self.acc[d]["lang"], int(rng.integers(1, 4)), .3):
                    self.block(b, d, self.t_between(FH0, FH1), live=True)

        # --- new accounts: first seen in the firehose
        first = {}
        for d in new_spam + new_troll + new_adult + new_normal:
            first[d] = self.t_between(FH0, FH1 - pd.Timedelta(days=3))
            self.events.append((first[d], d, "create", "app.bsky.actor.profile", "self",
                                {"$type": "app.bsky.actor.profile", "displayName": "new",
                                 **({"avatar": {"ref": "x"}} if rng.random() < .5 else {})}))
        for d in new_spam:
            if rng.random() < .5:   # instant takedown, nobody could have seen it coming
                self.label(first[d] + pd.Timedelta(hours=float(rng.uniform(.5, 20))), "!takedown", d)
                stop[d] = first[d] + pd.Timedelta(hours=10)
            else:
                outcome_after(d, first[d] + pd.Timedelta(days=1), .7, .8, 0, 0, (0, 10))
        for d in new_troll:
            en = self.acc[d]["lang"] == "en"
            outcome_after(d, first[d] + pd.Timedelta(days=1), .3 if en else .1, .5, .7 if en else .1, 0, (1, 20))
            for b in self.blockers_for(self.acc[d]["lang"], int(rng.integers(3, 35)), .7):
                self.block(b, d, first[d] + pd.Timedelta(hours=float(rng.exponential(40))), live=True)
        for d in new_adult:
            outcome_after(d, first[d] + pd.Timedelta(days=1), .15, .3, 0, .85, (0, 15))
        for d in new_normal:
            if rng.random() < .02:
                outcome_after(d, first[d] + pd.Timedelta(days=1), .5, .5, 0, 0, (1, 30))

        # posts / follows / likes in the firehose
        for d, a in self.acc.items():
            k = a["kind"]
            if k == "history":
                continue
            start = first.get(d, FH0)
            n = {"ordinary": rng.integers(0, 6), "troll": rng.integers(10, 30), "spam": rng.integers(15, 40),
                 "adult": rng.integers(5, 15), "disliked": rng.integers(5, 15), "new_spam": rng.integers(10, 40),
                 "new_troll": rng.integers(5, 25), "new_adult": rng.integers(3, 12), "new_normal": rng.integers(1, 8)}[k]
            for i in range(int(n)):
                # new accounts post a lot in their first day
                t = (start + pd.Timedelta(hours=float(rng.uniform(0, 24)))) if (k.startswith("new") and i < n // 2) \
                    else self.t_between(start, FH1)
                if d in stop and t >= stop[d]:
                    continue
                tox = ("troll" in k and rng.random() < .6) or rng.random() < .05
                reply = str(rng.choice(self.ordinary)) if ("troll" in k and rng.random() < .7) or rng.random() < .2 else None
                self.events.append((t, d, "create", "app.bsky.feed.post", self.rkey(d),
                                    self.post_record(d, t, a["lang"], tox, "spam" in k, reply)))
            for _ in range(int(rng.integers(0, 4 if "spam" not in k else 30))):
                t = self.t_between(start, FH1)
                if d in stop and t >= stop[d]:
                    continue
                s = str(rng.choice(self.ordinary))
                self.events.append((t, d, "create", "app.bsky.graph.follow", self.rkey(d),
                                    {"$type": "app.bsky.graph.follow", "subject": s, "createdAt": iso(t)}))
                self.events.append((t + pd.Timedelta(seconds=5), d, "create", "app.bsky.feed.like", self.rkey(d),
                                    {"$type": "app.bsky.feed.like", "subject": {"uri": f"at://{s}/app.bsky.feed.post/3kparent", "cid": "x"},
                                     "createdAt": iso(t)}))
        # a few unblocks (deletes) and list additions in the firehose
        live_blocks = [e for e in self.events if e[3] == "app.bsky.graph.block"]
        for e in [live_blocks[i] for i in rng.choice(len(live_blocks), 150, replace=False)]:
            self.events.append((e[0] + pd.Timedelta(days=float(rng.uniform(1, 10))), e[1], "delete", e[3], e[4], None))
        for s in rng.choice(new_troll + troll, 60, replace=False):
            t = first.get(s, T0) + pd.Timedelta(days=float(rng.uniform(1, 5)))
            self.events.append((t, owner, "create", "app.bsky.graph.listitem", self.rkey(owner),
                                {"$type": "app.bsky.graph.listitem", "subject": str(s), "list": modlist, "createdAt": iso(t)}))
        self.events = [e for e in self.events if FH0 <= e[0] < FH1]
        self.established = established

    # ---------- writers
    def write(self, out: Path):
        if out.exists():
            shutil.rmtree(out)
        snap, fh, lab = out / "snapshot", out / "firehose", out / "labelers"
        for p in (snap, fh, lab):
            p.mkdir(parents=True)
        rng = self.rng
        ids = {d: i + 1 for i, d in enumerate(self.established)}
        rev = {d: tid(self.t_between(CRAWL_END - pd.Timedelta(days=60), CRAWL_END)) for d in self.established}
        pds = {d: (4 if self.acc[d]["kind"] == "spam" and rng.random() < .5 else int(rng.integers(1, 4))) for d in self.established}
        pq.write_table(pa.table({"id": pa.array(list(ids.values()), pa.int32()), "pds": pa.array([pds[d] for d in ids], pa.int32()),
                                 "did": list(ids), "tombstoned_at": pa.nulls(len(ids), pa.timestamp("us", "UTC")),
                                 "last_error": pa.nulls(len(ids), pa.string())}), snap / "repos.parquet")
        coll = {"block": "app.bsky.graph.block", "follow": "app.bsky.graph.follow", "list": "app.bsky.graph.list",
                "listitem": "app.bsky.graph.listitem", "listblock": "app.bsky.graph.listblock",
                "profile": "app.bsky.actor.profile", "post": "app.bsky.feed.post"}
        for name, c in coll.items():
            rows = [r for r in self.snap[name] if r[0] in ids]       # authors missing from the snapshot have no records
            pq.write_table(pa.table({"repo_id": pa.array([ids[r[0]] for r in rows], pa.int64()),
                                     "collection": [c] * len(rows), "rkey": [r[1] for r in rows],
                                     "at_rev": [rev[r[0]] for r in rows],
                                     "content": [json.dumps({k: v for k, v in r[2].items() if v is not None}, ensure_ascii=False) for r in rows]}),
                           snap / f"{name}.parquet")
        # firehose: one gzip file per day, events in time order
        self.events.sort(key=lambda e: e[0])
        by_day = defaultdict(list)
        for seq, (t, did, op, c, rk, rec) in enumerate(self.events, start=1):
            o = {"operation": op, "collection": c, "rkey": rk}
            if rec is not None:
                o["record"] = rec
            by_day[t.strftime("%Y-%m-%d")].append(json.dumps(
                {"did": did, "time": iso(t), "received": iso(t + pd.Timedelta(milliseconds=200)), "seq": seq,
                 "commit": {"rev": tid(t), "ops": [o]}}, ensure_ascii=False))
        for day in pd.date_range(FH0, FH1 - pd.Timedelta(days=1), freq="D").strftime("%Y-%m-%d"):
            y, m, dd = day.split("-")
            p = fh / y / m / dd / f"{day}.json.gz"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:   # mtime=0: same bytes every run
                f.write(("\n".join(by_day.get(day, [])) + "\n").encode("utf-8"))
        # labels
        L = pd.DataFrame(sorted(self.labels, key=lambda x: x[0]), columns=["ts", "val", "uri", "neg", "exp"])
        L.insert(0, "seq", np.arange(1, len(L) + 1, dtype="int64"))
        L["src"], L["cid"], L["ver"] = LABELER, None, 1
        L = L[["seq", "ts", "src", "exp", "neg", "val", "uri", "cid", "ver"]]
        L["ts"] = L.ts.astype("datetime64[us, UTC]")
        schema = pa.schema([("seq", pa.int64()), ("ts", pa.timestamp("us", "UTC")), ("src", pa.string()), ("exp", pa.string()),
                            ("neg", pa.bool_()), ("val", pa.string()), ("uri", pa.string()), ("cid", pa.string()), ("ver", pa.int64())])
        pq.write_table(pa.Table.from_pandas(L, schema=schema, preserve_index=False), lab / "full_labeler.parquet")
        pq.write_table(pa.Table.from_pandas(L[L.ts >= FH0], schema=schema, preserve_index=False), lab / "official_labeler.parquet")
        kinds = pd.Series({d: a["kind"] for d, a in self.acc.items()}).value_counts()
        print(f"accounts: {dict(kinds)}")
        print(f"snapshot rows: { {k: len(v) for k, v in self.snap.items()} }")
        print(f"firehose events: {len(self.events):,} over {len(by_day)} days;  labels: {len(L):,}")
        print(f"written to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo/raw")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    w = World(args.seed)
    w.build()
    w.write(Path(args.out))


if __name__ == "__main__":
    main()
