#!/usr/bin/env python3
"""GENERATION ID TEST for ts->text (meeting 7/17 · §8 "用同一数据集 train/test split 先证明 ID 上有用").

The forecast direction (text->ts) already gets this ID test from the verification pipeline itself
(fine-tune MiGAS on a source's OWN train split, evaluate on its OWN held-out = in-distribution). This is the
ts->text analog: split each describe-the-series source 80/20 by time, and check whether a HELD-OUT description
still identifies its OWN (held-out) series against distractors drawn from the TRAIN pool. If discriminability
holds on the held-out split (not just in-sample), the ts->text pairing generalizes ID — the model would learn
a real series->text mapping, not memorize the training instances.

Non-circular (series as ground truth, not an LLM judge — 不自证). Reuses the matcher from verify_ts2text.py.

  micromamba run -n ts-language python flywheel/companion/verify_ts2text_heldout.py
"""
import os, sys, json, glob, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify_ts2text import raw_pool, stated_numbers, match_rate                      # reuse identical matcher
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "out")
K = 20


def load(src):
    recs = []
    for f in glob.glob(os.path.join(OUT, "*.jsonl")):
        if f.endswith("fred_fomc.v1.jsonl"):
            continue
        for line in open(f):
            r = json.loads(line)
            if r.get("task_type") == "text_desc" and (r.get("dataset") == src or src in f):
                recs.append(r)
    # order by event/knowledge time so the split is a real past->future ID hold-out
    recs.sort(key=lambda r: r.get("event_date") or r.get("knowledge_time") or r.get("series_id") or "")
    return recs


def disc_on(test_recs, pool_recs):
    """discriminability of each test description: own series match − mean match vs K random pool series."""
    random.seed(0)
    tnum = [stated_numbers(r["text"]) for r in test_recs]
    tpool = [raw_pool(r) for r in test_recs]
    ppool = [raw_pool(r) for r in pool_recs]
    out = []
    for i, r in enumerate(test_recs):
        o = match_rate(tnum[i], tpool[i])
        idx = random.sample(range(len(pool_recs)), min(K, len(pool_recs)))
        d = st.mean(match_rate(tnum[i], ppool[j]) for j in idx) if idx else 0.0
        out.append(o - d)
    return out


def main():
    srcs = ["sec_edgar", "wiki_pageviews", "cyber_epss", "fred_fomc", "usgs_quakes"]
    print("GENERATION ID TEST (ts->text) · held-out description identifies its own series vs TRAIN distractors")
    print("split = 80/20 by time; held-out discriminability >= 0.3 => the series->text pairing generalizes ID\n")
    print(f"{'source':16s} {'n':>5} {'train/test':>10} {'disc(full)':>11} {'disc(HELD-OUT)':>15} {'held ≥0.3':>10}  verdict")
    for src in srcs:
        recs = load(src)
        if len(recs) < 10:
            print(f"{src:16s} {len(recs):5d}  (too few text_desc records — skip)")
            continue
        cut = int(len(recs) * 0.8)
        train, test = recs[:cut], recs[cut:]
        full = disc_on(recs, recs)                                                   # in-sample (whole set)
        held = disc_on(test, train)                                                  # ID hold-out: test vs train pool
        mf, mh = st.median(full), st.median(held)
        spec = sum(x >= 0.3 for x in held)
        verdict = "ID-generalizes" if mh >= 0.3 else "weak on held-out"
        print(f"{src:16s} {len(recs):5d} {len(train):4d}/{len(test):<5d} {mf:11.2f} {mh:15.2f} "
              f"{spec:3d}/{len(test):<4d}  {verdict}")


if __name__ == "__main__":
    main()
