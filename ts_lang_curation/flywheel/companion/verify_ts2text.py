#!/usr/bin/env python3
"""VERIFICATION for ts->text (text_desc) sources — the analog of the forecast filter, for the OTHER direction.

MiGAS is a forecaster, so the fine-tune-on-own-split filter does not apply to ts->text (describe the series).
The usefulness question for a description is instead: does it carry series-SPECIFIC content, or is it generic?

Non-circular test (uses the SERIES as ground truth, not an LLM judge — 不自证): a useful description states
numbers that identify ITS OWN series but NOT a random distractor series. For each record:
  own_rate        = fraction of the description's stated numbers that match its own series (values or min/max/
                    mean/first/last, relative tol 2%);  should be ~1 (these passed the numeric gate at build).
  distractor_rate = same, averaged over K random OTHER series in the source.
  discriminability = own_rate - distractor_rate   (1 = fully series-specific, 0 = generic/could fit anything).
A source whose descriptions are broadly discriminable carries real series-grounded world knowledge.

  micromamba run -n ts-language python flywheel/companion/verify_ts2text.py <source> [K]
"""
import os, sys, re, json, glob, random, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "out")
_MON = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
# strip DATES first so date fragments (2022-07-08 -> -7,-8; "January 8, 2024" -> 8) are not read as series values
DATE = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b"
                  r"|\b(?:" + _MON + r")\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+(?:19|20)\d{2})?\b"
                  r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:" + _MON + r")\b", re.I)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
NUM = re.compile(r"-?\d+(?:\.\d+)?")
REL = 0.02


def raw_pool(r):
    """All matchable numbers of a record's series: every de-normalized value + per-span min/max/mean/first/last."""
    pool = []
    for sp, nz in zip(r["timeseries"], r["normalization"]["spans"]):
        mean, std = nz["mean"], nz["std"]
        vals = [v * std + mean for v in sp["values"]]
        if not vals:
            continue
        pool += vals + [min(vals), max(vals), st.mean(vals), vals[0], vals[-1]]
    return pool


def stated_numbers(text):
    """Numbers the assistant description asserts (drop years — they are not series values)."""
    ans = text.split("assistant", 1)[-1]
    ans = ans.split("</think>")[-1]
    ans = DATE.sub(" ", ans)                                # drop dates before extracting (their digits aren't series values)
    ans = YEAR.sub(" ", ans)
    return [float(x) for x in NUM.findall(ans)]


def match_rate(nums, pool):
    if not nums:
        return 0.0
    hit = 0
    for x in nums:
        if any(abs(x - p) <= REL * max(abs(p), 1e-9) or abs(x - p) <= 1e-6 for p in pool):
            hit += 1
    return hit / len(nums)


def main():
    src = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    random.seed(0)
    recs = []
    for f in glob.glob(os.path.join(OUT, "*.jsonl")):
        if f.endswith("fred_fomc.v1.jsonl"):
            continue
        for line in open(f):
            r = json.loads(line)
            if r.get("task_type") == "text_desc" and (r.get("dataset") == src or src in f):
                recs.append(r)
    if not recs:
        print(f"{src}: no text_desc records"); return
    pools = [raw_pool(r) for r in recs]
    nums = [stated_numbers(r["text"]) for r in recs]
    n = len(recs)
    own, disc, no_number = [], [], 0
    for i in range(n):
        if not nums[i]:
            no_number += 1
        o = match_rate(nums[i], pools[i])
        others = random.sample([j for j in range(n) if j != i], min(K, n - 1))
        d = st.mean(match_rate(nums[i], pools[j]) for j in others) if others else 0.0
        own.append(o); disc.append(o - d)
    med_disc = st.median(disc)
    specific = sum(x >= 0.3 for x in disc)
    print(f"{src}: {n} text_desc records · K={K} distractors")
    print(f"  own match-rate     median {st.median(own):.2f}   (sanity: should be high — these passed the numeric gate)")
    print(f"  discriminability   median {med_disc:.2f}   (own − distractor; 1=series-specific, 0=generic)")
    print(f"  series-specific    {specific}/{n} ({100*specific//n}%) with discriminability ≥ 0.3")
    print(f"  no-number descriptions {no_number}/{n}")
    verdict = "USEFUL (carries series-specific content)" if med_disc >= 0.3 else \
              "GENERIC (descriptions do not identify their own series — review)"
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
