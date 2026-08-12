#!/usr/bin/env python3
"""④ corpus-fit · cross-source DIVERSITY audit (ODiS/QuaDMix spirit): before merging a source, does it add
NEW coverage or duplicate an existing one? Represent each source's text as a TF-IDF centroid; pairwise
cosine between source centroids = how redundant their coverage is (1 = same topic space, 0 = orthogonal).
A source that is individually useful but near-duplicate of an existing one adds little to the corpus.

  micromamba run -n ts-language python flywheel/companion/diversity_audit.py
"""
import os, re, json, glob, math
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "out")
STOP = set("the a an and or of to in on for with by is are was were be been at as from that this it its "
           "over under between during percent pct value values series data at up down higher lower than".split())


def _tokens(s):
    return [w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOP]


def tfidf_vectors(docs):
    """Hand-rolled TF-IDF (pure Python): docs -> list of L2-normalized {term: weight} dicts over df>=2 vocab."""
    toks = [_tokens(d) for d in docs]
    df = Counter()
    for t in toks:
        df.update(set(t))
    idf = {w: math.log(len(docs) / c) + 1 for w, c in df.items() if c >= 2}
    vecs = []
    for t in toks:
        v = {w: n * idf[w] for w, n in Counter(t).items() if w in idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs


def cosine(a, b):
    return sum(a[w] * b.get(w, 0.0) for w in a)


def source_texts():
    """{source: [assistant/answer texts]} — the natural-language side of each record."""
    by = {}
    for f in glob.glob(os.path.join(OUT, "*.jsonl")):
        if f.endswith("fred_fomc.v1.jsonl"):
            continue
        for line in open(f):
            r = json.loads(line)
            src = r.get("dataset") or os.path.basename(f).split(".")[0]
            t = r.get("text", "")
            ans = t.split("assistant", 1)[-1].split("</think>")[-1]
            ans = ans.replace("<ts>", " ").replace("</ts>", " ").replace("<|im_end|>", " ").strip()
            if len(ans) > 20:
                by.setdefault(src, []).append(ans[:2000])
    return {s: v for s, v in by.items() if len(v) >= 5}


def main():
    by = source_texts()
    srcs = sorted(by)
    docs = [" ".join(by[s][:400]) for s in srcs]                     # one pooled doc per source (cap for speed)
    V = tfidf_vectors(docs)
    S = [[cosine(V[i], V[j]) for j in range(len(srcs))] for i in range(len(srcs))]
    print(f"cross-source coverage overlap (TF-IDF centroid cosine; 1=redundant, 0=orthogonal)\n")
    print("            " + "".join(f"{s[:9]:>10}" for s in srcs))
    for i, s in enumerate(srcs):
        print(f"{s[:11]:11} " + "".join(f"{S[i][j]:>10.2f}" for j in range(len(srcs))))
    print("\n  redundant pairs (cosine > 0.5 = overlapping coverage):")
    flagged = [(srcs[i], srcs[j], S[i][j]) for i in range(len(srcs)) for j in range(i + 1, len(srcs)) if S[i][j] > 0.5]
    for a, b, c in sorted(flagged, key=lambda x: -x[2]):
        print(f"    {a} ~ {b}: {c:.2f}")
    if not flagged:
        print("    none — every source covers a distinct topic space (good corpus diversity)")
    offs = [S[i][j] for i in range(len(srcs)) for j in range(len(srcs)) if i != j]
    mean_off = sum(offs) / len(offs)
    print(f"\n  mean off-diagonal overlap = {mean_off:.2f}  ({'diverse' if mean_off < 0.3 else 'some redundancy'})")
    print("  => low overlap = each source adds new coverage; high pairs = candidates to merge/down-weight (ODiS).")


if __name__ == "__main__":
    main()
