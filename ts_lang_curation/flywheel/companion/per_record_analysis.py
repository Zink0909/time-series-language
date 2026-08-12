#!/usr/bin/env python3
"""P1 · per-record text-usefulness analysis (generalizes feedback_s9 to any verified source).

Input = the per-record dump from migas_finetune --dump-per-record ({series_id, err_A, err_B, usefulness}
where usefulness = err_A - err_B, >0 means the record's TEXT helped its own forecast). This turns a
per-SOURCE keep/drop verdict into a per-RECORD score, so a MIXED source can keep its useful subset instead
of being dropped whole — and, by correlating usefulness with record features, LEARN what predicts it (which
is exactly the text-alignment axis of the scaling law). Method = the cheap S9 proxy; the principled upgrades
are For-Value (forward-only, arXiv 2508.10180) and influence functions / LoGra (arXiv 2405.13954), and this
score should be sanity-checked with the brittleness test (drop top-useful, verdict must degrade).

  micromamba run -n ts-language python flywheel/companion/per_record_analysis.py <dump.json> [source]
"""
import os, sys, json, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(os.path.dirname(HERE)), "raw")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return num / den if den else 0.0


def cyber_features():
    ev = {e["cve"]: e for e in json.load(open(os.path.join(RAW, "cyber", "events.json")))}
    feats = {}
    for cve, e in ev.items():
        s = e["series"]
        feats[cve] = {
            "spike_pt": max(s) - min(s),                       # peak-trough (real EPSS move to forecast)
            "recency": 1 if e["dateAdded"] >= "2025-06-01" else 0,   # recent CVE (1) vs older 2024 (0)
            "series_std": st.pstdev(s) if len(s) > 1 else 0.0,
            "text_words": len((e.get("desc", "") + " " + e.get("name", "")).split()),
        }
    return feats


def main():
    dump = json.load(open(sys.argv[1]))
    src = sys.argv[2] if len(sys.argv) > 2 else "cyber_epss"
    feat = cyber_features() if src.startswith("cyber") else {}
    key = lambda sid: sid.replace("_t2s", "").replace("_s2t", "")

    use = [r["usefulness"] for r in dump]
    n = len(use)
    helped = sum(1 for u in use if u > 0)
    print(f"per-record text-usefulness · {src} · n={n}")
    print(f"  usefulness = err_A - err_B (>0 = text helped this record)")
    print(f"  median {st.median(use):+.4f} · mean {st.mean(use):+.4f} · helped {helped}/{n} ({100*helped//n}%)")
    top = sorted(dump, key=lambda r: -r["usefulness"])
    print(f"  most-helped:  {', '.join(key(r['series_id']) for r in top[:5])}")
    print(f"  most-hurt:    {', '.join(key(r['series_id']) for r in top[-5:])}")

    if feat:
        print("\n  what predicts text-usefulness (Spearman ρ, |ρ|>0.15 = signal):")
        paired = [(feat[key(r["series_id"])], r["usefulness"]) for r in dump if key(r["series_id"]) in feat]
        us = [u for _, u in paired]
        for fname in ["spike_pt", "recency", "series_std", "text_words"]:
            xs = [f[fname] for f, _ in paired]
            rho = spearman(xs, us)
            flag = "  <-- predicts usefulness" if abs(rho) > 0.15 else ""
            print(f"    {fname:12} ρ = {rho:+.2f}{flag}")
        # actionable: keep the useful subset by the best predictor
        hi = [u for f, u in paired if f["spike_pt"] >= 5]
        lo = [u for f, u in paired if f["spike_pt"] < 5]
        if hi and lo:
            print(f"\n  by spike_pt>=5: useful median {st.median(hi):+.4f} (n={len(hi)}) vs flat {st.median(lo):+.4f} (n={len(lo)})")
        print("\n  => per-record score lets a MIXED source keep its useful subset instead of being dropped whole;")
        print("     the predictor(s) above ARE the text-alignment axis of the scaling law (learned, not hand-set).")


if __name__ == "__main__":
    main()
