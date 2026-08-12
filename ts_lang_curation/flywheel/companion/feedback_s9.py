#!/usr/bin/env python3
"""S9 v1 — which-data-helps feedback, the flywheel's closing loop.

Reuses the probe's CACHED LLMTime predictions (no new API calls) to compute a per-event TEXT GAIN
= MASE_A(no-text) - MASE_B(flywheel-text): positive => the text helped that event's forecast. Then
profiles the gain by data dimension (source, flat-context, spike magnitude, horizon) so the flywheel
knows which kinds of pairs to make MORE of and which HURT and should be down-weighted/dropped. This
is the steering signal proposal wants — driven by real downstream effect, not our own gates (不自证).
Cache-only (skips un-probed events). Run after `run.py --llm`:
  micromamba run -n ts-language python flywheel/companion/feedback_s9.py
"""
import os, sys, json, glob
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import data, arms, models                                          # noqa: E402

OUT_DIR = os.path.join(PKG, "out")
REPORT = os.path.join(HERE, "_s9_feedback.json")


def mase(pred, actual, history):
    scale = mean(abs(history[i] - history[i - 1]) for i in range(1, len(history))) or 1e-6
    H = min(len(pred), len(actual))
    return mean(abs(pred[i] - actual[i]) for i in range(H)) / scale


def _bucket_gain(rows, buckets):
    """mean gain per named bucket: buckets = [(name, predicate)]."""
    out = []
    for name, pred in buckets:
        g = [r["gain"] for r in rows if pred(r)]
        if g:
            out.append((name, round(mean(g), 3), len(g)))
    return out


def main():
    items = data.load(flywheel_only=True)
    train, test, _ = data.time_split(items, "2024-01-01", "2023-10-01")
    the_arms = arms.make_arms(test)
    llm = models.LLMTimeForecaster(os.path.join(HERE, "_llmtime_cache.json"))

    meta = {}
    for f in glob.glob(os.path.join(OUT_DIR, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("series_id"):
                meta[r["series_id"]] = r

    rows = []
    for a_it, b_it in zip(the_arms["A_no_text"], the_arms["B_flywheel_text"]):
        H = len(b_it["future"])
        ka = llm._key(a_it["history"], a_it["text"], H)
        kb = llm._key(b_it["history"], b_it["text"], H)
        if ka not in llm.cache or kb not in llm.cache:          # only events actually probed
            continue
        ma = mase(llm.cache[ka], a_it["future"], a_it["history"])
        mb = mase(llm.cache[kb], b_it["future"], b_it["history"])
        m = meta.get(b_it["series_id"], {})
        rows.append({"sid": b_it["series_id"], "gain": ma - mb, "dataset": b_it["dataset"],
                     "flat": bool(m.get("flat_context")),
                     "spike_z": m.get("spike_z") or m.get("spike_ratio") or 0, "H": H})

    if not rows:
        print("no cached probe events — run `companion/run.py --llm` first.")
        return
    n = len(rows)
    overall = mean(r["gain"] for r in rows)
    helped = sum(r["gain"] > 0 for r in rows)
    print(f"== S9 v1 which-data-helps · {n} probed events ==\n")
    print(f"overall mean text gain (MASE_A - MASE_B): {overall:+.3f}  "
          f"(positive = text helps; helped {helped}/{n} events)\n")

    dims = {
        "by source": _bucket_gain(rows, [
            ("flywheel_oil", lambda r: r["dataset"] == "flywheel_oil"),
            ("flywheel_wiki", lambda r: r["dataset"] == "flywheel_wiki"),
            ("flywheel_wiki_scaled", lambda r: r["dataset"] == "flywheel_wiki_scaled")]),
        "by spike size": _bucket_gain(rows, [
            ("small (<6)", lambda r: r["spike_z"] < 6),
            ("large (>=6)", lambda r: r["spike_z"] >= 6)]),
        "by horizon": _bucket_gain(rows, [
            ("short (<10)", lambda r: r["H"] < 10), ("long (>=10)", lambda r: r["H"] >= 10)]),
    }
    for dim, buckets in dims.items():
        print(f"  {dim}:")
        for name, g, k in sorted(buckets, key=lambda x: -x[1]):
            print(f"     {name:22} mean gain {g:+.3f}  (n={k})")

    tips = []
    for dim, buckets in dims.items():
        for name, g, k in buckets:
            if g < 0 and k >= 3:
                tips.append(f"down-weight/review {name} (gain {g:+.3f}, n={k})")
    best = max((b for bs in dims.values() for b in bs), key=lambda x: x[1])
    print(f"\n  S9 signal → make MORE of: {best[0]} (gain {best[1]:+.3f}). "
          f"{'; '.join(tips) if tips else 'no clearly harmful bucket at n>=3.'}")

    json.dump({"n": n, "overall_gain": overall, "helped": helped,
               "dims": {d: b for d, b in dims.items()}, "rows": rows},
              open(REPORT, "w"), indent=1)
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
