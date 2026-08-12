#!/usr/bin/env python3
"""Export held-out test events for the ChatTime zero-shot context-guided 3-arm probe — RAW VALUES.

ChatTime (AAAI'25, ChengsenWang/ChatTime-1-7B-Chat) is a text+series model whose predict(history,
context) is context-guided forecasting. The first run fed context-normalized (z) values and got a
null; ChatTime likely expects RAW magnitudes coherent with the event text ("oil price ..."), so this
version reconstructs the original values (raw = z*std + mean, using each record's context stats) and
feeds those, with the text UNCHANGED (we do NOT compress the cause to fit ChatTime's context style —
that would change what we are testing). Fixed shape (hist_len/pred_len). Text stays as-is.
  micromamba run -n ts-language python flywheel/companion/export_chattime_data.py
"""
import os, sys, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import data, arms                                                    # noqa: E402

OUT = os.path.join(HERE, "_chattime_export.json")
OUT_DIR = os.path.join(PKG, "out")
H, F = 10, 5


def _stats_map():
    """series_id -> (mean, std) = the context-window stats each flywheel record normalized with,
    so we can invert z back to the raw price/level."""
    m = {}
    for f in glob.glob(os.path.join(OUT_DIR, "*.jsonl")):
        if f.endswith("fred_fomc.v1.jsonl"):
            continue
        for line in open(f):
            r = json.loads(line)
            if not r.get("flywheel") or not r["task_type"].startswith("ts_forecast"):
                continue
            sp = (r.get("normalization", {}).get("spans") or [])
            if r.get("series_id") and sp:
                m[r["series_id"]] = (sp[0]["mean"], sp[0]["std"])
    return m


def main():
    items = data.load(flywheel_only=True)
    _, test, problems = data.time_split(items, "2024-01-01", "2023-10-01")
    A = arms.make_arms(test)
    STATS = _stats_map()
    exp, skipped = [], 0
    for a, b, c in zip(A["A_no_text"], A["B_flywheel_text"], A["C_shuffled_text"]):
        if len(b["history"]) < H or len(b["future"]) < F:
            continue
        st = STATS.get(b["series_id"])
        if not st:
            skipped += 1
            continue
        mean, std = st
        def raw(zs):
            return [round(z * std + mean, 4) for z in zs]
        exp.append({"series_id": b["series_id"], "dataset": b["dataset"],
                    "history": raw(b["history"][-H:]), "future": raw(b["future"][:F]),
                    "text": b["text"], "text_shuffled": c["text"]})
    json.dump({"hist_len": H, "pred_len": F, "space": "raw", "leakage_ok": not problems,
               "n": len(exp), "events": exp}, open(OUT, "w"), ensure_ascii=False)
    from collections import Counter
    print(f"exported {len(exp)} RAW-value ({H}->{F}) events (skipped {skipped} w/o stats) -> {OUT} "
          f"({os.path.getsize(OUT)//1024} KB), leakage_ok={not problems}")
    print("by dataset:", dict(Counter(e["dataset"] for e in exp)))
    print("sample raw history[0]:", exp[0]["history"], "->", exp[0]["future"])


if __name__ == "__main__":
    main()
