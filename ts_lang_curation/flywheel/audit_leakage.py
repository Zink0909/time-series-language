#!/usr/bin/env python3
"""W4 · independent, cross-source leakage audit for every text->ts record (不自证).

Does NOT trust each source's own cutoff claim — it re-derives, per record, the standard the whole
project lives on (CLAUDE §1.2): the conditioning text may only carry information dated BEFORE the
forecast origin. Programmatic + offline. Per reference papers: All-Leaks-Count temporal
contamination (2602.17234), lookahead-bias detection (2512.23847), point-in-time benchmark
(2601.13770), and "cut time BEFORE windowing" (2512.06932).

Checks each ts_forecast[_covariates] record:
  1. knowledge_time present + ISO — the honest "latest source timestamp actually used".
  2. knowledge_time <= forecast origin. Origin = spike_date (flywheel_wiki) else event_date.
     Flywheel sources (retrieved/synthesized text) must be STRICTLY before origin 00:00 UTC
     (the origin day is the first forecast step). Hand sources (verbatim same-day text: an FOMC
     statement, a 10-K) may equal the origin day.
  3. future-year screen (LLM implicit lookahead): a REVIEW FLAG, not a hard fail — the conditioning
     text naming a year later than the origin is a WEAK signal (a 2021 headline legitimately
     forecasts 2022 demand), high false-positive, so it is surfaced for human spot-check only.

Exit 0 iff the FLYWHEEL sources (oil + wiki) are 0-leak — that is W4's hard guarantee. Problems in
frozen hand sources are printed as WARN findings (changing their semantics is out of this scope).
Run:  micromamba run -n ts-language python flywheel/audit_leakage.py
"""
import os, sys, json, glob, re

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT_DIR = os.path.join(PKG, "out")
REPORT = os.path.join(HERE, "raw", "_leakage_audit.json")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def _user_text(rec):
    """The conditioning text = the user turn of the ChatML transcript."""
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", rec["text"], re.S)
    return m.group(1) if m else rec["text"]


def _origin(rec):
    return rec.get("spike_date") or rec.get("event_date")


def audit_record(rec):
    origin = _origin(rec)
    kt = rec.get("knowledge_time")
    is_fw = bool(rec.get("flywheel"))
    reasons = []
    if not origin:
        reasons.append("no_forecast_origin (no spike_date/event_date to check against)")
    if not kt:
        reasons.append("missing_knowledge_time")
    elif origin:
        # normalize: origin day 00:00 UTC is the first forecast step
        origin_z = origin + "T00:00:00Z"
        if is_fw:
            if not (kt < origin_z):                       # strictly before the origin day
                reasons.append(f"kt_not_before_origin (kt={kt} >= {origin_z})")
        else:
            if kt[:10] > origin:                          # hand source: same day allowed
                reasons.append(f"kt_after_origin (kt={kt[:10]} > {origin})")
    # future-year mention: REVIEW FLAG only (weak, high-FP signal), never a hard leak
    flags = []
    if origin:
        oy = int(origin[:4])
        fy = sorted({int(m.group()) for m in _YEAR.finditer(_user_text(rec)) if int(m.group()) > oy})
        if fy:
            flags.append(f"future_year_mention {fy} (origin year {oy})")
    return {"series_id": rec.get("series_id"), "dataset": rec.get("dataset"),
            "flywheel": is_fw, "origin": origin, "knowledge_time": kt,
            "leaked": bool(reasons), "reasons": reasons, "review_flags": flags}


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.jsonl"))):
        if f.endswith("fred_fomc.v1.jsonl"):
            continue
        for l in open(f):
            r = json.loads(l)
            if r["task_type"].startswith("ts_forecast"):
                rows.append(audit_record(r))

    fw = [r for r in rows if r["flywheel"]]
    hand = [r for r in rows if not r["flywheel"]]
    fw_leak = [r for r in fw if r["leaked"]]
    hand_leak = [r for r in hand if r["leaked"]]

    print(f"== W4 leakage audit · {len(rows)} text->ts records "
          f"({len(fw)} flywheel + {len(hand)} hand) ==\n")
    print(f"FLYWHEEL sources (hard guarantee): {len(fw)-len(fw_leak)}/{len(fw)} clean "
          f"({'0 leaks' if not fw_leak else str(len(fw_leak))+' LEAKS'})")
    for r in fw_leak:
        print(f"   LEAK {r['series_id']}: {r['reasons']}")

    # hand-source findings grouped by reason kind (WARN, not a hard fail)
    print(f"\nHAND sources: {len(hand)-len(hand_leak)}/{len(hand)} clean")
    kinds = {}
    for r in hand_leak:
        key = r["reasons"][0].split(" ")[0]
        kinds.setdefault(key, []).append(r["dataset"])
    for kind, dss in sorted(kinds.items()):
        import collections
        by = collections.Counter(dss)
        print(f"   WARN {kind}: {dict(by)}")

    flagged = [r for r in rows if r["review_flags"]]
    print(f"\nreview flags (future-year mention, human spot-check, non-blocking): {len(flagged)}")
    for r in flagged:
        print(f"   FLAG {r['series_id']}: {r['review_flags']}")

    json.dump({"n": len(rows), "flywheel_leaks": len(fw_leak), "hand_leaks": len(hand_leak),
               "review_flags": len(flagged), "rows": rows}, open(REPORT, "w"), indent=1,
              ensure_ascii=False)
    ok = not fw_leak
    print(f"\n== {'PASS' if ok else 'FAIL'}: flywheel {len(fw_leak)} leaks "
          f"(hand-source WARNs: {len(hand_leak)}, reported not blocking) ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
