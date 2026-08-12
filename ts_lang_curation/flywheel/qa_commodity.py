#!/usr/bin/env python3
"""Independent annotation-QA for the COMMODITY flywheel (Xinyue: 先查标注质量再训练).

Deliberately does NOT call core.chatml.validate() — re-derives the Data-Schema invariants
(CLAUDE.md §2) from scratch so a builder-validator bug can't hide (不自证). Per emitted text->ts:
  A. schema invariants (independent)                              -> 0 errors
  B. leakage: knowledge_time < forecast origin (event day 00:00Z) -> hard red line
  C. cause faithfulness: every named entity in the cause traces to a retrieved headline
     (core.faithfulness re-derives support from the grounding, independent of the generator) —
     the check that the text is REAL event context, not a generic bio.
Writes flywheel/raw/_commodity_qa.json. Run:
  micromamba run -n ts-language python flywheel/qa_commodity.py
"""
import os, sys, json, re

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
OUT = os.path.join(PKG, "out", "flywheel_commodity.jsonl")
QA_JSON = os.path.join(HERE, "raw", "_commodity_qa.json")

from core.faithfulness import report as faith_report


# ---------- A. independent schema check (re-derives CLAUDE §2, not chatml.validate) ----------
def independent_check(rec):
    e = []
    txt = rec.get("text", "")
    ts = rec.get("timeseries", [])
    if txt.count("<ts></ts>") != len(ts):
        e.append("placeholder/timeseries count mismatch")
    for i, s in enumerate(ts):
        v = s.get("values", [])
        if len(v) != s.get("len"):
            e.append(f"span[{i}] len != values")
        if any((x != x) or (x in (float("inf"), float("-inf"))) for x in v):
            e.append(f"span[{i}] non-finite")
        if not (0 <= s.get("loss_start", -1) <= s.get("len", -1)):
            e.append(f"span[{i}] bad loss_start")
    if not rec.get("task_type", "").startswith("ts_forecast"):
        e.append("not a forecast task")
    ctx = [s for s in ts if s.get("role") == "context"]
    tgt = [s for s in ts if s.get("role") == "target"]
    if len(tgt) != 1:
        e.append("must have exactly one target span")
    if len(ctx) != 1:
        e.append("expected exactly one context span (no covariates)")
    if tgt and ctx:
        c, g = ctx[0], tgt[0]
        H = c["len"]
        if g["values"][:H] != c["values"]:
            e.append("target history prefix != context (normalized)")
        if g.get("loss_start") != H or c.get("loss_start") != H:
            e.append("loss_start != H on context/target")
        if g.get("context_span_idx") != ts.index(c):
            e.append("target context_span_idx wrong")
    if rec.get("text_loss_char_ranges") != []:
        e.append("forecast must have empty text_loss_char_ranges")
    if not re.search(r"<stats>len=\d+, mean=[^<]+, std=[^<]+</stats> <ts></ts>", txt):
        e.append("missing prompt-side <stats> <ts>")
    if "assistant\n<think>\n</think>\n\n<ts></ts><|im_end|>" not in txt:
        e.append("missing bare assistant <ts>")
    for f in ("task_type", "source", "dataset", "series_id", "normalization", "knowledge_time"):
        if f not in rec:
            e.append(f"missing audit field {f}")
    nz = rec.get("normalization", {})
    if nz.get("scope") != "context_stats_for_assistant_targets" or "std_floor" not in nz:
        e.append("normalization block wrong")
    return e


def _cause(rec):
    """The synthesized cause = user text before the ' Given this context...' tail."""
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", rec["text"], re.S)
    user = m.group(1) if m else ""
    return re.split(r"\s+Given this context", user)[0].strip()


def main():
    if not os.path.exists(OUT):
        print("== commodity flywheel QA · no output file yet (run commodity_demo.py first) ==")
        sys.exit(1)
    recs = [json.loads(l) for l in open(OUT)]
    print(f"== commodity flywheel QA · {len(recs)} emitted text->ts records ==\n")

    n_bad = n_leak = n_unfaithful = 0
    by_commodity, per = {}, []
    covs = []
    for r in recs:
        errs = independent_check(r)
        n_bad += bool(errs)
        origin = (r.get("event_date") or "") + "T00:00:00Z"
        kt = r.get("knowledge_time", "")
        leaked = not (kt and kt < origin)
        n_leak += leaked
        ground = "\n".join(r.get("retrieved_titles", []))
        rep = faith_report(_cause(r), ground)
        faithful = rep["verdict"] == "ok"
        n_unfaithful += (not faithful)
        covs.append(rep["entity_coverage"])
        c = r.get("commodity", "?")
        by_commodity[c] = by_commodity.get(c, 0) + 1
        per.append({"series_id": r.get("series_id"), "commodity": c, "errs": errs,
                    "leaked": leaked, "faithful": faithful,
                    "entity_coverage": rep["entity_coverage"], "verdict": rep["verdict"]})

    covs_sorted = sorted(covs)
    print(f"by commodity: {by_commodity}\n")
    print(f"A. independent schema check : {len(recs)-n_bad}/{len(recs)} clean  "
          f"({'0 errors' if n_bad == 0 else str(n_bad)+' WITH ERRORS'})")
    print(f"B. leakage (kt < origin)    : {len(recs)-n_leak}/{len(recs)} clean  "
          f"({'0 leaks' if n_leak == 0 else str(n_leak)+' LEAKS'})")
    print(f"C. cause faithfulness       : {len(recs)-n_unfaithful}/{len(recs)} entity-traced  "
          f"(median entity-coverage {covs_sorted[len(covs_sorted)//2]:.2f})")
    if n_bad:
        for p in per:
            if p["errs"]:
                print(f"   ERR {p['series_id']}: {p['errs']}")
    if n_leak:
        for p in per:
            if p["leaked"]:
                print(f"   LEAK {p['series_id']}")
    if n_unfaithful:
        print("   unfaithful (should be 0 — every emitted cause passed the gate at build time):")
        for p in per:
            if not p["faithful"]:
                print(f"     {p['series_id']}: {p['verdict']}")

    json.dump({"n": len(recs), "schema_errors": n_bad, "leaks": n_leak,
               "unfaithful": n_unfaithful, "by_commodity": by_commodity,
               "coverage_median": covs_sorted[len(covs_sorted)//2], "per_record": per},
              open(QA_JSON, "w"), indent=1, ensure_ascii=False)
    ok = n_bad == 0 and n_leak == 0 and n_unfaithful == 0
    print(f"\n== {'PASS' if ok else 'FAIL'}: schema {n_bad} err, leakage {n_leak} leaks, "
          f"{n_unfaithful} unfaithful ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
