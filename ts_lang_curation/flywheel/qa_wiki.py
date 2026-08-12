#!/usr/bin/env python3
"""Independent annotation-QA for the Wikipedia flywheel (Xinyue: 先查标注质量再训练).

Deliberately does NOT call core.chatml.validate() — it re-derives the Data-Schema invariants
(CLAUDE.md §2) from scratch, so a bug in the builder's own validator can't hide (不自证).
Checks, per emitted text->ts record:
  A. schema invariants (independent)        -> must be 0 errors
  B. leakage: knowledge_time < forecast origin (spike day 00:00 UTC)  -> hard red line
  C. lead-event relevance: does the leakage-safe pre-spike lead actually mention the event that
     drove the spike? Proxy = distinctive-word overlap between the point-in-time lead and the
     article's CURRENT summary (which describes the event), excluding title words. Honest signal
     of whether the text can plausibly explain the forecast, or is just a generic entity bio.
Writes flywheel/raw/wiki/_wiki_qa.json for the demo. Run:
  micromamba run -n ts-language python flywheel/qa_wiki.py
"""
import os, sys, json, re

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT = os.path.join(PKG, "out", "flywheel_wiki.jsonl")
SUM_DIR = os.path.join(PKG, "raw", "wiki")
QA_JSON = os.path.join(HERE, "raw", "wiki", "_wiki_qa.json")

_STOP = set(("the a an and or of to in on for is are was were be been being as at by with from "
             "that this it its his her their he she they i an american is a s born who which had "
             "has have also known first second one two new de la el").split())
_WORD = re.compile(r"[a-z][a-z'-]{2,}")


def _content_words(text):
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


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
    if len(ctx) != 1:                                    # v1 wiki: no covariates -> [context, target]
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
    # user side must carry <stats>...</stats> <ts></ts>; assistant a bare <ts></ts>
    if not re.search(r"<stats>len=\d+, mean=[^<]+, std=[^<]+</stats> <ts></ts>", txt):
        e.append("missing prompt-side <stats> <ts>")
    if "assistant\n<think>\n</think>\n\n<ts></ts><|im_end|>" not in txt:
        e.append("missing bare assistant <ts>")
    for f in ("task_type", "source", "dataset", "series_id", "normalization"):
        if f not in rec:
            e.append(f"missing audit field {f}")
    nz = rec.get("normalization", {})
    if nz.get("scope") != "context_stats_for_assistant_targets" or "std_floor" not in nz:
        e.append("normalization block wrong")
    return e


def main():
    recs = [json.loads(l) for l in open(OUT)]
    print(f"== wiki flywheel QA · {len(recs)} emitted text->ts records ==\n")

    # A + B
    n_bad, n_leak, rel = 0, 0, []
    per = []
    for r in recs:
        errs = independent_check(r)
        n_bad += bool(errs)
        origin = r["spike_date"] + "T00:00:00Z"
        kt = r.get("knowledge_time", "")
        leaked = not (kt and kt < origin)
        n_leak += leaked
        # C. lead-event relevance
        slug = r["series_id"].replace("_flywheel_t2s", "")
        lead = r["text"].split(" Given this background")[0]
        title_words = _content_words(r.get("article", ""))
        lead_w = _content_words(lead) - title_words
        sump = os.path.join(SUM_DIR, f"{slug}_summary.json")
        score = None
        if os.path.exists(sump):
            ev_w = _content_words(json.load(open(sump)).get("extract", "")) - title_words
            if ev_w:
                score = round(len(lead_w & ev_w) / len(ev_w), 3)
                rel.append(score)
        per.append({"slug": slug, "errs": errs, "leaked": leaked, "relevance": score,
                    "align": r.get("detect_alignment_days"), "flat": r.get("flat_context")})

    print(f"A. independent schema check : {len(recs)-n_bad}/{len(recs)} clean  "
          f"({'0 errors' if n_bad == 0 else str(n_bad)+' WITH ERRORS'})")
    print(f"B. leakage (rev < spike)    : {len(recs)-n_leak}/{len(recs)} clean  "
          f"({'0 leaks' if n_leak == 0 else str(n_leak)+' LEAKS'})")
    rel_sorted = sorted(rel)
    hi = sum(s >= 0.15 for s in rel)
    print(f"C. lead-event relevance     : median {rel_sorted[len(rel)//2]:.2f}  "
          f"(>=0.15 'lead mentions the event': {hi}/{len(rel)})")
    print("   -> the honest annotation-quality caveat: leakage-safe leads split into event-aware")
    print("      (edited as news broke, high overlap) vs generic entity bios (low overlap).")

    # a few worst / best relevance for the report
    ranked = sorted([p for p in per if p["relevance"] is not None], key=lambda p: p["relevance"])
    print("\n   lowest-relevance (generic bio, weak text signal):")
    for p in ranked[:5]:
        print(f"     {p['slug']:14} rel={p['relevance']}")
    print("   highest-relevance (lead already reflects the event):")
    for p in ranked[-5:]:
        print(f"     {p['slug']:14} rel={p['relevance']}")

    json.dump({"n": len(recs), "schema_errors": n_bad, "leaks": n_leak,
               "relevance_median": rel_sorted[len(rel)//2], "relevance_ge_015": hi,
               "relevance_n": len(rel), "per_record": per},
              open(QA_JSON, "w"), indent=1, ensure_ascii=False)
    ok = n_bad == 0 and n_leak == 0
    print(f"\n== {'PASS' if ok else 'FAIL'}: schema {n_bad} err, leakage {n_leak} leaks ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
