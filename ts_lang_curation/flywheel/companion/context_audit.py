#!/usr/bin/env python3
"""CONTEXT AUDIT (meeting 7/17 · §4 "逐条自查 context 缺失") — INDEPENDENT of validate().

The 7/17 meeting flagged a common failure in public multimodal datasets: the conditioning text does not tell
the model (a) WHAT "now" is (the forecast origin) and (b) the TEMPORAL RELATION between the history and the
event/forecast — so the model cannot line up "now", the history, and the target. (Some public sets even order
economy tasks backwards = predicting the past.) This scans EVERY built record's conditioning text for an
explicit time anchor + forecast framing, per (source, direction). It is a CONTENT-level check, complementary
to the leakage audit (which checks timestamps, not whether the text SAYS the time).

  micromamba run -n ts-language python flywheel/companion/context_audit.py

Rule (text->ts, where the text CONDITIONS the forecast — anchor is REQUIRED):
  · now-anchor : an explicit date/point-in-time in the text (ISO date, "Month D, YYYY", "before YYYY-MM-DD",
                 "as of ...", "as known before ...", or "on YYYY-MM-DD").
  · framing    : forecast-relation wording ("forecast the", "given ... history", "over the next", "recent ...").
For ts->text (the text is the OUTPUT/description) the same anchor is checked on the user INTRO (which states
the series window) — a generic intro with no dates would make the description non-locatable.
"""
import os, re, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
ANCHOR = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"                                   # ISO date
    r"|\b(?:" + MONTHS + r")\s+\d{1,2},?\s+\d{4}\b"            # Month D, YYYY
    r"|\bas known before\b|\bas of\b|\bthrough\b\s+\d"         # explicit cutoff phrasing
    r"|\bbefore\s+\d{4}-\d{2}-\d{2}\b|\bon\s+\d{4}-\d{2}-\d{2}\b", re.I)
YEAR_RANGE = re.compile(r"\b(?:fiscal\s+)?\d{4}\s*(?:-|–|to)\s*\d{4}\b|\b\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}\b", re.I)
FRAMING = re.compile(r"\bforecast the\b|\bgiven .*?history\b|\bover the next\b|\brecent .*?(history|price|activity)\b"
                     r"|\bdescribe\b|\bconnect it to\b", re.I)


def user_text(rec):
    m = re.search(r"<\|im_start\|>user\s*(.*?)<\|im_end\|>", rec.get("text", ""), re.S)
    return (m.group(1).strip() if m else "")


def audit_file(path):
    src = os.path.basename(path)[:-6]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return None
    by = {}                                                    # (direction) -> stats
    for rec in rows:
        d = rec.get("direction", rec.get("meta", {}).get("direction", "?"))
        ut = user_text(rec)
        has_anchor = bool(ANCHOR.search(ut) or YEAR_RANGE.search(ut))
        has_framing = bool(FRAMING.search(ut))
        has_kt = bool(rec.get("knowledge_time") or rec.get("meta", {}).get("knowledge_time"))
        b = by.setdefault(d, dict(n=0, anchor=0, framing=0, kt=0, flags=[]))
        b["n"] += 1; b["anchor"] += has_anchor; b["framing"] += has_framing; b["kt"] += has_kt
        if not has_anchor and len(b["flags"]) < 3:
            b["flags"].append(ut[:150])
    return src, by


def main():
    print("CONTEXT AUDIT · does the conditioning text state 'now' + the history/forecast relation?")
    print("(independent of validate(); text->ts REQUIRES a now-anchor, ts->text checks the intro window)\n")
    tot = dict(n=0, anchor=0)
    problem = []
    for path in sorted(glob.glob(os.path.join(OUT, "*.jsonl"))):
        r = audit_file(path)
        if not r:
            continue
        src, by = r
        for d, b in by.items():
            pct = 100.0 * b["anchor"] / b["n"]
            tot["n"] += b["n"]; tot["anchor"] += b["anchor"]
            status = "OK " if pct >= 95 else ("WEAK" if pct >= 50 else "GAP ")
            print(f"[{status}] {src:26s} dir={d:10s} n={b['n']:5d}  now-anchor {b['anchor']:5d}/{b['n']} ({pct:5.1f}%)"
                  f"  framing {100.0*b['framing']/b['n']:5.1f}%  knowledge_time {100.0*b['kt']/b['n']:5.1f}%")
            if pct < 95:
                problem.append((src, d, pct, b["flags"]))
    print(f"\nTOTAL: {tot['anchor']}/{tot['n']} records ({100.0*tot['anchor']/tot['n']:.1f}%) carry an explicit time anchor in the text.")
    if problem:
        print("\n--- flagged (source,direction) with <95% anchored + example texts lacking a 'now' ---")
        for src, d, pct, flags in problem:
            print(f"\n  {src} · {d} · {pct:.1f}% anchored")
            for f in flags:
                print(f"     ✗ {f}")
    else:
        print("\nAll (source,direction) >=95% anchored.")


if __name__ == "__main__":
    main()
