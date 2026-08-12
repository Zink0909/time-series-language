#!/usr/bin/env python3
"""TX hazard · LAP leakage/contamination probe (Lookahead-Propensity, arXiv 2512.23847).

The most dangerous failure a source can have: its text (or the model's memory of the event) implicitly
ENCODES the outcome, so "text helps" is actually cheating — and you'd KEEP the poison thinking it's gold.
Timestamp checks miss this. LAP instead asks the model a DATE-ONLY recall query (NO conditioning text):
"on this date, did the series go up or down?" If the model COMMITS to a direction (not 'unknown') AND those
commitments PREDICT the real outcome, the model has memorized outcomes for these events -> any LLM-based use
of this source risks lookahead contamination. Note: the paper proves entity-masking does NOT remove this,
so this recall probe is the right instrument (stronger than our old anonymization probe).

  micromamba run -n ts-language python flywheel/companion/verify_lap.py <_verify_src.json> <subject> [date_regex]
  e.g. ... _verify_fred_fomc.json "the US 2-year Treasury yield"  'FOMC statement, (\\d{4}-\\d{2}-\\d{2})'
"""
import os, re, sys, json, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from core.compress import complete                                   # noqa: E402


def main():
    path, subject = sys.argv[1], sys.argv[2]
    dre = re.compile(sys.argv[3]) if len(sys.argv) > 3 else re.compile(r"(\d{4}-\d{2}-\d{2})")
    test = json.load(open(path))["test"]
    commit = correct = usable = 0
    rows = []
    for t in test:
        m = dre.search(t.get("text", ""))
        if not m:
            continue
        date = m.group(1)
        hist, fut = t["history"], t["future"]
        if not hist or not fut:
            continue
        actual = "up" if fut[-1] > hist[-1] else "down"             # did it rise over the horizon (date-only truth)
        usable += 1
        sys_p = ("You answer ONLY from memory of a specific date. If you do not genuinely recall, say 'unknown'. "
                 "Respond with exactly one word: up, down, or unknown.")
        user_p = (f"On {date}, over the trading days that followed, did {subject} go UP or DOWN relative to "
                  f"its level on {date}? Answer based ONLY on what you recall about that specific date "
                  f"(not general reasoning). One word: up, down, or unknown.")
        try:
            ans = complete(sys_p, user_p, max_tokens=4, temperature=0.0).strip().lower()
        except Exception:
            ans = "unknown"
        ans = "up" if "up" in ans else ("down" if "down" in ans else "unknown")
        if ans != "unknown":
            commit += 1
            if ans == actual:
                correct += 1
        rows.append((date, actual, ans))
    n = usable
    print(f"LAP leakage probe · {os.path.basename(path)} · subject='{subject}' · n={n} events")
    print(f"  commit rate (model gives a direction, not 'unknown') = {commit}/{n} ({100*commit//max(n,1)}%)  = LAP proxy")
    acc = correct / commit if commit else float("nan")
    print(f"  among committed: directional accuracy vs REAL outcome = {correct}/{commit} = {acc:.2f}  (chance 0.50)")
    # verdict: leakage requires BOTH meaningful commit AND committed-accuracy well above chance
    if commit / max(n, 1) < 0.25:
        v = "CLEAN — model abstains ('unknown'); no recalled outcomes -> no LLM-side lookahead hazard"
    elif commit and abs(acc - 0.5) < 0.15:
        v = "CLEAN — model commits but its recall is ~chance (guessing, not memorized) -> no leakage"
    else:
        v = "HAZARD — model recalls outcomes that predict the truth -> LLM use of this source risks contamination"
    print(f"  -> {v}")


if __name__ == "__main__":
    main()
