#!/usr/bin/env python3
"""①correctness · SEMANTIC direction-leak HARD GATE (two-stage: regex recall -> LLM-judge precision).

The regex `leak_screen.direction_leak` is high-recall but high-false-positive: it flags "the unemployment
rate will decline" (legitimate forward world-knowledge about ANOTHER quantity) the same as "the yield will
go up" (a real leak of the TARGET). Distinguishing the two needs SEMANTIC TARGET-BINDING — which the regex
can't do. This adds an LLM-judge (via the team vLLM) that, for a regex-flagged text, asks whether the
forward direction is about the FORECAST TARGET itself. Non-circular: it checks a factual property of the
EXTERNAL text ("does it state the target's next move?"), not the quality of any generated output.
Two-stage keeps it cheap: only regex hits reach the judge.

  micromamba run -n ts-language python flywheel/companion/semantic_leak_gate.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from core.leak_screen import direction_leak                          # noqa: E402
from core.compress import complete                                   # noqa: E402


def semantic_direction_leak(text, target):
    """LLM-judge: does `text` forward-assert the direction of the FORECAST TARGET itself (a real leak)?"""
    sysp = "You detect outcome leakage in forecasting inputs. Answer with exactly one word: yes or no."
    userp = (f"Forecast target series: {target}.\n\nText:\n{text}\n\n"
             f"Does the text state or clearly imply which direction the TARGET SERIES itself "
             f"({target}) will move in the FUTURE (e.g. will rise / will fall)? Answer 'yes' ONLY if the "
             f"forward direction is about the target series. Answer 'no' if it only discusses OTHER "
             f"quantities (e.g. unemployment, inflation), or only past/current conditions. One word: yes/no.")
    try:
        ans = complete(sysp, userp, max_tokens=4, temperature=0.0).strip().lower()
    except Exception:
        ans = "no"
    return "yes" in ans


def gate(text, target):
    """Full two-stage gate: regex pre-filter -> LLM semantic confirm. True = HARD leak (TX hazard)."""
    return direction_leak(text) and semantic_direction_leak(text, target)


def main():
    OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "out")
    # real FOMC (regex flags 26 — should be FALSE POSITIVES: about unemployment/inflation, not the yield)
    fomc = [json.loads(l) for l in open(os.path.join(OUT, "fred_fomc.jsonl"))
            if json.loads(l)["task_type"].startswith("ts_forecast")]
    fomc_flagged = [r for r in fomc if direction_leak(r.get("text", ""))]
    # planted poison (text literally = the target's direction)
    poison = json.load(open(os.path.join(HERE, "_verify_fomc_leaky.json")))["test"]
    tgt = "the US 2-year Treasury yield"

    print("SEMANTIC leak gate (regex recall -> LLM-judge precision) · target =", tgt, "\n")
    fp_kept = sum(1 for r in fomc_flagged if semantic_direction_leak(r["text"], tgt))
    print(f"  real FOMC regex-flagged {len(fomc_flagged)} -> LLM confirms leak on {fp_kept}  "
          f"(want ~0: these are 'unemployment will decline', not the yield)")
    pz = poison[:20]
    caught = sum(1 for r in pz if semantic_direction_leak(r["text"], tgt))
    print(f"  planted poison (n={len(pz)}) -> LLM confirms leak on {caught}  (want all: text = the yield's direction)")
    prec = "PRECISE" if fp_kept <= 2 and caught >= 0.8 * len(pz) else "needs tuning"
    print(f"\n  -> two-stage gate is {prec}: regex catches candidates, LLM-judge binds them to the target,")
    print(f"     turning a high-false-positive review-FLAG into a usable HARD gate (TX hazard).")


if __name__ == "__main__":
    main()
