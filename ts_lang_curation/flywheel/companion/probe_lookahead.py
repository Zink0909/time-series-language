#!/usr/bin/env python3
"""Implicit-lookahead probe — the honest stress test of the ③ signal.

The biggest threat to "text helps" (B<A) is that a general LLM RECOGNIZES the event ("this is the
OpenAI/Altman firing") from its pretraining rather than REASONING from the text. Test: build a third
arm B' = B with named entities ANONYMIZED (OpenAI -> "a major entity"), so the event can't be
recognized, only its structure reasoned about. Then:
  - if gain collapses (B' ~ A): the earlier gain was mostly event recognition -> lookahead-tainted;
  - if gain largely survives (B' still < A): there is real, transferable reasoning in the text.
Uses the cached A/B predictions + new B' calls via the team vLLM. Run:
  micromamba run -n ts-language python flywheel/companion/probe_lookahead.py [--limit N]
"""
import os, sys, re, argparse
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import data, arms, models                                          # noqa: E402
from core.faithfulness import extract_entities                     # noqa: E402


def mase(pred, actual, history):
    from statistics import mean
    scale = mean(abs(history[i] - history[i - 1]) for i in range(1, len(history))) or 1e-6
    H = min(len(pred), len(actual))
    return mean(abs(pred[i] - actual[i]) for i in range(H)) / scale


def anonymize(text):
    """Replace named entities (orgs, people, countries) with a generic placeholder so the specific
    event can't be recognized — the reasoning structure (a shock, a launch, a conflict) stays."""
    for e in sorted(set(extract_entities(text)), key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(e) + r"\b", "a major entity", text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()
    items = data.load(flywheel_only=True)
    _, test, _ = data.time_split(items, "2024-01-01", "2023-10-01")
    test = test[:a.limit]
    the_arms = arms.make_arms(test)
    llm = models.LLMTimeForecaster(os.path.join(HERE, "_llmtime_cache.json"))

    sA, sB, sBa = [], [], []
    for a_it, b_it in zip(the_arms["A_no_text"], the_arms["B_flywheel_text"]):
        H = len(b_it["future"])
        sA.append(mase(llm.predict(a_it["history"], "", H), a_it["future"], a_it["history"]))
        sB.append(mase(llm.predict(b_it["history"], b_it["text"], H), b_it["future"], b_it["history"]))
        anon = anonymize(b_it["text"])
        sBa.append(mase(llm.predict(b_it["history"], anon, H), b_it["future"], b_it["history"]))
    llm.save()

    mA, mB, mBa = median(sA), median(sB), median(sBa)
    gB = mA - mB
    gBa = mA - mBa
    ret = (gBa / gB * 100) if gB else 0.0
    n = len(test)
    print(f"== implicit-lookahead probe · {n} events ==\n")
    print(f"  A no-text           median MASE {mA:.3f}")
    print(f"  B flywheel-text     median MASE {mB:.3f}   gain vs A {gB:+.3f}")
    print(f"  B' anonymized-text  median MASE {mBa:.3f}   gain vs A {gBa:+.3f}\n")
    print(f"  gain retained after anonymizing entities: {ret:.0f}%")
    if ret >= 60:
        verdict = "MOSTLY REASONING — gain survives without recognizable entities (lookahead not the main driver)"
    elif ret >= 25:
        verdict = "MIXED — part reasoning, part event-recognition"
    else:
        verdict = "MOSTLY RECOGNITION — gain collapses when entities are hidden (lookahead-tainted)"
    print(f"  verdict: {verdict}")
    print("  (still a general LLM, not a TS base model — this bounds the lookahead risk, "
          "doesn't eliminate the need for a post-cutoff base-model run.)")


if __name__ == "__main__":
    main()
