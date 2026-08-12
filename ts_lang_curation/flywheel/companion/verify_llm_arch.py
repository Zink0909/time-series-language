#!/usr/bin/env python3
"""③ verdict-trust · CROSS-ARCHITECTURE robustness. The per-source verdict (does text help) is measured with
MiGAS (frozen-Chronos + FinBERT + trained gated-attention = LATE-FUSION / ALIGNING). This re-runs the same
3-arm test with a completely different paradigm — a zero-shot PROMPTING forecaster (LLMTime-style via the
team vLLM): feed normalized history + text, parse the continuation. If "text helps" survives BOTH paradigms
it is architecture-robust; if it only shows up under one, the verdict is architecture-specific (a moderator
axis of the scaling law, not a universal property).

  micromamba run -n ts-language python flywheel/companion/verify_llm_arch.py <_verify_src.json> [label]
"""
import os, sys, json, random, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from models import LLMTimeForecaster                                  # noqa: E402


def znorm(hist, fut):
    mu = st.mean(hist); sd = st.pstdev(hist) or 1.0
    return [(x - mu) / sd for x in hist], [(x - mu) / sd for x in fut]


def mase(pred, actual, histn):
    raw = st.mean(abs(histn[i] - histn[i - 1]) for i in range(1, len(histn)))
    scale = max(raw, 1e-3)
    n = min(len(pred), len(actual))
    return st.mean(abs(pred[i] - actual[i]) for i in range(n)) / scale


def boot(deltas, iters=2000):
    n = len(deltas); ms = sorted(st.median([deltas[random.randrange(n)] for _ in range(n)]) for _ in range(iters))
    return st.median(deltas), ms[int(.025 * iters)], ms[int(.975 * iters)]


def main():
    d = json.load(open(sys.argv[1]))
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(sys.argv[1])
    test = [t for t in d["test"] if len(t["history"]) >= 5 and len(t["future"]) >= 3]
    random.seed(0)
    texts = [t["text"] for t in test]
    shuf = texts[:]
    while any(shuf[i] == texts[i] for i in range(len(shuf))):        # derangement: no record keeps its own text
        random.shuffle(shuf)
    fc = LLMTimeForecaster(os.path.join(HERE, f"_llmarch_cache_{label}.json"))
    err = {"A": [], "B": [], "C": []}
    for i, t in enumerate(test):
        hn, fn = znorm(t["history"], t["future"])
        H = len(fn)
        for arm, txt in [("A", ""), ("B", t["text"]), ("C", shuf[i])]:
            err[arm].append(mase(fc.predict(hn, txt, H), fn, hn))
    fc.save()
    n = len(test)
    dBA = [err["B"][i] - err["A"][i] for i in range(n)]
    dBC = [err["B"][i] - err["C"][i] for i in range(n)]
    mba, lba, hba = boot(dBA); mbc, lbc, hbc = boot(dBC)
    print(f"cross-architecture (PROMPTING / LLMTime via vLLM) · {label} · n={n}")
    print(f"  median MASE   A(no-text) {st.median(err['A']):.3f} | B(text) {st.median(err['B']):.3f} | C(shuffled) {st.median(err['C']):.3f}")
    print(f"  Δ(B−A) {mba:+.3f}  95% CI [{lba:+.3f}, {hba:+.3f}]  {'SIG<0 text helps' if hba < 0 else 'ns'}")
    print(f"  Δ(B−C) {mbc:+.3f}  95% CI [{lbc:+.3f}, {hbc:+.3f}]  {'SIG<0 content helps' if hbc < 0 else 'ns'}")
    print(f"  per-event B<A {sum(x < 0 for x in dBA)}/{n} · B<C {sum(x < 0 for x in dBC)}/{n}")
    mig = "MiGAS(aligning): FOMC text-helps SIG / Cyber not-robust"
    print(f"  vs {mig}")


if __name__ == "__main__":
    main()
