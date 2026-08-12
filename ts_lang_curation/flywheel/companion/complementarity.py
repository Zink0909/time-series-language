#!/usr/bin/env python3
"""② value · COMPLEMENTARY vs REDUNDANT text (the Q4 check; 2506.21611's unique-vs-redundant, per-record).

Text is COMPLEMENTARY if it carries information the series does NOT already show — then it should help most
exactly where the numeric-only arm (A) fails. It is REDUNDANT if it only restates what the series shows —
then it adds nothing beyond A. Using the per-record dump ({err_A, err_B, err_C}):
  - concentration: does the text-gain (err_A - err_B) CONCENTRATE on the hardest-A records? (complementary)
  - decoupling: corr(err_A, err_B) — if text rescues hard cases, err_B decouples from err_A (low corr)
  - content-vs-channel: err_C - err_B (correct text beating SHUFFLED = the content, not just a text channel)

  micromamba run -n ts-language python flywheel/companion/complementarity.py <dump.json> [label]
"""
import sys, json, statistics as st


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        for p, i in enumerate(o):
            r[i] = p
        return r
    rx, ry = rank(xs), rank(ys); n = len(xs); mx, my = st.mean(rx), st.mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return num / den if den else 0.0


def main():
    d = json.load(open(sys.argv[1]))
    label = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
    A = [r["err_A"] for r in d]; B = [r["err_B"] for r in d]; C = [r.get("err_C", r["err_B"]) for r in d]
    n = len(d)
    order = sorted(range(n), key=lambda i: A[i])                     # by numeric-only difficulty
    q = n // 4
    easy, hard = order[:q], order[-q:]
    gain = lambda idx: st.median(A[i] - B[i] for i in idx)           # text-gain (err_A - err_B)
    print(f"complementarity · {label} · n={n}")
    print(f"  text-gain (err_A-err_B): hardest-A quartile {gain(hard):+.4f}  vs  easiest-A quartile {gain(easy):+.4f}")
    print(f"  corr(err_A, err_B) Spearman = {spearman(A, B):+.2f}   (low/negative = text decouples difficulty = complementary)")
    content = st.median(C[i] - B[i] for i in range(n))               # correct vs shuffled
    print(f"  content-vs-channel (err_C-err_B) median = {content:+.4f}   (>0 = correct text beats shuffled = real content)")
    has_content = content > 0.001                                    # correct text must beat shuffled (real info, not channel)
    noisy = gain(easy) < -0.05                                       # text HARMS where the series is already easy = noisy channel
    if has_content and gain(hard) > 0 and gain(hard) > gain(easy):
        verdict = "COMPLEMENTARY (real content, helps most where the series fails — carries info not in the numbers)"
    elif not has_content or noisy:
        verdict = ("REDUNDANT / NOISY CHANNEL (correct text does NOT beat shuffled = no real content"
                   + ("; and it harms easy cases)" if noisy else ")"))
    else:
        verdict = "WEAK / inconclusive"
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
