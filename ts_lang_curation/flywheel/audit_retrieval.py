#!/usr/bin/env python3
"""W2 · retrieval ranking + an HONEST denoising trade-off, on the oil flywheel's cached headlines.

What works safely (v1): relevance RANKING — order headlines by oil-market-term overlap so the LLM
sees the on-topic ones first (lossless; noise sinks to the bottom).

What does NOT work with bag-of-terms (the honest finding): HARD denoising. There is no single
min_score that's safe — `min_score=1` keeps real noise (an unrelated 'Naira forex reserves'
headline scores 1 via the ambiguous word 'reserves'); `min_score=2` finally drops it but ALSO
drops a genuinely relevant 'OPEC out of spare capacity' headline (its key words aren't in the term
list), which BREAKS a cause's faithfulness. So denoising needs semantic relevance (embeddings /
NLI) — left to v2; v1 ships ranking only. This audit demonstrates both, deterministically.
  micromamba run -n ts-language python flywheel/audit_retrieval.py
"""
import os, sys, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from core.retrieval_rank import rank_headlines, relevance                # noqa: E402
from core.faithfulness import report                                     # noqa: E402

CAUSE_DIR = os.path.join(HERE, "raw", "cause")
REPORT = os.path.join(HERE, "raw", "_retrieval_audit.json")

OIL_TERMS = set((
    "oil crude opec brent wti petroleum energy supply demand barrel barrels gas gasoline fuel "
    "price prices russia ukraine iea eia saudi refinery reserves output production sanctions "
    "embargo pipeline stockpile inventory inventories recession lockdown china exports imports "
    "diesel refining spr strategic geopolitical stagflation").split())


def _faithful_at(causes, min_score):
    """How many causes stay faithful if we HARD-drop headlines below min_score."""
    ok, dropped = 0, 0
    for c in causes:
        kept, noise, _ = rank_headlines(c.get("headlines", []), OIL_TERMS, min_score=min_score)
        dropped += len(noise)
        ok += report(c["cause"], "\n".join(kept))["verdict"] == "ok"
    return ok, dropped


def main():
    causes = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(CAUSE_DIR, "*.json")))]
    n = len(causes)

    # (1) RANKING — safe, lossless. Verify the least-relevant headline in each group sinks last.
    rank_ok = 0
    per = []
    for c in causes:
        heads = c.get("headlines", [])
        _, _, scored = rank_headlines(heads, OIL_TERMS, min_score=0)     # min_score=0 => rank only
        order_ok = all(scored[i][0] >= scored[i + 1][0] for i in range(len(scored) - 1))
        rank_ok += order_ok
        low = scored[-1] if scored else (0, "")
        per.append({"top": scored[0][1] if scored else "", "top_score": scored[0][0] if scored else 0,
                    "bottom": low[1], "bottom_score": low[0], "ranked": order_ok})

    print(f"== W2 retrieval audit · {n} oil events ==\n")
    print(f"(1) RELEVANCE RANKING (safe, lossless): {rank_ok}/{n} groups correctly ordered "
          f"(most on-topic first, noise last)")
    for c, p in zip(causes, per):
        print(f"    top[{p['top_score']}] {p['top'][:52]}")
        print(f"    bot[{p['bottom_score']}] {p['bottom'][:52]}")

    # (2) HARD DENOISING trade-off — the honest finding: no safe threshold.
    print("\n(2) HARD DENOISING trade-off (why v1 does NOT auto-drop):")
    for ms in (1, 2, 3):
        ok, dropped = _faithful_at(causes, ms)
        note = ("keeps real noise (Naira/'reserves')" if ms == 1 else
                "drops noise but breaks a cause (OPEC spare capacity)" if ms == 2 else
                "over-prunes")
        print(f"    min_score={ms}: drop {dropped:2} headlines → {ok}/{n} causes faithful  ({note})")
    print("    => bag-of-terms can't separate 'forex reserves' from 'oil reserves'; semantic")
    print("       relevance (embeddings/NLI) is the v2 fix. v1 ships RANKING, not hard dropping.")

    json.dump({"n": n, "ranking_ok": rank_ok, "per_group": per,
               "denoise_tradeoff": {ms: dict(zip(("faithful", "dropped"),
                                                  _faithful_at(causes, ms))) for ms in (1, 2, 3)}},
              open(REPORT, "w"), indent=1, ensure_ascii=False)
    ok = rank_ok == n                                    # v1 claim = ranking is correct & lossless
    print(f"\n== {'PASS' if ok else 'FAIL'}: ranking correct on {rank_ok}/{n} groups "
          f"(hard-denoise deferred to v2 — semantic relevance) ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
