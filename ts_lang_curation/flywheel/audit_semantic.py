#!/usr/bin/env python3
"""W2 v2 · semantic-denoising audit — does LLM-judged relevance fix what bag-of-terms couldn't?

The keyword version had no safe threshold (min_score=1 kept the unrelated 'Naira forex reserves';
min_score=2 dropped it but ALSO dropped the relevant 'OPEC out of spare capacity', breaking a cause).
Semantic denoising should resolve BOTH: drop Naira (topically irrelevant) AND keep OPEC-spare-capacity
(topically relevant despite few oil keywords), with every cause still faithful against the kept set.
Uses the team vLLM (cached, temp 0). Run:
  micromamba run -n ts-language python flywheel/audit_semantic.py
"""
import os, sys, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from core.retrieval_rank import semantic_rank                        # noqa: E402
from core.faithfulness import report                                 # noqa: E402

CAUSE_DIR = os.path.join(HERE, "raw", "cause")
CACHE = os.path.join(HERE, "raw", "semantic")
REPORT = os.path.join(HERE, "raw", "_semantic_audit.json")
TOPIC = "crude oil prices and the global oil market"


def main():
    rows, faithful, total_drop = [], 0, 0
    naira, opec = None, None
    for p in sorted(glob.glob(os.path.join(CAUSE_DIR, "*.json"))):
        c = json.load(open(p))
        date = os.path.basename(p)[:-5]
        heads = c.get("headlines", [])
        kept, dropped = semantic_rank(heads, TOPIC, CACHE)
        total_drop += len(dropped)
        f = report(c["cause"], "\n".join(kept))["verdict"] == "ok"
        faithful += f
        rows.append({"date": date, "kept": len(kept), "dropped": dropped, "faithful": f})
        for h in dropped:
            if "naira" in h.lower():
                naira = "dropped ✓"
            if "spare capacity" in h.lower():
                opec = "dropped ✗ (real signal lost)"
        for h in kept:
            if "naira" in h.lower():
                naira = "kept ✗ (noise survived)"
            if "spare capacity" in h.lower():
                opec = "kept ✓"

    n = len(rows)
    print(f"== W2 v2 semantic denoising audit · {n} oil events ==\n")
    print(f"dropped as off-topic (semantic): {total_drop}")
    print(f"causes still faithful after denoise: {faithful}/{n}\n")
    print(f"  the two cases bag-of-terms couldn't get right:")
    print(f"    'Naira forex reserves' (real noise)      -> {naira}")
    print(f"    'OPEC out of spare capacity' (real signal)-> {opec}\n")
    for r in rows:
        if r["dropped"]:
            print(f"   {r['date']}: dropped {len(r['dropped'])}"
                  f"{'' if r['faithful'] else '  <- FAITHFULNESS BROKE'}")
            for d in r["dropped"]:
                print(f"        · {d}")

    json.dump({"n": n, "dropped": total_drop, "faithful": faithful,
               "naira": naira, "opec_spare": opec, "rows": rows},
              open(REPORT, "w"), indent=1, ensure_ascii=False)
    ok = naira == "dropped ✓" and opec == "kept ✓" and faithful == n
    print(f"\n== {'PASS' if ok else 'PARTIAL'}: Naira {naira}; OPEC-spare {opec}; "
          f"faithful {faithful}/{n} ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
