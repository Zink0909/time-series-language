#!/usr/bin/env python3
"""THE DISCRIMINATOR · one orchestrator that runs the whole verification battery on a source and prints a
single SCORECARD + tier. Turns the scattered tools (verify_export/migas_finetune, per_record_analysis,
complementarity, q8_contamination, verify_lap, verify_llm_arch, diversity_audit, leak_screen, source_registry)
into one machine: the "system, not a pile of scripts".

Two kinds of check:
  - RECORDED (need a box GPU run): the value verdict, Q8 base-contamination, cross-architecture, compound-vs-
    saturate — stored per source in source_registry.SOURCES (updated when a box run is done).
  - LIVE (run here, no GPU): outcome-leak regex scan, cross-source diversity, ts->text discriminability.
The tier (T1 admit / T2 conditional / T3 quarantine / T4 reject / TX hazard) is assigned by source_registry.tier.

  micromamba run -n ts-language python flywheel/companion/discriminator.py [source]
"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import source_registry as reg                                        # recorded verdicts + tier logic
from core.leak_screen import direction_leak                          # noqa: E402
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "out")
TIER_LABEL = {"T1": "T1 放行", "T2": "T2 有条件", "T3": "T3 隔离观察", "T4": "T4 拒绝(内在)", "TX": "TX 危险退回"}


def _src_records(base):
    """All ts_forecast + text_desc records whose dataset/file matches `base` (e.g. 'fred_fomc')."""
    recs = []
    for f in glob.glob(os.path.join(OUT, "*.jsonl")):
        if f.endswith("fred_fomc.v1.jsonl") or base not in os.path.basename(f):
            continue
        recs += [json.loads(l) for l in open(f)]
    return recs


def live_checks(base):
    """LIVE, no-GPU checks over the source's built records."""
    recs = _src_records(base)
    fc = [r for r in recs if r.get("task_type", "").startswith("ts_forecast")]
    td = [r for r in recs if r.get("task_type") == "text_desc"]
    out = {}
    if fc:                                                            # text->ts: outcome-leak regex scan
        hits = sum(1 for r in fc if direction_leak(r.get("text", "")))
        out["leak_scan"] = f"{hits}/{len(fc)} regex-flagged" + (" → LLM-judge review" if hits else " → clean")
    if td:                                                           # ts->text: discriminability (reuse verify_ts2text)
        try:
            from verify_ts2text import raw_pool, stated_numbers, match_rate
            import random, statistics as st
            random.seed(0)
            pools = [raw_pool(r) for r in td]; nums = [stated_numbers(r["text"]) for r in td]
            disc = []
            for i in range(len(td)):
                o = match_rate(nums[i], pools[i])
                others = random.sample([j for j in range(len(td)) if j != i], min(20, len(td) - 1))
                d = st.mean(match_rate(nums[i], pools[j]) for j in others) if others else 0.0
                disc.append(o - d)
            out["ts2text_discriminability"] = f"{st.median(disc):.2f} (>=0.3 useful)"
        except Exception as e:
            out["ts2text_discriminability"] = f"n/a ({e})"
    return out


def scorecard(key):
    s = reg.SOURCES[key]
    base = key.split("(")[0]
    t, reason, retest = reg.tier(s)
    rec = {"value_robust": s["value_robust"], "content_real": s["content_real"], "correct": s["correct"],
           "base_clean": s["base_clean"], "leakage": s["leakage"]}
    return {"tier": t, "reason": reason, "retest": retest, "note": s["note"],
            "recorded": rec, "live": live_checks(base)}


def main():
    keys = [k for k in reg.SOURCES if len(sys.argv) < 2 or sys.argv[1] in k]
    order = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "TX": 4}
    cards = sorted(((k, scorecard(k)) for k in keys), key=lambda kv: order[kv[1]["tier"]])
    print("=" * 74 + "\nTHE DISCRIMINATOR · source scorecards (recorded box verdicts + live checks)\n" + "=" * 74)
    for k, c in cards:
        print(f"\n[{TIER_LABEL[c['tier']]}]  {k}")
        print(f"    recorded: " + " · ".join(f"{n}={v}" for n, v in c["recorded"].items()))
        if c["live"]:
            print(f"    live:     " + " · ".join(f"{n}={v}" for n, v in c["live"].items()))
        print(f"    verdict:  {c['reason']}")
        print(f"    retest:   {c['retest']}")
    print("\n" + "-" * 74)
    tally = {}
    for _, c in cards:
        tally[c["tier"]] = tally.get(c["tier"], 0) + 1
    print("tally: " + "  ".join(f"{TIER_LABEL[t]}={tally[t]}" for t in sorted(tally, key=lambda x: order[x])))
    print("(recorded = needs a box run; live = run here. One command = the whole battery's current picture.)")


if __name__ == "__main__":
    main()
