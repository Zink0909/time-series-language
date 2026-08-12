#!/usr/bin/env python3
"""Independent text↔series MATCH audit for the whole flywheel.

Scores every conditioning-text record's correspondence to why its series moves, and reports the
distribution per source. This both (a) surfaces where the flywheel's world-knowledge data is strongly
vs weakly coupled (the new quality dimension Defu's CPT needs), and (b) validates the scorer the honest
way: event-grounded commodity causes should skew 'coupled', standing wiki bios should skew 'weak'. If
that ordering holds, the scorer is measuring what it claims to.

  micromamba run -n ts-language python flywheel/audit_match.py
"""
import os, sys, json, glob, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from core.match_score import match_score, cause_text_of                # noqa: E402

OUT = os.path.join(PKG, "out")


def _is_text_to_ts(r):
    if r.get("direction"):
        return r["direction"] == "text_to_ts"
    return r.get("task_type", "").startswith("ts_forecast")            # forecast = text conditions the series


def audit():
    print(f"{'source':34} {'n':>5} {'coupled':>9} {'weak':>8}  {'median':>6}  example weak")
    print("-" * 100)
    grand = []
    frac = {}
    for f in sorted(glob.glob(os.path.join(OUT, "flywheel_*.jsonl"))):
        scored, weak_ex = [], ""
        for line in open(f):
            r = json.loads(line)
            if not _is_text_to_ts(r):
                continue
            ed = r.get("spike_date") or r.get("event_date")
            txt = cause_text_of(r)
            if not txt:
                continue
            s, lab, _ = match_score(txt, ed)
            scored.append((s, lab))
            if lab == "weak" and not weak_ex:
                weak_ex = txt[:52].replace("\n", " ")
        if not scored:
            continue
        sc = [s for s, _ in scored]
        coupled = sum(1 for _, l in scored if l == "coupled")
        weak = len(scored) - coupled
        grand += scored
        frac[os.path.basename(f)] = coupled / len(scored)
        print(f"{os.path.basename(f):34} {len(scored):5} {coupled:5} ({100*coupled/len(scored):3.0f}%) "
              f"{weak:5} ({100*weak/len(scored):3.0f}%)  {st.median(sc):6.2f}  {weak_ex}")
    if grand:
        sc = [s for s, _ in grand]
        c = sum(1 for _, l in grand if l == "coupled")
        print("-" * 100)
        print(f"{'TOTAL':34} {len(grand):5} {c:5} ({100*c/len(grand):3.0f}%) "
              f"{len(grand)-c:5} ({100*(len(grand)-c)/len(grand):3.0f}%)  {st.median(sc):6.2f}")

    # 不自证 self-check: event-grounded commodity text must rank clearly above standing wiki bios,
    # otherwise the scorer has stopped measuring coupling. (Validates the scorer on known-coupling data.)
    com = frac.get("flywheel_commodity.jsonl")
    wik = frac.get("flywheel_wiki_big.jsonl")
    if com is not None and wik is not None:
        ok = com > wik + 0.2
        print(f"\nMATCH SCORER {'OK' if ok else 'DEGRADED'}: commodity coupled {com:.0%} vs "
              f"wiki_big coupled {wik:.0%} (need commodity > wiki_big + 20pts)")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(audit())
