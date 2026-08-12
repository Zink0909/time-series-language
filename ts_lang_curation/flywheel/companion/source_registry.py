#!/usr/bin/env python3
"""verification · SOURCE REGISTRY — the tiered "customs + quarantine" decision layer.

Not every source is a binary pass/fail. Each source's verification results (value / content / correctness /
base-contamination / leakage / fixability) map to a TIER, so a source that fails for a FIXABLE reason is
quarantined-and-retested rather than discarded, and a HAZARD (leakage/contamination) is hard-rejected. This
makes rejection REVERSIBLE and REASONED — strict at the door (never admit poison) without permanent false
negatives. Verdict is relative to the deployment reference (an aligning multimodal model, MiGAS-family).

  T1 admit · T2 conditional · T3 quarantine(fixable) · T4 reject(intrinsic no-signal) · TX hazard(leakage)

  micromamba run -n ts-language python flywheel/companion/source_registry.py
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))

# Each source's measured verification state (from this project's runs; None = not-applicable/not-measured).
# value_robust: text-help significant AND holds across metric+scale.  content_real: correct text beats shuffled
# (P2, real info not just a channel).  base_clean: Q8 no memorization.  leakage: P3/contamination detected.
# fixable: if not useful, is the failure a FIXABLE artifact (processing/scale/model) rather than no-signal?
SOURCES = {
    "fred_fomc(text->ts)": dict(value_robust=True, content_real=True, correct=True, base_clean=True,
                                leakage=False, fixable=None, arch_specific=True,
                                note="text-help SIG n17->30 both metrics + Time-MMD; complementary; but aligning-only (zero-shot prompting ns)"),
    "cyber_epss(text->ts)": dict(value_robust=False, content_real=False, correct=True, base_clean=True,
                                 leakage=False, fixable=False, arch_specific=None,
                                 note="RETEST LOOP DONE: logit-EPSS unlocked a text-CHANNEL benefit (B<A huge even at long horizon) but CONTENT still absent (B~C, shuffled works equally) across raw AND logit -> no world-knowledge content; confident reject-for-CPT (a channel effect is not world knowledge)"),
    "usgs_quakes(text->ts)": dict(value_robust=False, content_real=False, correct=True, base_clean=None,
                                  leakage=False, fixable=False, arch_specific=None,
                                  note="aftershock decay is a numeric law (Omori) -> text adds no info; intrinsic, keep numbers-only (n=9 also underpowered)"),
    "sec_edgar(ts->text)": dict(value_robust=True, content_real=True, correct=True, base_clean=None,
                                leakage=False, fixable=None, arch_specific=None,
                                note="ts->text discriminability 0.74 (strongly series-specific)"),
    "wiki_pageviews(ts->text)": dict(value_robust=True, content_real=True, correct=True, base_clean=None,
                                     leakage=False, fixable=None, arch_specific=None,
                                     note="ts->text discriminability 0.83 (peak-view pins the series); text->ts direction removed earlier (was leaky)"),
    # per-DIRECTION: the same source can pass one direction and fail the other (surfaced by discriminator.py).
    "cyber_epss(ts->text)": dict(value_robust=True, content_real=True, correct=True, base_clean=None,
                                 leakage=False, fixable=None, arch_specific=None,
                                 note="ts->text discriminability 0.75 (the EPSS trajectory -> grounded description is series-specific) — USEFUL, even though the SAME source's text->ts forecast direction is T4"),
    "fred_fomc(ts->text)": dict(value_robust=True, content_real=True, correct=True, base_clean=None,
                                leakage=False, fixable=None, arch_specific=None,
                                note="ts->text discriminability 0.36 (borderline, just above the 0.3 bar) — yield-window description is weakly series-specific"),
    "energy_gas(text->ts)": dict(value_robust=False, content_real=False, correct=True, base_clean=None,
                                 leakage=False, fixable=False, arch_specific=None,
                                 note="NEW DOMAIN (energy): EIA weekly gas report -> Henry Hub price. Text CHANNEL helps a lot (B<A 39/39) but CONTENT absent (B~C, complementarity=noisy channel; 46% targets trivially persistent). Same channel-not-content pattern as cyber/flywheel -> 6th domain confirming precise content pairing has weak forecasting value; only FOMC had content. Discriminator generalized cleanly to a new domain."),
    "usgs_quakes(ts->text)": dict(value_robust=False, content_real=False, correct=True, base_clean=None,
                                  leakage=False, fixable=False, arch_specific=None,
                                  note="ts->text discriminability 0.19 (< 0.3): the aftershock-decay description is generic — weak in BOTH directions"),
}


def tier(s):
    if s["leakage"] or s["base_clean"] is False:
        return "TX", "hazard: leakage/contamination detected (would poison the model)", "never — hard reject"
    if not s["correct"]:
        return "TX", "hazard: data not correct (schema/faithfulness fail)", "fix generation, then re-audit"
    if s["value_robust"] and s["content_real"]:
        if s.get("arch_specific"):
            return "T2", "useful + real content, but value is architecture-specific (aligning only)", "monitor if deployment arch changes"
        return "T1", "robustly useful, real content, clean", "periodic re-audit only"
    # not (robustly) useful:
    if s["fixable"]:
        return "T3", "not shown useful, FIXABLE reason (processing/scale/model)", "re-test after the fix in the note"
    if s["fixable"] is False:
        return "T4", "intrinsic: no usable text signal (numeric prior governs) — keep unimodal", "only if the task/target changes"
    return "T3", "inconclusive — hold for more evidence", "re-test with more data / better extractor"


def main():
    order = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "TX": 4}
    rows = [(name, *tier(s), s["note"]) for name, s in SOURCES.items()]
    rows.sort(key=lambda r: order[r[1]])
    print("SOURCE REGISTRY · tiered admission (relative to the deployment reference = aligning multimodal model)\n")
    label = {"T1": "T1 放行", "T2": "T2 有条件", "T3": "T3 隔离观察", "T4": "T4 拒绝(内在)", "TX": "TX 危险退回"}
    for name, t, reason, retest, note in rows:
        print(f"[{label[t]}]  {name}")
        print(f"    why:    {reason}")
        print(f"    retest: {retest}")
        print(f"    note:   {note}\n")
    reg = {name: dict(tier=t, reason=reason, retest=retest, **SOURCES[name])
           for name, t, reason, retest, _ in rows}
    json.dump(reg, open(os.path.join(HERE, "source_registry.json"), "w"), ensure_ascii=False, indent=1)
    counts = {}
    for _, t, *_ in rows:
        counts[t] = counts.get(t, 0) + 1
    print("tally:", "  ".join(f"{label[t]}={counts[t]}" for t in sorted(counts, key=lambda x: order[x])))
    print("-> registry written to source_registry.json (rejection is REVERSIBLE: each non-T1 carries a re-test trigger)")


if __name__ == "__main__":
    main()
