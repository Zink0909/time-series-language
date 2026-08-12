#!/usr/bin/env python3
"""W3 · independent faithfulness audit of the flywheel's SYNTHESIZED causes (read-only baseline).

Scans every cached oil-flywheel cause against the exact headlines it was distilled from, using
core.faithfulness (entity tracing + factual-consistency taxonomy) — which does NOT reuse the
generator's own validate()/verify loop (不自证). Prints a per-cause verdict + a rejection-reason
log and writes flywheel/raw/_faithfulness.json. This is the baseline BEFORE deciding thresholds
or wiring the gate into oil_demo.py's S7. Run:
  micromamba run -n ts-language python flywheel/audit_faithfulness.py
"""
import os, sys, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from core.faithfulness import report                                   # noqa: E402

CAUSE_DIR = os.path.join(HERE, "raw", "cause")
OUT = os.path.join(HERE, "raw", "_faithfulness.json")


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(CAUSE_DIR, "*.json"))):
        c = json.load(open(path))
        date = os.path.basename(path)[:-5]
        grounding = "\n".join(c.get("headlines", []))
        r = report(c["cause"], grounding)
        r["date"] = date
        rows.append(r)

    print(f"== flywheel cause faithfulness audit · {len(rows)} synthesized causes ==\n")
    ok = sum(r["verdict"] == "ok" for r in rows)
    cov = sorted(r["entity_coverage"] for r in rows)
    print(f"verdict OK           : {ok}/{len(rows)}")
    print(f"entity coverage      : min {cov[0]:.2f}  median {cov[len(cov)//2]:.2f}  max {cov[-1]:.2f}")
    print(f"entities traced      : {sum(r['entities_traced'] for r in rows)}"
          f"/{sum(r['entities_total'] for r in rows)}\n")
    for r in rows:
        flag = "OK " if r["verdict"] == "ok" else r["verdict"].upper()
        print(f"  [{flag}] {r['date']}  cov={r['entity_coverage']:.2f} "
              f"({r['entities_traced']}/{r['entities_total']} ents)")
        for reason in r["reasons"]:
            print(f"        · {reason}")

    # negative control: an INJECTED cause with a hallucinated entity + ungrounded number must be
    # rejected — proves the gate actually discriminates, not just passes everything (不自证).
    neg = report(
        "The Federal Reserve and Venezuela jointly announced production cuts of 3.5 million "
        "barrels, tightening the market.",
        "IEA warns of supply deficits\nOPEC spare capacity at historic lows\nChina demand concerns")
    neg_rejected = neg["verdict"] != "ok"
    print(f"\nnegative control (injected hallucination): "
          f"{'REJECTED ✓' if neg_rejected else 'PASSED ✗ (gate not discriminating!)'}  "
          f"-> {neg['reasons']}")

    json.dump({"causes": rows, "negative_control": neg}, open(OUT, "w"), indent=1,
              ensure_ascii=False)
    print(f"wrote {OUT}")
    # healthy iff every real cause is faithful AND the negative control is caught.
    all_ok = all(r["verdict"] == "ok" for r in rows) and neg_rejected
    print(f"\n== {'PASS' if all_ok else 'FAIL'}: {ok}/{len(rows)} causes faithful, "
          f"negative control {'caught' if neg_rejected else 'MISSED'} ==")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
