#!/usr/bin/env python3
"""Stage-2 (empirical forecasting-lift) adapter + runner for Faisal-style CPT world-knowledge data.

Stage-1 (verify_cpt.py) answers "is the record well-formed and are its claims honest?" — CPU, laptop.
Stage-2 (this) answers the real question: "does the paired text actually LOWER forecast error?" —
the 3-arm A/B/C test. It reuses the existing flywheel harness (arms.py + models.py) and adds the
piece that CPT data specifically needs:

  A CPT record's text DESCRIBES the whole co-located series. If we carve a future tail out of that
  series to forecast, a `recites` text may literally state the tail value -> the text leaks the
  answer. So before any 3-arm test we run a LEAKAGE SCREEN: drop every record whose conditioning
  text contains a held-out future value. How many survive is itself a finding (recites-heavy data
  tends to survive poorly — which is exactly why CPT describe-data != clean forecasting data).

WHAT RUNS ON A LAPTOP (here, now):
  - adapt CPT records -> forecast items (context-window z-normalised history/future)
  - the leakage screen (+ survival report)
  - the numeric-baseline FLOOR (A=B=C by construction, text-blind) — proves the plumbing
WHAT NEEDS A GPU/BOX (the actual verdict):
  - the TEXT arm (models.TextConditionedStub): fine-tune / run a semantic base so B can beat A and C.
    Swap it in without touching this adapter, the screen, the arms, or the split.

Usage:
    python flywheel/companion/stage2_cpt.py CPT.jsonl [CPT2.jsonl ...]
           [--horizon 4] [--min-history 6] [--all-channels] [--json out.json]
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from statistics import mean, median, pstdev

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arms, models  # noqa: E402  (reuse the exact 3-arm construction + baselines)

STD_FLOOR_COEF = 1e-3  # Xinyue's robust floor: std_eff = max(std, 1e-3*max(|mean|,1))
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
TS_TOKEN = "<ts></ts>"


def _load(path):
    txt = open(path, encoding="utf-8").read()
    try:
        d = json.loads(txt)
        return d if isinstance(d, list) else [d]
    except Exception:
        pass
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def _cond_text(rec):
    return (rec.get("text") or "").replace(TS_TOKEN, " ").strip()


def _nums(s):
    out = set()
    for m in _NUM.findall(s):
        try:
            out.add(float(m.replace(",", "")))
        except Exception:
            pass
    return out


def _z(hist, fut):
    """context-window z-score with the robust floor (history stats only — no future leakage)."""
    m = mean(hist)
    sd = pstdev(hist) if len(hist) > 1 else 0.0
    sd_eff = max(sd, STD_FLOOR_COEF * max(abs(m), 1.0))
    z = lambda xs: [(x - m) / sd_eff for x in xs]
    return z(hist), z(fut)


def _text_leaks_future(text_nums, fut_raw):
    """True if any held-out future value appears in the conditioning text (rounded forms)."""
    for v in fut_raw:
        if v is None:
            continue
        for cand in {v, round(v, 1), round(v, 2), round(v)}:
            for t in text_nums:
                if cand and abs(t - cand) <= max(0.01 * abs(cand), 0.01):
                    return True
                if cand == 0 and t == 0:
                    return True
    return False


def adapt(records, horizon, min_history, all_channels):
    """CPT records -> forecast items, with a per-item leakage flag. Returns (items, stats)."""
    items, n_short, n_leak = [], 0, 0
    for r in records:
        text = _cond_text(r)
        tnums = _nums(text)
        chans = r.get("timeseries") or []
        chosen = chans if all_channels else chans[:1]
        for ci, ch in enumerate(chosen):
            vals = [v for v in (ch.get("values") or []) if v is not None]
            if len(vals) < min_history + horizon:
                n_short += 1
                continue
            hist_raw, fut_raw = vals[:-horizon], vals[-horizon:]
            leak = _text_leaks_future(tnums, fut_raw)
            if leak:
                n_leak += 1
            hz, fz = _z(hist_raw, fut_raw)
            items.append({
                "series_id": r.get("series_id", f"{r.get('dataset','?')}:{id(r)}") + f"#c{ci}",
                "dataset": r.get("dataset"),
                "alignment": r.get("alignment"),
                "history": hz, "future": fz, "text": text,
                "leak": leak,
            })
    return items, {"n_short": n_short, "n_leak": n_leak}


def _mase(pred, actual, history):
    scale = mean(abs(history[i] - history[i - 1]) for i in range(1, len(history))) or 1e-6
    H = min(len(pred), len(actual))
    return mean(abs(pred[i] - actual[i]) for i in range(H)) / scale


def floor_eval(clean_items):
    """Numeric-baseline floor: text-blind, so A=B=C by construction. Establishes the no-text floor
    and proves the split/arms/eval plumbing on THIS data before the GPU text arm plugs in."""
    three = arms.make_arms(clean_items)
    out = {}
    for base in (models.LastValue(), models.LinearTrend()):
        per_arm = {}
        for arm, its in three.items():
            sc = [_mase(base.predict(x["history"], x["text"], len(x["future"])),
                        x["future"], x["history"]) for x in its]
            per_arm[arm] = round(median(sc), 4) if sc else None
        out[base.name] = per_arm
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--min-history", type=int, default=6)
    ap.add_argument("--all-channels", action="store_true", help="forecast every channel (default: first)")
    ap.add_argument("--json", help="write full result JSON")
    a = ap.parse_args()

    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)) if os.path.isdir(p) \
            else (glob.glob(p, recursive=True) if any(c in p for c in "*?[") else [p])
    files = [f for f in files if os.path.isfile(f)]

    recs = []
    for f in files:
        recs += _load(f)
    items, stt = adapt(recs, a.horizon, a.min_history, a.all_channels)
    clean = [x for x in items if not x["leak"]]

    # leakage survival, broken down by the record's own alignment tag
    from collections import Counter
    by_align_total = Counter(x["alignment"] for x in items)
    by_align_leak = Counter(x["alignment"] for x in items if x["leak"])

    print(f"CPT stage-2 adapter  |  records={len(recs)}  forecast-items={len(items)}  "
          f"(too_short_skipped={stt['n_short']})")
    print(f"LEAKAGE SCREEN: {stt['n_leak']}/{len(items)} items' text states a held-out future value "
          f"-> dropped.  clean={len(clean)}")
    print("  by alignment (leaked/total):")
    for al in sorted(by_align_total, key=str):
        print(f"    {str(al):<15} {by_align_leak.get(al,0):>4}/{by_align_total[al]:<4}")
    if len(clean) < 8:
        print("\n! too few leakage-clean items for a meaningful 3-arm run on this input "
              "(expected on recites-heavy CPT data — that's the finding).")
    floor = floor_eval(clean) if clean else {}
    if floor:
        print("\nNUMERIC FLOOR (text-blind -> A=B=C by construction; establishes the no-text floor):")
        for base, pa in floor.items():
            print(f"  {base:<14} " + "  ".join(f"{k.split('_')[0]}={v}" for k, v in pa.items()))
    print("\nTEXT ARM (B vs A/C, the actual verdict) = models.TextConditionedStub -> PENDING: "
          "needs a semantic base model + GPU (box). Plug it in; screen/arms/split unchanged.")

    if a.json:
        json.dump({"stats": stt, "n_items": len(items), "n_clean": len(clean),
                   "leak_by_alignment": {k: [by_align_leak.get(k, 0), by_align_total[k]] for k in by_align_total},
                   "floor": floor}, open(a.json, "w"), ensure_ascii=False, indent=2)
        print(f"\nJSON -> {a.json}")


if __name__ == "__main__":
    main()
