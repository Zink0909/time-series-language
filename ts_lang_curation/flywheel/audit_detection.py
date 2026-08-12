#!/usr/bin/env python3
"""W1 · detection audit — fixed-threshold vs robust multi-scale on the real WTI series.

Shows the multi-scale detector's payoff: it recovers salient MULTI-DAY moves the single-day |ret|
threshold misses (slow drifts where each day is under the bar), i.e. lower miss-rate, WITHOUT
lowering the single-day bar into noise. Read-only + deterministic; does not touch the emitted 8/8
oil pairs (those keep their cached _salient.json). Run:
  micromamba run -n ts-language python flywheel/audit_detection.py
"""
import os, sys, csv, json

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from core.detect import detect_threshold, detect_robust_multiscale       # noqa: E402

PRICES = os.path.join(HERE, "raw", "DCOILWTICO.csv")
REPORT = os.path.join(HERE, "raw", "_detection_audit.json")


def _prices():
    rows = []
    for r in csv.reader(open(PRICES)):
        if not r or not r[0][:4].isdigit() or len(r) < 2 or not r[1].strip() or r[1] == ".":
            continue
        rows.append((r[0], float(r[1])))
    return rows


def main():
    rows = _prices()
    dates = [d for d, _ in rows]
    prices = [p for _, p in rows]

    thr = detect_threshold(prices, ret_min=0.06, spacing=10, top_k=None)
    multi = detect_robust_multiscale(prices, scales=(1, 3, 5), z_min=2.8, spacing=10)

    thr_idx = {i for i, *_ in thr}
    # coverage: multi-scale should still catch every move the threshold flags (no regression).
    covered = sum(1 for j in thr_idx if any(abs(i - j) < 10 for i, *_ in multi))
    # multi-scale points NOT within spacing of any threshold hit = what the threshold missed.
    recovered = [(i, k, r, z) for i, k, r, z in multi
                 if all(abs(i - j) >= 10 for j in thr_idx)]
    multiday = [x for x in recovered if x[1] > 1]        # the slow-drift wins (scale > 1 day)

    print(f"== W1 detection audit · WTI, {len(rows)} days ==\n")
    print(f"fixed threshold (|1d ret|>=6%)       : {len(thr)} salient days")
    print(f"robust multi-scale (1/3/5d, |z|>=2.8): {len(multi)} salient days")
    print(f"coverage of threshold hits           : {covered}/{len(thr)} (no regression)")
    print(f"recovered by multi-scale (threshold missed): {len(recovered)} "
          f"({len(multiday)} are genuine MULTI-DAY moves the 1-day bar can't see)\n")
    for i, k, r, z in recovered[:12]:
        print(f"   {dates[i]}  {k}d move {r*100:+.1f}%  z={z:+.1f}"
              f"{'  <- multi-day drift' if k > 1 else ''}")

    json.dump({"n_days": len(rows), "threshold": len(thr), "multiscale": len(multi),
               "recovered": [{"date": dates[i], "scale_days": k, "ret": round(r, 4),
                              "z": round(z, 2)} for i, k, r, z in recovered]},
              open(REPORT, "w"), indent=1)
    # healthy = no regression (covers every threshold hit) AND recovers multi-day moves it missed.
    ok = covered == len(thr) and len(multiday) > 0
    print(f"\n== {'PASS' if ok else 'FAIL'}: covers {covered}/{len(thr)} threshold hits, "
          f"recovers {len(recovered)} missed ({len(multiday)} multi-day) ==")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
