#!/usr/bin/env python3
"""History-length sweep on ForecastBench (2024-07-21 FRED+Yahoo). Chronos-2 was run at several context
lengths (CTX = 32/64/128/256/512); this scores each by HORIZON bucket to decouple the long-history
advantage from the forecast horizon. Question: does more history help, and does it interact with horizon?

Superforecasters are horizon-bucketed too (constant across CTX) as the world-knowledge reference.
Metric = Brier (lower better); persistence(0.5) = 0.25 floor.

  micromamba run -n ts-language python forecastbench_eval/score_ctx.py
"""
import json, os, glob, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKETS = [("≤1wk", 0, 7), ("≤1mo", 8, 30), ("≤1qtr", 31, 90), (">1qtr", 91, 10000)]
NAIVE = {"Always 0", "Always 0.5", "Always 1", "Imputed Forecaster", "Naive Forecaster",
         "Random Uniform", "Public median forecast", "Superforecaster median forecast"}


def llm_median_by_bucket(ref_rows):
    """Median Brier across the official LLM configs, per horizon bucket, from the real-time round data
    (no lookahead — these are the forecasts submitted in 2024). Joined to horizons via ref_rows."""
    hz = {(r["id"], r["resolution_date"][:10]): r for r in ref_rows}       # key -> {horizon, resolved_to}
    per_bucket = {b[0]: [] for b in BUCKETS}                                # bucket -> list of per-config Brier
    for f in glob.glob(os.path.join(HERE, "raw", "llm", "*.json")):
        d = json.load(open(f))
        if d.get("model", "").split(" (")[0] in NAIVE:
            continue
        by_b = {b[0]: [] for b in BUCKETS}
        for x in d.get("forecasts", []):
            if x.get("source") not in ("fred", "yfinance") or not isinstance(x.get("id"), str):
                continue
            k = (x["id"], str(x.get("resolution_date", ""))[:10])
            if k not in hz:
                continue
            try:
                p = float(x["forecast"])
            except (ValueError, TypeError):
                continue
            by_b[bucket(hz[k]["horizon"])].append((p - hz[k]["resolved_to"]) ** 2)
        for b, v in by_b.items():
            if v:
                per_bucket[b].append(sum(v) / len(v))                       # this config's Brier on the bucket
    return {b: (st.median(v) if v else None) for b, v in per_bucket.items()}


def bucket(h):
    for name, lo, hi in BUCKETS:
        if lo <= h <= hi:
            return name
    return ">1qtr"


def brier(rows, pred_key):
    v = [(r[pred_key] - r["resolved_to"]) ** 2 for r in rows if r.get(pred_key) is not None]
    return sum(v) / len(v) if v else None


def load(ctx):
    tag = "" if ctx == 512 else f"_ctx{ctx}"
    p = os.path.join(HERE, f"ts_forecast{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ctxs = [c for c in (32, 64, 128, 256, 512) if load(c) is not None]
    if not ctxs:
        print("no ts_forecast_ctx*.json found — run the sweep on the box first"); return
    cols = [b[0] for b in BUCKETS]
    print(f"=== Chronos-2 Brier by history length × horizon (lower=better) · buckets {cols} ===\n")
    hdr = f"{'CTX (history)':16}" + "".join(f"{c:>9}" for c in cols) + f"{'ALL':>9}{'n':>6}"
    print(hdr); print("-" * len(hdr))
    for ctx in ctxs:
        rows = load(ctx)
        line = f"{ctx:<16}"
        for c in cols:
            br = brier([r for r in rows if bucket(r["horizon"]) == c], "p_up")
            line += f"{(f'{br:.3f}' if br is not None else '—'):>9}"
        line += f"{brier(rows,'p_up'):>9.3f}{len(rows):>6}"
        print(line)
    # superforecaster reference (constant across CTX; use the 512 rows that carry human_super)
    ref = [r for r in load(ctxs[-1]) if r.get("human_super") is not None]
    line = f"{'Superforecasters':16}"
    for c in cols:
        br = brier([r for r in ref if bucket(r["horizon"]) == c], "human_super")
        line += f"{(f'{br:.3f}' if br is not None else '—'):>9}"
    line += f"{brier(ref,'human_super'):>9.3f}{len(ref):>6}"
    print(line)
    # median official LLM (real-time forecasts, no lookahead), by horizon
    llm = llm_median_by_bucket(load(ctxs[-1]))
    line = f"{'Median LLM':16}"
    for c in cols:
        line += f"{(f'{llm[c]:.3f}' if llm[c] is not None else '—'):>9}"
    print(line + f"{'':>9}{'':>6}")
    print(f"{'persistence':16}" + "".join(f"{0.250:>9.3f}" for _ in cols) + f"{0.250:>9.3f}")
    print("\nRead: across a CTX column top-to-bottom = does more history help at that horizon;")
    print("Chronos vs Superforecasters per bucket = where the numeric prior beats world-knowledge judgment.")


if __name__ == "__main__":
    main()
