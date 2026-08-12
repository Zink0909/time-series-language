#!/usr/bin/env python3
"""RUNS ON THE BOX. Chronos-2 forecasts each FRED/Yahoo series to each ForecastBench resolution date and
reads P(value goes up) = fraction of the forecast distribution above the CURRENT value.

Threshold = the series' own last observed value (NOT bench freeze_value): for split stocks the freeze is
the raw price while our series is split-adjusted, and ForecastBench resolves splits, so the series' own
last is the consistent, leakage-safe threshold. Horizons beyond ~2 trading years are skipped (Chronos is
unreliable that far out); those long-horizon questions are out of scope for a pure-TS extrapolation.

  cd /data/jiaju/synthefy-migas && CUDA_VISIBLE_DEVICES=2 uv run python /data/jiaju/fbench/chronos_forecast.py
"""
import os, json, datetime as dt
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
bench = json.load(open(os.path.join(HERE, "bench_fred_yahoo.json")))
series = json.load(open(os.path.join(HERE, "series.json")))
DUE = dt.date(2024, 7, 21)
CTX = int(os.environ.get("FBENCH_CTX", "512"))                     # history length (env-swept: 32/64/128/256/512)
MAX_H = 504                                                        # cap horizon at ~2 trading years

from chronos import BaseChronosPipeline                            # noqa: E402
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda:0",
                                           torch_dtype=torch.bfloat16)


def p_up(vals, horizon, last):
    """P(series value at `horizon` steps ahead > last). Chronos-2 predict returns quantiles; we read the
    quantile level whose value crosses `last` -> that's P(value <= last), so P(up) = 1 - that level."""
    ctx = torch.tensor(vals[-CTX:], dtype=torch.float32).reshape(1, 1, -1)   # (n_series, n_variates, hist)
    q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles, _mean = pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=q_levels)
    qh = quantiles[0][0, -1].float().cpu().numpy()                 # (n_quantiles,) at the final horizon step
    # interpolate the CDF at `last`
    if last <= qh[0]:
        return 1.0 - q_levels[0]
    if last >= qh[-1]:
        return 1.0 - q_levels[-1]
    for i in range(1, len(qh)):
        if last <= qh[i]:
            frac = (last - qh[i - 1]) / (qh[i] - qh[i - 1] + 1e-9)
            level = q_levels[i - 1] + frac * (q_levels[i] - q_levels[i - 1])
            return float(1.0 - level)
    return 0.5


def main():
    out, skip = [], 0
    for q in bench["questions"]:
        s = series.get(q["id"])
        if not s or len(s["values"]) < 30:
            continue
        vals = s["values"]
        last = vals[-1]                                            # consistent threshold (adjusted scale)
        for row in q["resolution_rows"]:
            rd = dt.date.fromisoformat(row["resolution_date"])
            h = int(np.busday_count(DUE, rd))
            if h < 1 or h > MAX_H:
                skip += 1
                continue
            try:
                p = p_up(vals, h, last)
            except Exception as e:
                print(f"  predict fail {q['id']} h={h}: {repr(e)[:50]}", flush=True)
                continue
            out.append({"id": q["id"], "source": s["source"], "resolution_date": row["resolution_date"],
                        "horizon": h, "p_up": round(p, 4), "last": last,
                        "resolved_to": row["resolved_to"], "human_super": row["human_super"]})
        print(f"  {q['source']:8} {q['id']:12} done ({sum(1 for o in out if o['id']==q['id'])} horizons)", flush=True)
    tag = "" if CTX == 512 else f"_ctx{CTX}"
    path = os.path.join(HERE, f"ts_forecast{tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nChronos-2 forecasts (CTX={CTX}): {len(out)} (id,horizon) points · {skip} skipped (>2y) -> {path}")


if __name__ == "__main__":
    main()
