# forecastbench_eval

Where a **pure time-series model lands against LLMs and human forecasters** on ForecastBench, and
what that says about when text helps forecasting.

Headline (round `2024-07-21`, 213 resolved points, base rate 0.50): a pure time-series model
(Chronos-2, **no text at all**) beats the median news-reading LLM; the best humans/LLMs win only on
macro questions where structured world knowledge applies; Wikipedia world-knowledge questions are
not forecastable from a series at all. Full write-up in **`RESULTS.md`**.

## Pipeline

```
fetch_bench.py     pull the round, resolutions, human baselines
fetch_series.py    pull FRED + Yahoo histories up to the due date (nothing dated after it — leakage-safe)
chronos_forecast.py  Chronos-2 on GPU, one forecast per series per horizon  -> ts_forecast*.json (cached)
score.py           three-way Brier table (Chronos vs superforecasters vs 133 LLM configs)
analyze_domain_edge.py   per-domain + history×horizon + edge-case deep-dive
```

## Reproduce

The Chronos forecasts are cached (`ts_forecast.json` = 512-ctx, `ts_forecast_ctx{32,64,128,256}.json`),
so the analysis re-runs **offline, no GPU**:

```bash
micromamba run -n ts-language python analyze_domain_edge.py
```

Re-running the forecasts themselves needs a GPU and re-fetching `raw/` (git-ignored).

## Key findings (`analyze_domain_edge.py`)

- The ">1 quarter gap re-opens" pattern is a **macro (FRED) phenomenon** — at long horizons human
  structured priors keep improving while Chronos decays; stocks stay near a random walk.
- The aggregate non-monotonicity is a **horizon-axis** effect, not a lookback-window one: more history
  is monotonically better at every horizon (falsifies the "lookback local-optimum" guess).
- **No edge case where both Chronos and humans fail badly** — every hard-for-Chronos point is one a
  human got, i.e. "the information was available, the series just couldn't reach it".
