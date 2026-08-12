# ForecastBench: how a pure time-series model does vs LLMs and humans

**Question (from Xinyue): has ForecastBench been run — and where does a pure time-series model land?**

ForecastBench (Karger et al., 2024) scores forecasters on 500 questions/round with difficulty-adjusted
Brier. About half are **dataset questions** auto-generated from real numeric series (FRED, Yahoo/yfinance,
Wikipedia, ACLED, DBnomics) with the template *"will the value at {resolution_date} be higher than at
{forecast_due_date}?"*. Those are the ones a pure time-series model **can** answer: forecast the underlying
series and read `P(future > current)` off the predictive distribution. The `2024-07-21` round is the only
one with published baseline forecasts (superforecasters, public crowd, and 133 LLM configs), so it is the
one round where a three-way comparison is possible.

## Pipeline (reproducible)

```
micromamba run -n ts-language python forecastbench_eval/fetch_bench.py     # pull round + resolutions + human baselines
micromamba run -n ts-language python forecastbench_eval/fetch_series.py    # pull FRED+Yahoo histories UP TO 2024-07-21 (leakage-safe)
# on the team box (ds-serv10, card 2), Chronos-2:
cd /data/jiaju/synthefy-migas && CUDA_VISIBLE_DEVICES=2 uv run python /data/jiaju/fbench/chronos_forecast.py
micromamba run -n ts-language python forecastbench_eval/score.py           # three-way Brier table
```

- **Leakage control**: `fetch_series.py` fetches nothing dated after the `2024-07-21` forecast-due date;
  Chronos sees only history a forecaster would have had. Threshold for `P(up)` is the series' own last
  observed value (split-adjusted), not the bench freeze (which is raw price for split stocks).
- **Scope**: FRED + Yahoo dataset questions (id = series id / ticker, rule = "future > current"). Horizons
  beyond ~2 trading years are dropped (Chronos is unreliable that far out). Result: **213 resolved
  (id, resolution_date) points**, base rate 0.50.
- **Chronos ran on the team GPU** (ds-serv10, `CUDA_VISIBLE_DEVICES=2`), not locally — the local env has no
  torch. Everything else (fetch, score) is pure-urllib and runs locally.

## Result (Brier, lower = better; 213 points, base rate 0.50)

| Forecaster                                         | ALL   | FRED (macro) | Yahoo (stocks) |
|----------------------------------------------------|-------|--------------|----------------|
| Superforecasters (human crowd)                     | 0.204 | **0.157**    | 0.252          |
| **best LLM** — Claude-3.5-Sonnet (superfc.+news)   | 0.203 | 0.182        | **0.225**      |
| **Chronos-2 (pure TS, no text at all)**            | 0.232 | 0.197        | 0.268          |
| median of 133 LLM configs                          | 0.248 | 0.254        | 0.240          |
| persistence (always 0.5)                           | 0.250 | 0.250        | 0.250          |

## What it says

1. **The median world-knowledge LLM barely beats a coin flip on TS-driven questions** (0.248 vs the 0.250
   floor). News access + world knowledge, by itself, is not enough for "will this number go up".
2. **A pure time-series model with zero world knowledge beats the typical news-reading LLM** (Chronos 0.232 <
   median LLM 0.248). On these questions the *channel* — the raw series — already carries most of the signal
   an average LLM extracts from text.
3. **The best forecasters (humans 0.204, best LLM 0.203) do beat pure TS**, and the edge is concentrated on
   **FRED macro** (humans 0.157 vs Chronos 0.197): structured world-knowledge / mean-reversion priors help
   there. On **Yahoo stocks** everyone clusters near the floor (0.225–0.268) — near-random-walk, little to add.
4. **Wikipedia questions are the world-knowledge pole and are not TS-forecastable at all** (see below).

## Wikipedia: the non-TS bucket (documented, not skipped)

Of the 22 Wikipedia dataset questions in this round:

- **11** are binary world-knowledge facts — *"will a vaccine have been developed for {disease}?"* (freeze = "No").
  No time series exists.
- **8** are FIDE Elo / ranking values — numeric, but ForecastBench only snapshots them per round; there is no
  accessible daily history to feed a TS model, and the rule is "≥1% higher" / "ranking as high as", not a clean
  series threshold.
- **3** are "will {athlete} still hold the {event} world record?" — a record-holder fact, not a series.

So **none** of the Wikipedia questions are cleanly forecastable by a pure TS model. This is not a coverage gap
to paper over — it is the point: Wikipedia is exactly the **world-knowledge** slice where a time-series channel
has zero traction, and it is the slice a world-knowledge data flywheel is built to serve.

## Feeds both tracks

- **【论文】 "Content or Channel?"**: on TS-driven questions the channel already captures most of what an average
  news-reading LLM gets from text; a pure-TS baseline (Chronos 0.232, beating the median LLM) is the strong
  control any claim about the *marginal* value of precise text↔series pairing must clear.
- **【飞轮】**: the questions a TS model structurally cannot touch (Wikipedia world-knowledge) are precisely what
  world-knowledge continued-pretraining data targets — concrete external evidence for why the flywheel exists.
