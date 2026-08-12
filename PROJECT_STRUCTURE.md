# Project structure

Two self-contained engineering bodies. Everything else at the repo root (reports, notes,
references, raw caches, built data) is local-only and git-ignored — see `.gitignore`.

```
.
├── ts_lang_curation/        # (1) text <-> time-series data pipeline + verification
├── forecastbench_eval/      # (2) ForecastBench: pure-TS baseline + horizon/domain analysis
├── environment.yml          # micromamba env  (env name: ts-language)
└── README.md
```

---

## 1. `ts_lang_curation/` — data pipeline + verification

Builds ChatML text↔time-series records (schema in `DATASHEET.md`) from first-party sources and
from an automated "flywheel", then verifies whether the paired text actually carries value.

```
build.py                     # CLI: python build.py --source <name>
core/                        # shared engine (one schema, reused by every source)
  schema.py  chatml.py       #   record dataclasses; ChatML assembly + z-norm + loss masks + validate
  compress.py generate.py    #   LLM calls (endpoint/key/model via env vars — never hardcoded)
  verify.py  faithfulness.py #   numeric reflection; entity-tracing faithfulness gate
  detect.py  retrieval_rank.py leak_screen.py match_score.py writer.py
sources/                     # one thin adapter per source (core untouched when adding one)
  fred_fomc.py sec_edgar.py cyber_epss.py usgs_quakes.py wiki_pageviews.py fnspid.py
flywheel/                    # automated engine: series -> detect -> retrieve -> annotate -> gate -> emit
  oil_demo.py wiki_demo.py wiki_scale.py commodity_demo.py bq_pool.py
  audit_*.py                 #   independent QA/leakage/faithfulness audits (don't trust validate())
  companion/                 # the VERIFICATION pipeline (two stages)
    verify_cpt.py            #     stage 1 — data-quality verifier (CPU, laptop, full-scale)
    stage2_cpt.py            #     stage 2 — forecasting-lift adapter + leakage screen (GPU verdict pending)
    data.py arms.py models.py run.py     # the 3-arm A/B/C harness (no-text / correct / shuffled)
    discriminator.py migas_finetune.py source_registry.py
    verify_*.py export_*.py colab_*.py   # robustness battery + base-model export/probes
scripts/regress.py           # minimum regression (build 2 sources + replay flywheel + audits)
dev/                         # small dev-set samples (the shipped data; full out/ is git-ignored)
```

Run: `micromamba run -n ts-language python build.py --source fred_fomc`
Regression: `micromamba run -n ts-language python scripts/regress.py`

## 2. `forecastbench_eval/` — ForecastBench analysis

Scores a pure time-series model (Chronos-2) against human superforecasters and LLMs on the
`2024-07-21` round, then decomposes by history length, forecast horizon, and domain.

```
fetch_bench.py fetch_series.py    # pull the round + FRED/Yahoo histories (leakage-safe cutoff)
chronos_forecast.py               # Chronos-2 forecasts (GPU)  ->  cached ts_forecast*.json
score.py score_ctx.py             # 3-way Brier tables
analyze_domain_edge.py            # per-domain + lookback + edge-case deep-dive (offline, from cache)
RESULTS.md                        # write-up
*.json                            # cached forecasts/series (raw/ is git-ignored — 600M)
```

Reproduce the analysis (no GPU): `python analyze_domain_edge.py`

---

## Not in the repo (git-ignored, local only)

`raw/` `.cache/` (downloads) · `out/` (built jsonl) · `reports/` `notion_pages/` `dev_set_review/`
(local write-ups) · `references/` (papers) · `scratchpad/` (backups) · `discord_exports/` &
`CLAUDE.md` (team-internal) · `**/_*.json` (regenerable verification caches) · `__pycache__/`.
