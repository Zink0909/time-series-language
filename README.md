# time-series-language

Research code for **building and *verifying* text ↔ time-series data** for a unified
time-series + language model, plus a **ForecastBench** study of *when* text actually helps
forecasting.

The repo has two independent engineering bodies. A full file map is in
[`PROJECT_STRUCTURE.md`](./PROJECT_STRUCTURE.md).

```
ts_lang_curation/     # (1) text↔series data pipeline + a two-stage verification system
forecastbench_eval/   # (2) where a pure time-series model lands vs LLMs & human forecasters
```

---

## 1. `ts_lang_curation/` — data pipeline + verification

**Sources.** One thin adapter per first-party source (`sources/`), each pairing real official
text with its time series at the source's own granularity: FOMC statement ↔ 2-year Treasury
yield, SEC 10-K MD&A ↔ XBRL fundamentals, CVE text ↔ EPSS exploit-probability, Wikipedia event
lead ↔ pageview attention, earthquake report ↔ aftershock decay, FNSPID news ↔ prices. Adding a
source touches only `sources/<name>.py`; the shared `core/` is untouched.

**Flywheel** (`flywheel/`). An automated engine that *manufactures* pairs for series with **no
pre-existing text**: detect salient moves → retrieve news → LLM-distill a **leakage-safe** cause →
faithfulness + leakage gates → emit. Two links built (oil via GDELT/BigQuery; Wikipedia via
point-in-time revisions, no external retrieval). The old oil/commodity outputs use FRED-backed
series and remain reproducible research artifacts, but are blocked from the release package by the
central license registry.

**Schema.** Every record is a Qwen ChatML transcript with `<ts></ts>` placeholders mapped to
z-normalized time-series spans, with loss masks (`text_desc`, `ts_forecast`,
`ts_forecast_covariates`, …). Format spec in [`ts_lang_curation/DATASHEET.md`](./ts_lang_curation/DATASHEET.md).

**Verification pipeline** (`flywheel/companion/`) — correctness and usefulness are separate:

| stage | file | question | compute |
|-------|------|----------|---------|
| ① data-quality | `verify_cpt.py` | is the record well-formed, leakage-safe, its alignment claim honest, non-redundant, trainable? | CPU |
| ② forecasting-lift | `stage2_cpt.py` + the 3-arm harness (`arms.py`/`models.py`/`run.py`) | does the text actually **lower forecast error**? | GPU |

Stage ② is a **3-arm test** — **A** no-text / **B** correct-text / **C** shuffled-text (text
permuted onto the wrong series). A real gain needs **B to beat both A and C**: beating A but not C
means the gain is the *channel*, not the *content*.

`verify_any.py` has two explicit profiles. `structural` checks trainability and format invariants;
`release` additionally enforces the reviewed source-license decision, attribution conditions,
source, and evidence URLs. A structural pass
must not be described as publication-ready.

> **Experiment validity notice (2026-09-05).** Historical MiGAS numbers in local reports predate the
> removal of future-window normalization and symmetric A/B/C training. Treat those numbers as
> invalidated pending rerun. The corrected runner uses history-normalized targets, a strict shuffled
> derangement, both temporal-ID and entity-OOD exports, three seeds by default, and cluster bootstrap CIs.

**Shared framework & format bridge** (for merging two producers' data). The curation framework is
one shared `core/` + one thin adapter per source — see [`ts_lang_curation/FRAMEWORK.md`](./ts_lang_curation/FRAMEWORK.md).
Data can be persisted as the lossless `pair@1` canonical IR, then emitted directly as ChatML or the
instruction-free CPT form; `flywheel/companion/verify_any.py` verifies both output formats through
one gate. The older converters remain compatibility tools, not the primary architecture.

**Discipline.** Independent audits (`flywheel/audit_*.py`) re-derive quality from the spec rather
than trusting the builder's own `validate()`; the leakage cutoff (`knowledge_time` < forecast
origin) is enforced in code and audited per record; everything is reproducible under a minimum
regression (`scripts/regress.py`).

## 2. `forecastbench_eval/` — where a pure TS model lands

Scores a pure time-series model (Chronos-2, **no text at all**) against human superforecasters and
133 LLM configurations on ForecastBench, then decomposes the result by history length, forecast
horizon, and domain. Details + reproduction in
[`forecastbench_eval/README.md`](./forecastbench_eval/README.md) and `RESULTS.md`.

---

## Setup

```bash
micromamba create -f environment.yml           # env: ts-language (Python 3.11)
```

## Run

```bash
# build a source into ChatML records
micromamba run -n ts-language python ts_lang_curation/build.py --source treasury_fomc
micromamba run -n ts-language python ts_lang_curation/build.py --source sec_edgar --format ir
micromamba run -n ts-language python ts_lang_curation/build.py --source sec_edgar --format cpt

# minimum regression (build 2 sources + replay flywheel + independent audits)
micromamba run -n ts-language python ts_lang_curation/scripts/regress.py

# verify a CPT corpus (stage 1, CPU) / probe forecasting-lift (stage 2)
micromamba run -n ts-language python ts_lang_curation/flywheel/companion/verify_cpt.py <corpus_dir> --report out.html
micromamba run -n ts-language python ts_lang_curation/flywheel/companion/verify_any.py <corpus_dir> --profile release --strict
micromamba run -n ts-language python ts_lang_curation/release.py ts_lang_curation/out/release_candidate_v05
micromamba run -n ts-language python ts_lang_curation/flywheel/companion/stage2_cpt.py <corpus.jsonl>

# dependency-free invariant tests
python -m unittest discover -s ts_lang_curation/tests -v

# corrected GPU experiment; writes a provenance-bearing result manifest
python ts_lang_curation/flywheel/companion/migas_finetune.py \
  --migas-dir /path/to/synthefy-migas --seeds 3 --results-json experiment_result.json

# ForecastBench analysis (offline, from cached forecasts — no GPU)
micromamba run -n ts-language python forecastbench_eval/analyze_domain_edge.py
```

---

## Notes

- Built data (`out/`), raw caches (`**/raw/`), and large regenerable artifacts are **not** tracked —
  run the pipeline to regenerate them. Small dev-set samples live in `ts_lang_curation/dev/`.
- The pipeline's LLM calls read the endpoint / key / model from environment variables
  (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`) — nothing is hardcoded.
