# ts_lang_curation — unified stage-2 data pipeline

One shared engine + a thin adapter per source. Adding a dataset does **not** mean writing a
new pipeline — you write one small `sources/<name>.py` and the shared core turns it into
full **Data-Schema** training records (ChatML transcript + z-score spans + loss masks).

```
core/         # written ONCE, shared by every source
  schema.py     adapter contract: TsToText / TextToTs  ->  build_record(), validate()
  chatml.py     emits full Data-Schema records (ChatML, <stats><ts></ts>, z-score, loss masks)
  compress.py   Qwen vLLM API client (chat / summarize)
  generate.py   grounded ts->text: generate -> numeric verify -> retry -> fallback
  verify.py     numeric "reflection" check (every stated figure must match the series)
  writer.py     jsonl output
sources/      # one thin adapter per dataset — fetch / parse / pair only
  fred_fomc.py        FOMC statement  <->  2y Treasury yield (+10y covariate)
  sec_edgar.py        10-K MD&A       <->  XBRL annual revenue / net income
  wiki_pageviews.py   event article   <->  daily pageview attention
  usgs_quakes.py      mainshock text  <->  aftershock-sequence decay
  cyber_epss.py       CVE description <->  EPSS exploit-probability rise
  fnspid.py           demo only (Faisal owns FNSPID)
build.py      # unified entry: python build.py --source fred_fomc
raw/          # cached source downloads (gitignore)
out/          # built .jsonl records
```

## Setup
```
micromamba create -f ../environment.yml          # one-time: builds the `ts-language` env
# or, into any Python 3.11:  pip install -r requirements.txt
```

## Run
```
python build.py --list                  # show available sources
python build.py --source fred_fomc       # -> out/fred_fomc.jsonl
```
(Use the project env: `micromamba run -n ts-language python build.py ...`)

## Two task contracts (per references/Data Schema.docx)

Adapters yield one of two example types; the core maps them to the loader's task contracts:

- **`TsToText`  → `task_type: text_desc`** (ts → text). The series are user-side observed spans
  (`<stats>len,mean,std</stats> <ts></ts>`); the assistant writes the description; **text loss**
  is on the assistant answer (`text_loss_char_ranges`).
- **`TextToTs`  → `task_type: ts_forecast`** (text → ts; `ts_forecast_covariates` with covariates).
  Conditioning text + history context span → assistant target span; **TS loss only on the future
  suffix** (`loss_start = H`). The target's first H values equal the context span exactly after
  z-score normalization.

All series values are z-scored (`raw = z*std + mean`); per-span stats live in `normalization`.

## Add a new source (the whole job)
Create `sources/<name>.py` exposing `pairs()` that yields `TsToText` / `TextToTs`:
```python
from core.schema import TsToText, TextToTs

def pairs():
    # ts -> text  (text_desc): describe a series
    yield TsToText(
        user_intro="<series description + the ask>",
        series=[{"name":"...", "values":[...raw...], "unit":"...", "freq":"..."}],
        answer="<assistant description>",
        meta={"dataset":"...", "series_id":"...", "direction":"ts_to_text"},
        text_source="...", is_generated="real|derived_generated")

    # text -> ts  (ts_forecast): condition on text + history, predict the future
    yield TextToTs(
        user_text="<world-knowledge conditioning text + the ask>",
        history=[...raw...], future=[...raw...],
        series_name="...", unit="...", freq="...",
        covariates=[],                                   # optional -> ts_forecast_covariates
        meta={"dataset":"...", "series_id":"...", "direction":"text_to_ts"},
        text_source="...", is_generated="real", knowledge_time="ISO8601")
```
`core/` z-score-normalizes, assembles the ChatML transcript, sets loss masks / `loss_start`,
computes `text_loss_char_ranges`, and validates. Same schema across all sources.

## For LLM-generated ts→text
`core/generate.grounded_describe()` generates with Qwen, then **verifies every number** against
the series (`core/verify.py`); on mismatch it retries once, then falls back to a deterministic
template. Use `rel_tol` for large-magnitude series (e.g. pageviews). Demos can run deterministic
(template) without the API; full runs use Qwen.

## Status
- Active sources (ours): `fred_fomc` (166), `sec_edgar` (387), `wiki_pageviews` (134),
  `usgs_quakes` (86), `cyber_epss` (160). `fnspid` = demo (Faisal owns FNSPID).
- Dev-Set freeze phase: ~80 records/source in `dev/` (mirrored in `../dev_set_review/`).
- `flywheel/` = open-loop data-flywheel v1 (WTI crude, GDELT retrieval) → `out/flywheel_oil.jsonl`.
- Output is the full Data-Schema (ChatML) format, ready for the team loader.
- Deliverables & collaboration live on Notion (`../notion_pages/`), not a shared code repo.
