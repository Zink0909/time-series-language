# The shared data-curation framework (core + adapters)

The team framework for producing text↔time-series data: **one shared `core/`, one thin adapter per
source.** Adding a data source is one file; the engine (schema, normalization, gates, validation)
is never touched. This is the reusable "工序" both producers migrate onto.

## The pattern

```
core/                 # shared engine — DO NOT fork per source
  schema.py           #   TsToText / TextToTs record classes + build_record() + validate()
  chatml.py           #   ChatML assembly + z-normalization (robust floor) + loss masks
  compress.py         #   LLM calls (endpoint/key/model via env vars)
  verify.py / faithfulness.py / leak_screen.py   # correctness + leakage + faithfulness gates
sources/
  <name>.py           # ONE thin adapter per source — the only file you add
build.py              # python build.py --source <name>   (core untouched)
```

## Add a source in one file

`sources/<name>.py` exposes a single `pairs()` generator that yields
`core.schema.TsToText` (understanding, ts→text) and/or `core.schema.TextToTs`
(forecasting, text→ts) objects. That is the whole contract.

```python
from core.schema import TsToText, TextToTs

def pairs():
    for event in _load_my_source():           # your fetch/parse logic (private helpers)
        # understanding: describe the series
        yield TsToText(series=event.series, unit="...", freq="1d",
                       answer=event.description, series_id=event.id, dataset="<name>", ...)
        # forecasting: predict the series from text
        yield TextToTs(text=event.text, history=event.hist, future=event.future,
                       unit="...", freq="1d", knowledge_time=event.kt, series_id=event.id, ...)
```

`build.py` then does everything else: `build_record()` assembles the ChatML transcript, applies
context-window z-normalization, sets the loss masks, and `validate()` runs the schema invariants;
`writer.py` writes `out/<name>.jsonl`. **No change anywhere in `core/`.**

```
python build.py --list                 # list sources
python build.py --source <name>        # build one
python scripts/regress.py              # minimum regression (all sources + flywheel + audits)
```

## Two formats, one gate, converters between them

Understanding data can live in either the **ChatML** form (this framework's `text_desc`, with loss
masks) or the instruction-free **CPT** form. They are bridged, and both pass one verification gate:

| tool (`flywheel/companion/`) | does |
|------|------|
| `verify_any.py` | one verification gate that reads **both** CPT and ChatML |
| `verify_cpt.py` | CPU data-quality checks for CPT records (structural / leakage / alignment-honesty / redundancy / licence) |
| `chatml_to_cpt.py` | ChatML `text_desc` → CPT world_knowledge (**canonical direction; CPT first**) |
| `cpt_to_chatml.py` | CPT world_knowledge → ChatML `text_desc` (reverse direction) |

Conversions are lossy in the expected way: CPT is instruction-free, so ChatML→CPT drops the task
prompt / loss masks / think scaffold; the prose + series + provenance are preserved. Both
converters self-check their output against an independent validator.

## Migrating an existing per-source codebase onto this

For a codebase that currently has one script per dataset: keep your fetch/parse logic as private
helpers, and wrap the final pairing step in a `pairs()` generator that yields `TsToText`/`TextToTs`.
Everything downstream (normalization, ChatML assembly, loss masks, validation, verification) is then
shared, so per-dataset scripts shrink to the source-specific parsing only.
