# The shared data-curation framework (core + adapters)

The team framework for producing text↔time-series data: **one shared `core/`, one thin adapter per
source.** Adding a data source is one file; the engine (schema, normalization, gates, validation)
is never touched. This is the reusable "工序" both producers migrate onto.

## The pattern

```
core/                 # shared engine — DO NOT fork per source
  schema.py           #   TsToText / TextToTs adapter contract
  ir.py               #   lossless pair@1 persistence
  emitters.py         #   peer Pair -> IR / ChatML / CPT outputs
  governance.py       #   central source-license registry policy
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
        yield TsToText(
            user_intro="Describe the following series.",
            series=[{"name": event.name, "values": event.series,
                     "unit": "...", "freq": "daily"}],
            answer=event.description,
            meta={"series_id": event.id, "dataset": "<name>",
                  "source": event.source, "license": event.license})
        # forecasting: predict the series from text
        yield TextToTs(
            user_text=event.text, history=event.hist, future=event.future,
            series_name=event.name, unit="...", freq="daily",
            knowledge_time=event.kt,
            meta={"series_id": event.id, "dataset": "<name>",
                  "source": event.source, "license": event.license,
                  "text_url": event.text_url, "ts_url": event.ts_url})
```

`build.py` then validates the Pair and either persists IR or invokes a peer emitter. ChatML applies
context-window normalization and loss masks; CPT accepts understanding (`TsToText`) rows only.
`writer.py` atomically writes artifacts and ledgers. **No change anywhere in `core/`.**

```
python build.py --list                 # list sources
python build.py --source <name>        # build one (for example treasury_fomc)
python build.py --source <name> --format ir
python build.py --source <name> --format cpt
python scripts/regress.py              # integration smoke test (selected sources + flywheel audits)
python -m unittest discover -s tests -v # dependency-free invariant tests
python release.py out/release_candidate # strict gate + cryptographic release catalog
```

## Canonical IR, peer outputs, one gate

Adapter examples can be persisted losslessly as `pair@1`, then emitted directly as **ChatML** or
instruction-free **CPT**. Both outputs pass one verification gate:

| tool (`flywheel/companion/`) | does |
|------|------|
| `verify_any.py` | one verification gate that reads **both** CPT and ChatML |
| `verify_cpt.py` | CPU data-quality checks for CPT records (structural / leakage / alignment-honesty / redundancy / licence) |
| `chatml_to_cpt.py` | ChatML `text_desc` → CPT world_knowledge (**canonical direction; CPT first**) |
| `cpt_to_chatml.py` | CPT world_knowledge → ChatML `text_desc` (reverse direction) |

The converters are retained for old artifacts and are lossy in the expected way. New builds do not
convert one training format into another. CPT cannot represent forecasting rows, which are recorded
in the build's `.unsupported.jsonl` ledger.

`verify_any.py --profile structural` checks structural trainability. Use
`verify_any.py --profile release --strict` before publication; it additionally enforces the central
source-license decision, attribution conditions, source, and evidence URLs. Unknown formats and
unregistered sources fail closed. ChatML→CPT also fails closed when the normalization metadata
needed to reconstruct raw values is absent.

## Migrating an existing per-source codebase onto this

For a codebase that currently has one script per dataset: keep your fetch/parse logic as private
helpers, and wrap the final pairing step in a `pairs()` generator that yields `TsToText`/`TextToTs`.
Everything downstream (normalization, ChatML assembly, loss masks, validation, verification) is then
shared, so per-dataset scripts shrink to the source-specific parsing only.
