# Commodity flywheel — v1 spec

> **Governance status:** historical research spec. The current cached series are FRED-backed and
> the `flywheel_commodity*` datasets are blocked from release. The detection, retrieval and audit
> code remains usable; a release rebuild requires approved first-party series replacements.

**Why this exists (the diagnosis it answers).** The fine-tune "cheat upper-bound" probe showed the
flywheel mechanism is sound — when the conditioning text carries *real information about what happened*,
the model uses it (MASE 2.84→2.16). But ~95 % of the Wikipedia point-in-time text is a **generic
standing description** ("Altman is an entrepreneur"), not "what happened that day", so the model
learns to ignore it and arm B can't beat arm A. Cure per Xinyue's meeting: move to **dense sources
that carry real event text** — commodities / FX / crypto — where each salient move has genuine news
behind it (like the WTI oil demo, whose cause text is retrieved event news, not a bio).

This is the generalization of `oil_demo.py` (one commodity, 8 events) to **several dense series ×
many moves each**, to manufacture a few hundred *event-grounded* text→ts pairs.

## IN
- Dense daily price series from FRED (no key, curl + UA, cache-first):
  Brent `DCOILBRENTEU`, Henry-Hub natural gas `DHHNGSP`, Bitcoin `CBBTCUSD`, EUR/USD `DEXUSEU`.
  (Gold has no working *daily* FRED series — AM/PM fixes discontinued — deferred to a non-FRED add.)
- GDELT DOC 2.0 news (no key), per-move retrieval with backoff, cached to a local per-commodity pool.

## OUT
- `out/flywheel_commodity.jsonl` — ChatML `ts_forecast` text→ts records, `flywheel:true`,
  `dataset:"flywheel_commodity"`, one `commodity` tag per record. Same schema as `flywheel_oil`
  (reuses `core.schema`/`core.chatml`), so the companion loader and every audit pick it up unchanged.

## Pipeline (reuses the oil template end-to-end)
- **S1 ingest** — FRED daily CSV per commodity → `raw/commodity/<fred>.csv`.
- **S2 detect** — `core.detect.detect_robust_multiscale` (W1, z≥2.8, ≥10d apart, all scales) →
  many salient moves per series. Cached `raw/_salient_<key>.json`.
- **S3/S4 retrieve** — per-commodity GDELT query (+ one widening fallback), window
  `[move−4d, cutoff)`, backoff ×3, deduped, cached `raw/_gdelt_<key>.json` (the local news pool).
- **S5/S6 cause** — `core.generate.grounded_describe` distills the leakage-safe **cause** from
  pre-cutoff headlines (no price, no direction, no ungrounded number). Cached per event.
- **S7 gate** — (a) time cutoff = move day 00:00 UTC (move day is the first forecast step ⇒ strictly
  T-1 news); (b) numeric reflection; (c) outcome-direction screen (per-commodity price nouns);
  (d) **entity-tracing faithfulness** (`core.faithfulness.is_faithful`). Fail → retry once → **drop**
  (never emit/ cache empty text). `knowledge_time` = latest retained article.
- **S8 emit** — `TextToTs` → `build_record`/`validate` → jsonl. HIST=10, FUT=5 (short, well within
  Xinyue's ≤512/≤128 ceiling; long history would drown the text signal — the mistake we already hit).

## Leakage (the lifeline — CLAUDE §1.2)
The conditioning text may only carry information dated **before** the forecast origin (the move day).
Enforced programmatically at retrieval (cutoff) **and** re-audited post-hoc by the independent
cross-source `flywheel/audit_leakage.py` (which does not trust this module's own cutoff claim).
Each record carries an honest `knowledge_time`.

## Acceptance criteria
1. Independent QA (`qa_commodity.py`, re-derives Data-Schema §2 — does **not** call `chatml.validate`):
   0 schema errors, 0 leaks over every emitted record.
2. `flywheel/audit_leakage.py` reports the commodity records 100 % clean (kt < origin).
3. Every emitted cause passes the faithfulness gate (entity-traced) — no bio-style generic text.
4. Volume: order of a few hundred moves detected; emitted = whatever survives retrieval + gates
   (honest coverage reported, gaps categorized — no padding).
5. Re-export the 3-arm fine-tune data (`export_finetune_data.py` auto-includes the new dataset) and
   re-run the Colab fine-tune probe (A no-text / B flywheel-text / C shuffled) — the real test of
   "does the event text help downstream". (GPU step is Xinyue's; framework is ready.)

## Non-goals / documented next steps
- **Batch GDELT→BigQuery local event DB** (Xinyue's scale path): v1 uses per-move DOC calls with a
  cached local pool — highest cause quality, rate-friendly at a-few-hundred scale. BigQuery bulk pull
  is the thousands-scale upgrade.
- Gold / equities (need a non-FRED daily source, e.g. Stooq/yfinance) — deferred.
- Multi-scale *trainable* changepoint detector (W1 v2) — the robust-MAD rung ships now.
