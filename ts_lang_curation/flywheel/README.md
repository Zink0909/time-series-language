# Data Flywheel (open-loop v1 — WTI crude demo)

> **Governance status:** research-only / blocked from release. The cached WTI series came through
> FRED. Keep this demo for offline reproducibility and migration testing; do not include its output
> in training or redistribution until the series leg is rebuilt from an approved original source.

Proves the flywheel: manufacture `text→ts` pairs for a series that has **no pre-paired text**, by
retrieving the real events that explain its moves and grounding them with an LLM. Reuses `../core/`.

Pipeline (all stages **in this file**, cache-first, reproducible offline):
S1 ingest (FRED DCOILWTICO, auto-fetch) → S2 detect salient moves (top-8 |ret|≥6%, ≥10d apart) →
S3 leakage-aware query (pre-event window only) → S4 GDELT DOC retrieval → S5/S6 LLM cause
extraction → S7 gate (cutoff audit + numeric check + outcome-direction screen; drop on failure) →
S8 emit ChatML.

**Leakage policy**: the salient move day is the *first forecast step*, so retrieval ends at the
event day 00:00 UTC (`--cutoff prior_day`, default; strictly T-1 news). Cached articles from
older, leakier windows are re-filtered at build time and counted in `_trace.json`
(`dropped_for_leakage`). `knowledge_time` = latest retained article. A generated cause is dropped
(never emitted empty) if it states an un-grounded number or narrates the price move itself.

Run:  `micromamba run -n ts-language python flywheel/oil_demo.py`  → `out/flywheel_oil.jsonl`
Flags: `--refresh-retrieval` (re-query GDELT for every event; gaps are retried on every run),
`--cutoff same_day` (ablation only, tagged in meta).
Demo: `reports/flywheel_demo.html`   ·   Proposal: `reports/data_flywheel_proposal*.html`

Caches: `raw/DCOILWTICO.csv` (S1) · `raw/_salient.json` (S2) · `raw/_gdelt.json` (S4) ·
`raw/cause/<date>.json` (S5/S6, keyed on the headline set — changes to retrieval auto-invalidate)
· `raw/_trace.json` (full per-event audit: kept/dropped articles, cause, emit reason).
GDELT rate-limits (~1 req/5s; be patient — empty windows refill on later runs).
