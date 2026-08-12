# W5 · Wikipedia point-in-time flywheel — v1 spec

Second flywheel source, generalizing `oil_demo.py` to public-attention series **without GDELT**.
Xinyue's constraints: 配对非交错 · 自动测 spike · 暂不用 GDELT · 先查标注质量再训练.

## Core idea (why it differs from `sources/wiki_pageviews.py`)
The hand source is *seeded*: it trusts a known `event_date` and uses the article's **current** lead
(hindsight → a leakage wrinkle). The flywheel must (a) **auto-detect** the spike from the series
alone and (b) condition on **leakage-safe** text. With no GDELT, the retrieval target is the
article's **own revision as it existed just before the spike** — genuinely "what was knowable then".

This only works on **persistent-entity** articles (existed before attention spiked, e.g. Nvidia,
OpenAI, DeepSeek). **Event-created** articles (born at the event: Turkey–Syria earthquakes,
ChatGPT-at-launch) have no pre-event pageviews and no prior revision — they auto-fail S2's
baseline gate and are reported as an honest gap. The gate *is* the persistence filter (no hand-labeling).

## IN
- Candidate slugs + `event_date` from `raw/wiki/_meta.json` (event_date used ONLY to bound the
  fetch window and to *score* detection — never as model input).
- Public Wikimedia APIs (no key): pageviews REST + MediaWiki revisions/parse.

## Stages
- **S1 ingest** — daily pageviews over `[event_date − PRE(21), event_date + POST(14)]`.
  Cache `flywheel/raw/wiki/<slug>_pv.json`.
- **S2 auto-detect spike** — robust MAD z-score over the window; spike day `D` = argmax views.
  Keep iff: `≥ MIN_PRE(10)` pre-`D` baseline days exist AND `peak / median(pre-baseline) ≥ SPIKE_RATIO(3)`.
  Else drop → reason `no_pre_event_state` (event-created) or `no_clear_spike`.
  Report alignment `|D − event_date|` (detector quality metric, not a gate).
- **S3 point-in-time text** — last revision with timestamp `< D 00:00 UTC` (strict T-1); parse
  section-0 (lead) wikitext → plain first paragraph. `knowledge_time` = revision timestamp.
  Cache `flywheel/raw/wiki/rev/<slug>.json`. No prior revision → drop, reason `no_prior_revision`.
- **S6 annotate** — conditioning text = the point-in-time lead (real text; `text_source=wikipedia_lead_pit`).
  No LLM, no GDELT.
- **S7 gate** — per record: assert revision ts `< D` (leakage); series finite; history/future length OK.
- **S8 emit** — `text→ts` (ts_forecast): history = views **before** `D` (baseline + run-up),
  future = views from `D` onward (peak + decay); origin `D` = first forecast step (mirrors oil's
  "move day = first forecast step"). Reuse `core/`. Flag `flat_context` when baseline std floored.

## OUT
- `out/flywheel_wiki.jsonl` — text→ts records, **0 invalid** (independent validator, not `validate()`).
- `flywheel/raw/wiki/_wiki_trace.json` — per-candidate emitted/dropped + reason + metrics.
- `reports/flywheel_wiki_demo.html` — stage-by-stage demo, honest coverage (X/Y + gap reasons).
- **Annotation-quality (QA) section** (Xinyue's 先查标注质量): per record — leakage assert,
  detection alignment days, spike ratio, lead-relevance (event-keyword overlap), flat_context.
  NO downstream training (that is W6).

## Acceptance
1. Runs cache-first, offline-deterministic; `regress.py` extended to replay it (0 invalid).
2. Every emitted record: revision timestamp strictly `< D` (leakage-clean by construction).
3. Honest coverage: emitted count + each drop reason enumerated (target ≥ ~8 emitted from the
   persistent-entity subset; event-created gaps disclosed).
4. Independent validator (Data-Schema checklist) → 0 errors on the new jsonl.
5. Demo HTML + QA metrics rendered; progress report timeline updated.

## Known limitations to disclose (not hide)
- Pre-spike lead may be generic (not yet mention the event) → weak conditioning; measured as
  lead-relevance, kept honest rather than filtered to inflate quality.
- Flat baseline + large spike → z-target magnitude large (robust floor + model asinh handle it;
  flagged `flat_context`, same regime as cyber).
- "Attention spike" ≠ proven causal driver; and this demo answers "can we build it", not
  "is it useful" (→ W6 companion study).
