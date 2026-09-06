# W6 · Companion study — "is the manufactured text actually useful?"

The flywheel's demos prove text→ts data can be **built** (leakage-clean, faithful). They do NOT
prove it **helps** downstream forecasting. That is this study's job — and per *The Data-Quality
Illusion* (2510.00866), passing our own quality gates is **not** evidence; only an independent
downstream gain counts (不自证).

This directory is the **executable scaffold**. Everything except the text-reading model runs today
with no GPU; the real base model plugs into one documented interface.

## Design

**Three arms** (separate "the text CONTENT helps" from "just having a text channel helps"):
- **A `no_text`** — series only.
- **B `flywheel_text`** — the manufactured world-knowledge text, correctly paired.
- **C `shuffled_text`** — same texts permuted onto the WRONG series (content decoupled).

**Success criterion:** B must beat **both** A and C. B > A but B ≈ C ⇒ the gain is the channel /
alignment, not the content (this answers the MiGAS relabel critique).

**Leakage discipline** (the result is void without it):
- **Time split** — `train origin < cutoff ≤ test origin`; asserted in `data.time_split` (cut time
  before windowing, 2512.06932).
- **Two named split estimands** — the standard temporal-ID export allows the same persistent entity
  on opposite sides of the cutoff; a second `entity_ood` export purges test entities from train.
  Results must identify which split they use rather than calling both simply "generalization".
- **Target normalization** — targets remain in the history-context-normalized space stored in the
  record; future-window mean/std are never consulted during training or inference.
- **Base-model stratum** — test events are tagged `after_base_cutoff`; the honest gain is on events
  the base model could NOT have memorized (implicit lookahead, 2512.23847).

**Metric:** median MASE (robust to the long tail of near-zero-baseline spike events), with confidence
intervals cluster-bootstrapped by persistent entity/series.

The MiGAS runner trains the same fusion stack in all three arms, defaults to three seeds, and can
write a machine-readable run receipt with `--results-json PATH`.

## Files
- `data.py` — load text→ts into (history_z, future_z, text); time split + leakage assertion.
- `arms.py` — build A/B/C.
- `models.py` — `Forecaster` interface; numeric baselines (`LastValue`, `LinearTrend`, text-blind);
  `TextConditionedStub` (the slot for the real base model).
- `run.py` — end-to-end; prints median-MASE table + leakage check; writes `_companion_result.json`.

## Status (numeric baselines, no GPU)
Runs on the 47 flywheel text→ts. Numeric baselines are text-blind ⇒ **A=B=C by construction**, which
establishes the **no-text floor** (median MASE ~6.5 linear_trend / ~13 last_value on the spike test
events — deliberately hard, that's the headroom text should fill) and proves the plumbing +
leakage discipline.

## First signal — zero-GPU LLMTime probe (`run.py --llm`)
Rather than wait for a GPU, the text arm runs TODAY via the team vLLM (LLMTime-style zero-shot,
`models.LLMTimeForecaster`, cached → offline-reproducible). On the 14 held-out test events:

| arm | median MASE (n=40) | per-event win vs B |
|-----|--------------------|--------------------|
| A no-text | 4.69 | B beats A **24/40 (60%)** |
| **B flywheel-text** | **3.93** | — |
| C shuffled-text | 4.89 | B beats C **20/40 (50%)** |

B beats both A and C on the MEDIAN. But the per-event win rates tell the honest story: **"text helps"
(B<A) is fairly solid (60%), while "content beats channel" (B<C) is a coin-flip per event (50%) — the
median edge is carried by some events, not systematic.** (A smaller n=14 slice looked rosier at
71%/57%; the bigger sample after scale-up is more sober — good.) So a GENERAL LLM can't cleanly
separate *content* from *channel* — which is exactly why a real multimodal base model is needed.
Directional signal, NOT proof: possible implicit lookahead (may have seen these events; z-space
weakens value-memory, not event-memory). The scale-up (47 → 343 flywheel pairs, test 14 → 79) is what
makes even this probe statistically meaningful; the definitive answer is the GPU/base-model run this
scaffold plugs into.

## Closing the loop — S9 v1 (`feedback_s9.py`)
The probe's per-event text gain (MASE_A − MASE_B) IS the "which data helps" signal — zero extra cost.
Profiled over the 40 probed events:
- **by source:** flywheel_wiki (seed-event) **+1.82** > wiki_scaled (multi-peak) +0.47 > oil (n=1) —
  seed-event text is more useful than the generic point-in-time leads of extra spikes (honest: scale-up
  adds volume at slightly lower text quality).
- **by spike size:** large (≥6σ) +1.01 > small +0.10; **by horizon:** long +0.92 > short.
- S9 steering: make more seed-event-like pairs; no clearly harmful bucket yet. This takes ⑤
  (self-evolution) from 0 → a v1, driven by real downstream effect — not our own gates (不自证).

## Implicit-lookahead probe (`probe_lookahead.py`)
The biggest threat to B<A is that a general LLM RECOGNIZES the event rather than reasoning from the
text. Test: a B' arm with named entities anonymized (OpenAI → "a major entity"). Result on 40 events:
gain vs A is +0.76 (B) vs **+0.77 (B', anonymized) — 102% retained**. The gain SURVIVES hiding the
entities ⇒ it is mostly REASONING from text structure, not event recognition, so lookahead is NOT the
main driver. This BOUNDS the lookahead risk (doesn't remove the need for a post-cutoff base-model run).

## Wiring the real base model (when GPU is available)
Subclass `models.Forecaster` (`uses_text=True`) wrapping **Moirai-MoE** (2410.10469) or **Time-MMD /
ChatTime** (2406.08627 / 2412.11376):

```python
class MyBase(Forecaster):
    uses_text = True
    def predict(self, history, text, H):
        return base_model.forecast(history=history, context_text=text, horizon=H)  # GPU
```

Drop it into `run.py`'s model list. No change to split/arms/eval — the harness already guarantees
the time split, the arms, the metric, and the leakage assertion.

## Honest limitations
- **Scale**: 47 flywheel pairs is a scaffold, not a powered experiment. Real conclusions need the
  flywheel scaled up (thousands of pairs) so the A/B/C gaps are statistically meaningful.
- **External replication**: re-run the same 3-arm test on **Time-MMD** — a gain that only shows on
  our own data isn't trustworthy.
- Numeric baselines are intentionally weak; they set the floor, not the ceiling.
