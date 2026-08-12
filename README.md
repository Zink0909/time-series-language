# Time-Series & LLM — World-Knowledge Data (stage-2)

Curating real-world **"time-series ↔ text"** data to train the **stage-2 (world-knowledge)** phase
of a time-series + language unified model.

> Model pipeline = `① alignment → ② world-knowledge (stage-2) → ③ SFT`.
> The model team currently skips ②; this project builds the data that fills it.

**Team (Discord "MMTSFM - Data"):** Charon = Xinyue (coordinator) · Defu Cao (source sheet) ·
Faisal/`wheretfiam` (curation) · Oliver/`olv0909` = me. **All coordination/deliverables happen in
Notion** (the "Process Protocol" pages) — no shared code repo.

---

## Status — 2026-06-27

- **Unified pipeline** (`ts_lang_curation/`): one shared core + a thin adapter per source.
- **5 active sources (ours)** across 5 domains — **934 records, 0 invalid**:

  | source | # | pairing | domain |
  |---|---|---|---|
  | `sec_edgar` (#13) | 387 | 10-K MD&A ↔ XBRL revenue / net income (382 cos) | finance fundamentals |
  | `fred_fomc` (#9) | 166 | FOMC statement ↔ 2y Treasury yield (+10y covariate) | macro |
  | `cyber_epss` (#16) | 160 | CVE text ↔ EPSS exploit-probability rise (80 CVEs) | cybersecurity |
  | `wiki_pageviews` (#21) | 134 | event article ↔ daily pageview attention (73 events) | public attention |
  | `usgs_quakes` | 86 | mainshock text ↔ aftershock decay (43 quakes, USGS ComCat) | geophysics |
  | `fnspid` | 1 | demo only (Faisal owns FNSPID) | — |

- **Output = full Data-Schema (ChatML)** records: `text_desc` (ts→text) + `ts_forecast` /
  `ts_forecast_covariates` (text→ts), z-score spans + loss masks. Univariate + multivariate mix,
  varied window lengths.
- **Quality:** every generated `ts→text` passes a numeric **reflection** check (+ integer-series
  and year-attribution checks); shipped numeric-error rate **0%**.
- **Phase:** building a **Dev Set** (≈80 samples/source) for the team to review and **freeze the
  format** before scaling. See `dev_set_review/`.

---

## Directory structure

```
Time-Series&LLM/
├── README.md                    📌 project hub (this file)
├── environment.yml              🔧 env spec → `micromamba create -f environment.yml` (env: ts-language)
│
├── references/                  📚 inputs from the team (read-only)
│   ├── 2505.14683v3.pdf            BAGEL paper (model blueprint)
│   ├── Data Schema.docx            official ChatML data-format spec
│   ├── Qwen vLLM (…).docx           team LLM API details
│   └── source-catalogs/            141-source catalog + Defu's live sheet (.xlsx)
│
├── reports/                     📊 human-facing reports (HTML)
│   ├── 项目进展报告.html             progress report (updated at the end of each work day)
│   ├── 开会讲点_*.html               meeting talking points (CN/EN)
│   └── dev_set_preview*.html        dev-set preview (CN + EN)
│
├── notion_pages/                📝 Notion "Process Protocol" page drafts (one per source)
│   └── notion_<source>_page.md
│
├── discord_exports/             💬 team channel export + dce.sh wrapper
│
├── dev_set_review/              ⭐ team handoff: 5 × *_dev.jsonl + English preview html
│
└── ts_lang_curation/            ⭐ the pipeline
    ├── README.md · DATASHEET.md · requirements.txt · build.py
    ├── core/        🔧 shared engine: schema · chatml · compress · generate · verify · writer
    ├── sources/     🔌 one adapter per source: fred_fomc · sec_edgar · wiki_pageviews ·
    │                   usgs_quakes · cyber_epss (+ fnspid demo)
    ├── raw/         💾 cached downloads        out/   📤 built .jsonl (full scale)
    └── dev/         🧪 ≈80-sample dev sets     scripts/  scale helpers
```

**Data flow:** `raw/ → sources/<x>.py (adapter) → core/ (z-score + ChatML + loss masks + validate) → out/<x>.jsonl`
**Add a source** = one new `sources/<name>.py` yielding `TsToText` / `TextToTs`; `core/` is untouched.

---

## Setup

```
micromamba create -f environment.yml          # builds the `ts-language` env (Python 3.11 + openai)
micromamba run -n ts-language python ts_lang_curation/build.py --list
```

Runtime dependency is just `openai` (the pipeline calls the team's OpenAI-compatible vLLM API —
Qwen 35B-a3b on :8003 / Gemma 31B on :8004). See `ts_lang_curation/README.md` for details.

---

## Decisions (resolved with the team)

- ✅ **Output = full Data-Schema (ChatML)** records (z-score spans, `loss_start`, `text_loss_char_ranges`).
- ✅ **Both directions**: `text→ts` (`ts_forecast`) + `ts→text` (`text_desc`); interleaved later.
- ✅ **Loss masks, not trimming**: text→ts computes loss only on the future suffix; ts→text only on the text.
- ✅ **Mix** univariate + multivariate (covariates); **varied, domain-aware window lengths**.
- ✅ Generated text kept as a **soft gate** (generate → numeric-verify → retry → template fallback); tagged.
- ✅ **All deliverables in Notion** (no shared repo). One page per source under "Process Protocol".

---

## Next

- **Dev Set** (≈80/source) under team review to **freeze the format** (target: this weekend).
- After freeze, **scale** is mostly window-length / multivariate-ratio tuning (infra is data-driven).
- Open unclaimed P0 sources on Defu's sheet (wildfire ICS-209/NIFC, energy ERCOT/CAISO) if assigned.

---

## Log

- **2026-06-27** — Numeric-check extended (integer-series + year-attribution); shipped error 0%.
  Built Dev Sets (5 × ~80) + English preview for the team format-freeze. Project cleanup
  (removed superseded caches / old panorama / review screenshots; added `notion_pages/`,
  `dev_set_review/`, `environment.yml`; env renamed `ts-lang-curation` → **`ts-language`**).
- **2026-06-26** — Scaled toward ceiling (EDGAR→387, FRED→166, Wiki→134); added two non-finance
  sources **`usgs_quakes`** (geophysics) and **`cyber_epss`** (cybersecurity). Team swapped the
  LLM (Qwen → Gemma 31B, then a3b back on :8003); `compress.py` made model-aware. **934 records.**
- **2026-06-25** — Migrated all sources to the **full Data-Schema (ChatML)** output per Xinyue's
  Notion review (`core/chatml.py`); mapped directions to `text_desc` / `ts_forecast`; added
  covariates / multivariate / varied windows.
- **2026-06-22** — Built `wiki_pageviews`; factored shared `core/generate.py`.
- **2026-06-21** — Numeric "reflection" check (`core/verify.py`); read 3 reference papers.
- **2026-06-20** — Joined team Discord; built the unified pipeline (core + adapters); switched
  active source FNSPID → FRED+FOMC; wired the team's vLLM API.
- **2026-06-18/19** — First meetings w/ Xinyue; received the Data Schema + 141-source catalog;
  scoped stage-2 paired data, v1 = single-turn.
