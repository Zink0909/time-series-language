# Datasheet — ts_lang_curation (stage-2 world-knowledge)

Per-source documentation for the curated time-series ↔ text records. One section per source.

**Output format = full Data-Schema (ChatML) records** (per `references/Data Schema.docx`): a Qwen
ChatML transcript with `<stats>len,mean,std</stats> <ts></ts>` prompt spans, z-score-normalized
`timeseries` spans (`raw = z*std + mean`), a `normalization` block, and loss masks. Each pair maps
to a task contract: **ts→text → `text_desc`** (loss on the assistant text), **text→ts →
`ts_forecast`** (loss only on the future suffix, `loss_start = H`). Per-record provenance
(`builder`, `text_source`, `is_generated_text`, `knowledge_time`, source URLs) is preserved.

---

## Source: `fred_fomc`  (ACTIVE — ours)

**Motivation.** Pair official U.S. monetary-policy language with the market's interest-rate
reaction, so the model learns real text ↔ time-series alignment and macro world knowledge.

**Composition.**
- **166 records** = 83 FOMC meetings (2016 → 2026) × 2 directions. (Pre-2016 statements use an
  older HTML layout the extractor doesn't parse yet — ~40 more meetings recoverable.)
- `text→ts` → **`ts_forecast`** (39): FOMC statement (conditioning) + pre-decision 2-year yield
  history → post-decision future; loss only on the future suffix. **History length varies** per
  meeting (20/30/45/60 trading days). ~Half also carry the **10-year yield as a covariate**
  (→ `ts_forecast_covariates`).
- `ts→text` → **`text_desc`** (39): 2-year yield window (±10 days) as a user span + a grounded,
  **Qwen-generated**, numerically-verified 2–3 sentence description (cached in `raw/fomc_s2t/`).
- Text: `text→ts` is real/verbatim (`is_generated_text=real`); `ts→text` is grounded
  LLM-generated (`is_generated_text=derived_generated`, machine-verified).

**Numeric reflection check (quality gate).** Every generated `ts→text` description passes
`core/verify.py`: each decimal it states must match a real series value/stat or a number in the
policy decision (target-rate fractions like `1-1/2` are converted to `1.5`). Loop = generate →
verify → one stricter retry → fall back to the verbatim decision; only verified text is cached.
Current batch: **39/39 pass, 0 fall back, 0 hallucinated figures.** Mirrors the consistency filter
in HORAI (arXiv 2602.05646) and the reflection loop in From-News-to-Forecast (arXiv 2409.17515).

**Leakage handling (per schema).** The decision boundary is the `loss_start` of the target span:
the series is kept whole and loss is computed only on the post-decision future, so the aftermath
is conditioning-safe by construction (no series trimming, no custom cutoff field).

**Sources & license.**
- Text: Federal Reserve FOMC statements, federalreserve.gov (public domain). Per-record `text_url`.
- Series: FRED `DGS2` (2y) + `DGS10` (10y covariate), public. Per-record `ts_url`.

**Collection / preprocessing.**
- FOMC statements fetched per meeting, body extracted (paragraphs between the release line and
  the voting roster). FRED `DGS2` fetched once (2018–2025) and windowed per meeting; weekend/
  holiday blanks dropped. Cached under `raw/` for reproducibility. Build: `python build.py --source fred_fomc`.

**Known limitations / TODO.**
- `ts→text` text is **Qwen-generated** (grounded + every stated number machine-verified against
  the series). Tagged `derived_generated` and down-weightable — not native text.
- `DGS2`/`DGS10` are yields (not revised), so no ALFRED vintage needed; revised macro series would.

**Provenance fields on every record.** `builder`, `source`, `series_id`, `fred_series`,
`event_date`, `text_source`, `text_url`, `ts_url`, `knowledge_time` (text→ts).

---

## Source: `sec_edgar`  (ACTIVE — ours, #13 on the sheet)

**Motivation.** SEC EDGAR's distinctive asset (vs FRED macro rates and FNSPID news) is
**XBRL structured fundamentals** — multi-year company financials as a native time series,
paired with the company's own 10-K text. Teaches the model to align reported fundamentals
with management's narrative.

**Composition.**
- **387 records** over **382 large-caps** (data-driven: any company with a cached XBRL revenue file).
- `ts→text` → **`text_desc`** (30): annual revenue (USD billions, ~7 fiscal years) → grounded,
  **Qwen-generated**, verified description. **Mix of univariate (revenue) and multivariate
  (revenue + net income, 13 records)** per Xinyue's "mix uni/multivariate" note.
- `text→ts` → **`ts_forecast`** (4: AAPL, MSFT, DIS, NKE): real **10-K MD&A** revenue-driver text
  + prior-year history → latest year. One (DIS) adds **net income as a covariate**
  (→ `ts_forecast_covariates`). Emitted only where clean MD&A sentences are found (no fabrication).
- Series scaled to USD billions so generated figures are machine-verifiable.

**Sources & license.**
- Series: SEC EDGAR XBRL `companyconcept` / `companyfacts` (`us-gaap` revenue), public, no key.
  Per-record `ts_url` (companyfacts API), `cik`.
- Text: the company's 10-K (MD&A), federal filing, public domain. Per-record `text_url` (filing).
- SEC fair-access requires a descriptive `User-Agent` when fetching (set at fetch time, cached).

**Collection / preprocessing.**
- XBRL revenue facts filtered to full fiscal years (form 10-K, `fp=FY`, ~350–380-day period),
  deduped by fiscal-year-end, last 7 years, scaled to USD billions.
- 10-K HTML fetched per company; MD&A located (last "Management's Discussion and Analysis"),
  revenue-driver sentences extracted (state a change + a reason). Cached under `raw/sec/`.
- `ts→text` description passes the same numeric reflection check as `fred_fomc`
  (`core/verify.py` via `core/generate.py`); only verified text cached in `raw/sec/s2t/`.

**Known limitations / TODO.**
- `ts→text` scales to any company with XBRL revenue (cheap to add more). `text→ts` is gated by
  **MD&A extraction** — only 4/17 companies with cached 10-Ks yield clean revenue-driver sentences
  (filers phrase results very differently); a more general extractor is the next iteration.
- NVDA dropped: its `RevenueFromContract…`/`Revenues` XBRL tags are incomplete for recent years.
- Annual granularity (~7 points/series); quarterly would lengthen series at the cost of more noise.

---

## Source: `wiki_pageviews`  (ACTIVE — ours, P0 backbone)

**Motivation.** A **public-attention** time series (daily English-Wikipedia pageviews) around a
real-world event — a modality distinct from finance. Also the cleanest prototype of the team's
**data flywheel**: a single-modal attention series whose spike is explained by the driving event.

**Composition.**
- **134 records** over **73 event articles** (earthquakes, Olympics/World Cup, eclipses, ChatGPT,
  SVB collapse, JWST, films, hurricanes, assassinations, deaths, product launches, …) many domains.
- `text→ts` → **`ts_forecast`** (14): article real **lead paragraph** + the attention *rise*
  (history through the peak) → the *decay* (future). Events created at the spike (history < 3)
  are skipped for this direction.
- `ts→text` → **`text_desc`** (20): the pageview series + a grounded, **Qwen-generated**,
  verified description of the attention pattern (peak value/date, link to the event).
- **Window length varies** per event (±20/30/45 days) for length diversity.
- Series scaled to **pageviews/day in thousands** (1 dp).

**Sources & license.**
- Series: Wikimedia REST pageviews API (`per-article` daily), public, no key. Per-record `ts_url`.
- Text: English Wikipedia article lead via REST `page/summary` (CC BY-SA). Per-record `text_url`.
- Wikimedia policy requires a descriptive `User-Agent` when fetching (set at fetch time, cached).

**Collection / preprocessing.**
- Per event: fetch the page summary (canonical title + lead extract) and daily pageviews for
  ±30 days around the event date; cache under `raw/wiki/`.
- `ts→text` uses the shared grounded-generation + reflection path (`core/generate.py`), with a
  **relative numeric tolerance** (`rel_tol=0.01`, `core/verify.py`) since a faithful description
  of large counts rounds (e.g. "about 670k"). Result: **5/5 verified, 0 hallucinations.**

**Known limitations / TODO.**
- Scales easily by adding events (both directions generalize); curated event list for now, the
  flywheel testbed would auto-detect spikes in arbitrary article series.
- Pattern mix (measured): sharp impulse+decay (breaking news), anticipation ramp + multi-peak
  plateau (scheduled multi-day events), double peak (multi-stage events) — not all single spikes.
- Lead text reflects the article's current (post-event) state — fine for world-knowledge.

---

## Source: `usgs_quakes`  (ACTIVE — ours, geophysics)

**Motivation.** A physically-meaningful intrinsic series — the **aftershock sequence** (daily
M3.5+ counts after a mainshock, an Omori-law decay) — paired with the mainshock's text. First
non-finance domain beyond Wikipedia; distinct from Faisal's weather/storm sources.

**Composition.**
- **86 records** over **43 mainshocks** (M6.8+, 2016–2025, with a real aftershock sequence).
- `text→ts` → **`ts_forecast`** (43): real USGS mainshock description (verbatim title + magnitude,
  place, date, depth) + the first 5 days of aftershocks → forecast the rest of the decay.
- `ts→text` → **`text_desc`** (43): the aftershock-count series → grounded, **Gemma-generated**,
  numerically-verified description (onset, peak, taper, total). 43/43 verified.

**Sources & license.**
- Series + text: **USGS FDSN event API** (`earthquake.usgs.gov/fdsnws/event/1/query`), public
  domain, **no key**. Per-record `event_id`, USGS event-page `text_url`/`ts_url`.

**Collection / preprocessing.**
- Query mainshocks (M6.8+); for each, query aftershocks within 250 km and 21 days (M3.5+) and
  bin to daily counts from day 0. Keep events with a real sequence (≥6 days, ≥20 aftershocks).
  Cached in `raw/usgs/events.json`.

**Known limitations / TODO.**
- ~462 M6.5+ mainshocks exist 2015–2025 (thousands at M6.0+) → scales to **thousands of pairs**;
  v1 uses 43.
- Counts are integers, so the numeric check (decimals only) is weaker here; the tight prompt +
  template fallback keep counts faithful. Text could be enriched with USGS impact text / Wikipedia.

---

## Source: `cyber_epss`  (ACTIVE — ours, #16 on the sheet, cybersecurity)

**Motivation.** A vulnerability's **EPSS exploit-probability trajectory** (daily %, which spikes
when the CVE becomes actively exploited) paired with the CVE's official text. New domain
(cybersecurity); a clean risk-signal ↔ advisory-text alignment.

**Composition.**
- **160 records** over **80 CVEs** (recently-added CISA KEV entries with real EPSS dynamics,
  ≥5-point rise in a 30-day window).
- `text→ts` → **`ts_forecast`** (80): real **CISA KEV** text (vendor/product, vuln name,
  short description) + the early EPSS days → forecast the exploit-probability rise.
- `ts→text` → **`text_desc`** (80): the EPSS series → grounded, **Gemma-generated**,
  numerically-verified description of how the exploitation risk evolved. 80/80 verified.

**Sources & license.**
- Text + series: **CISA KEV** (`cisa.gov`, public domain) + **FIRST EPSS** API
  (`api.first.org`), both public, no key. Per-record `cve`, NVD `text_url`, KEV `ts_url`.

**Collection / preprocessing.**
- Take the most recently-added KEV CVEs; for each, fetch the EPSS 30-day time-series (scaled to
  percent) and keep those with a real rise (≥5 pp). Cached in `raw/cyber/events.json`.

**Known limitations / TODO.**
- The EPSS API gives only the **last 30 days**, so dynamics are visible only for *recently-added*
  CVEs (older ones are flat-high today). Full historical trajectories (spike at each CVE's own
  disclosure) need the **EPSS daily archive** — that unlocks ~1,600 KEV pairs / tens of thousands
  across all CVEs.
- Generated descriptions sometimes cite an intermediate value rather than the peak (grounded, but
  may understate the rise).

---

## Source: `fnspid`  (DEMO ONLY — owned by Faisal)

1 record (AAPL 2020-07-30 earnings, `ts_forecast`), kept only to prove the shared core handles
>1 source. Not maintained here; FNSPID is Faisal's on the team sheet.
