# sources/wiki_pageviews.py — thin adapter for Wikimedia pageviews (P0 backbone source).
# Distinctive asset: a *public-attention* time series (daily English-Wikipedia pageviews)
# around a real-world event — a different modality from finance (FRED rates, EDGAR
# fundamentals). Also the cleanest prototype of the team's "data flywheel": a single-modal
# attention series whose spike is explained by a retrieved real event.
#   text->ts : the event article's real lead paragraph -> the pageview series
#   ts->text : the pageview series -> grounded, numerically-verified description of the spike
# Seeded with EVENT articles (the page *is* the event), so the lead text describes the event
# and the pageview spike coincides with it — tight text<->series alignment.
# Reads raw/wiki/ cache (pageviews + page summary; see README to refresh). Public, no API key.
import os, json, re
from core.schema import TsToText      # text->ts moved to flywheel/wiki_demo.py (leakage-clean)
from core.generate import grounded_describe
from core.verify import is_consistent

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "raw", "wiki")
S2T_DIR = os.path.join(RAW, "s2t")
SCALE = 1e3                                   # pageviews -> thousands/day (verifiable round figures)
REL_TOL = 0.01                                # a faithful description of large counts rounds ~1%

# Data-driven: every event in raw/wiki/_meta.json with cached pageviews + summary is built.


def _series(slug):
    """(canonical_title, [(yyyymmdd, views_thousands), ...]) from the cached pageview window."""
    items = json.load(open(os.path.join(RAW, f"{slug}_pv.json")))["items"]
    return [(x["timestamp"][:8], round(x["views"] / SCALE, 1)) for x in items]


def _lead(slug):
    sm = json.load(open(os.path.join(RAW, f"{slug}_summary.json")))
    return sm.get("title", ""), re.sub(r"\s+", " ", sm.get("extract", "")).strip()


def _fmt(d8):
    return f"{d8[:4]}-{d8[4:6]}-{d8[6:8]}"


def _describe(slug, title, ev_date, dates, vals):
    cache = os.path.join(S2T_DIR, f"{slug}.txt")
    if os.path.exists(cache):
        cached = open(cache).read().strip()
        if is_consistent(cached, vals, rel_tol=REL_TOL):
            return cached, "derived_generated", True
    peak = max(vals); peak_date = _fmt(dates[vals.index(peak)])
    facts = (f"- Topic: English Wikipedia article '{title}' (event around {ev_date})\n"
             f"- Metric: daily pageviews, in thousands\n"
             f"- Window {_fmt(dates[0])} to {_fmt(dates[-1])}: start {vals[0]}k, end {vals[-1]}k, "
             f"min {min(vals)}k, max(peak) {peak}k on {peak_date}")
    template = (f"Daily pageviews for '{title}' peaked at about {peak}k on {peak_date}, "
                f"from {vals[0]}k at the start of the window to {vals[-1]}k at the end, "
                f"reflecting the surge of public attention around the {ev_date} event.")
    desc, tag, verified = grounded_describe(
        "You write a concise, factual description for a dataset. Use ONLY the provided numbers; "
        "invent no other figures. Approximate phrasing of the given values is fine.",
        "Facts:\n" + facts + "\n\nIn 2-3 sentences, describe the attention spike (when it peaked "
        "and how high) and connect it to the event. State only the pageview figures given.",
        vals, fallback=(template, "derived_template"), rel_tol=REL_TOL)
    if tag == "derived_generated":
        os.makedirs(S2T_DIR, exist_ok=True)
        open(cache, "w").write(desc)
    return desc, tag, verified


def pairs():
    meta = json.load(open(os.path.join(RAW, "_meta.json")))
    for slug in sorted(meta):
        if not (os.path.exists(os.path.join(RAW, f"{slug}_pv.json"))
                and os.path.exists(os.path.join(RAW, f"{slug}_summary.json"))):
            continue
        m = meta.get(slug, {})
        ev = m.get("event_date", "")
        title, lead = _lead(slug)
        series = _series(slug)
        if len(series) < 5 or not lead:
            continue
        dates = [d for d, _ in series]
        vals = [v for _, v in series]
        d0, d1 = _fmt(dates[0]), _fmt(dates[-1])
        canon = m.get("canonical", title.replace(" ", "_"))
        page_url = f"https://en.wikipedia.org/wiki/{canon}"
        ts_url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                  "en.wikipedia/all-access/all-agents/" + canon + "/daily")
        common = {"dataset": "wiki_pageviews", "source": "Wikimedia pageviews + English Wikipedia",
                  "article": title, "event_date": ev, "text_url": page_url, "ts_url": ts_url}

        # A — text->ts REMOVED (W4 leakage cleanup, 2026-07-08): it conditioned on the article's
        # CURRENT lead (hindsight — the lead describes the whole event incl. its aftermath) and
        # carried no knowledge_time, so the cross-source leakage audit (flywheel/audit_leakage.py)
        # flagged all 61. The leakage-clean text->ts for Wikipedia is flywheel/wiki_demo.py
        # (point-in-time revision strictly before the spike day; knowledge_time = revision
        # timestamp; 0 leak). This hand source now emits ts->text only.

        # B — ts -> text (text_desc): attention series -> grounded, numerically-verified description
        desc, tag, verified = _describe(slug, title, ev, dates, vals)
        yield TsToText(
            user_intro=(f"Daily English Wikipedia pageviews for '{title}' (thousands), {d0} to {d1}. "
                        f"Describe the attention pattern and connect it to the event."),
            series=[{"name": "daily pageviews (thousands)", "values": vals,
                     "unit": "pageviews_per_day_thousands", "freq": "daily"}],
            answer=desc,
            meta={**common, "series_id": f"{slug}_s2t", "direction": "ts_to_text",
                  "grounding": "pageview_series", "numeric_verified": verified},
            text_source=("wiki_grounded_qwen" if tag == "derived_generated" else "wiki_template"),
            is_generated=tag)
