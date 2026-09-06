#!/usr/bin/env python3
"""Data-flywheel open-loop v1 — Wikipedia public attention (second source, NO GDELT).

Generalizes oil_demo.py to attention series. Xinyue's constraints: paired (non-interleaved),
auto-detect the spike, no GDELT, check annotation quality before any training.

Unlike sources/wiki_pageviews.py (seeded: trusts a known event_date, uses the CURRENT lead =
hindsight), this manufactures LEAKAGE-SAFE text->ts pairs from a series alone:

  S1 ingest   daily pageviews over [event-21, event+14]        -> flywheel/raw/wiki/<slug>_pv.json
              (event_date bounds the fetch + scores detection only; never a model input)
  S2 detect   robust-MAD spike day D = argmax; keep iff >=MIN_PRE pre-D baseline days AND
              peak/median(baseline) >= SPIKE_RATIO. No baseline -> event-created article -> drop.
  S3 text     the article's OWN revision timestamped < D 00:00 UTC (strict T-1); parse the lead
              -> "what the page said before the spike"        -> flywheel/raw/wiki/rev/<slug>.json
  S6 annotate conditioning text = that point-in-time lead (real text, no LLM, no GDELT)
  S7 gate     assert revision ts < D (leakage-clean by construction); finite series; lengths ok
  S8 emit     text->ts (ts_forecast): history = views before D (baseline+run-up),
              future = views from D (peak+decay); D = first forecast step   -> out/flywheel_wiki.jsonl

Only PERSISTENT-ENTITY articles (existed before attention spiked: Nvidia, OpenAI, DeepSeek...) have
a pre-spike baseline AND a prior revision. EVENT-CREATED articles (born at the event: earthquakes,
ChatGPT-at-launch) auto-fail S2's baseline gate and are reported as an honest gap -- the gate IS the
persistence filter (no hand-labeling). See flywheel/WIKI_DEMO_SPEC.md.

Cache-first (offline re-runs deterministic). Run:
  micromamba run -n ts-language python flywheel/wiki_demo.py [--refresh]
"""
import os, sys, json, re, time, argparse, datetime as dt
import urllib.request, urllib.parse
from statistics import median, pstdev, mean

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from core.schema import TextToTs, build_record, validate           # noqa: E402
from core.writer import write_jsonl                                 # noqa: E402
from core.chatml import ROBUST_COEF                                  # noqa: E402

META = os.path.join(PKG, "raw", "wiki", "_meta.json")               # candidate slugs + event_date
FW_WIKI = os.path.join(HERE, "raw", "wiki")                         # flywheel-only cache (separate!)
REV_DIR = os.path.join(FW_WIKI, "rev")
TRACE = os.path.join(FW_WIKI, "_wiki_trace.json")
OUT = os.path.join(PKG, "out", "flywheel_wiki.jsonl")

PRE, POST = 21, 14                  # S1 fetch window around event_date (bound only)
MIN_PRE = 10                        # S2: need >= this many pre-D baseline days (persistence gate)
SPIKE_RATIO = 3.0                   # S2: peak must be >= 3x the pre-D baseline median
MIN_HIST, MIN_FUT = 5, 5           # S8: usable history / horizon
SCALE = 1e3                         # pageviews -> thousands/day (verifiable round figures)
UA = "ts-language research (majiaju89@gmail.com)"
PV_URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
          "en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}")
API = "https://en.wikipedia.org/w/api.php"


# ---------- tiny HTTP ----------
def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "ignore")


def _d8(date):
    return date.replace("-", "")


def _shift(date, days):
    y, m, d = map(int, date.split("-"))
    return (dt.date(y, m, d) + dt.timedelta(days=days)).strftime("%Y%m%d")


# ---------- S1 · ingest wide pageview window ----------
def _pv(slug, canon, event_date, refresh):
    cache = os.path.join(FW_WIKI, f"{slug}_pv.json")
    if os.path.exists(cache) and not refresh:
        items = json.load(open(cache))
    else:
        url = PV_URL.format(article=urllib.parse.quote(canon, safe=""),
                            start=_shift(event_date, -PRE), end=_shift(event_date, POST))
        try:
            items = json.loads(_get(url)).get("items", [])
        except Exception:
            items = []
        os.makedirs(FW_WIKI, exist_ok=True)
        json.dump(items, open(cache, "w"))
        time.sleep(0.4)
    # [(yyyymmdd, views_thousands)]
    return [(x["timestamp"][:8], round(x["views"] / SCALE, 3)) for x in items]


# ---------- S2 · auto spike detection ----------
def _mad_z(vals, x):
    m = median(vals)
    mad = median([abs(v - m) for v in vals]) or 1e-9
    return 0.6745 * (x - m) / mad


def _detect(series, event_d8):
    """Return (idx_D, metrics) or (None, reason). D = peak day; require a real pre-D baseline."""
    if len(series) < MIN_HIST + MIN_FUT:
        return None, {"reason": "series_too_short"}
    vals = [v for _, v in series]
    pk = vals.index(max(vals))
    pre = vals[:pk]
    if len(pre) < MIN_PRE:
        return None, {"reason": "no_pre_event_state"}          # event-created / no baseline
    base = median(pre) or 1e-9
    ratio = vals[pk] / base
    if ratio < SPIKE_RATIO:
        return None, {"reason": "no_clear_spike", "spike_ratio": round(ratio, 2)}
    d8 = series[pk][0]
    align = abs((dt.datetime.strptime(d8, "%Y%m%d") - dt.datetime.strptime(event_d8, "%Y%m%d")).days)
    return pk, {"spike_date": f"{d8[:4]}-{d8[4:6]}-{d8[6:8]}", "spike_ratio": round(ratio, 2),
                "mad_z_peak": round(_mad_z(pre + [vals[pk]], vals[pk]), 1),
                "baseline_median_k": round(base, 3), "peak_k": round(vals[pk], 3),
                "detect_alignment_days": align}


# ---------- S3 · point-in-time revision lead (strict T-1) ----------
def _strip_wikitext(wt):
    s = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S | re.I)
    for _ in range(12):                                   # nested {{templates}}
        new = re.sub(r"\{\{[^{}]*\}\}", "", s)
        if new == s:
            break
        s = new
    for _ in range(12):                                   # [[File:...]] / [[Image:...]]
        new = re.sub(r"\[\[(?:File|Image):[^\[\]]*\]\]", "", s, flags=re.I)
        if new == s:
            break
        s = new
    s = re.sub(r"\[\[[^\[\]|]*\|([^\[\]]*)\]\]", r"\1", s)   # [[a|b]] -> b
    s = re.sub(r"\[\[([^\[\]]*)\]\]", r"\1", s)              # [[a]] -> a
    s = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", s)
    s = re.sub(r"\[https?://[^\s\]]+\]", "", s)
    s = re.sub(r"\{\|.*?\|\}", "", s, flags=re.S)           # {| tables |}
    s = s.replace("'''", "").replace("''", "")
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _lead_para(wt, cap=520):
    clean = _strip_wikitext(wt)
    para = next((ln.strip() for ln in clean.split("\n") if len(ln.strip()) >= 40), "")
    out = ""
    for sent in re.split(r"(?<=[.!?]) ", para):
        if out and len(out) + len(sent) > cap:
            break
        out += sent + " "
    return out.strip()


def _revision(slug, title, spike_date, refresh):
    """Last revision timestamped strictly before the spike day (00:00 UTC), + its lead paragraph."""
    cache = os.path.join(REV_DIR, f"{slug}.json")
    if os.path.exists(cache) and not refresh:
        return json.load(open(cache))
    q = urllib.parse.urlencode({
        "action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
        "rvstart": spike_date + "T00:00:00Z", "rvdir": "older",
        "rvprop": "timestamp|ids", "format": "json", "formatversion": 2})
    res = {"revid": None, "timestamp": None, "lead": None}
    try:
        page = json.loads(_get(f"{API}?{q}"))["query"]["pages"][0]
        rev = page["revisions"][0]                          # KeyError -> no prior revision
        q2 = urllib.parse.urlencode({"action": "parse", "oldid": rev["revid"], "prop": "wikitext",
                                     "section": 0, "format": "json", "formatversion": 2})
        wt = json.loads(_get(f"{API}?{q2}"))["parse"]["wikitext"]
        res = {"revid": rev["revid"], "timestamp": rev["timestamp"], "lead": _lead_para(wt)}
    except Exception:
        pass
    os.makedirs(REV_DIR, exist_ok=True)
    json.dump(res, open(cache, "w"), indent=1)
    time.sleep(0.4)
    return res


def _flat_context(hist):
    m = mean(hist)
    return pstdev(hist) < ROBUST_COEF * max(abs(m), 1.0)


# ---------- S8 · build ----------
def pairs_and_trace(refresh=False):
    meta = json.load(open(META))
    examples, trace = [], []
    for slug in sorted(meta):
        m = meta[slug]
        title, canon, ev = m.get("title", ""), m.get("canonical", ""), m.get("event_date", "")
        t = {"slug": slug, "title": title, "event_date": ev, "emitted": False, "reason": ""}
        trace.append(t)
        if not (title and canon and ev):
            t["reason"] = "meta_incomplete"
            continue
        series = _pv(slug, canon, ev, refresh)
        if len(series) < MIN_HIST + MIN_FUT:
            t["reason"] = "no_pageviews"
            continue
        pk, info = _detect(series, _d8(ev))
        t.update(info)
        if pk is None:
            continue                                        # reason already set by _detect
        rev = _revision(slug, title, info["spike_date"], refresh)
        if not rev.get("lead") or not rev.get("timestamp"):
            t["reason"] = "no_prior_revision"
            continue
        # S7 leakage assert: revision strictly before the forecast origin D
        if rev["timestamp"] >= info["spike_date"] + "T00:00:00Z":
            t["reason"] = "revision_not_before_spike"
            continue
        vals = [v for _, v in series]
        hist, fut = vals[:pk], vals[pk:]
        if len(hist) < MIN_HIST or len(fut) < MIN_FUT:
            t["reason"] = "insufficient_history_or_horizon"
            continue
        flat = _flat_context(hist)
        ex = TextToTs(
            user_text=(f"{rev['lead']} Given this background (as known before {info['spike_date']}) "
                       f"and the recent daily Wikipedia attention, forecast the pageviews."),
            history=hist, future=fut,
            series_name=f"daily English-Wikipedia pageviews for '{title}' (thousands)",
            unit="pageviews_per_day_thousands", freq="daily",
            meta={"dataset": "flywheel_wiki",
                  "source": "Wikimedia pageviews + English-Wikipedia point-in-time revision",
                  "series_id": f"{slug}_flywheel_t2s", "article": title, "event_date": ev,
                  "direction": "text_to_ts", "spike_date": info["spike_date"],
                  "spike_ratio": info["spike_ratio"], "detect_alignment_days": info["detect_alignment_days"],
                  "revision_id": rev["revid"], "leakage_cutoff": "prior_day_revision",
                  "flat_context": flat, "flywheel": True,
                  "text_url": f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}",
                  "ts_url": PV_URL.format(article=urllib.parse.quote(canon, safe=""),
                                          start=_shift(ev, -PRE), end=_shift(ev, POST))},
            text_source="wikipedia_lead_pit", is_generated="real",
            knowledge_time=rev["timestamp"])
        examples.append(ex)
        t.update({"emitted": True, "revision_timestamp": rev["timestamp"],
                  "rev_age_days": (dt.datetime.strptime(info["spike_date"], "%Y-%m-%d")
                                   - dt.datetime.strptime(rev["timestamp"][:10], "%Y-%m-%d")).days,
                  "flat_context": flat, "hist_len": len(hist), "fut_len": len(fut),
                  "lead_preview": rev["lead"][:160]})
    return examples, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch pageviews + revisions (else cache)")
    a = ap.parse_args()
    examples, trace = pairs_and_trace(a.refresh)
    recs = [build_record(example) for example in examples]
    invalid = [validate(record) for record in recs if validate(record)]
    if invalid:
        raise ValueError(f"canonical flywheel examples emitted invalid ChatML: {invalid[:3]}")
    n = write_jsonl(recs, OUT)
    json.dump(trace, open(TRACE, "w"), indent=1, ensure_ascii=False)
    emitted = sum(1 for t in trace if t["emitted"])
    gaps = {}
    for t in trace:
        if not t["emitted"]:
            gaps[t["reason"]] = gaps.get(t["reason"], 0) + 1
    gap_str = ", ".join(f"{k}={v}" for k, v in sorted(gaps.items()))
    print(f"flywheel_wiki: {n} text->ts pairs -> out/flywheel_wiki.jsonl  "
          f"({emitted}/{len(trace)} candidates emitted; gaps: {gap_str})")


if __name__ == "__main__":
    main()
