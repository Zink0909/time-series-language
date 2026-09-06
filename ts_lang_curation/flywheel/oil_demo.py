#!/usr/bin/env python3
"""Data-flywheel open-loop v1 — WTI crude oil. Full S1–S8, reproducible end-to-end.

Proves the flywheel on a series we have NO pre-paired text for: it manufactures text→ts pairs by
RETRIEVING the real events that explain each salient move and grounding them with an LLM.

  S1 ingest    WTI daily price (FRED DCOILWTICO; fetched if missing) -> flywheel/raw/DCOILWTICO.csv
  S2 detect    top-K salient daily moves (|ret| >= 6%, >=10d apart)  -> flywheel/raw/_salient.json
  S3 query     entity keywords + a PRE-EVENT time window (leakage-aware by construction)
  S4 retrieve  GDELT DOC API articles strictly before the cutoff     -> flywheel/raw/_gdelt.json
  S5/S6 ground LLM reads pre-event headlines -> the CAUSE            -> flywheel/raw/cause/<date>.json
  S7 gate      leakage cutoff + numeric check + outcome screen + entity-tracing faithfulness (W3);
               every entity in the cause must trace to a headline, else drop
  S8 emit      build + validate text→ts (reuse core schema)          -> out/flywheel_oil.jsonl

Leakage policy (S7): the salient move day is the FIRST forecast step, so only articles seen
before the cutoff may condition the forecast. Default cutoff "prior_day" = event day 00:00 UTC
(strictly T-1 news); "same_day" (event day 23:59) exists for ablation only and is tagged in meta.
knowledge_time = the latest retained article. The generated cause is dropped if it states any
number not grounded in the headlines, or narrates the price move itself (direction words).

All stages are cache-first (offline re-runs are deterministic). Run:
  micromamba run -n ts-language python flywheel/oil_demo.py [--refresh-retrieval] [--cutoff same_day]
"""
import os, csv, json, sys, re, time, hashlib, argparse, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
RAW = os.path.join(HERE, "raw")
CAUSE_DIR = os.path.join(RAW, "cause")
PRICES_CSV = os.path.join(RAW, "DCOILWTICO.csv")
SALIENT_JSON = os.path.join(RAW, "_salient.json")
GDELT_JSON = os.path.join(RAW, "_gdelt.json")

from core.schema import TextToTs, build_record, validate
from core.writer import write_jsonl
from core.generate import grounded_describe
from core.faithfulness import is_faithful      # W3: entity-tracing faithfulness gate
from core.leak_screen import forecast_leak, future_year_leak   # content-layer soft-leak screens (shared)

HIST, FUT = 10, 5
RET_MIN, SPACING, TOP_K = 0.06, 10, 8       # S2: top-K |daily return| moves, >=SPACING rows apart
LOOKBACK_DAYS = 4                            # S3: retrieval window = [event - LOOKBACK, cutoff)
QUERIES = [                                  # S3 query expansion: try narrower first, widen on empty
    '("crude oil" OR "oil prices" OR OPEC) sourcelang:english',
    '("oil price" OR "crude oil" OR "oil demand" OR OPEC OR petroleum) sourcelang:english',
]
MAX_ARTICLES = 8
UA = "ts-language research (majiaju89@gmail.com)"
FRED_URL = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"
            "&cosd=2021-06-01&coed=2024-12-31")


# ---------- S1 · ingest ----------
def _fetch_prices():
    req = urllib.request.Request(FRED_URL, headers={"User-Agent": UA})
    data = urllib.request.urlopen(req, timeout=45).read()
    open(PRICES_CSV, "wb").write(data)


def _prices():
    if not os.path.exists(PRICES_CSV):
        _fetch_prices()
    rows = []
    for r in csv.reader(open(PRICES_CSV)):
        if not r or not r[0][:4].isdigit() or len(r) < 2 or not r[1].strip() or r[1] == ".":
            continue
        rows.append((r[0], float(r[1])))
    return rows


# ---------- S2 · salient detection (reproduces _salient.json exactly) ----------
def _detect(rows):
    if os.path.exists(SALIENT_JSON):
        return json.load(open(SALIENT_JSON))
    cand = [(i, rows[i][0], rows[i][1] / rows[i - 1][1] - 1) for i in range(1, len(rows))]
    cand = [c for c in cand if abs(c[2]) >= RET_MIN]
    picked = []
    for i, d, r in sorted(cand, key=lambda c: -abs(c[2])):        # greedy by magnitude
        if len(picked) < TOP_K and all(abs(i - j) >= SPACING for j, _, _ in picked):
            picked.append((i, d, r))
    picked.sort()
    out = [{"date": d, "idx": i, "ret": r} for i, d, r in picked]
    json.dump(out, open(SALIENT_JSON, "w"), indent=1)
    return out


# ---------- S3/S4 · leakage-aware retrieval ----------
def _cutoff(date, policy):
    """Article-timestamp cutoff (GDELT compact format). The move day is the first forecast
    step, so 'prior_day' (event day 00:00 UTC) admits strictly T-1 news."""
    d8 = date.replace("-", "")
    return d8 + ("000000" if policy == "prior_day" else "235959")


def _minus_days(date, n):
    import datetime as dt
    y, m, d = map(int, date.split("-"))
    return (dt.date(y, m, d) - dt.timedelta(days=n)).strftime("%Y%m%d") + "000000"


def _gdelt_call(query, date, policy):
    params = urllib.parse.urlencode({
        "query": query, "mode": "artlist", "maxrecords": 25, "format": "json",
        "sort": "hybridrel",
        "startdatetime": _minus_days(date, LOOKBACK_DAYS), "enddatetime": _cutoff(date, policy)})
    req = urllib.request.Request("https://api.gdeltproject.org/api/v2/doc/doc?" + params,
                                 headers={"User-Agent": UA})
    for attempt in range(3):                        # rate-limit answers are plain text -> retry
        try:
            body = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
            return json.loads(body).get("articles", [])
        except Exception:
            time.sleep(10 * (attempt + 1))
    return []


def _gdelt_fetch(date, policy):
    """S4 with S3 query expansion: try the narrow query, widen when a window comes back empty.
    Returns [{date,title,domain,url}]."""
    arts = []
    for qi, q in enumerate(QUERIES):
        if qi:
            time.sleep(6)
        arts = _gdelt_call(q, date, policy)
        if arts:
            break
    out, seen = [], set()
    for a in arts:
        t = a.get("title", "").strip()
        k = re.sub(r"\W+", "", t.lower())
        if not t or k in seen:
            continue
        seen.add(k)
        out.append({"date": a.get("seendate", ""), "title": t, "domain": a.get("domain", ""),
                    "url": a.get("url", "")})
        if len(out) >= MAX_ARTICLES:
            break
    return out


def _retrieval(salient, policy, refresh=False, offline=False):
    """Cache-first S4: fill gaps (or refresh) via live GDELT, 6s apart (API rate limit)."""
    cache = json.load(open(GDELT_JSON)) if os.path.exists(GDELT_JSON) else {}
    if offline:
        return cache
    dirty = False
    for e in salient:
        d = e["date"]
        if refresh or not cache.get(d):
            time.sleep(6)
            got = _gdelt_fetch(d, policy)
            if got or d not in cache:
                cache[d] = got
                dirty = True
    if dirty:
        json.dump(cache, open(GDELT_JSON, "w"), indent=1)
    return cache


def _apply_cutoff(arts, date, policy):
    """S7 audit: keep only articles time-stamped before the cutoff (cache may hold later ones
    from older, leakier retrieval windows). Returns (kept, n_dropped)."""
    lim = _cutoff(date, policy) + "Z"
    kept = [a for a in arts if a.get("date", "") and a["date"] <= lim]
    return kept, len(arts) - len(kept)


# ---------- S5/S6 · grounded cause + S7 outcome screen ----------
_OUTCOME = re.compile(
    r"\b(price|prices|market|markets|wti|crude|oil|barrel)s?\b[^.]{0,60}?"
    r"\b(fell|falls?|dropp?e?d?s?|plunged?s?|tumbled?s?|slid|slides?|sank|slumped?s?|crashed|"
    r"rose|rises?|risen|surged?s?|jumped?s?|rallied|climbed|soared|gained|declined?s?|dived)\b"
    r"|\b(falling|rising|surging|plunging|tumbling|soaring|sliding|crashing)\s+"
    r"(?:\w+\s+){0,2}(price|prices|oil|crude|market)", re.I)


def _leaks_outcome(text):
    return bool(_OUTCOME.search(text))


def _cause(date, headlines):
    """S5+S6: LLM distills the real-world CAUSE from pre-cutoff headlines — leakage-safe (no
    price, no direction, no un-grounded number). Cached per event, keyed on the headline set;
    returns None (event dropped) when generation is unavailable or never passes the gate."""
    heads = headlines[:6]
    hkey = hashlib.sha1("\n".join(heads).encode()).hexdigest()[:12]
    cache = os.path.join(CAUSE_DIR, f"{date}.json")
    ground = "\n".join("- " + h for h in heads)
    if os.path.exists(cache):
        c = json.load(open(cache))
        if (c.get("headlines_key") == hkey and c.get("cause")
                and not _leaks_outcome(c["cause"]) and not forecast_leak(c["cause"])
                and not future_year_leak(c["cause"], date)
                and is_faithful(c["cause"], ground)):
            return c["cause"]
    sys_p = ("You summarize the real-world cause of a commodity-price move for a forecasting "
             "dataset, using ONLY the given news headlines. Write the context/cause (geopolitics, "
             "OPEC, supply, demand, macro) — do NOT state the price, the direction, or the size "
             "of the move.")
    user_p = (f"WTI crude had a notable move around {date}. Headlines from the days before it:\n"
              f"{ground}\n\nIn 2-3 sentences, describe the real-world context behind this period. "
              "Do not describe the price move itself.")
    for extra in ("", "\n\nIMPORTANT: never say prices/the market rose, fell, or moved — "
                      "describe only the underlying events and conditions."):
        # values=[0.0]: the cause may state no series number; headline numbers are grounded.
        desc, _, _ = grounded_describe(sys_p, user_p + extra, [0.0], grounding_text=ground,
                                       fallback=None, max_tokens=160, temperature=0.4)
        if (desc and not _leaks_outcome(desc) and not forecast_leak(desc)
                and not future_year_leak(desc, date) and is_faithful(desc, ground)):
            os.makedirs(CAUSE_DIR, exist_ok=True)
            json.dump({"headlines_key": hkey, "cause": desc, "headlines": heads},
                      open(cache, "w"), indent=1)
            return desc
    return None


def _knowledge_time(arts):
    """Latest retained article timestamp, ISO-8601 (honest provenance for the cutoff)."""
    ts = max(a["date"] for a in arts)                         # '20241014T144500Z'
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}Z"


# ---------- S8 · emit ----------
def pairs_and_trace(policy="prior_day", refresh=False, offline=False):
    rows = _prices()
    idx = {d: i for i, (d, _) in enumerate(rows)}
    salient = _detect(rows)
    gdelt = _retrieval(salient, policy, refresh, offline)
    examples, trace = [], []
    for e in salient:
        d = e["date"]
        t = {"date": d, "ret": e["ret"], "articles": [], "dropped_for_leakage": 0,
             "cause": None, "emitted": False, "reason": ""}
        trace.append(t)
        arts, dropped = _apply_cutoff(gdelt.get(d, []), d, policy)
        t["dropped_for_leakage"] = dropped
        if not arts:                                  # S4 returned nothing -> a real flywheel gap
            t["reason"] = "no_retrieval"
            continue
        i = idx[d]
        if i < HIST or i + FUT > len(rows):
            t["reason"] = "series_boundary"
            continue
        cause = _cause(d, [a["title"] for a in arts])
        if not cause:                                 # S7 gate never passed -> drop, never emit ""
            t["articles"] = arts[:3]
            t["reason"] = "no_verified_cause"
            continue
        hist = [p for _, p in rows[i - HIST:i]]
        fut = [p for _, p in rows[i:i + FUT]]         # move day = first forecast step
        ex = TextToTs(
            user_text=f"As of {d}: {cause} Given this context and the recent WTI crude price history, "
                      f"forecast the price.",
            history=hist, future=fut, series_name="WTI crude oil price (USD/bbl)",
            unit="usd_per_barrel", freq="business_daily",
            meta={"dataset": "flywheel_oil", "source": "FRED DCOILWTICO + GDELT-retrieved news",
                  "series_id": f"DCOILWTICO_{d}_flywheel_t2s",
                  "event_date": d, "direction": "text_to_ts", "fred_series": "DCOILWTICO",
                  "leakage_cutoff": policy, "articles_used": len(arts),
                  "retrieved_from": [a["domain"] for a in arts[:6]],
                  "retrieved_titles": [a["title"] for a in arts[:3]],
                  "flywheel": True, "text_synthesized_from": "gdelt_headlines"},
            text_source="flywheel_gdelt_synth", is_generated="derived_generated",
            knowledge_time=_knowledge_time(arts))
        examples.append(ex)
        t.update({"articles": arts[:3], "cause": cause, "hist": hist, "fut": fut,
                  "emitted": True})
    return examples, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", choices=["prior_day", "same_day"], default="prior_day",
                    help="article cutoff: prior_day = event day 00:00 UTC (default, strict T-1)")
    ap.add_argument("--refresh-retrieval", action="store_true",
                    help="re-query GDELT for every event (else only fills gaps)")
    ap.add_argument("--offline", action="store_true", help="use cached retrieval only")
    a = ap.parse_args()
    examples, trace = pairs_and_trace(a.cutoff, a.refresh_retrieval, a.offline)
    recs = [build_record(example) for example in examples]
    invalid = [validate(record) for record in recs if validate(record)]
    if invalid:
        raise ValueError(f"canonical flywheel examples emitted invalid ChatML: {invalid[:3]}")
    n = write_jsonl(recs, os.path.join(PKG, "out", "flywheel_oil.jsonl"))
    json.dump(trace, open(os.path.join(RAW, "_trace.json"), "w"), indent=1)
    got = sum(1 for t in trace if t["emitted"])
    gaps = {t["reason"] for t in trace if not t["emitted"]}
    dropped = sum(t["dropped_for_leakage"] for t in trace)
    print(f"flywheel_oil: {n} text→ts pairs -> out/flywheel_oil.jsonl  "
          f"({got}/{len(trace)} events emitted · {dropped} articles dropped by {a.cutoff} cutoff"
          f"{' · gaps: ' + ', '.join(sorted(gaps)) if gaps else ''})")


if __name__ == "__main__":
    main()
