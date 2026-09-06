#!/usr/bin/env python3
"""Commodity data-flywheel — the DENSE, event-text-rich generalization of oil_demo.py.

Diagnosis it answers (Xinyue meeting + fine-tune probe): the flywheel mechanism works when the
conditioning text carries REAL event information, but Wikipedia point-in-time leads are generic
standing bios the model learns to ignore. Dense financial/commodity series are the "sweet spot":
every salient move has genuine news behind it. This module manufactures event-grounded text→ts
pairs for several dense series × many moves each, reusing the proven oil pipeline end-to-end.

  S1 ingest    FRED daily price per commodity                      -> raw/commodity/<fred>.csv
  S2 detect    robust MULTI-SCALE salient moves (core.detect, W1)  -> raw/_salient_<key>.json
  S3 query     per-commodity keywords + a PRE-EVENT window (leakage-aware by construction)
  S4 retrieve  GDELT DOC API BATCHED to a local monthly pool (Xinyue: pull broad once, match
               locally — one call per needed month, not per move, to stay under the rate limit),
               then locally cut to [move-LOOKBACK, move) -> raw/_pool_<key>.json
  S5/S6 ground LLM distills the leakage-safe CAUSE (core.generate) -> raw/cause_commodity/<id>.json
  S7 gate      cutoff + numeric + outcome screen + entity-tracing faithfulness (drop, never fake)
  S8 emit      build + validate text→ts (reuse core.schema)        -> out/flywheel_commodity.jsonl

Leakage (CLAUDE §1.2): the move day is the FIRST forecast step, so only articles before the cutoff
(move day 00:00 UTC = strictly T-1) may condition the forecast. knowledge_time = latest retained
article. Cache-first; offline re-runs are deterministic. Run:
  micromamba run -n ts-language python flywheel/commodity_demo.py [--refresh-retrieval] [--only brent]
"""
import os, csv, sys, json, re, time, hashlib, argparse, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
RAW = os.path.join(HERE, "raw")
COMMODITY_CSV_DIR = os.path.join(RAW, "commodity")
CAUSE_DIR = os.path.join(RAW, "cause_commodity")
CAUSE_RICH_DIR = os.path.join(RAW, "cause_commodity_rich")   # TimesX-style multi-event context cache
SEM_DIR = os.path.join(RAW, "sem_commodity")     # W2 v2 semantic-relevance cache (per headline set)

from core.schema import TextToTs, build_record, validate
from core.writer import write_jsonl
from core.generate import grounded_describe
from core.faithfulness import is_faithful
from core.detect import detect_robust_multiscale
from core.retrieval_rank import semantic_rank   # W2 v2: LLM semantic relevance filter for headlines
from core.leak_screen import forecast_leak, future_year_leak   # content-layer soft-leak screens (shared)

HIST, FUT = 10, 5
Z_MIN, SPACING = 2.2, 10                     # S2: multi-scale robust-MAD detector (core.detect, W1)
LOOKBACK_DAYS = 4                            # S3: retrieval window = [move - LOOKBACK, cutoff)
MAX_ARTICLES = 8
UA = "ts-language research (majiaju89@gmail.com)"
COSD, COED = "2021-06-01", "2024-12-31"

# Each dense series: FRED id + display metadata + a retrieval query (narrow, then a widening
# fallback) + the price NOUNS used by the per-commodity outcome-direction screen + a plain-English
# subject for the cause prompt.
COMMODITIES = [
    dict(key="brent", fred="DCOILBRENTEU", subject="Brent crude oil",
         name="Brent crude oil price (USD/bbl)", unit="usd_per_barrel", freq="business_daily",
         nouns=r"oil|crude|brent|petroleum|barrel|fuel",
         queries=['("crude oil" OR "oil prices" OR OPEC OR Brent) sourcelang:english',
                  '("oil price" OR "crude oil" OR OPEC OR petroleum OR "oil demand") sourcelang:english']),
    dict(key="natgas", fred="DHHNGSP", subject="US natural gas (Henry Hub)",
         name="US natural gas spot price, Henry Hub (USD/MMBtu)", unit="usd_per_mmbtu",
         freq="business_daily", nouns=r"gas|lng|heating|storage|pipeline",
         queries=['("natural gas" OR LNG OR "gas prices" OR "Henry Hub") sourcelang:english',
                  '("natural gas" OR LNG OR "gas market" OR "gas storage" OR heating) sourcelang:english']),
    dict(key="bitcoin", fred="CBBTCUSD", subject="Bitcoin",
         name="Bitcoin price (USD)", unit="usd", freq="daily",
         nouns=r"bitcoin|btc|crypto|cryptocurrency|ether|ethereum|token|coin",
         queries=['(Bitcoin OR cryptocurrency OR crypto OR BTC) sourcelang:english',
                  '(Bitcoin OR crypto OR cryptocurrency OR blockchain OR "digital asset") sourcelang:english']),
    dict(key="eurusd", fred="DEXUSEU", subject="the euro-dollar exchange rate",
         name="US dollar to euro exchange rate (USD per EUR)", unit="usd_per_eur",
         freq="business_daily", nouns=r"euro|dollar|ecb|forex|currency|exchange",
         queries=['(euro OR "European Central Bank" OR ECB OR "US dollar" OR forex) sourcelang:english',
                  '(euro OR dollar OR ECB OR "Federal Reserve" OR inflation OR currency) sourcelang:english']),
    dict(key="gbpusd", fred="DEXUSUK", subject="the pound-dollar exchange rate",
         name="US dollar to British pound exchange rate (USD per GBP)", unit="usd_per_gbp",
         freq="business_daily", nouns=r"pound|sterling|gbp|dollar|forex|bank of england|boe",
         queries=['(pound OR sterling OR "Bank of England" OR "US dollar" OR forex) sourcelang:english',
                  '(pound OR sterling OR GBP OR dollar OR inflation OR currency) sourcelang:english']),
    dict(key="jpyusd", fred="DEXJPUS", subject="the yen-dollar exchange rate",
         name="Japanese yen to US dollar exchange rate (JPY per USD)", unit="jpy_per_usd",
         freq="business_daily", nouns=r"yen|jpy|dollar|forex|bank of japan|boj",
         queries=['(yen OR "Bank of Japan" OR BOJ OR "US dollar" OR forex) sourcelang:english',
                  '(yen OR JPY OR dollar OR inflation OR currency) sourcelang:english']),
    dict(key="ethereum", fred="CBETHUSD", subject="Ethereum",
         name="Ethereum price (USD)", unit="usd", freq="daily",
         nouns=r"ethereum|ether|crypto|cryptocurrency|eth|token|coin|blockchain",
         queries=['(Ethereum OR ether OR cryptocurrency OR crypto) sourcelang:english',
                  '(Ethereum OR crypto OR cryptocurrency OR blockchain OR "digital asset") sourcelang:english']),
]

_DIR = (r"fell|falls?|dropp?e?d?s?|plunged?s?|tumbled?s?|slid|slides?|sank|slumped?s?|crashed|"
        r"rose|rises?|risen|surged?s?|jumped?s?|rallied|climbed|soared|gained|declined?s?|dived|"
        r"weakened?s?|strengthened?s?|rebounded?s?")


def _outcome_re(nouns):
    """Per-commodity outcome-direction screen: reject a cause that narrates the move itself
    (e.g. 'oil prices fell', 'bitcoin surged') — that leaks the answer."""
    return re.compile(
        r"\b(price|prices|market|markets|" + nouns + r")s?\b[^.]{0,60}?\b(" + _DIR + r")\b"
        r"|\b(falling|rising|surging|plunging|tumbling|soaring|sliding|crashing|rallying)\s+"
        r"(?:\w+\s+){0,2}(price|prices|market|" + nouns + r")", re.I)


# Stricter gate for the RICH multi-event context: the per-commodity outcome_re only knows THIS
# commodity's nouns, so a multi-event cause can leak via (a) a CROSS-asset move ("Bitcoin surged"
# inside an Ethereum cause) or (b) an absolute price level/number ("$103k", "above $4,000") — both
# of which reveal the answer. This independent check (found 26 such cases in the first rich build)
# rejects any direction verb next to ANY financial asset/price term, or any explicit price figure.
_ASSET_ANY = re.compile(r"\b(oil|crude|brent|petroleum|gas|lng|bitcoin|btc|ether\w*|crypto\w*|"
                        r"coin|token|blockchain|pound|sterling|euro|dollar|yen|currency|forex|"
                        r"price|prices|market|markets|rate|rates|index|indices)\b", re.I)
_PXNUM = re.compile(r"\$\s?\d|\b\d{1,3},\d{3}\b|"
                    r"\b(?:above|below|past|reached|reaching|hit|hitting|breaking|topping|"
                    r"level|levels|threshold|mark)\b[^.]{0,18}\$?\d", re.I)


def _rich_leaky(text):
    dr = re.compile(_DIR, re.I)
    if _PXNUM.search(text):
        return True
    for m in dr.finditer(text):
        if _ASSET_ANY.search(text[max(0, m.start() - 40):m.end() + 15]):
            return True
    return False


# forecast_leak (forward-looking projection + direction/price = soft leak) lives in core/leak_screen.py


# ---------- S1 · ingest ----------
def _prices(c):
    path = os.path.join(COMMODITY_CSV_DIR, f"{c['fred']}.csv")
    if not os.path.exists(path):
        os.makedirs(COMMODITY_CSV_DIR, exist_ok=True)
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={c['fred']}"
               f"&cosd={COSD}&coed={COED}")
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        open(path, "wb").write(urllib.request.urlopen(req, timeout=45).read())
    rows = []
    for r in csv.reader(open(path)):
        if not r or not r[0][:4].isdigit() or len(r) < 2 or not r[1].strip() or r[1] == ".":
            continue
        rows.append((r[0], float(r[1])))
    return rows


# ---------- S2 · salient detection (multi-scale, W1) ----------
def _detect(c, rows):
    cache = os.path.join(RAW, f"_salient_{c['key']}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    prices = [p for _, p in rows]
    picked = detect_robust_multiscale(prices, z_min=Z_MIN, spacing=SPACING, top_k=None)
    out = [{"date": rows[i][0], "idx": i, "ret": round(r, 4), "scale": k, "z": round(z, 2)}
           for i, k, r, z in picked]
    json.dump(out, open(cache, "w"), indent=1)
    return out


# ---------- S3/S4 · leakage-aware retrieval ----------
def _cutoff(date):
    return date.replace("-", "") + "000000"                    # move day 00:00 UTC (strict T-1)


def _minus_days(date, n):
    import datetime as dt
    y, m, d = map(int, date.split("-"))
    return (dt.date(y, m, d) - dt.timedelta(days=n)).strftime("%Y%m%d") + "000000"


POOL_MAX, POOL_SLEEP = 250, 20


def _months_needed(salient):
    """Every year-month a move's retrieval window [move-LOOKBACK, move) can touch."""
    import datetime as dt
    months = set()
    for e in salient:
        y, m, d = map(int, e["date"].split("-"))
        base = dt.date(y, m, d)
        for off in (0, LOOKBACK_DAYS):
            dd = base - dt.timedelta(days=off)
            months.add(f"{dd.year:04d}-{dd.month:02d}")
    return sorted(months)


def _month_bounds(month):
    y, m = map(int, month.split("-"))
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y:04d}{m:02d}01000000", f"{ny:04d}{nm:02d}01000000"


def _pool_call(c, month):
    """One wide GDELT call covering a whole month (batched retrieval, datedesc for time coverage)."""
    start, end = _month_bounds(month)
    params = urllib.parse.urlencode({
        "query": c["queries"][0], "mode": "artlist", "maxrecords": POOL_MAX, "format": "json",
        "sort": "datedesc", "startdatetime": start, "enddatetime": end})
    req = urllib.request.Request("https://api.gdeltproject.org/api/v2/doc/doc?" + params,
                                 headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            body = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "ignore")
            arts = json.loads(body).get("articles", [])
            return [{"date": a.get("seendate", ""), "title": a.get("title", "").strip(),
                     "domain": a.get("domain", ""), "url": a.get("url", "")}
                    for a in arts if a.get("title")]
        except Exception:
            time.sleep(10 * (attempt + 1))
    return []


def _pool(c, salient, refresh=False, offline=False):
    """Cache-first local news pool {month: [articles]}. Only NON-EMPTY months are cached — an empty
    result under GDELT rate-limiting is a FAILURE, not 'no news that month', so it is never cached
    (CLAUDE §1.5); the pool is saved after each successful fetch, so repeated runs resume and
    converge as the rate limit allows. Fetches only the months the moves need.
    offline=True: use only the already-built local pool (e.g. from bq_pool.py's BigQuery pull) —
    no network, no DOC-API rate limit."""
    path = os.path.join(RAW, f"_pool_{c['key']}.json")
    pool = json.load(open(path)) if os.path.exists(path) else {}
    if offline:
        return pool
    for m in _months_needed(salient):
        if not refresh and pool.get(m):
            continue
        time.sleep(POOL_SLEEP)
        arts = _pool_call(c, m)
        if arts:                                       # cache only verified, non-empty fetches
            pool[m] = arts
            json.dump(pool, open(path, "w"), indent=1)
    return pool


def _art_ts(a):
    return a.get("date", "").replace("T", "").replace("Z", "")[:14]       # 'YYYYMMDDHHMMSS'


def _match_local(pool, date, noun_re):
    """S4 (local): from the pool, keep articles strictly inside [move-LOOKBACK, cutoff), dedupe by
    title, rank by commodity-term overlap then recency, take the top MAX_ARTICLES. Leakage-safe by
    construction: the upper bound is the move day 00:00 UTC (the first forecast step)."""
    lo, hi = _minus_days(date, LOOKBACK_DAYS), _cutoff(date)               # both 'YYYYMMDD000000'
    seen, cand = set(), []
    for a in (x for month in pool.values() for x in month):
        ts = _art_ts(a)
        if not (lo <= ts < hi):
            continue
        k = re.sub(r"\W+", "", a["title"].lower())
        if not a["title"] or k in seen:
            continue
        seen.add(k)
        cand.append((len(noun_re.findall(a["title"])), ts, a))
    cand.sort(key=lambda t: (t[0], t[1]), reverse=True)                    # relevance, then recency
    return [a for _, _, a in cand[:MAX_ARTICLES]]


# ---------- S5/S6 · grounded cause + S7 gate ----------
def _rank_titles(c, arts, cache_only=False):
    """W2 v2: keep only the headlines an LLM judges TOPICALLY relevant to this commodity, before
    grounding the cause — fixes cross-source contamination (a EUR/GBP headline leaking into a
    euro-dollar move) and clickbait noise that a keyword match can't tell apart. Cache-keyed on the
    exact title set (deterministic), so a frozen replay reads the cache with no LLM call; if the
    cache is absent in cache_only mode it falls back to the raw titles (never blocks)."""
    titles = [a["title"] for a in arts]
    if len(titles) <= 2:
        return titles
    if cache_only:
        key = hashlib.sha1((c["subject"] + "||" + "\n".join(titles)).encode()).hexdigest()[:16]
        cache = os.path.join(SEM_DIR, f"sem_{key}.json")
        return (json.load(open(cache)).get("kept") or titles) if os.path.exists(cache) else titles
    kept, _ = semantic_rank(titles, c["subject"], cache_dir=SEM_DIR)
    return kept or titles


def _cause(c, date, headlines, outcome_re, cache_only=False, style="single"):
    """style='single' → the original 2-3 sentence cause; style='multi' → a TimesX-style NUMBERED
    multi-event context (<1>...<2>...), the higher-quality event text the TimesX 7.3% finding says
    matters. Both go through the SAME faithfulness (is_faithful vs headlines) + outcome-leakage gates;
    only gated text is cached. The two styles use separate caches/keys so they never collide."""
    heads = headlines[:6]
    hkey = hashlib.sha1(("\n".join(heads) + "|" + c["key"]
                         + ("" if style == "single" else "|" + style)).encode()).hexdigest()[:12]
    cdir = CAUSE_DIR if style == "single" else CAUSE_RICH_DIR
    cache = os.path.join(cdir, f"{c['key']}_{date}.json")
    ground = "\n".join("- " + h for h in heads)
    if os.path.exists(cache):
        j = json.load(open(cache))
        if (j.get("headlines_key") == hkey and j.get("cause")
                and not outcome_re.search(j["cause"])
                and not (style != "single" and _rich_leaky(j["cause"]))
                and not forecast_leak(j["cause"])
                and not future_year_leak(j["cause"], date)
                and is_faithful(j["cause"], ground)):
            return j["cause"]
    if cache_only:                       # frozen replay: no LLM, only already-verified cached causes
        return None
    if style == "single":
        sys_p = (f"You summarize the real-world cause of a price move in {c['subject']} for a forecasting "
                 "dataset, using ONLY the given news headlines. Write the context/cause (geopolitics, "
                 "policy, supply, demand, macro, regulation) — do NOT state the price, the direction, or "
                 "the size of the move.")
        user_p = (f"{c['subject'].capitalize()} had a notable move around {date}. Headlines from the days "
                  f"before it:\n{ground}\n\nIn 2-3 sentences, describe the real-world context behind this "
                  "period. Do not describe the price move itself.")
        maxtok = 160
    else:                                # TimesX-style numbered multi-event context
        sys_p = (f"You build the event CONTEXT for a {c['subject']} price-forecasting dataset, using ONLY "
                 "the given news headlines. List the distinct real-world events and conditions "
                 "(geopolitics, policy, supply, demand, macro, regulation) behind this period. Do NOT "
                 "state the price, the direction, or the size of the move. NEVER mention a specific "
                 "price level or number (no '$103k', no 'above $4,000'), and NEVER say ANY asset, "
                 "currency, or market rose, fell, surged, dropped, strengthened, or weakened — describe "
                 "only the underlying events, not any market reaction.")
        user_p = (f"{c['subject'].capitalize()} had a notable period around {date}. Headlines from the "
                  f"days before it:\n{ground}\n\nWrite EXACTLY this format: the line 'Recent related "
                  "events (events are separated by numbered tags):' then 2 to 5 numbered events "
                  "'<1> ... <2> ...', each a concrete event with its specifics drawn ONLY from the "
                  "headlines. Do not describe the price move itself.")
        maxtok = 340
    for extra in ("", "\n\nIMPORTANT: never say the price/market rose, fell, or moved — describe "
                      "only the underlying events and conditions."):
        desc, _, _ = grounded_describe(sys_p, user_p + extra, [0.0], grounding_text=ground,
                                       fallback=None, max_tokens=maxtok, temperature=0.4)
        if (desc and not outcome_re.search(desc)
                and not (style != "single" and _rich_leaky(desc))
                and not forecast_leak(desc)
                and not future_year_leak(desc, date)
                and is_faithful(desc, ground)):
            os.makedirs(cdir, exist_ok=True)
            json.dump({"headlines_key": hkey, "cause": desc, "headlines": heads, "style": style},
                      open(cache, "w"), indent=1)
            return desc
    return None


def _knowledge_time(arts):
    ts = max(a["date"] for a in arts)                                # GDELT date = YYYYMMDDHHMMSS (14 chars)
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}Z"


# ---------- S8 · emit ----------
def build_commodity(c, refresh=False, offline=False, cache_only=False, style="single"):
    rows = _prices(c)
    idx = {d: i for i, (d, _) in enumerate(rows)}
    salient = _detect(c, rows)
    pool = _pool(c, salient, refresh, offline)
    outcome_re = _outcome_re(c["nouns"])
    noun_re = re.compile(c["nouns"], re.I)
    examples, trace = [], []
    for e in salient:
        d = e["date"]
        t = {"commodity": c["key"], "date": d, "ret": e["ret"], "emitted": False, "reason": ""}
        arts = _match_local(pool, d, noun_re)
        if not arts:
            t["reason"] = "no_retrieval"
            trace.append(t)
            continue
        i = idx[d]
        if i < HIST or i + FUT > len(rows):
            t["reason"] = "series_boundary"
            trace.append(t)
            continue
        ranked = _rank_titles(c, arts, cache_only)          # W2 v2 semantic filter before grounding
        cause = _cause(c, d, ranked, outcome_re, cache_only, style)
        if not cause:
            t["reason"] = "no_verified_cause"
            trace.append(t)
            continue
        hist = [p for _, p in rows[i - HIST:i]]
        fut = [p for _, p in rows[i:i + FUT]]
        ex = TextToTs(
            user_text=f"As of {d}: {cause} Given this context and the recent {c['subject']} price history, "
                      f"forecast the price.",
            history=hist, future=fut, series_name=c["name"], unit=c["unit"], freq=c["freq"],
            meta={"dataset": "flywheel_commodity" if style == "single" else "flywheel_commodity_rich",
                  "commodity": c["key"], "text_style": "single_cause" if style == "single"
                  else "timesx_multi_event",
                  "source": f"FRED {c['fred']} + GDELT-retrieved news",
                  "series_id": f"{c['fred']}_{d}_flywheel_{'rich_' if style != 'single' else ''}t2s",
                  "event_date": d,
                  "direction": "text_to_ts", "fred_series": c["fred"], "leakage_cutoff": "prior_day",
                  "articles_used": len(arts), "retrieved_from": [a["domain"] for a in arts[:6]],
                  # store the SAME (semantically-ranked) headlines the cause was grounded+gated on
                  # (heads=first 6), so an independent audit verifies against the real evidence.
                  "retrieved_titles": ranked[:6],
                  "detect_scale": e.get("scale"), "detect_z": e.get("z"),
                  "flywheel": True, "text_synthesized_from": "gdelt_headlines"},
            text_source="flywheel_gdelt_synth", is_generated="derived_generated",
            knowledge_time=_knowledge_time(arts))
        examples.append(ex)
        t.update({"emitted": True, "cause": cause, "articles": arts[:3]})
        trace.append(t)
    return examples, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="build a single commodity by key (brent/natgas/bitcoin/eurusd)")
    ap.add_argument("--refresh-retrieval", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="use only the local pool (e.g. built by bq_pool.py); no network")
    ap.add_argument("--frozen", action="store_true",
                    help="cache-only causes: no LLM, replay already-verified causes (deterministic)")
    ap.add_argument("--rich", action="store_true",
                    help="TimesX-style NUMBERED multi-event context (higher-quality event text); "
                         "writes out/flywheel_commodity_rich.jsonl (leaves the single-cause set intact)")
    a = ap.parse_args()
    style = "multi" if a.rich else "single"
    outfile = "flywheel_commodity_rich.jsonl" if a.rich else "flywheel_commodity.jsonl"
    todo = [c for c in COMMODITIES if not a.only or c["key"] == a.only]
    all_examples, all_trace = [], []
    for c in todo:
        examples, trace = build_commodity(c, a.refresh_retrieval, a.offline, a.frozen, style)
        got = sum(1 for t in trace if t["emitted"])
        gaps = {}
        for t in trace:
            if not t["emitted"]:
                gaps[t["reason"]] = gaps.get(t["reason"], 0) + 1
        print(f"  {c['key']:8s}: {got}/{len(trace)} events emitted"
              + (f"  gaps={gaps}" if gaps else ""))
        all_examples += examples
        all_trace += trace
    all_recs = [build_record(example) for example in all_examples]
    invalid = [validate(record) for record in all_recs if validate(record)]
    if invalid:
        raise ValueError(f"canonical flywheel examples emitted invalid ChatML: {invalid[:3]}")
    n = write_jsonl(all_recs, os.path.join(PKG, "out", outfile))
    tracefile = "_trace_commodity_rich.json" if a.rich else "_trace_commodity.json"
    json.dump(all_trace, open(os.path.join(RAW, tracefile), "w"), indent=1)
    got = sum(1 for t in all_trace if t["emitted"])
    print(f"{outfile[:-6]}: {n} text→ts pairs -> out/{outfile} "
          f"({got}/{len(all_trace)} events emitted across {len(todo)} commodities)")


if __name__ == "__main__":
    main()
