#!/usr/bin/env python3
"""Scale the flywheel via MANY persistent Wikipedia entities — attention series DO respond to text
(unlike near-efficient prices, per the MiGAS diagnosis), and the point-in-time lead is REAL text so
NO LLM generation is needed. Collects hundreds of persistent entities from the Wikimedia top-pageviews
API, fetches a fixed 2021-2025 pageview window each, multi-peak detects, and emits one leakage-clean
point-in-time text->ts per spike (the article revision's lead strictly before the spike). Reuses the
wiki_scale / wiki_demo helpers.
  micromamba run -n ts-language python flywheel/wiki_scale_big.py [--entities 600] [--refresh]
"""
import os, sys, json, time, argparse, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PKG)
import wiki_demo as wd                                                # noqa: E402
import wiki_scale as ws                                               # noqa: E402
from core.schema import TextToTs, build_record, validate              # noqa: E402
from core.writer import write_jsonl                                   # noqa: E402

BIG = os.path.join(wd.FW_WIKI, "big")
ENT_CACHE = os.path.join(BIG, "_entities.json")
OUT = os.path.join(PKG, "out", "flywheel_wiki_big.jsonl")
START, END = "20210101", "20250101"
PEAKS_PER = 8                                                         # cap spikes per entity
TOP_DATES = [(2022, 3, 15), (2022, 6, 15), (2022, 9, 15), (2023, 3, 15), (2023, 6, 15),
             (2023, 9, 15), (2024, 3, 15), (2024, 6, 15), (2024, 9, 15), (2024, 11, 6)]


def collect_entities(n, refresh=False):
    if os.path.exists(ENT_CACHE) and not refresh:
        return json.load(open(ENT_CACHE))[:n]
    seen, slugs = set(), []
    for y, m, d in TOP_DATES:
        try:
            url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/"
                   f"all-access/{y}/{m:02d}/{d:02d}")
            arts = json.loads(wd._get(url))["items"][0]["articles"]
        except Exception:
            continue
        for a in arts:
            s = a["article"]
            if (":" in s or s.startswith("List_of") or s.startswith("Wikipedia")
                    or s in ("Main_Page", "-") or len(s) < 4):
                continue
            if s not in seen:
                seen.add(s)
                slugs.append(s)
        time.sleep(0.3)
    os.makedirs(BIG, exist_ok=True)
    json.dump(slugs, open(ENT_CACHE, "w"))
    return slugs[:n]


def _long_pv(slug, refresh):
    cache = os.path.join(BIG, slug.replace("/", "_") + "_pv.json")
    if os.path.exists(cache) and not refresh:
        items = json.load(open(cache))
    else:
        url = wd.PV_URL.format(article=urllib.parse.quote(slug, safe=""), start=START, end=END)
        try:
            items = json.loads(wd._get(url)).get("items", [])
        except Exception:
            items = []
        os.makedirs(BIG, exist_ok=True)
        json.dump(items, open(cache, "w"))
        time.sleep(0.3)
    return [(x["timestamp"][:8], round(x["views"] / wd.SCALE, 3)) for x in items]


def build(n_entities, refresh=False):
    slugs = collect_entities(n_entities, refresh)
    recs, trace = [], []
    for k, slug in enumerate(slugs):
        series = _long_pv(slug, refresh)
        if len(series) < ws.HIST + ws.FUT:
            trace.append({"slug": slug, "emitted": 0, "reason": "no_pv"})
            continue
        dates = [d for d, _ in series]
        vals = [v for _, v in series]
        peaks = ws._detect_peaks(vals)[:PEAKS_PER]
        title = slug
        n_emit = 0
        for i, z in peaks:
            if i < ws.MIN_HIST or i + ws.MIN_FUT > len(vals):
                continue
            spike_date = f"{dates[i][:4]}-{dates[i][4:6]}-{dates[i][6:8]}"
            rev = ws._rev_before(slug, title, spike_date, refresh)
            if not rev.get("lead") or not rev.get("timestamp"):
                continue
            if rev["timestamp"] >= spike_date + "T00:00:00Z":         # leakage assert
                continue
            hist, fut = vals[max(0, i - ws.HIST):i], vals[i:i + ws.FUT]
            if len(hist) < ws.MIN_HIST or len(fut) < ws.MIN_FUT:
                continue
            ex = TextToTs(
                user_text=(f"{rev['lead']} Given this background (as known before {spike_date}) "
                           f"and the recent daily Wikipedia attention, forecast the pageviews."),
                history=hist, future=fut,
                series_name=f"daily English-Wikipedia pageviews for '{title.replace('_', ' ')}' (thousands)",
                unit="pageviews_per_day_thousands", freq="daily",
                meta={"dataset": "flywheel_wiki_big",
                      "source": "Wikimedia pageviews + English-Wikipedia point-in-time revision",
                      "series_id": f"{slug}_{spike_date}_big_t2s", "article": title,
                      "direction": "text_to_ts", "spike_date": spike_date, "spike_z": round(z, 1),
                      "revision_id": rev["revid"], "leakage_cutoff": "prior_day_revision",
                      "flat_context": ws._flat(hist), "flywheel": True,
                      "text_url": f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}"},
                text_source="wikipedia_lead_pit", is_generated="real", knowledge_time=rev["timestamp"])
            rec = build_record(ex)
            if validate(rec):
                continue
            recs.append(rec)
            n_emit += 1
        trace.append({"slug": slug, "emitted": n_emit})
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(slugs)} entities · {len(recs)} records so far", flush=True)
    return recs, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=600)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    recs, trace = build(a.entities, a.refresh)
    n = write_jsonl(recs, OUT)
    ent = sum(1 for t in trace if t.get("emitted", 0) > 0)
    print(f"flywheel_wiki_big: {n} text->ts from {ent}/{len(trace)} entities -> {OUT}")


if __name__ == "__main__":
    main()
