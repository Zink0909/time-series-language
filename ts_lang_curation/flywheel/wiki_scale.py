#!/usr/bin/env python3
"""Scale the Wikipedia point-in-time flywheel — zero GPU, zero GDELT, zero new curation.

Reuses the 39 PROVEN persistent entities (the ones wiki_demo already emitted) but fetches a LONGER
pageview history and detects ALL salient attention spikes (robust MAD peaks, multi-scale in spirit),
not just the one seeded event. Each spike → one leakage-clean point-in-time text→ts (the article's
revision strictly before that spike). One Nvidia yields several samples (each earnings/AI surge), so
output multiplies from one workflow. Cache-first; caches live in flywheel/raw/wiki/scaled/ (separate).

  micromamba run -n ts-language python flywheel/wiki_scale.py [--refresh] [--limit N]
"""
import os, sys, json, time, argparse, datetime as dt, urllib.parse, urllib.request
from statistics import median, mean, pstdev

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import wiki_demo as wd                                              # reuse pure helpers  # noqa: E402
from core.schema import TextToTs, build_record, validate           # noqa: E402
from core.writer import write_jsonl                                # noqa: E402
from core.chatml import ROBUST_COEF                                 # noqa: E402

with open(wd.META, encoding="utf-8") as _meta_handle:
    META = json.load(_meta_handle)
SCALED = os.path.join(wd.FW_WIKI, "scaled")
REVDIR = os.path.join(SCALED, "rev")
TRACE = os.path.join(SCALED, "_scale_trace.json")
OUT = os.path.join(PKG, "out", "flywheel_wiki_scaled.jsonl")

YEARS_BACK, MONTHS_FWD = 2, 4       # long fetch window around the seed event
Z_MIN, PEAK_SPACING = 4.0, 21       # MAD peak threshold + min days between distinct spikes
HIST, FUT = 21, 14                  # per-spike local history / horizon
MIN_HIST, MIN_FUT = 8, 6
SCALE = 1e3
MAXZ = 50                           # health gate: drop near-zero-baseline records whose normalized
                                    # target explodes (|z|>50). Both S9 feedback and the rigorous
                                    # probe flagged these as low-quality (denominator->0, unlearnable).


def _persistent_slugs():
    """The slugs wiki_demo emitted = confirmed persistent entities (had pre-event state)."""
    tr = json.load(open(wd.TRACE))
    return [t["slug"] for t in tr if t.get("emitted")]


# ---------- S1 · long pageview history ----------
def _long_pv(slug, canon, event_date, refresh):
    cache = os.path.join(SCALED, f"{slug}_pv.json")
    if os.path.exists(cache) and not refresh:
        items = json.load(open(cache))
    else:
        y, m, d = map(int, event_date.split("-"))
        start = (dt.date(y, m, d) - dt.timedelta(days=365 * YEARS_BACK)).strftime("%Y%m%d")
        end = (dt.date(y, m, d) + dt.timedelta(days=30 * MONTHS_FWD)).strftime("%Y%m%d")
        url = wd.PV_URL.format(article=urllib.parse.quote(canon, safe=""), start=start, end=end)
        try:
            items = json.loads(wd._get(url)).get("items", [])
        except Exception:
            items = []
        os.makedirs(SCALED, exist_ok=True)
        json.dump(items, open(cache, "w"))
        time.sleep(0.4)
    return [(x["timestamp"][:8], round(x["views"] / SCALE, 3)) for x in items]


# ---------- S2 · multi-peak robust detection (level-based, right for attention counts) ----------
def _detect_peaks(vals):
    med = median(vals)
    mad = median([abs(v - med) for v in vals]) or 1e-9
    cand = [(i, 0.6745 * (vals[i] - med) / mad) for i in range(len(vals))]
    cand = [(i, z) for i, z in cand if z >= Z_MIN]           # positive attention spikes only
    cand.sort(key=lambda c: -c[1])
    picked = []
    for i, z in cand:
        if all(abs(i - j) >= PEAK_SPACING for j, _ in picked):
            picked.append((i, z))
    return sorted(picked)


# ---------- S3 · point-in-time revision strictly before this spike (per-spike cache) ----------
def _rev_before(slug, title, spike_date, refresh):
    cache = os.path.join(REVDIR, f"{slug}_{spike_date}.json")
    if os.path.exists(cache) and not refresh:
        return json.load(open(cache))
    q = urllib.parse.urlencode({
        "action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
        "rvstart": spike_date + "T00:00:00Z", "rvdir": "older",
        "rvprop": "timestamp|ids", "format": "json", "formatversion": 2})
    res = {"revid": None, "timestamp": None, "lead": None}
    try:
        page = json.loads(wd._get(f"{wd.API}?{q}"))["query"]["pages"][0]
        rev = page["revisions"][0]
        q2 = urllib.parse.urlencode({"action": "parse", "oldid": rev["revid"], "prop": "wikitext",
                                     "section": 0, "format": "json", "formatversion": 2})
        wt = json.loads(wd._get(f"{wd.API}?{q2}"))["parse"]["wikitext"]
        res = {"revid": rev["revid"], "timestamp": rev["timestamp"], "lead": wd._lead_para(wt)}
    except Exception:
        pass
    os.makedirs(REVDIR, exist_ok=True)
    json.dump(res, open(cache, "w"), indent=1)
    time.sleep(0.4)
    return res


def _flat(hist):
    return pstdev(hist) < ROBUST_COEF * max(abs(mean(hist)), 1.0)


def build(refresh=False, limit=0):
    slugs = _persistent_slugs()
    if limit:
        slugs = slugs[:limit]
    examples, trace = [], []
    for slug in slugs:
        m = META.get(slug, {})
        title, canon, ev = m.get("title", ""), m.get("canonical", ""), m.get("event_date", "")
        series = _long_pv(slug, canon, ev, refresh)
        if len(series) < HIST + FUT:
            trace.append({"slug": slug, "spikes": 0, "emitted": 0, "reason": "no_long_pv"})
            continue
        dates = [d for d, _ in series]
        vals = [v for _, v in series]
        peaks = _detect_peaks(vals)
        n_emit = 0
        for i, z in peaks:
            if i < MIN_HIST or i + MIN_FUT > len(vals):
                continue
            spike_date = f"{dates[i][:4]}-{dates[i][4:6]}-{dates[i][6:8]}"
            rev = _rev_before(slug, title, spike_date, refresh)
            if not rev.get("lead") or not rev.get("timestamp"):
                continue
            if rev["timestamp"] >= spike_date + "T00:00:00Z":       # leakage assert
                continue
            hist, fut = vals[max(0, i - HIST):i], vals[i:i + FUT]
            if len(hist) < MIN_HIST or len(fut) < MIN_FUT:
                continue
            ex = TextToTs(
                user_text=(f"{rev['lead']} Given this background (as known before {spike_date}) "
                           f"and the recent daily Wikipedia attention, forecast the pageviews."),
                history=hist, future=fut,
                series_name=f"daily English-Wikipedia pageviews for '{title}' (thousands)",
                unit="pageviews_per_day_thousands", freq="daily",
                meta={"dataset": "flywheel_wiki_scaled",
                      "source": "Wikimedia pageviews + English-Wikipedia point-in-time revision",
                      "series_id": f"{slug}_{spike_date}_scaled_t2s", "article": title,
                      "seed_event_date": ev, "direction": "text_to_ts", "spike_date": spike_date,
                      "spike_z": round(z, 1), "revision_id": rev["revid"],
                      "leakage_cutoff": "prior_day_revision", "flat_context": _flat(hist),
                      "flywheel": True,
                      "text_url": f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}"},
                text_source="wikipedia_lead_pit", is_generated="real",
                knowledge_time=rev["timestamp"])
            examples.append(ex)
            n_emit += 1
        trace.append({"slug": slug, "title": title, "spikes": len(peaks), "emitted": n_emit})
    return examples, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    examples, trace = build(a.refresh, a.limit)
    recs = [build_record(example) for example in examples]
    invalid = [validate(record) for record in recs if validate(record)]
    if invalid:
        raise ValueError(f"canonical flywheel examples emitted invalid ChatML: {invalid[:3]}")
    n = write_jsonl(recs, OUT)
    os.makedirs(SCALED, exist_ok=True)
    json.dump(trace, open(TRACE, "w"), indent=1, ensure_ascii=False)
    ent = sum(1 for t in trace if t["emitted"] > 0)
    spikes = sum(t["spikes"] for t in trace)
    print(f"flywheel_wiki_scaled: {n} text->ts from {ent}/{len(trace)} entities "
          f"({spikes} spikes detected) -> out/flywheel_wiki_scaled.jsonl")


if __name__ == "__main__":
    main()
