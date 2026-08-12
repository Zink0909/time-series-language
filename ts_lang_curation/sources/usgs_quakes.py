# sources/usgs_quakes.py — thin adapter for USGS earthquakes (non-finance, geophysics domain).
# Distinctive asset: an intrinsic, physically-meaningful time series — the AFTERSHOCK SEQUENCE
# (daily M3.5+ counts after a mainshock, an Omori-law decay) — paired with the mainshock's text.
#   text->ts (ts_forecast): mainshock description + early aftershock days -> the rest of the decay.
#   ts->text (text_desc):   aftershock sequence -> grounded, numerically-verified description.
# Data: USGS FDSN event API (public, no key). Cached in raw/usgs/events.json (see README).
import os, json, re
from core.schema import TsToText, TextToTs
from core.generate import grounded_describe
from core.verify import is_consistent


def _place(e):
    """USGS 'place' is the event title for named quakes (e.g. '2025 Aomori …, Japan Earthquake').
    Strip the leading year and trailing 'Earthquake' so it reads as a location, not the event name
    (avoids 'struck near 2025 … Earthquake' / 'Earthquake earthquake')."""
    p = re.sub(r"\s+Earthquake$", "", re.sub(r"^\d{4}\s+", "", e.get("place", "").strip()))
    return p or e.get("place", "")

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "raw", "usgs")
S2T_DIR = os.path.join(RAW, "s2t")
UNIT, FREQ = "aftershocks_per_day_m3.5", "daily"


def _mainshock_text(e):
    """Real USGS facts (title is verbatim) describing the mainshock."""
    ts = " A tsunami was observed or warned." if e.get("tsunami") else ""
    return (f"{e['title']}. The magnitude {e['mag']} earthquake struck near {_place(e)} on "
            f"{e['date']} at {e['depth_km']} km depth.{ts}")


def _describe(e):
    s = e["series"]; eid = e["id"]
    cache = os.path.join(S2T_DIR, f"{eid}.txt")
    # grounding holds the legit non-count numbers (magnitude/depth/threshold/date/radius/day range)
    ground = (f"magnitude {e['mag']}, depth {e['depth_km']} km, aftershocks above M3.5, "
              f"on {e['date']}, within 250 km, days 0 to {len(s) - 1}")
    if os.path.exists(cache):
        c = open(cache).read().strip()
        if is_consistent(c, s, grounding_text=ground, ints=True):   # counts are integers
            return c, "derived_generated", True
    facts = (f"- M{e['mag']} earthquake near {_place(e)} on {e['date']}\n"
             f"- daily M3.5+ aftershock counts, days 0-{len(s)-1}: {', '.join(map(str, s))}\n"
             f"- day-0 count {s[0]}, peak {max(s)}, last-day {s[-1]}, total {sum(s)} over {len(s)} days")
    template = (f"Following the M{e['mag']} {_place(e)} earthquake on {e['date']}, daily M3.5+ "
                f"aftershock counts began at {s[0]}, peaked at {max(s)}, and decayed to {s[-1]} "
                f"by day {len(s)-1}, totaling {sum(s)} aftershocks.")
    desc, tag, verified = grounded_describe(
        "You write a concise, factual description for a dataset. Use ONLY the provided counts; "
        "invent no numbers.",
        "Facts:\n" + facts + "\n\nIn 2-3 sentences, describe the aftershock decay (how it started, "
        "peaked, and tapered). State only the given counts.",
        s, grounding_text=ground, fallback=(template, "derived_template"), ints=True)
    if tag == "derived_generated":
        os.makedirs(S2T_DIR, exist_ok=True); open(cache, "w").write(desc)
    return desc, tag, verified


def pairs():
    events = json.load(open(os.path.join(RAW, "events.json")))
    for e in events:
        s = e["series"]
        if len(s) < 6:
            continue
        url = f"https://earthquake.usgs.gov/earthquakes/eventpage/{e['id']}"
        common = {"dataset": "usgs_quakes", "source": "USGS FDSN event API (aftershock sequence)",
                  "event_id": e["id"], "place": e["place"], "magnitude": e["mag"],
                  "event_date": e["date"], "text_url": url, "ts_url": url}

        # A — text -> ts (ts_forecast): mainshock text + early aftershocks -> rest of the decay
        if len(s) >= 8:
            yield TextToTs(
                user_text=_mainshock_text(e) + " Given the first few days of aftershock activity, "
                          "forecast the daily M3.5+ aftershock count.",
                history=[float(x) for x in s[:5]], future=[float(x) for x in s[5:]],
                series_name="daily M3.5+ aftershock count", unit=UNIT, freq=FREQ,
                meta={**common, "series_id": f"{e['id']}_t2s", "direction": "text_to_ts"},
                text_source="usgs_event", is_generated="real",
                knowledge_time=f"{e['date']}T00:00:00Z")

        # B — ts -> text (text_desc): aftershock sequence -> grounded, verified description
        desc, tag, verified = _describe(e)
        yield TsToText(
            user_intro=(f"Daily M3.5+ aftershock counts within 250 km of the M{e['mag']} "
                        f"{_place(e)} earthquake ({e['date']}), days 0-{len(s)-1}. "
                        f"Describe the aftershock sequence."),
            series=[{"name": "daily aftershock count", "values": [float(x) for x in s],
                     "unit": UNIT, "freq": FREQ}],
            answer=desc,
            meta={**common, "series_id": f"{e['id']}_s2t", "direction": "ts_to_text",
                  "grounding": "aftershock_series", "numeric_verified": verified},
            text_source=("usgs_grounded_qwen" if tag == "derived_generated" else "usgs_template"),
            is_generated=tag)
