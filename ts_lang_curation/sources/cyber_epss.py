# sources/cyber_epss.py — thin adapter for Cybersecurity (#16 on the sheet, P1).
# Distinctive asset: the EPSS exploit-probability TRAJECTORY of a vulnerability (daily %),
# which spikes when a CVE becomes actively exploited — paired with the CVE's official text.
#   text->ts (ts_forecast): CVE description + early EPSS days -> forecast the exploitation-risk rise.
#   ts->text (text_desc):   EPSS trajectory -> grounded, numerically-verified risk description.
# Sources (all public, no key): CISA KEV (known-exploited catalog + text) + FIRST EPSS (series).
# Cached in raw/cyber/events.json (see README). Series = EPSS as a percent (0-100).
import os, json
from core.schema import TsToText, TextToTs
from core.generate import grounded_describe
from core.verify import is_consistent

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "raw", "cyber")
S2T_DIR = os.path.join(RAW, "s2t")
UNIT, FREQ = "epss_exploit_probability_pct", "daily"
HIST_CYCLE = [5, 7, 10, 14]                 # vary the history length (Defu §3: not fixed at 7)


def _vp(e):
    """vendor + product, de-duplicated (KEV often repeats: vendor 'Mirasvit', product
    'Mirasvit Knowledge Base' -> 'Mirasvit Knowledge Base')."""
    v, p = e.get("vendor", "").strip(), e.get("product", "").strip()
    if p.lower().startswith(v.lower()):
        return p
    return (v + " " + p).strip()


def _cve_text(e):
    """Real CISA KEV text for the vulnerability."""
    bits = [_vp(e), e.get("name", "")]
    head = " — ".join(b for b in bits if b)
    return f"{e['cve']}: {head}. {e.get('desc','')}".strip()


def _describe(e):
    s = e["series"]
    cache = os.path.join(S2T_DIR, f"{e['cve']}.txt")
    if os.path.exists(cache):
        c = open(cache).read().strip()
        if is_consistent(c, s):
            return c, "derived_generated", True
    facts = (f"- CVE {e['cve']} ({_vp(e)})\n"
             f"- daily EPSS exploit-probability (percent), {e['dates'][0]} to {e['dates'][-1]}: "
             f"{', '.join(map(str, s))}\n"
             f"- start {s[0]}%, end {s[-1]}%, min {min(s)}%, max {max(s)}%")
    template = (f"The EPSS exploit probability for {e['cve']} rose from {s[0]}% to {s[-1]}% "
                f"between {e['dates'][0]} and {e['dates'][-1]}, peaking at {max(s)}% — indicating "
                f"sharply increasing likelihood of active exploitation.")
    desc, tag, verified = grounded_describe(
        "You write a concise, factual description. Use ONLY the provided percentages; invent no "
        "numbers. Describe the subject directly — never refer to 'this dataset', 'this series', "
        "or 'the data'.",
        "Facts:\n" + facts + "\n\nIn 2-3 sentences, describe how this CVE's exploitation risk "
        "evolved over the window. Vary your wording and sentence structure; you need not follow a "
        "fixed template. State only the given percentages; do not mention 'dataset' or 'series'.",
        s, fallback=(template, "derived_template"), temperature=0.6, max_tokens=220)
    if tag == "derived_generated":
        os.makedirs(S2T_DIR, exist_ok=True); open(cache, "w").write(desc)
    return desc, tag, verified


def pairs():
    events = json.load(open(os.path.join(RAW, "events.json")))
    for i, e in enumerate(events):
        s = e["series"]
        if len(s) < 10:
            continue
        h = max(4, min(HIST_CYCLE[i % len(HIST_CYCLE)], len(s) - 5))   # varied history; ≥5 future
        flat_ctx = (max(s[:h]) - min(s[:h])) < 0.5                     # near-flat pre-spike history
        d0, d1 = e["dates"][0], e["dates"][-1]
        url = f"https://www.cisa.gov/known-exploited-vulnerabilities-catalog"
        nvd = f"https://nvd.nist.gov/vuln/detail/{e['cve']}"
        common = {"dataset": "cyber_epss", "source": "CISA KEV + FIRST EPSS",
                  "cve": e["cve"], "vendor": e["vendor"], "product": e["product"],
                  "event_date": e["dateAdded"], "text_url": nvd, "ts_url": url}

        # A — text -> ts (ts_forecast): CVE text + early EPSS days -> the rise
        yield TextToTs(
            user_text=_cve_text(e) + (
                      f" As of {e['dateAdded']}, given this vulnerability and the early EPSS "
                      f"exploit-probability history, forecast the daily EPSS probability (%)."
                      if e.get("dateAdded") else
                      " Given this vulnerability and the early EPSS exploit-"
                      "probability history, forecast the daily EPSS probability (%)."),
            history=s[:h], future=s[h:],
            series_name="daily EPSS exploit probability (%)", unit=UNIT, freq=FREQ,
            meta={**common, "series_id": f"{e['cve']}_t2s", "direction": "text_to_ts",
                  "history_days": h, "flat_context": flat_ctx},
            text_source="cisa_kev", is_generated="real",
            knowledge_time=f"{e['dateAdded']}T00:00:00Z" if e.get("dateAdded") else "")

        # B — ts -> text (text_desc): EPSS trajectory -> grounded, verified description
        desc, tag, verified = _describe(e)
        yield TsToText(
            user_intro=(f"Daily EPSS exploit-probability (%) for {e['cve']} ({_vp(e)}), {d0} to "
                        f"{d1}. Describe how the exploitation risk evolved."),
            series=[{"name": "EPSS exploit probability (%)", "values": s, "unit": UNIT, "freq": FREQ}],
            answer=desc,
            meta={**common, "series_id": f"{e['cve']}_s2t", "direction": "ts_to_text",
                  "grounding": "epss_series", "numeric_verified": verified},
            text_source=("epss_grounded_qwen" if tag == "derived_generated" else "epss_template"),
            is_generated=tag)
