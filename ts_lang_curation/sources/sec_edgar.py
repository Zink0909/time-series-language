# sources/sec_edgar.py — thin adapter for SEC EDGAR (#13 on the sheet).
# Distinctive asset vs FRED/FNSPID: XBRL *structured fundamentals* (annual revenue, net income)
# as native time series, paired with the company's own 10-K text.
#   ts->text  (text_desc): revenue [+ net income] series -> grounded, verified description.
#             Mix of univariate (revenue) and multivariate (revenue + net income) rows.
#   text->ts  (ts_forecast[_covariates]): real 10-K MD&A revenue-driver text + prior-year
#             history -> latest year; some rows add net income as a covariate.
# Data-driven: every company with a cached raw/sec/<TK>_revenue.json is built. Net income
# (<TK>_netincome.json) is used as covariate/2nd channel when its fiscal years align.
# Public data, no API key; SEC requires a descriptive User-Agent (set at fetch time).
import os, re, json, glob, html as _html
from datetime import date
from core.schema import TsToText, TextToTs
from core.generate import grounded_describe
from core.verify import is_consistent, year_value_errors

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "raw", "sec")
S2T_DIR = os.path.join(RAW, "s2t")

_DRIVER = re.compile(
    r"(?:net sales|total net sales|total revenues?|revenues?|total revenue)\b[^.]{0,40}?"
    r"(?:increased|decreased|grew|rose|declined)\b[^.]*?"
    r"(?:due to|driven by|primarily|reflect|as a result|attributable to)[^.]*\.", re.I)


def _days(a, b):
    y1, m1, d1 = map(int, a.split("-")); y2, m2, d2 = map(int, b.split("-"))
    return (date(y2, m2, d2) - date(y1, m1, d1)).days


def _annual(path):
    """{fiscal_year_end: value_usd_billions} for full fiscal years, deduped."""
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    if "USD" not in d.get("units", {}):
        return {}
    fy = {}
    for x in d["units"]["USD"]:
        if x.get("form") != "10-K" or x.get("fp") != "FY":
            continue
        s, e = x.get("start"), x.get("end")
        if s and e and 350 <= _days(s, e) <= 380:
            fy[e] = round(x["val"] / 1e9, 2)
    return fy


def _entity(path):
    return json.load(open(path)).get("entityName", "")


def _plain(htm_path):
    raw = open(htm_path, encoding="utf-8", errors="ignore").read()
    return _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)))


def _mda_drivers(ticker, max_sentences=4):
    fp = os.path.join(RAW, f"{ticker}_10K.htm")
    if not os.path.exists(fp):
        return ""
    txt = _plain(fp)
    i = txt.rfind("Management" + chr(0x2019) + "s Discussion and Analysis")
    if i < 0:
        i = txt.rfind("Management's Discussion and Analysis")
    if i < 0:
        i = txt.rfind("Results of Operations")
    region = txt[i:i + 16000] if i > 0 else txt
    seen, out = set(), []
    for m in _DRIVER.finditer(region):
        s = re.sub(r"\s+", " ", m.group(0)).strip()
        k = s.lower()
        if k not in seen and 30 < len(s) < 320:
            seen.add(k); out.append(s)
        if len(out) >= max_sentences:
            break
    return " ".join(out) if len(out) >= 1 else ""     # >=1 clean total-revenue driver sentence


def _facts(name, years, channels):
    """channels = [(label, vals)]; one fact block per channel over the aligned years."""
    lines = [f"- Company: {name}", f"- Fiscal years: {years[0][:4]} to {years[-1][:4]} (USD billions)"]
    for label, vals in channels:
        lines.append(f"- {label}: " + ", ".join(f"{e[:4]}=${v}" for e, v in zip(years, vals)))
        lines.append(f"  start ${vals[0]}B, end ${vals[-1]}B, min ${min(vals)}B, max ${max(vals)}B, "
                     f"net change ${round(vals[-1] - vals[0], 2)}B")
    return "\n".join(lines)


def _describe(ticker, name, years, channels):
    """ts->text description over 1 (revenue) or 2 (revenue+net income) channels. Cached."""
    multi = len(channels) > 1
    cache = os.path.join(S2T_DIR, f"{ticker}{'_mv' if multi else ''}.txt")
    prim = channels[0][1]
    extra = [v for _, v in channels[1:]]
    year_map = {int(y[:4]): v for y, v in zip(years, prim)}    # catch wrong fiscal-year labels
    if os.path.exists(cache):
        cached = open(cache).read().strip()
        if is_consistent(cached, prim, extra_values=extra) and not year_value_errors(cached, year_map):
            return cached, "derived_generated", True
    labels = " and ".join(l for l, _ in channels)
    template = "; ".join(
        f"{l} went from ${v[0]}B ({years[0][:4]}) to ${v[-1]}B ({years[-1][:4]})"
        for l, v in channels) + "."
    ask = ("In 2-3 sentences, describe the trajectory of each series and how they relate. "
           if multi else "In 2-3 sentences, describe the revenue trajectory (start, end, change, peak/dip). ")
    desc, tag, verified = grounded_describe(
        "You write a concise, factual description for a dataset. Use ONLY the provided numbers; "
        "invent no figures, growth rates, or events.",
        "Facts:\n" + _facts(name, years, channels) + f"\n\n{ask}State dollar figures only; no percentages.",
        prim, fallback=(template, "derived_template"), extra_values=extra, year_map=year_map)
    if tag == "derived_generated":
        os.makedirs(S2T_DIR, exist_ok=True)
        open(cache, "w").write(desc)
    return desc, tag, verified


def pairs():
    cik = json.load(open(os.path.join(RAW, "_cik.json")))
    meta = json.load(open(os.path.join(RAW, "_10k_meta.json")))
    for i, rev_path in enumerate(sorted(glob.glob(os.path.join(RAW, "*_revenue.json")))):
        tk = os.path.basename(rev_path).replace("_revenue.json", "")
        rev = _annual(rev_path)
        if len(rev) < 3:
            continue
        name = _entity(rev_path)
        years = sorted(rev)[-7:]
        rev_vals = [rev[e] for e in years]
        y0, y1 = years[0][:4], years[-1][:4]
        c = cik.get(tk, "")
        ts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c}.json" if c else ""
        # net income aligned to the same fiscal-year-ends (covariate / 2nd channel)
        ni = _annual(os.path.join(RAW, f"{tk}_netincome.json"))
        ni_vals = [ni[e] for e in years] if all(e in ni for e in years) else None

        # --- A: ts -> text (text_desc) — mix univariate / multivariate (revenue + net income) ---
        multivar = ni_vals is not None and i % 2 == 0
        channels = [("annual total revenue", rev_vals)] + ([("net income", ni_vals)] if multivar else [])
        desc, tag, verified = _describe(tk, name, years, channels)
        series = [{"name": "annual total revenue", "values": rev_vals, "unit": "usd_billions", "freq": "annual"}]
        if multivar:
            series.append({"name": "net income", "values": ni_vals, "unit": "usd_billions", "freq": "annual"})
        intro = (f"{name} fundamentals (USD billions), fiscal {y0}-{y1}. Describe the trajectories."
                 if multivar else
                 f"{name} annual total revenue (USD billions), fiscal {y0}-{y1}. Describe the revenue trajectory.")
        yield TsToText(
            user_intro=intro, series=series, answer=desc,
            meta={"dataset": "sec_edgar", "source": "SEC EDGAR XBRL us-gaap revenue/net income",
                  "ticker": tk, "cik": c, "series_id": f"{tk}_{y0}_{y1}_s2t",
                  "event_date": years[-1], "direction": "ts_to_text",
                  "channels": [l for l, _ in channels], "grounding": "fundamentals",
                  "numeric_verified": verified, "text_url": ts_url, "ts_url": ts_url},
            text_source=("xbrl_grounded_qwen" if tag == "derived_generated" else "xbrl_template"),
            is_generated=tag)

        # --- B: text -> ts (ts_forecast[_covariates]) — only where 10-K MD&A is clean ---
        drivers = _mda_drivers(tk)
        if drivers and len(rev_vals) >= 4:
            m = meta.get(tk, {})
            filed, accn = m.get("filed", ""), m.get("accn", "").replace("-", "")
            filing_url = (f"https://www.sec.gov/Archives/edgar/data/{int(c)}/{accn}/{m.get('doc','')}"
                          if accn and c else ts_url)
            covs = []
            if ni_vals is not None and i % 2 == 1:        # covariate mixup: ~half of forecast rows
                covs = [{"name": "net income", "values": ni_vals[:-1],
                         "unit": "usd_billions", "freq": "annual"}]
            yield TextToTs(
                user_text=(f"From {name}'s Form 10-K (filed {filed}), management's discussion of "
                           f"results: {drivers} Given this and the prior-year history, forecast "
                           f"{name}'s revenue."),
                history=rev_vals[:-1], future=rev_vals[-1:],
                series_name="annual total revenue (USD billions)", unit="usd_billions", freq="annual",
                covariates=covs,
                meta={"dataset": "sec_edgar", "source": "SEC EDGAR 10-K (MD&A) + XBRL us-gaap revenue",
                      "ticker": tk, "cik": c, "series_id": f"{tk}_{y0}_{y1}_t2s",
                      "event_date": filed, "direction": "text_to_ts",
                      "text_url": filing_url, "ts_url": ts_url},
                text_source="10k_mdna", is_generated="real",
                knowledge_time=f"{filed}T00:00:00Z" if filed else "")
