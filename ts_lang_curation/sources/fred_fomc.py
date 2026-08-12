# sources/fred_fomc.py — thin adapter. Knows ONLY how to fetch/parse/pair FRED+FOMC.
# Format/validation/output are handled by core/. Reads from raw/ cache (see README to refresh).
import os, re, glob, html as _html
from core.schema import TsToText, TextToTs
from core.verify import is_consistent
from core.generate import grounded_describe

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "raw")
DGS2_CACHE = os.path.join(RAW, "DGS2_2008-01-01_2026-06-30.csv")
DGS10_CACHE = os.path.join(RAW, "DGS10_2008-01-01_2026-06-30.csv")
FOMC_DIR = os.path.join(RAW, "fomc")
S2T_DIR = os.path.join(RAW, "fomc_s2t")        # cached Qwen-generated ts->text descriptions
BEFORE_CYCLE = [20, 30, 45, 60]                # varied history lengths (rates have long memory)


def _s2t_description(date, b0, b1, vals, decision):
    """Grounded ts->text description via Qwen, with a numeric "reflection" check
    (every figure it states must match the real series or the policy target).
    Returns (text, is_generated_tag, verified). Generate -> verify -> one stricter
    retry -> fall back to the verbatim decision. Only verified text is cached.
    Reflection idea: arXiv 2409.17515 (reflection loop) + 2602.05646 (consistency filter)."""
    cache = os.path.join(S2T_DIR, f"{date}.txt")
    if os.path.exists(cache):
        cached = open(cache).read().strip()
        if is_consistent(cached, vals, decision):        # re-verify, never trust stale cache
            return cached, "derived_generated", True
    facts = (f"- FOMC meeting date: {date}\n"
             f"- Window: {b0} to {b1} (daily, business days)\n"
             f"- 2-year Treasury yield: start {vals[0]}%, end {vals[-1]}%, min {min(vals)}%, "
             f"max {max(vals)}%, net change {vals[-1] - vals[0]:+.2f} percentage points\n"
             f'- FOMC decision (verbatim): "{decision}"')
    desc, tag, verified = grounded_describe(
        "You write a concise, factual market description for a dataset. Use ONLY the "
        "provided facts; invent no numbers or events.",
        "Facts:\n" + facts + "\n\nIn 2-3 sentences, describe how the 2-year Treasury yield "
        "moved over this window and connect it to the FOMC decision. Be concise.",
        vals, grounding_text=decision,
        fallback=(decision, "real"))                      # verbatim decision is grounded by definition
    if tag == "derived_generated":
        os.makedirs(S2T_DIR, exist_ok=True)
        open(cache, "w").write(desc)
    return desc, tag, verified


def _load(path):
    rows = []
    for line in open(path).read().splitlines()[1:]:
        d, v = line.split(",")
        if v.strip() and v.strip() != ".":
            rows.append((d, float(v)))
    dates = [d for d, _ in rows]
    return rows, dates, {d: i for i, d in enumerate(dates)}


def _load_dgs2():
    return _load(DGS2_CACHE)


def _extract(htmltext):
    """Return (full_statement_text, decision_sentence) from a Fed press-release page."""
    paras = []
    for p in re.findall(r'<p[^>]*>(.*?)</p>', htmltext, re.S):
        t = _html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', p)).strip())
        if len(t) > 40:
            paras.append(t)
    body, started = [], False
    for p in paras:
        if not started:                                   # statement starts after "...Share"
            # both "For release at <time> ... Share" (2016+) and "For immediate release Share" (2008-2015)
            m = re.search(r'For .*?release.*?\bShare\b\s*(.*)', p)
            if m:
                started = True
                if m.group(1).strip():
                    body.append(m.group(1).strip())
            continue
        if p.lower().startswith("voting for") or p.startswith("Board of Governors"):
            break                                         # stop before voting roster / footer
        body.append(p)
    statement = " ".join(body).strip()
    decision = ""
    for sent in re.split(r'(?<=[.])\s+', statement):
        if "target range for the federal funds rate" in sent.lower():
            decision = sent.strip(); break
    return statement, decision


def _compress(statement, decision, target=300):
    """Extractive, verbatim compression of a long FOMC statement to ~target words so the text fits the
    text encoder (FinBERT truncates ~512 tokens) instead of being silently cut. Keep the leading economic-
    assessment sentences up to the budget, and GUARANTEE the policy-decision sentence is present."""
    if len(statement.split()) <= target:
        return statement
    sents = re.split(r'(?<=[.])\s+', statement)
    out, wc = [], 0
    for s in sents:
        w = len(s.split())
        if wc + w > target and out:
            break
        out.append(s); wc += w
    text = " ".join(out).strip()
    if decision and decision not in text:                 # never drop the rate decision
        text = text + " " + decision
    return text.strip()


def _window(rows, dates, idx, date, before, after):
    if date not in idx:                                   # snap to nearest trading day <= date
        cand = [d for d in dates if d <= date]
        if not cand:
            return None
        date = cand[-1]
    i = idx[date]
    seg = rows[max(0, i - before): min(len(rows), i + after + 1)]
    return [v for _, v in seg], seg[0][0], seg[-1][0]


def _split(rows, dates, idx, date, before, after):
    """History (up to & incl the decision day) + future (after) for a forecast row."""
    if date not in idx:
        cand = [d for d in dates if d <= date]
        if not cand:
            return None
        date = cand[-1]
    i = idx[date]
    hist = rows[max(0, i - before): i + 1]
    fut = rows[i + 1: i + after + 1]
    if len(hist) < 3 or len(fut) < 1:
        return None
    return ([v for _, v in hist], [v for _, v in fut], [d for d, _ in hist], hist[0][0], fut[-1][0])


def pairs():
    rows, dates, idx = _load_dgs2()
    dgs10 = {d: v for d, v in _load(DGS10_CACHE)[0]} if os.path.exists(DGS10_CACHE) else {}
    for mi, fp in enumerate(sorted(glob.glob(os.path.join(FOMC_DIR, "*.html")))):
        d8 = os.path.basename(fp)[:8]
        date = f"{d8[:4]}-{d8[4:6]}-{d8[6:8]}"
        statement, decision = _extract(open(fp).read())
        if not statement:
            continue
        stmt_url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{d8}a.htm"
        ts_url = "https://fred.stlouisfed.org/series/DGS2"

        # A — text -> ts (ts_forecast[_covariates]): FOMC statement + pre-decision history ->
        #     post-decision future. Loss only on the post-decision suffix (loss_start=H).
        #     Window length varies per meeting; ~half add the 10y yield as a covariate.
        before = BEFORE_CYCLE[mi % len(BEFORE_CYCLE)]
        sp = _split(rows, dates, idx, date, before, 10)
        if sp:
            hist, fut, hdates, h0, f1 = sp
            covs = []
            if mi % 2 == 0 and dgs10:                         # covariate mixup
                c10 = [dgs10[d] for d in hdates if d in dgs10]
                if len(c10) == len(hist):
                    covs = [{"name": "10-year U.S. Treasury yield (%)", "values": c10,
                             "unit": "pct_yield_10y_treasury", "freq": "business_daily"}]
            cov_txt = " The 10-year Treasury yield over the same history is also given." if covs else ""
            yield TextToTs(
                user_text=(f"FOMC statement, {date}: {_compress(statement, decision)} Given this "
                           f"statement and the 2-year U.S. Treasury yield history through {date}, "
                           f"forecast the yield.{cov_txt}"),
                history=hist, future=fut,
                series_name="2-year U.S. Treasury yield (%)",
                unit="pct_yield_2y_treasury", freq="business_daily", covariates=covs,
                meta={"dataset": "fred_fomc", "source": "FRED DGS2 + Federal Reserve FOMC statement",
                      "series_id": f"DGS2_{date}_FOMC_t2s", "fred_series": "DGS2",
                      "event_date": date, "direction": "text_to_ts", "history_days": before,
                      "history_range": [h0, date], "future_range_end": f1,
                      "text_url": stmt_url, "ts_url": ts_url},
                text_source="fomc_statement", is_generated="real",
                knowledge_time=f"{date}T18:00:00Z")

        # B — ts -> text (text_desc): yield window -> grounded, numerically-verified description.
        wB = _window(rows, dates, idx, date, 10, 10)
        if wB and decision:
            vals, b0, b1 = wB
            desc, gen, verified = _s2t_description(date, b0, b1, vals, decision)
            yield TsToText(
                user_intro=(f"2-year U.S. Treasury yield (%), daily, {b0} to {b1}. Describe how the "
                            f"yield moved over this window and connect it to the FOMC decision."),
                series=[{"name": "2-year Treasury yield", "values": vals,
                         "unit": "pct_yield_2y_treasury", "freq": "business_daily"}],
                answer=desc,
                meta={"dataset": "fred_fomc", "source": "FRED DGS2 + Federal Reserve FOMC statement",
                      "series_id": f"DGS2_{date}_FOMC_s2t", "fred_series": "DGS2",
                      "event_date": date, "direction": "ts_to_text",
                      "grounding": "series_stats+fomc_decision", "numeric_verified": verified,
                      "text_url": stmt_url, "ts_url": ts_url},
                text_source=("fomc_grounded_qwen" if gen != "real" else "fomc_statement_decision_sentence"),
                is_generated=gen)
