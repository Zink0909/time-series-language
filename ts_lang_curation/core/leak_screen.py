#!/usr/bin/env python3
"""Content-layer leakage screens for flywheel causes — SHARED so every source that synthesizes a cause
defines the screen once (CLAUDE §core: shared logic in core).

Beyond the TIMESTAMP cutoff (knowledge_time < forecast origin, enforced by the leakage audit), the TEXT
itself must not leak the answer. `forecast_leak` catches a SOFT leak the timestamp check misses: a
forward-looking projection/estimate coupled with a direction or price — an expert "forecast that prices
will rise", a valuation target, an EIA price estimate — tells the model which way the series goes, even
though the projection was published before the origin. Deliberately NOT tripped by a bare future date
like "tax season 2023 approached" (no projection + direction), so genuine pre-origin context survives.

  from core.leak_screen import forecast_leak
"""
import re

# A soft leak is a forward-looking projection ABOUT PRICE / VALUATION / a directional move — NOT any
# mention of a "forecast". "OPEC's demand growth forecast" or "analysts' scenarios" are legitimate
# fundamentals context and must survive; "price forecast", "valuation target", "EIA estimates for
# prices", "projecting a rally/reach $X" reveal the answer. So we match the SPECIFIC price/direction
# projection, not the bare cue.
_PRICE_PROJ = re.compile(
    r"\b(price|valuation)\s+(target|forecast|estimate|projection|outlook)s?\b"          # price/valuation target
    r"|\b(forecast|estimate|projection|outlook)s?\s+(for|of)\s+[\w\s,'-]{0,25}\bprices?\b"  # estimates for ... prices
    r"|\bprice\s+targets?\b|\bvaluation\s+targets?\b"
    r"|\bprojecting\b[\w\s,'-]{0,45}\b(rise|rising|rally|surge|surging|reach\w*|higher|increase|"
    r"appreciat\w*|upside|bullish|climb\w*)\b"                                           # projecting ... a move
    r"|\b(forecast|estimat|project|predict|expect|anticipat)\w*\b[\w\s,'-]{0,35}\b(will|to|would|could)\s+"
    r"(rise|rising|rally|surge|reach|climb|hit|increase|jump|soar|higher|double|appreciate)\b",
    re.I)


def forecast_leak(text):
    """True if the text contains a forward-looking PRICE/valuation/directional projection = soft leak.
    Bare fundamentals forecasts ('demand growth forecast', 'OPEC outlook') are intentionally allowed."""
    return bool(_PRICE_PROJ.search(text or ""))


# GENERAL outcome-direction leak: the conditioning text ASSERTS which way the target series moves next
# ("... will go up over the coming days", "expected to rise", "set to fall"). This is the most direct
# poison — it hands the model the answer — yet the price-specific forecast_leak misses it. Deliberately
# NOT tripped by past/descriptive moves ("the yield rose in December", "inflation has increased"), only by
# FORWARD-looking direction (a modal/future cue + a directional verb).
_DIR_LEAK = re.compile(
    r"\b(will|won'?t|going to|gonna|expected to|set to|likely to|about to|poised to|forecast to|"
    r"projected to|on track to|due to)\s+(?:go\s+|move\s+|head\s+|trend\s+|continue\s+)?"
    r"(up|down|higher|lower|rise|rising|risen|fall|falling|fallen|drop|dropping|climb|climbing|"
    r"increase|increasing|decrease|decreasing|surge|surging|plunge|plunging|jump|soar|rally|decline|"
    r"gain|gaining|weaken|strengthen|spike|spiking)\b"
    r"|\b(next|coming|following|subsequent|upcoming)\s+(day|days|week|weeks|month|months|period|quarter)s?"
    r"[\w\s,'-]{0,40}\b(rise|rising|fall|falling|higher|lower|increase|decrease|up|down|surge|plunge)\b",
    re.I)


def direction_leak(text):
    """HIGH-RECALL / HIGH-FALSE-POSITIVE review FLAG, NOT a hard gate. True if the text forward-asserts A
    direction — but a regex CANNOT tell "the TARGET series will move" (a real leak) from "unemployment /
    inflation will rise" (legitimate forward world-knowledge about ANOTHER quantity). Meta-validation: it
    caught a planted poison 30/30 (text literally = the answer) but also flagged 26/147 real FOMC records
    (all 'the unemployment rate will decline...' = fine content). So use it to SURFACE candidates for human
    / LLM-judge review with semantic target-binding, never to auto-reject. (Same high-FP fate as the
    future-year and price-projection screens.)"""
    return bool(_DIR_LEAK.search(text or ""))


def future_year_leak(text, origin_date):
    """True if the text names any year AFTER the forecast origin's year. For PRICE series (commodity /
    FX / crypto), a cause that reaches into a future year is almost always a projection/outlook that
    leaks direction — the phrasings are endless ('2025 outlook', 'parity by 2025', 'targets for 2026'),
    so instead of chasing them with regex we forbid future years outright. Do NOT use this for Wikipedia
    attention series, where a future year is usually just the article's own title (e.g. '2022 World Cup')
    and harmless to a pageview forecast."""
    if not origin_date or len(origin_date) < 4 or not origin_date[:4].isdigit():
        return False
    oy = int(origin_date[:4])
    return any(int(y) > oy for y in re.findall(r"\b(20\d{2})\b", text or ""))
