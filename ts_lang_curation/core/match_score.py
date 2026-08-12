#!/usr/bin/env python3
"""Text↔series CORRESPONDENCE (match) score — a new, INDEPENDENT quality dimension for the flywheel.

The existing gates check: is the text faithful to its evidence (faithfulness), does it leak the answer
(leakage), do its numbers reconcile (verify). None of them ask the question Defu's world-knowledge data
actually needs: **does the conditioning text correspond to WHY this series moves at the event, or is it
generic standing description?** A Wikipedia lead like "Scott Hall (born 1958) is a retired wrestler"
is faithful, leakage-safe, and numerically fine — yet it says nothing about why the pageviews spiked on
a given day. That is a weak text↔series match, and for world-knowledge continued-pretraining (numeric ↔
text must "match up") it should be measured, not silently mixed in.

Design (不自证 / reproducible): the score uses ONLY the text's own structure + the record's event
timestamp — no LLM re-judging the LLM's output, no dependence on the generator. It is deterministic, so
a frozen replay reproduces it exactly. It is validated the honest way: on data with KNOWN coupling
(event-grounded commodity causes should score high; standing wiki bios should score low), not by the
generator's own say-so.

Two independent signals:
  1. EVENT-TIME ANCHORING — does the text refer to the event's time window at all? A bio's years
     (born 1958, …) sit far from the spike year; a real cause names the period of the move.
  2. EVENTIVE vs STATIVE language — "announced / halted / died / resigned" (something happened) vs
     "is / was / born / known for" (what something durably is).

  from core.match_score import match_score
  score, label, detail = match_score(user_text, event_date="2021-04-11")
"""
import re, calendar

# Something HAPPENED (event-driven, the kind of text that explains a move)
_EVENTIVE = re.compile(
    r"\b(announced?|declared?|died|passed away|launched?|released?|unveiled?|reported|resigned?|"
    r"elected?|arrested?|charged|signed|acquired?|merged?|filed|sued|banned?|approved?|rejected?|"
    r"halted?|suspended?|shut|reopened?|struck|hit|erupted?|crashed?|won|lost|defeated|beat|"
    r"recalled?|voted?|passed|introduced?|imposed?|cut|raised?|hiked?|slashed?|warned?|confirmed?|"
    r"discovered?|attacked?|invaded?|withdrew|resumed?|delayed?|postponed?|canceled|cancelled|"
    r"agreed?|rejected?|nominated?|appointed?|fired|laid off|struck down|blocked?)\b", re.I)

# What something durably IS (standing description / biography)
_STATIVE = re.compile(
    r"\b(is|are|was|were|been|born|is a|is an|is the|was a|was an|known for|based in|located in|"
    r"refers to|is a type|is a form|consists of|comprises|is one of|has been|had been|is named|"
    r"is best known|is home to|is situated)\b", re.I)

_RECENCY = re.compile(r"\b(recent(ly)?|this week|this month|latest|current(ly)?|amid|following|"
                      r"in response to|days before|ahead of|as .* (concerns|tensions|fears|hopes))\b", re.I)

# Standing-description / biography signatures — the core of WEAK coupling. These patterns are few and
# stable (unlike the open-ended set of event verbs), so detecting THEM and defaulting everything else to
# 'coupled' is far more robust than enumerating what "happened".
_BIO_OPEN = re.compile(r"^\s*[A-Z][^.\n]{0,90}?\(\s*;?\s*born\b", re.I)          # "Name (born …" lead
_BIO_DEF = re.compile(
    r"\bis (?:a|an|the) (?:[a-z]+ ){0,3}"
    r"(wrestler|singer|actor|actress|rapper|politician|footballer|player|athlete|writer|author|"
    r"musician|band|group|company|corporation|city|town|country|state|film|movie|song|album|series|"
    r"novel|video game|character|team|league|university|river|mountain|island|festival|holiday|"
    r"television|drama|franchise|organization|organisation|breed|species)\b", re.I)


def _years(text):
    return set(int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text))


def match_score(text, event_date=None):
    """Return (score in [0,1], label 'coupled'|'weak', detail dict). event_date = 'YYYY-MM-DD'.

    Coupling = does the text describe SOMETHING THAT HAPPENED (event-driven, explains the move) vs.
    WHAT SOMETHING DURABLY IS (standing bio). We deliberately do NOT reward the presence of a year in
    the text: leakage-safe causes strip dates, while bios carry an (irrelevant) birth year — so a
    year-anchor would penalize exactly the coupled records and reward the weak ones. Instead:
      - EVENTIVE vs STATIVE language (the main signal),
      - a recency marker ("amid", "following", "ahead of"), and
      - the event's MONTH named (e.g. "in July"), the one temporal anchor a leakage-safe cause keeps."""
    text = text or ""
    n_ev = len(_EVENTIVE.findall(text))
    n_st = len(_STATIVE.findall(text))
    has_recency = bool(_RECENCY.search(text))

    month_named = False
    if event_date and len(event_date) >= 7 and event_date[5:7].isdigit():
        mon = calendar.month_name[int(event_date[5:7])]
        month_named = bool(mon and re.search(r"\b" + mon + r"\b", text, re.I))

    bio_like = bool(_BIO_OPEN.search(text) or _BIO_DEF.search(text))
    event_signal = (n_ev > 0) or has_recency or month_named
    stative_dominant = (n_st >= 2 and n_ev == 0)
    # WEAK = looks like a standing bio/definition AND carries no event signal. Everything else defaults
    # to coupled — we only demote records that clearly describe "what something is", not "what happened".
    weak = (bio_like or stative_dominant) and not event_signal

    # Continuous coupling score (for ranking / down-weighting), roughly consistent with the label.
    score = round(max(0.0, min(1.0,
                      0.5 * min(1, n_ev) + 0.2 * (n_ev >= 2) + 0.2 * has_recency
                      + 0.15 * month_named - (0.55 if bio_like else 0.0) - 0.06 * max(0, n_st - 1))), 3)
    label = "weak" if weak else "coupled"
    return score, label, {"eventive_n": n_ev, "stative_n": n_st, "recency": has_recency,
                          "event_month_named": month_named, "bio_like": bio_like}


def cause_text_of(rec):
    """Extract the conditioning text (user turn up to the '(history=' / 'Given ...' marker) from a record."""
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", rec.get("text", ""), re.S)
    user = m.group(1) if m else ""
    for cut in ("\n(history=", " Given this", " Given the", "Given this background"):
        user = user.split(cut)[0]
    return user.strip()
