# core/faithfulness.py — W3 faithfulness gate: raise the quality bar from "numeric reflection"
# (core/verify.py) to "factual consistency + entity tracing", per the reference papers:
#   - Identifying Factual Inconsistencies via Task Taxonomy (arXiv 2402.12821): classify HOW a
#     generated text is inconsistent, not just yes/no.
#   - ETF: Entity Tracing Framework for Hallucination Detection (arXiv 2410.14748): every entity a
#     summary states must be traceable to the source; untraceable entity = hallucination.
#
# Scope: only text that is LLM-SYNTHESIZED FROM RETRIEVED TEXT (the flywheel's `cause`). Real text
# (Wikipedia leads, verbatim FOMC/10-K) and pure-numeric descriptions need no entity tracing.
# Programmatic + offline + deterministic + independent of the generator (不自证): it re-derives
# support from the grounding headlines, never trusts the model that wrote the cause.
import re
from core.verify import unmatched_numbers

# Known single-token entities worth tracing in a news-cause context (countries, oil producers,
# institutions, macro concepts). Multi-word Title-Case names and ALL-CAPS acronyms are caught
# structurally; this table only rescues salient SINGLE Title-Case words that structure would miss.
_KNOWN = {w.lower() for w in (
    "Russia Ukraine China Iran Iraq Israel Palestine Gaza Saudi Venezuela Libya Nigeria Germany "
    "France Britain Japan India Brazil Turkey Syria Yemen Qatar Kuwait Norway Canada Mexico "
    "OPEC IEA EIA Fed Fed's OPEC+ EU UN NATO WTI Brent COVID Omicron Delta Coronavirus Pandemic "
    "Biden Putin Trump Powell Zelensky Xi").split()}

_ACRONYM = re.compile(r"\b[A-Z][A-Z&+]{1,6}\b")                       # IEA, OPEC, OPEC+, EU, WTI
_MULTIWORD = re.compile(r"\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+)+\b")       # Russia-Ukraine, Silicon Valley
_TITLE1 = re.compile(r"\b[A-Z][a-z]{2,}\b")                           # single Title-Case word
# Ordinary sentence-initial / connective words that are Title-Case but not entities.
_NOTENT = {"the", "this", "that", "these", "those", "given", "based", "while", "with", "after",
           "before", "during", "amid", "as", "in", "on", "for", "and", "but", "its", "their",
           "a", "an", "meanwhile", "however", "additionally", "overall", "context", "prices",
           "price", "market", "markets", "demand", "supply", "energy", "oil", "crude", "fuel",
           "government", "governments", "reserves", "war", "risks", "costs"}


def extract_entities(text):
    """Salient named entities a faithful cause may mention: ALL-CAPS acronyms, multi-word
    Title-Case names, and known single-word countries/institutions. Deduped, order-preserving."""
    ents, seen = [], set()

    def add(e):
        e = re.sub(r"^(?:The|A|An)\s+", "", e).strip()       # drop leading article
        k = e.lower()
        if e and k not in seen:
            seen.add(k)
            ents.append(e)

    for m in _ACRONYM.findall(text):
        add(m)
    for m in _MULTIWORD.findall(text):
        add(m)
    for m in _TITLE1.findall(text):
        if m.lower() in _KNOWN and m.lower() not in _NOTENT:
            add(m)
    return ents


def _match_forms(e):
    """All surface forms that count as tracing entity `e`: its content tokens (>=4 letters), the
    acronym as-is, and — for a multi-word Title-Case name — its initialism (International Energy
    Agency -> 'iea'), so a full name traces to a headline that only uses the acronym."""
    forms = [t for t in re.split(r"[ /-]+", e.lower()) if len(t) >= 4]
    if e.isupper() and len(e) >= 2:
        forms.append(e.lower())
    words = [w for w in re.split(r"[ -]+", e) if w[:1].isupper()]
    if len(words) >= 2:
        forms.append("".join(w[0] for w in words).lower())         # initialism
    return forms or [e.lower()]


def _is_weak(e):
    """Two-letter ALL-CAPS abbreviations (US, UK, EU, UN) are too ambiguous/low-value to trace —
    kept for coverage stats but an untraced weak entity does not fail the record."""
    return e.isupper() and len(e) == 2


def trace_entities(text, grounding):
    """Split the entities of `text` into (traced, untraced) against `grounding`. An entity is
    traced if ANY of its surface forms appears in the grounding — so 'Russian'/'Russia-Ukraine'
    trace to 'Russia', and 'International Energy Agency' traces to 'IEA'."""
    g = re.sub(r"[^a-z0-9]+", " ", grounding.lower())
    traced, untraced = [], []
    for e in extract_entities(text):
        hit = any(re.search(r"\b" + re.escape(t[:6]), g) for t in _match_forms(e))
        (traced if hit else untraced).append(e)
    return traced, untraced


def report(text, grounding, values=(0.0,), rel_tol=0.0):
    """Independent faithfulness verdict for one synthesized text against its grounding headlines.
    Returns a dict: entity coverage, unsupported numbers, a taxonomy-classified verdict, and a
    human-readable reason log (why it would be rejected)."""
    traced, untraced = trace_entities(text, grounding)
    total = len(traced) + len(untraced)
    strong_untraced = [e for e in untraced if not _is_weak(e)]        # weak (US/UK/EU) don't fail
    bad_nums = unmatched_numbers(text, list(values), grounding, rel_tol)
    reasons = []
    if strong_untraced:
        reasons.append(f"unsupported_entity: {strong_untraced} not traceable to grounding")
    if bad_nums:
        reasons.append(f"unsupported_number: {bad_nums} match neither series nor grounding")
    verdict = "ok" if not reasons else ("unsupported_entity" if strong_untraced
                                        else "unsupported_number")
    return {"verdict": verdict,
            "entity_coverage": round(len(traced) / total, 3) if total else 1.0,
            "entities_total": total, "entities_traced": len(traced),
            "untraced_entities": untraced, "unsupported_numbers": bad_nums,
            "reasons": reasons}


def is_faithful(text, grounding, values=(0.0,), rel_tol=0.0):
    return report(text, grounding, values, rel_tol)["verdict"] == "ok"
