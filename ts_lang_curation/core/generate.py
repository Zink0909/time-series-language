# core/generate.py — shared "grounded generation" used by any source that needs an
# LLM-written ts->text annotation. Generate -> numeric reflection check -> one stricter
# retry -> caller-supplied fallback. Only verified text is returned as generated.
# Standard from the reference papers: HORAI/MM-TS (2602.05646) consistency filtering +
# From-News-to-Forecast (2409.17515) reflection loop.
import logging

from core.verify import is_consistent, year_value_errors

LOG = logging.getLogger(__name__)
_UNAVAILABLE_WARNED = False

_STRICTER = ("\n\nState ONLY numbers that appear in the facts above; introduce no other figures.")


def grounded_describe(system, user, values, grounding_text="", fallback=None,
                      max_tokens=200, attempts=2, rel_tol=0.0, extra_values=None,
                      ints=False, year_map=None, temperature=0.3):
    """Returns (text, is_generated_tag, verified).
    `fallback` is (text, tag) used when generation is unavailable or never verifies;
    a fallback built from the real numbers/text is grounded by construction (verified=True).
    `rel_tol` forwarded to the numeric check (see core.verify) for large-magnitude series.
    `extra_values` = extra series allowed in the check (multivariate descriptions).
    `ints` = also verify standalone integers (integer-valued series).
    `year_map` = {year: value}; if given, also reject descriptions that misattribute a value to
    the wrong labelled year."""
    def good(desc):
        return (is_consistent(desc, values, grounding_text, rel_tol, extra_values, ints)
                and not (year_map and year_value_errors(desc, year_map)))
    # Steer away from meta-phrasing ("This dataset tracks…") across every source.
    system = system + (" Describe the subject directly; never write 'this dataset', 'this series', "
                       "'the data', or similar meta-references.")
    global _UNAVAILABLE_WARNED
    rejected = 0
    try:
        from core.compress import complete
        for k in range(attempts):
            desc = complete(system, user + (_STRICTER if k else ""), max_tokens=max_tokens,
                            temperature=temperature)
            if desc and good(desc):
                return desc, "derived_generated", True
            rejected += 1
    except Exception as exc:
        if not _UNAVAILABLE_WARNED:
            LOG.warning("grounded generation unavailable; using deterministic fallbacks: %s", exc)
            _UNAVAILABLE_WARNED = True
    if rejected:
        LOG.warning("grounded generation rejected %d candidate(s) by numeric checks", rejected)
    if fallback is not None:
        text, tag = fallback
        return text, tag, True
    return None, None, False
