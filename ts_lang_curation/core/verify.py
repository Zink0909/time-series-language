# core/verify.py — numeric "reflection" check for generated ts->text annotations.
#
# Flywheel standard (from the team's reference papers): generated text must be
# logically consistent with the series it describes — never hallucinate values.
#   - HORAI / MM-TS (arXiv 2602.05646): "logical consistency filtering" discards
#     generated text conflicting with the underlying series.
#   - From-News-to-Forecast (arXiv 2409.17515): a "reflection" loop verifies the
#     model's output against ground truth before it is accepted.
#
# This module makes that programmatic for any source: every decimal number a
# generated description states must match a real value/stat of the series (or a
# number that literally appears in the grounding text, e.g. a policy target).
import re

_DECIMAL = re.compile(r"\d+\.\d+")          # yields look like 1.58, 1.44 — not years/days
TOL = 0.02                                  # ±0.02 absorbs rounding to 2 dp


def series_facts(values):
    """The set of numbers a faithful description is allowed to state about `values`."""
    v0, v1 = values[0], values[-1]
    derived = {round(v0, 2), round(v1, 2), round(min(values), 2), round(max(values), 2),
               round(v1 - v0, 2), round(abs(v1 - v0), 2)}
    return {round(x, 2) for x in values} | derived


_INT = re.compile(r"(?<![\d.])\d+(?![\d.])")        # standalone integers (not part of a decimal)


def _decimals(text):
    return [float(m) for m in _DECIMAL.findall(text)]


def _integers(text):
    return [int(m) for m in _INT.findall(text)]


# Policy-rate targets are quoted as fractions ("1-1/2 to 1-3/4 percent") but a
# faithful description may state them as decimals (1.5, 1.75). Convert the target
# range out of the grounding text so those decimals count as grounded, not invented.
_MIXED = re.compile(r"(\d+)-(\d+)/(\d+)")            # 1-3/4
_FRAC = re.compile(r"(?<!\d-)(\d+)/(\d+)")          # 3/4 (not part of a mixed number)
_RATE_INT = re.compile(r"\b(\d+)\s+(?:to\b|percent\b)")  # bare bound: "5 percent", "5 to"


def policy_numbers(grounding_text):
    out = set()
    grounding_text = re.sub(r"[‐-―−]", "-", grounding_text)  # unicode hyphens -> ASCII
    for w, n, d in _MIXED.findall(grounding_text):
        out.add(round(int(w) + int(n) / int(d), 2))
    for n, d in _FRAC.findall(_MIXED.sub(" ", grounding_text)):
        out.add(round(int(n) / int(d), 2))
    for w in _RATE_INT.findall(_MIXED.sub(" ", grounding_text)):
        out.add(round(float(w), 2))
    return out


def unmatched_numbers(text, values, grounding_text="", rel_tol=0.0, extra_values=None, ints=False):
    """Return the numbers in `text` that match neither the series nor the grounding text.
    Empty list == the description is numerically consistent with the series.
    `rel_tol` adds a relative tolerance (e.g. 0.01) for large-magnitude series like pageviews,
    where a faithful description rounds ("about 670 thousand"); 0 keeps it strictly absolute
    (yields/revenue, where ±0.02 must hold exactly).
    `extra_values` = list of additional series (for multivariate descriptions); their facts are
    allowed too.
    `ints=True` also checks standalone integers (for integer-valued series like aftershock counts):
    allowed = series values/stats + day-indices 0..len + grounding integers + any 4-digit year."""
    series = [values] + list(extra_values or [])
    allowed = set(_decimals(grounding_text)) | policy_numbers(grounding_text)
    for s in series:
        allowed |= series_facts(s)
    bad = []
    for n in _decimals(text):
        if not any(abs(n - a) <= max(TOL, rel_tol * abs(a)) for a in allowed):
            bad.append(n)
    if ints:
        ok_ints = set(_integers(grounding_text))
        for s in series:
            ok_ints |= {int(round(v)) for v in s}
            ok_ints |= {int(round(min(s))), int(round(max(s))), int(round(sum(s)))}
            ok_ints |= set(range(0, len(s) + 1))            # day / step indices
        for n in _integers(text):
            if 1900 <= n <= 2099:                           # years are not value claims
                continue
            if n not in ok_ints:
                bad.append(n)
    return bad


def is_consistent(text, values, grounding_text="", rel_tol=0.0, extra_values=None, ints=False):
    return not unmatched_numbers(text, values, grounding_text, rel_tol, extra_values, ints)


def year_value_errors(text, year_to_value, tol_rel=0.01, tol_abs=0.5):
    """Catch year-misattribution (e.g. "$394.3 billion in 2021" when 394.3 belongs to 2022).
    Conservative: only flags a (value, year) pair when the stated value does NOT match that
    year's value BUT DOES match some other labelled year's value — a confident mislabel, not a
    stray figure. Returns list of (year, stated_value, correct_value_for_year)."""
    yv = {str(y): v for y, v in year_to_value.items()}
    allvals = list(year_to_value.values())

    def near(a, b):
        return abs(a - b) <= max(tol_abs, tol_rel * abs(b))

    # Only the "value ... year" direction ("$394.3 billion in 2021") is reliable; the reverse
    # ("2019 to 416.2") mis-pairs a year with the *next* value, so it is intentionally not used.
    pairs = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)[^.\d\n]{0,12}?((?:19|20)\d{2})", text):
        pairs.append((float(m.group(1)), m.group(2)))
    errs = []
    for val, yr in pairs:
        if yr in yv and not near(val, yv[yr]) and any(near(val, o) for o in allvals):
            errs.append((yr, val, yv[yr]))
    return errs
