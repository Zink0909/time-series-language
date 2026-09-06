#!/usr/bin/env python3
"""verify_cpt.py — Stage-1 data-quality verifier for CPT world-knowledge records.

Producer-INDEPENDENT verification for instruction-free CPT JSONL (the format:
one <ts></ts> in `text`, `timeseries` = [{values, unit, freq}], task_type=world_knowledge,
plus provenance/alignment/license fields). Runs on CPU, on a laptop, at full scale.

WHAT IT CHECKS (beyond a plain schema validator):
  ERROR  (breaks the required contract → file fails):
    - text missing/empty; <ts></ts> count != 1 (or != len(timeseries) if multi_series)
    - a channel missing values/unit/freq; illegal freq token; same-freq channels unequal length
    - task_type != world_knowledge; text_quality not in enum
  WARN   (quality/claim flags → reported, do not fail unless --strict):
    - recite_unsupported : alignment=="recites" but NO series value appears in the text
    - freq_period_mismatch: freq token implies a span wildly inconsistent with period_start/end
                            (catches e.g. `1m`=minute mislabelling a monthly series)
    - single_point_series / short_series (< --min-window)
    - duplicate_text     : identical text shared across records in one dataset (redundancy)
    - license_nontrainable / license_nonstandard
    - generated_text     : text_quality=="generated" (needs team sign-off)
    - missing_alignment / missing_provenance

WHAT IT DOES NOT DO: the empirical "does the text actually improve forecasting" verdict
(the 3-arm A/B/C tier). That stage fine-tunes a base model and needs a GPU — it is a
separate box job, not a laptop script. See the FULL PIPELINE note in --help.

stdlib only. Reads JSONL, pretty-printed JSON arrays, or concatenated JSON objects.

Usage:
    python verify_cpt.py PATH [PATH ...] [--report out.html] [--json out.json]
                         [--min-window 8] [--strict] [--quiet]
Exit 0 if no ERRORs (and no WARNs when --strict), else 1.
"""
from __future__ import annotations
import argparse, glob, html, json, os, re, statistics as st, sys
from collections import Counter, defaultdict

TS_TOKEN = "<ts></ts>"
FREQ_RE = re.compile(r"^(\d+)(ms|s|m|h|d|w|W|M|q|y|over|play|prd)$")
URL_RE = re.compile(r"^https?://", re.I)
TASK_TYPES = {"world_knowledge"}
TEXT_QUALITY = {"real", "generated"}
ALIGNMENT = {"recites", "describes", "contextualizes"}
LICENSE_TRAINABLE = {"public-domain-us-gov", "cc-by-4.0", "cc0"}
LICENSE_GATED = {"proprietary-review", "unknown"}
# freq unit -> seconds (approx; only used for the span-consistency heuristic)
UNIT_SEC = {"ms": 1e-3, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800,
            "W": 604800, "M": 2629800, "q": 7889400, "y": 31557600}
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def load_records(path):
    """Robust loader: JSONL / pretty-printed array / concatenated objects."""
    txt = open(path, encoding="utf-8").read()
    try:
        d = json.loads(txt)
        return d if isinstance(d, list) else [d]
    except Exception:
        pass
    recs, ok = [], True
    for line in txt.splitlines():
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            ok = False
            break
    if ok and recs:
        return recs
    dec, i, recs = json.JSONDecoder(), 0, []
    while i < len(txt):
        while i < len(txt) and txt[i] in " \n\r\t":
            i += 1
        if i >= len(txt):
            break
        obj, i = dec.raw_decode(txt, i)
        recs.append(obj)
    return recs


def _nums(s):
    out = set()
    for m in _NUM.findall(s):
        try:
            out.add(float(m.replace(",", "")))
        except Exception:
            pass
    return out


def _recites_a_value(rec):
    """Independent check of a 'recites' claim: does any series value appear in the text?"""
    tn = _nums(rec.get("text", ""))
    if not tn:
        return False
    for ch in rec.get("timeseries", []):
        if not isinstance(ch, dict):
            continue
        for v in ch.get("values", []):
            if v is None:
                continue
            for cand in {v, round(v, 1), round(v, 2), round(v)}:
                for t in tn:
                    if cand and abs(t - cand) <= max(0.01 * abs(cand), 0.01):
                        return True
                    if cand == 0 and t == 0:
                        return True
    return False


def _span_days(rec):
    a, b = rec.get("period_start"), rec.get("period_end")
    if not (isinstance(a, str) and isinstance(b, str)):
        return None
    m = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
    ma, mb = m.match(a), m.match(b)
    if not (ma and mb):
        return None
    import datetime
    try:
        da = datetime.date(*map(int, ma.groups()))
        db = datetime.date(*map(int, mb.groups()))
        return abs((db - da).days)
    except Exception:
        return None


def check_record(rec, min_window):
    """Return (errors, warns) lists of (code, detail)."""
    errs, warns = [], []
    text = rec.get("text")
    if not isinstance(text, str) or not text:
        errs.append(("text_missing", "text missing or empty"))
        text = ""
    ntok = text.count(TS_TOKEN)
    ts = rec.get("timeseries")
    if not isinstance(ts, list) or not ts:
        errs.append(("timeseries_missing", "timeseries missing or empty"))
        ts = []
    multi = bool(rec.get("multi_series"))
    want = len(ts) if multi else 1
    if ntok != want:
        errs.append(("ts_token_count", f"<ts></ts> x{ntok} != {want}"))

    lens_by_freq = defaultdict(list)
    span_days = _span_days(rec)
    for i, ch in enumerate(ts):
        if not isinstance(ch, dict) or "values" not in ch or "unit" not in ch or "freq" not in ch:
            errs.append(("channel_shape", f"channel[{i}] missing values/unit/freq"))
            continue
        fr = str(ch["freq"])
        mfr = FREQ_RE.match(fr)
        if not mfr:
            errs.append(("bad_freq", f"channel[{i}] illegal freq '{fr}'"))
        vals = ch["values"]
        if not isinstance(vals, list) or not vals:
            errs.append(("empty_values", f"channel[{i}] empty values"))
            continue
        lens_by_freq[fr].append(len(vals))
        if len(vals) == 1:
            warns.append(("single_point_series", f"channel[{i}] '{ch['unit']}' has 1 point"))
        elif len(vals) < min_window:
            warns.append(("short_series", f"channel[{i}] '{ch['unit']}' len={len(vals)} < {min_window}"))
        # freq label vs period span consistency (catches 1m-minute mislabelling a monthly series)
        if mfr and span_days is not None and span_days >= 2 and len(vals) >= 2:
            n, unit = int(mfr.group(1)), mfr.group(2)
            if unit in UNIT_SEC:
                implied_days = n * UNIT_SEC[unit] * (len(vals) - 1) / 86400.0
                if implied_days > 0:
                    ratio = span_days / implied_days
                    if ratio > 50 or ratio < 1 / 50:
                        warns.append(("freq_period_mismatch",
                                      f"channel[{i}] freq '{fr}' + {len(vals)} pts imply "
                                      f"~{implied_days:.2f}d but period spans {span_days}d (x{ratio:.0f})"))
    for fr, ls in lens_by_freq.items():
        if len(set(ls)) > 1:
            errs.append(("channel_len_mismatch", f"same-freq '{fr}' channels differ: {sorted(set(ls))}"))

    if rec.get("task_type") not in TASK_TYPES:
        errs.append(("task_type", f"task_type={rec.get('task_type')!r} != world_knowledge"))
    tq = rec.get("text_quality")
    if tq not in TEXT_QUALITY:
        errs.append(("text_quality_enum", f"text_quality={tq!r} not in {sorted(TEXT_QUALITY)}"))
    elif tq == "generated":
        warns.append(("generated_text", "text_quality=generated (needs team sign-off)"))

    align = rec.get("alignment")
    if align is None:
        warns.append(("missing_alignment", "no alignment field"))
    elif align not in ALIGNMENT:
        warns.append(("alignment_enum", f"alignment={align!r} not in {sorted(ALIGNMENT)}"))
    elif align == "recites" and text and ts and not _recites_a_value(rec):
        warns.append(("recite_unsupported",
                      "alignment='recites' but no series value found in text"))

    lic = rec.get("license")
    if lic is not None:
        if rec.get("license_status") in {"approved", "conditional"}:
            pass  # the central governance registry supersedes this legacy enum
        elif lic in LICENSE_GATED or (isinstance(lic, str) and "nc" in lic.lower().replace("-", "")):
            warns.append(("license_nontrainable", f"license={lic!r} (release-gated / non-commercial)"))
        elif lic not in LICENSE_TRAINABLE:
            warns.append(("license_nonstandard", f"license={lic!r} not a standard trainable enum"))

    if not (URL_RE.match(str(rec.get("source", ""))) or rec.get("text_url")
            or rec.get("ts_url") or rec.get("period_start")):
        warns.append(("missing_provenance", "no source URL and no period_start"))

    return errs, warns


def audit_file(path, min_window):
    recs = load_records(path)
    dataset = os.path.basename(os.path.dirname(os.path.dirname(path))) or path
    n = len(recs)
    err_c, warn_c = Counter(), Counter()
    per_record = []
    texts, serielens, chans, freqs = [], [], [], set()
    recite_claimed = recite_ok = 0
    for idx, rec in enumerate(recs):
        e, w = check_record(rec, min_window)
        for code, _ in e:
            err_c[code] += 1
        for code, _ in w:
            warn_c[code] += 1
        if e or w:
            per_record.append({"i": idx, "series_id": rec.get("series_id"),
                               "errors": e, "warns": w})
        texts.append(rec.get("text", ""))
        ts = rec.get("timeseries", []) or []
        chans.append(len(ts))
        for ch in ts:
            if isinstance(ch, dict) and isinstance(ch.get("values"), list):
                serielens.append(len(ch["values"]))
                freqs.add(str(ch.get("freq")))
        if rec.get("alignment") == "recites":
            recite_claimed += 1
            if _recites_a_value(rec):
                recite_ok += 1
    # duplicate text within this file/dataset
    tc = Counter(texts)
    dups = sum(c for t, c in tc.items() if c > 1 and t)
    summary = {
        "dataset": dataset, "path": path, "n": n,
        "errors": dict(err_c), "n_error_records": sum(1 for r in per_record if r["errors"]),
        "warns": dict(warn_c),
        "channels_median": st.median(chans) if chans else 0,
        "series_len_min": min(serielens) if serielens else 0,
        "series_len_median": st.median(serielens) if serielens else 0,
        "series_len_max": max(serielens) if serielens else 0,
        "freqs": sorted(freqs),
        "text_unique_ratio": round(len(set(texts)) / n, 3) if n else 0,
        "duplicate_text_records": dups,
        "recite_claimed": recite_claimed,
        "recite_supported": recite_ok,
        "recite_support_rate": round(recite_ok / recite_claimed, 3) if recite_claimed else None,
    }
    return summary, per_record


def find_files(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        elif any(ch in p for ch in "*?["):
            out += sorted(glob.glob(p, recursive=True))
        else:
            out.append(p)
    return [f for f in out if os.path.isfile(f)]


def write_html(summaries, out_path, min_window):
    rows = []
    for s in summaries:
        flags = []
        if s["errors"]:
            flags.append(f"<b style='color:#f87171'>{sum(s['errors'].values())} ERR</b>")
        for k in ("freq_period_mismatch", "recite_unsupported", "duplicate_text_records",
                  "single_point_series", "license_nontrainable", "generated_text"):
            v = s["warns"].get(k) if k in s["warns"] else (s["duplicate_text_records"] if k == "duplicate_text_records" else 0)
            if v:
                flags.append(f"<span style='color:#fbbf24'>{k}:{v}</span>")
        rr = s["recite_support_rate"]
        recite_cell = "" if rr is None else f"{s['recite_supported']}/{s['recite_claimed']} ({rr})"
        lenmm = f"{s['series_len_min']}/{s['series_len_median']:.0f}/{s['series_len_max']}"
        freqs_cell = html.escape(",".join(s["freqs"]))
        rows.append(
            f"<tr><td class=l>{html.escape(str(s['dataset']))}</td><td>{s['n']}</td>"
            f"<td>{s['n_error_records']}</td><td>{s['channels_median']:.0f}</td>"
            f"<td>{lenmm}</td>"
            f"<td>{freqs_cell}</td>"
            f"<td>{s['text_unique_ratio']}</td>"
            f"<td>{recite_cell}</td>"
            f"<td class=l>{' '.join(flags)}</td></tr>")
    css = ("body{background:#0f1117;color:#e7e9ee;font:14px/1.6 -apple-system,sans-serif;max-width:1100px;margin:0 auto;padding:30px}"
           "table{width:100%;border-collapse:collapse;font-size:12.5px}th,td{border:1px solid #262b36;padding:5px 7px;text-align:center}"
           "th{background:#10141d;color:#9aa3b2}td.l{text-align:left}h1{font-size:20px}.sub{color:#9aa3b2}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html><head><meta charset=utf-8><style>{css}</style></head><body>"
                f"<h1>CPT Corpus — Stage-1 data-quality verification</h1>"
                f"<p class=sub>producer-independent · CPU · min_window={min_window} · "
                f"empirical forecasting-lift verdict NOT included (box job)</p>"
                f"<table><tr><th>dataset</th><th>n</th><th>err recs</th><th>ch</th>"
                f"<th>len min/med/max</th><th>freqs</th><th>text uniq</th>"
                f"<th>recites supported</th><th>flags</th></tr>{''.join(rows)}</table></body></html>")


def main():
    ap = argparse.ArgumentParser(
        description="Stage-1 CPU data-quality verifier for CPT world-knowledge JSONL. "
                    "FULL PIPELINE = this (stage 1, laptop) + empirical 3-arm forecasting-lift "
                    "tier (stage 2, needs a GPU/base-model; run separately).")
    ap.add_argument("paths", nargs="+", help="jsonl files, globs, or dirs (recursive)")
    ap.add_argument("--report", help="write an HTML summary here")
    ap.add_argument("--json", help="write the full JSON result here")
    ap.add_argument("--min-window", type=int, default=8, help="warn on series shorter than this")
    ap.add_argument("--strict", action="store_true", help="warnings also fail (exit 1)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    files = find_files(a.paths)
    if not files:
        print("no .jsonl files found", file=sys.stderr)
        sys.exit(2)

    summaries, all_records, tot_err, tot_warn = [], {}, 0, 0
    for path in files:
        s, per = audit_file(path, a.min_window)
        summaries.append(s)
        all_records[path] = per
        tot_err += sum(s["errors"].values())
        tot_warn += sum(s["warns"].values())

    if not a.quiet:
        print(f"{'dataset':<28}{'n':>6}{'err':>5}{'ch':>4}{'len(min/med/max)':>17}"
              f"{'uniq':>6}{'recite':>9}  flags")
        for s in summaries:
            rr = s["recite_support_rate"]
            flags = []
            if s["errors"]:
                flags.append(f"{sum(s['errors'].values())}ERR")
            for k in ("freq_period_mismatch", "recite_unsupported", "single_point_series",
                      "license_nontrainable", "generated_text", "missing_alignment"):
                if s["warns"].get(k):
                    flags.append(f"{k}:{s['warns'][k]}")
            if s["duplicate_text_records"]:
                flags.append(f"dup_text:{s['duplicate_text_records']}")
            ld = f"{s['series_len_min']}/{s['series_len_median']:.0f}/{s['series_len_max']}"
            rc = "" if rr is None else f"{s['recite_supported']}/{s['recite_claimed']}"
            print(f"{s['dataset'][:27]:<28}{s['n']:>6}{s['n_error_records']:>5}"
                  f"{s['channels_median']:>4.0f}{ld:>17}{s['text_unique_ratio']:>6}{rc:>9}  "
                  f"{' '.join(flags)}")
        print(f"\n{len(files)} file(s) · {tot_err} ERROR(s) · {tot_warn} WARN(s)")
        print("note: empirical forecasting-lift tier (T1-T4) is a separate GPU/box job, not run here.")

    if a.report:
        write_html(summaries, a.report, a.min_window)
        if not a.quiet:
            print(f"HTML report -> {a.report}")
    if a.json:
        json.dump({"summaries": summaries, "flagged_records": all_records},
                  open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        if not a.quiet:
            print(f"JSON result -> {a.json}")

    sys.exit(1 if (tot_err or (a.strict and tot_warn)) else 0)


if __name__ == "__main__":
    main()
