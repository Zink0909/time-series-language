#!/usr/bin/env python3
"""verify_any.py — ONE verification entry point for BOTH data formats.

The two producers emit different formats:
  * CPT world-knowledge  (instruction-free: text + <ts></ts> + timeseries[{values,unit,freq}],
    task_type=world_knowledge)                                   -> handled by verify_cpt.py
  * ChatML forecasting   (Qwen transcript + <ts></ts> + timeseries spans with role/loss_start/
    normalization, task_type=ts_forecast/text_desc/...)          -> handled here

This is the shared VERIFICATION GATE for a merged pipeline: whatever the format, a record is
detected, parsed into a common view, and checked against one standard (structural contract,
provenance, leakage field, text-quality, redundancy) plus format-specific rules. It does NOT
change either format on disk --- it verifies both. (If the team later wants a single training
format, the same detect+parse layer is the read-half of a converter.)

stdlib only. Usage:
    python verify_any.py PATH [PATH ...] [--min-window 4] [--json out.json] [--strict]
Exit 0 if no ERRORs.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from collections import Counter
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_cpt as vcpt  # reuse the CPT checker + robust loader

TS = "<ts></ts>"
CHATML_MARK = "<|im_start|>"
ROLES = {"context", "target", "observed", "covariate"}
# text_quality vocab differs by format/producer — check against the right one.
TQ_BY_FMT = {
    "cpt": {"real", "generated"},
    "chatml": {"real", "derived", "generated", "derived_generated", "template"},
}


def detect_format(rec):
    """Return 'chatml' or 'cpt' for a single record."""
    if not isinstance(rec, dict):
        return "unknown"
    if CHATML_MARK in str(rec.get("text", "")):
        return "chatml"
    ts = rec.get("timeseries")
    if isinstance(ts, list) and ts and isinstance(ts[0], dict) and "role" in ts[0] and "loss_start" in ts[0]:
        return "chatml"
    if rec.get("task_type") == "world_knowledge":
        return "cpt"
    if isinstance(ts, list) and ts and isinstance(ts[0], dict) and "unit" in ts[0] and "freq" in ts[0] and "role" not in ts[0]:
        return "cpt"
    return "cpt"  # default to the simpler contract


# ---------- ChatML-specific checker ----------
def check_chatml(rec, min_window):
    errs, warns = [], []
    text = rec.get("text")
    if not isinstance(text, str) or not text:
        errs.append(("text_missing", "text missing/empty")); text = ""
    ts = rec.get("timeseries")
    if not isinstance(ts, list) or not ts:
        errs.append(("timeseries_missing", "timeseries missing/empty")); ts = []
    ntok = text.count(TS)
    if ntok != len(ts):
        errs.append(("ts_token_count", f"<ts></ts> x{ntok} != {len(ts)} spans"))
    roles = []
    for i, sp in enumerate(ts):
        if not isinstance(sp, dict):
            errs.append(("span_shape", f"span[{i}] not an object")); continue
        vals = sp.get("values")
        if not isinstance(vals, list) or not vals:
            errs.append(("empty_values", f"span[{i}] empty values")); continue
        if sp.get("len") != len(vals):
            errs.append(("len_mismatch", f"span[{i}] len={sp.get('len')} != {len(vals)} values"))
        ls = sp.get("loss_start")
        if not isinstance(ls, int) or ls < 0 or ls > len(vals):
            errs.append(("loss_start", f"span[{i}] loss_start={ls} out of [0,{len(vals)}]"))
        role = sp.get("role")
        roles.append(role)
        if role not in ROLES:
            warns.append(("role_enum", f"span[{i}] role={role!r}"))
        if len(vals) < min_window and role in ("context", "target"):
            warns.append(("short_series", f"span[{i}] len={len(vals)} < {min_window}"))
    # forecast contract: exactly [context, target], target[:H] == context, loss_start == H
    tt = rec.get("task_type", "")
    if tt == "ts_forecast":
        ctx = [s for s in ts if s.get("role") == "context"]
        tgt = [s for s in ts if s.get("role") == "target"]
        if len(ctx) == 1 and len(tgt) == 1:
            H = ctx[0].get("len")
            if tgt[0].get("values", [])[:H] != ctx[0].get("values", []):
                errs.append(("forecast_prefix", "target[:H] != context after normalization"))
            if ctx[0].get("loss_start") != H or tgt[0].get("loss_start") != H:
                warns.append(("forecast_loss_start", "context/target loss_start != H"))
        else:
            warns.append(("forecast_spans", f"expected [context,target], got roles {roles}"))
    if "normalization" not in rec:
        warns.append(("no_normalization", "missing normalization block"))
    if rec.get("task_type", "").startswith("text_desc") and not rec.get("text_loss_char_ranges"):
        warns.append(("no_text_loss", "text_desc without text_loss_char_ranges"))
    return errs, warns


# ---------- shared checks (both formats) ----------
def common_view(rec, fmt):
    """Uniform fields used by the shared checks + reporting."""
    ts = rec.get("timeseries") or []
    lens = [len(s.get("values", [])) for s in ts if isinstance(s, dict) and isinstance(s.get("values"), list)]
    return {
        "text": rec.get("text", ""),
        "n_series": len(ts),
        "series_len": lens,
        "series_id": rec.get("series_id"),
        "dataset": rec.get("dataset"),
        "task_type": rec.get("task_type"),
        "knowledge_time": rec.get("knowledge_time"),
        "text_quality": rec.get("text_quality"),
        "direction": rec.get("direction"),
        "fmt": fmt,
    }


def shared_checks(cv):
    warns = []
    if not (cv["series_id"] and cv["dataset"]):
        warns.append(("missing_provenance", "no series_id/dataset"))
    # leakage field: a text->ts / forecasting record should carry knowledge_time
    d = str(cv["direction"] or "")
    if (cv["fmt"] == "chatml" and "forecast" in str(cv["task_type"] or "")) or d == "text_to_ts":
        if not cv["knowledge_time"]:
            warns.append(("no_knowledge_time", "forecasting record without knowledge_time (leakage field)"))
    tq_ok = TQ_BY_FMT.get(cv["fmt"], set())
    if cv["text_quality"] and cv["text_quality"] not in tq_ok:
        warns.append(("text_quality_enum", f"text_quality={cv['text_quality']!r} (fmt={cv['fmt']})"))
    return warns


def audit_file(path, min_window):
    recs = vcpt.load_records(path)
    fmts = Counter(); err_c = Counter(); warn_c = Counter()
    texts = []; serielens = []; n_err_rec = 0
    for rec in recs:
        fmt = detect_format(rec)
        fmts[fmt] += 1
        if fmt == "chatml":
            e, w = check_chatml(rec, min_window)
        else:
            e, w = vcpt.check_record(rec, min_window)
        cv = common_view(rec, fmt)
        w = list(w) + shared_checks(cv)
        for c, _ in e: err_c[c] += 1
        for c, _ in w: warn_c[c] += 1
        if e: n_err_rec += 1
        texts.append(cv["text"]); serielens += cv["series_len"]
    dom_fmt = fmts.most_common(1)[0][0] if fmts else "?"
    uniq = round(len(set(texts)) / len(texts), 3) if texts else 0
    return {
        "path": path, "dataset": os.path.basename(path).replace(".jsonl", ""),
        "format": dom_fmt, "mixed_format": len(fmts) > 1, "n": len(recs),
        "errors": dict(err_c), "n_error_records": n_err_rec, "warns": dict(warn_c),
        "series_len_min": min(serielens) if serielens else 0,
        "series_len_median": median(serielens) if serielens else 0,
        "text_unique_ratio": uniq,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--min-window", type=int, default=4)
    ap.add_argument("--json")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    files = []
    for p in a.paths:
        files += (sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)) if os.path.isdir(p)
                  else glob.glob(p, recursive=True) if any(c in p for c in "*?[") else [p])
    files = [f for f in files if os.path.isfile(f)]

    summaries, tot_e, tot_w = [], 0, 0
    print(f"{'dataset':<26}{'fmt':>7}{'n':>7}{'err':>5}{'uniq':>6}  flags")
    for f in files:
        s = audit_file(f, a.min_window)
        summaries.append(s); tot_e += sum(s["errors"].values()); tot_w += sum(s["warns"].values())
        flags = ([f"{sum(s['errors'].values())}ERR"] if s["errors"] else []) + \
                [f"{k}:{v}" for k, v in list(s["warns"].items())[:4]]
        mix = " [MIXED]" if s["mixed_format"] else ""
        print(f"{s['dataset'][:25]:<26}{s['format']:>7}{s['n']:>7}{s['n_error_records']:>5}"
              f"{s['text_unique_ratio']:>6}  {' '.join(flags)}{mix}")
    print(f"\n{len(files)} file(s) · {tot_e} ERROR(s) · {tot_w} WARN(s) · "
          f"one gate, both formats (CPT + ChatML).")
    if a.json:
        json.dump(summaries, open(a.json, "w"), ensure_ascii=False, indent=2)
        print(f"JSON -> {a.json}")
    sys.exit(1 if (tot_e or (a.strict and tot_w)) else 0)


if __name__ == "__main__":
    main()
