#!/usr/bin/env python3
"""TimesX → external GENERALIZATION test set for the MiGAS fine-tune experiment (2nd, stronger).

Why TimesX (haoxin1998.github.io/TimesX-project, "Rethinking Multimodal TS Forecasting Evaluation"):
Xinyue flagged it as the current largest multimodal-forecasting benchmark. 190 variables / 19 domains
/ 2018-2025, with STRICT temporal isolation (events dated before the forecast origin; the paper shows
leakage-free methods degrade ~2% vs ~13% for contaminated LLM methods). Crucially, its three groups
map onto OUR two series types: Currency + CommodityPrice = price series (our commodity flywheel),
SearchTrend = attention series (our wiki). So it is both a fair external test AND the reference format
for what "high-quality event context" should look like (its 7.3% MASE gain from low- vs high-quality
context is the external evidence behind our line-B upgrade).

Each TimesX sample already carries: past_time/future_time ({timestamp,value}), a `scenario` string
("Prediction target period ... Recent related events <1>...<2>..."), `covariates_info` (per-covariate
stats over the history window) and `holiday_info` (upcoming holidays in the forecast window). We map:
  fact  <- the numbered-events portion of `scenario`  (== MiGAS h_fact / FACTUAL SUMMARY)
  preds <- covariates_info + holiday_info             (== MiGAS h_pred / PREDICTIVE SIGNALS)
and emit (history, future, fact, preds, text) in the SAME shape as timemmd_export.py, so the MiGAS
scaffold consumes it unchanged (--gen-test timesx).

Leakage: we TRUST TimesX's temporal isolation (events precede origin; covariates_info window ends the
day before origin; holiday_info is a deterministic future calendar) and record knowledge_time=origin
with an explicit note — we do NOT independently re-audit each embedded event date (TimesX's own
control is documented). Only samples with event_count>=1 are kept (TimesX excludes event_count==0).

  micromamba run -n ts-language python flywheel/companion/timesx_export.py \
      --timesx-dir /tmp/TimesX-project/Datasets [--per-series-cap 40]
"""
import os, sys, glob, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_timesx_test.json")
H, F = 10, 5
EVENTS_MARKER = "Recent related events"


def _fact_from_scenario(scenario):
    """Keep the numbered-events portion (drop the boilerplate 'Prediction target period ...' prefix)."""
    if not scenario:
        return ""
    i = scenario.find(EVENTS_MARKER)
    return (scenario[i:] if i >= 0 else scenario).strip()


def _preds(sample):
    parts = []
    cov = (sample.get("covariates_info") or "").strip()
    hol = (sample.get("holiday_info") or "").strip()
    if cov and cov.lower() != "unknown":
        parts.append(cov)
    if hol and hol.lower() != "unknown":
        parts.append(hol)
    return "  ".join(parts)


def _even_sample(items, cap):
    if cap <= 0 or len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(k * step)] for k in range(cap)]


def export_file(path, group, per_series_cap):
    try:
        d = json.load(open(path))
    except Exception:
        return []
    info = d.get("dataset_info", {})
    name = info.get("dataset_name", os.path.basename(path))
    domain = info.get("domain", group)
    out = []
    for s in d.get("samples", []):
        meta = s.get("metadata", {})
        if meta.get("event_count", 0) < 1:                    # need real event context (TimesX rule)
            continue
        hist = (s.get("past_time") or {}).get("value") or []
        fut = (s.get("future_time") or {}).get("value") or []
        if len(hist) < H or len(fut) < F:
            continue
        fact = _fact_from_scenario(s.get("scenario", ""))
        if not fact:
            continue
        origin = s.get("date", "")
        out.append({"domain": domain, "group": group,
                    "series_id": f"{name}_{origin}", "dataset": "timesx",
                    "origin": origin, "knowledge_time": origin,
                    "history": [round(float(v), 4) for v in hist[-H:]],
                    "future": [round(float(v), 4) for v in fut[:F]],
                    "fact": fact[:1200], "preds": _preds(s)[:1200],
                    "text": (fact + " " + _preds(s)).strip()[:1600],
                    "event_count": meta.get("event_count", 0)})
    return _even_sample(out, per_series_cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesx-dir", default="/tmp/TimesX-project/Datasets",
                    help="path to a cloned haoxin1998/TimesX-project Datasets folder")
    ap.add_argument("--groups", nargs="+",
                    default=["Currency", "CommodityPrice", "SearchTrend"])
    ap.add_argument("--per-series-cap", type=int, default=40,
                    help="max samples kept per series file (even-spread) to balance the test set")
    a = ap.parse_args()
    ev, per = [], {}
    for g in a.groups:
        files = sorted(glob.glob(os.path.join(a.timesx_dir, g, "**", "*.json"), recursive=True))
        ge = []
        for f in files:
            ge += export_file(f, g, a.per_series_cap)
        per[g] = len(ge)
        ev += ge
    json.dump({"hist_len": H, "pred_len": F, "space": "raw", "source": "TimesX (external)",
               "leakage": "trusted TimesX temporal isolation; knowledge_time=origin (events precede "
                          "origin per TimesX pipeline; not independently re-audited per event)",
               "n": len(ev), "by_group": per, "events": ev}, open(OUT, "w"), ensure_ascii=False)
    print(f"exported {len(ev)} TimesX generalization-test events -> {OUT} "
          f"({os.path.getsize(OUT)//1024} KB)")
    print("by group:", per)
    if ev:
        e = ev[0]
        print(f"sample [{e['domain']}] origin={e['origin']} events={e['event_count']}")
        print("  hist:", e["history"], "-> fut:", e["future"])
        print("  fact:", e["fact"][:160])
        print("  preds:", e["preds"][:160])


if __name__ == "__main__":
    main()
