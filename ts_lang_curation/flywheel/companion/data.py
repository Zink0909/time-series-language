# companion/data.py — load text->ts records into a forecasting-ready form and split them by TIME.
# The whole companion study answers "does the manufactured text help downstream forecasting?" —
# so the data plumbing must (a) expose history/future + the conditioning text, and (b) split
# strictly by time with a leakage assertion (train cutoff < every test event), per the reference
# papers on evaluation leakage (2512.06932 "cut time before windowing", 2510.13654 TSFM leakage).
import os, json, glob, re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "out")


def _conditioning_text(rec):
    """The world-knowledge text the forecaster may read = the user turn up to the '(history=...'
    marker (drops the <stats>/<ts> scaffolding, keeps the cause/lead)."""
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", rec["text"], re.S)
    user = m.group(1) if m else ""
    return user.split("\n(history=")[0].strip()


def _spans(rec):
    """z-space (history, future). Everything stays in the record's context-normalized space, so
    forecasting/MASE are scale-free and need no external raw data."""
    ts = rec["timeseries"]
    ctx = next(s for s in ts if s["role"] == "context")
    tgt = next(s for s in ts if s["role"] == "target")
    H = ctx["len"]
    return list(ctx["values"]), list(tgt["values"][H:])       # history_z, future_z


def load(datasets=None, flywheel_only=True):
    """Load forecast items: {series_id, dataset, origin, kt, history, future, text, flywheel}."""
    items = []
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.jsonl"))):
        if f.endswith("fred_fomc.v1.jsonl"):
            continue
        for line in open(f):
            r = json.loads(line)
            if not r["task_type"].startswith("ts_forecast"):
                continue
            if flywheel_only and not r.get("flywheel"):
                continue
            if datasets and r.get("dataset") not in datasets:
                continue
            hist, fut = _spans(r)
            if len(hist) < 2 or len(fut) < 1:
                continue
            items.append({
                "series_id": r.get("series_id"), "dataset": r.get("dataset"),
                "origin": r.get("spike_date") or r.get("event_date"),
                "kt": r.get("knowledge_time"), "history": hist, "future": fut,
                "text": _conditioning_text(r), "flywheel": bool(r.get("flywheel"))})
    return items


def time_split(items, cutoff, base_pretrain_cutoff=None):
    """Split by forecast origin: train = origin < cutoff, test = origin >= cutoff. Asserts no
    leakage (every train event strictly before cutoff; every test knowledge_time >= cutoff, i.e.
    nothing the model trained on could have seen the test events). Optionally tags each test item
    with `after_base_cutoff` — the honest stratum where gains can't be base-model memorization."""
    train = [x for x in items if x["origin"] and x["origin"] < cutoff]
    test = [x for x in items if x["origin"] and x["origin"] >= cutoff]
    problems = []
    for x in train:
        if x["kt"] and x["kt"][:10] >= cutoff:
            problems.append(f"train {x['series_id']} kt {x['kt']} >= cutoff {cutoff}")
    if train and test:
        max_train = max(x["origin"] for x in train)
        min_test = min(x["origin"] for x in test)
        if not (max_train < cutoff <= min_test):
            problems.append(f"boundary violation: max_train {max_train} / min_test {min_test}")
    for x in test:
        x["after_base_cutoff"] = bool(base_pretrain_cutoff and x["origin"] > base_pretrain_cutoff)
    return train, test, problems
