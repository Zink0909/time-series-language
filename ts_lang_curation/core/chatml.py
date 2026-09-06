# core/chatml.py — emit FULL training-loader records per references/Data Schema.docx.
#
# Two task contracts cover our world-knowledge pairs:
#   ts -> text  =>  task_type "text_desc"  (Understanding): series are user-side observed
#                   spans (loss_start=len, no TS loss); assistant outputs the description;
#                   text loss is on the assistant answer (text_loss_char_ranges).
#   text -> ts  =>  task_type "ts_forecast" (Forecasting): user gives conditioning text +
#                   history context span; assistant emits the target span; TS loss only on
#                   the future suffix (loss_start=H). With covariates => ts_forecast_covariates.
#
# Everything here follows the doc: ChatML transcript, prompt-side <stats>len,mean,std</stats>
# <ts></ts>, bare assistant <ts></ts>, z-score spans (raw = z*std+mean), context-stats-for-
# assistant-targets normalization, and char-range text loss. Alignment-stage empty <think>.
import math
from statistics import mean as _mean, pstdev as _pstdev

# Low-variance robust scaler (the team's online normalization, per Xinyue 2026-06-29):
# context-window z-score, but floor the std at a scale-relative value so a near-flat context
# can't explode the horizon: std_eff = max(raw_std, ROBUST_COEF * max(|mean|, 1.0)).
ROBUST_COEF = 1e-3
STD_FLOOR = ROBUST_COEF        # reported in the normalization block
CHATML_SCHEMA_VERSION = "chatml-ts@1"
SYS_FORECAST = "You are a helpful assistant specialized in time-series forecasting."
SYS_UNDERSTAND = "You are a helpful assistant specialized in time-series understanding."


def _fmt(x):
    return f"{x:.6g}"


def _z(raw):
    """Return (z_values, mean, std_eff, std_floor_applied) using the robust context-window scaler:
    std_eff = max(raw_std, ROBUST_COEF * max(|mean|, 1.0))."""
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("time-series values must be a non-empty list")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in raw):
        raise ValueError("time-series values must all be finite numbers")
    m = _mean(raw)
    s = _pstdev(raw)
    floor = ROBUST_COEF * max(abs(m), 1.0)
    applied = s < floor
    s_eff = max(s, floor)
    return [(x - m) / s_eff for x in raw], m, s_eff, applied


def _span_norm(idx, speaker, role, m, s, raw_len, loss_start, applied,
               renorm_source, ref_idx, stats_visible, **extra):
    d = {"span_idx": idx, "speaker": speaker, "role": role, "mean": m, "std": s,
         "raw_len": raw_len, "loss_start": loss_start, "std_floor_applied": applied,
         "renorm_source": renorm_source, "reference_span_idx": ref_idx,
         "stats_visible": stats_visible, "stats_supervised": False}
    d.update(extra)
    return d


def build_text_desc(user_intro, series, answer, meta):
    """ts -> text. `series` = [{"name","values"(raw),"unit","freq"}]; `answer` = assistant text."""
    # --- assemble user turn: intro, then one labeled <stats> <ts></ts> per observed span ---
    user = user_intro.rstrip()
    spans, norm_spans = [], []
    for i, s in enumerate(series):
        zz, m, sd, applied = _z(s["values"])
        label = s.get("name", f"series {i}")
        user += f"\n{label}:\n<stats>len={len(zz)}, mean={_fmt(m)}, std={_fmt(sd)}</stats> <ts></ts>"
        spans.append({"len": len(zz), "role": "observed", "values": [round(v, 6) for v in zz],
                      "loss_start": len(zz), "unit": s.get("unit"), "freq": s.get("freq")})
        norm_spans.append(_span_norm(i, "user", "observed", m, sd, len(zz), len(zz), applied,
                                     "self_visible_span_stats", None, True,
                                     variable_name=s.get("name")))
    # --- assistant turn: empty think scaffold (alignment stage) + answer; record loss range ---
    head = (f"<|im_start|>system\n{SYS_UNDERSTAND}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n</think>\n\n")
    a0 = len(head)
    text = head + answer + "<|im_end|>\n"
    rec = {"schema_version": CHATML_SCHEMA_VERSION,
           "text": text, "timeseries": spans, "task_type": "text_desc",
           "text_loss_char_ranges": [[a0, a0 + len(answer)]],
           "normalization": {"method": "zscore", "scope": "context_stats_for_assistant_targets",
                             "num_spans": len(norm_spans), "std_floor": STD_FLOOR, "spans": norm_spans}}
    rec.update(meta)
    return rec


def build_ts_forecast(user_text, history, future, series_name, unit, freq, meta, covariates=None):
    """text -> ts. history/future = raw lists; loss only on the future suffix of the target."""
    covariates = covariates or []
    H, F = len(history), len(future)
    if H < 2:
        raise ValueError("forecast history must contain at least two values")
    if F < 1:
        raise ValueError("forecast future must contain at least one value")
    # primary history (context) uses its own visible stats; target reuses them.
    hist_z, m, sd, applied = _z(history)
    hist_z = [round(v, 6) for v in hist_z]
    fut_z = [round((x - m) / sd, 6) for x in future]      # same stats -> first H of target == context
    user = user_text.rstrip() + f"\n(history={H}, horizon={F})\n{series_name} history:\n"
    user += f"<stats>len={H}, mean={_fmt(m)}, std={_fmt(sd)}</stats> <ts></ts>"

    spans, norm_spans = [], []
    spans.append({"len": H, "role": "context", "values": hist_z, "loss_start": H,
                  "unit": unit, "freq": freq})
    norm_spans.append(_span_norm(0, "user", "context", m, sd, H, H, applied,
                                 "self_visible_span_stats", None, True, series_name=series_name))
    # optional covariate spans (user-side conditioning; no TS loss)
    for c in covariates:
        zz, cm, cs, ca = _z(c["values"])
        user += f"\nCovariate: {c['name']}.\n<stats>len={len(zz)}, mean={_fmt(cm)}, std={_fmt(cs)}</stats> <ts></ts>"
        idx = len(spans)
        spans.append({"len": len(zz), "role": "context", "values": [round(v, 6) for v in zz],
                      "loss_start": len(zz), "unit": c.get("unit"), "freq": c.get("freq")})
        norm_spans.append(_span_norm(idx, "user", "context", cm, cs, len(zz), len(zz), ca,
                                     "self_visible_span_stats", None, True, covariate_name=c["name"]))
    # assistant target span: [history_z | future_z], loss_start=H, reuses context stats
    tgt_idx = len(spans)
    target = {"len": H + F, "role": "target", "values": hist_z + fut_z, "loss_start": H,
              "unit": unit, "freq": freq, "context_span_idx": 0}
    spans.append(target)
    norm_spans.append(_span_norm(tgt_idx, "assistant", "target", m, sd, H + F, H, applied,
                                 "history_context_span_stats", 0, False, target_name=series_name))

    text = (f"<|im_start|>system\n{SYS_FORECAST}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n</think>\n\n<ts></ts><|im_end|>\n")
    task = "ts_forecast_covariates" if covariates else "ts_forecast"
    rec = {"schema_version": CHATML_SCHEMA_VERSION,
           "text": text, "timeseries": spans, "task_type": task, "text_loss_char_ranges": [],
           "normalization": {"method": "zscore", "scope": "context_stats_for_assistant_targets",
                             "num_spans": len(norm_spans), "std_floor": STD_FLOOR, "spans": norm_spans}}
    rec.update(meta)
    return rec


def validate(rec):
    """Schema invariants. Returns list of problems ([] == ok)."""
    errs = []
    if not isinstance(rec, dict):
        return ["record must be an object"]
    text = rec.get("text")
    ts = rec.get("timeseries")
    if not isinstance(text, str) or not text:
        errs.append("text missing/empty")
        text = ""
    if not isinstance(ts, list) or not ts:
        errs.append("timeseries missing/empty")
        ts = []
    n_ph = text.count("<ts></ts>")
    if n_ph != len(ts):
        errs.append(f"<ts></ts> count {n_ph} != timeseries {len(ts)}")
    for i, s in enumerate(ts):
        if not isinstance(s, dict):
            errs.append(f"span[{i}] must be an object")
            continue
        v = s.get("values", [])
        if not isinstance(v, list) or not v:
            errs.append(f"span[{i}] values missing/empty")
            continue
        if len(v) != s.get("len"):
            errs.append(f"span[{i}] len {s.get('len')} != values {len(v)}")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in v):
            errs.append(f"span[{i}] has non-finite or non-numeric values")
        if s.get("role") not in {"context", "target", "observed", "covariate"}:
            errs.append(f"span[{i}] invalid role {s.get('role')!r}")
        ls = s.get("loss_start")
        if not isinstance(ls, int) or not 0 <= ls <= len(v):
            errs.append(f"span[{i}] invalid loss_start {ls!r}")
    # forecast: first H of target must equal context exactly (normalized)
    task_type = rec.get("task_type", "")
    if task_type.startswith("ts_forecast"):
        tgt = next((s for s in ts if isinstance(s, dict) and s.get("role") == "target"), None)
        if tgt:
            ci = tgt.get("context_span_idx", 0)
            if not isinstance(ci, int) or not 0 <= ci < len(ts) or not isinstance(ts[ci], dict):
                errs.append(f"target context_span_idx {ci!r} out of range")
                ci = None
            if ci is not None:
                H = ts[ci].get("len", 0)
                if tgt.get("values", [])[:H] != ts[ci].get("values", []):
                    errs.append("target history prefix != context span")
                if tgt.get("loss_start") != H:
                    errs.append(f"target loss_start {tgt.get('loss_start')} != H {H}")
        else:
            errs.append("forecast row missing target span")
    # text_desc: must have a text loss range; forecast: must be empty
    tlr = rec.get("text_loss_char_ranges")
    if tlr is None:
        errs.append("missing text_loss_char_ranges")
        tlr = []
    elif not isinstance(tlr, list):
        errs.append("text_loss_char_ranges must be a list")
        tlr = []
    elif task_type == "text_desc" and not tlr:
        errs.append("text_desc has empty text_loss_char_ranges")
    elif task_type.startswith("ts_forecast") and tlr:
        errs.append("forecast row should have empty text_loss_char_ranges")
    for r in (tlr or []):
        if not (isinstance(r, list) and len(r) == 2 and all(isinstance(x, int) for x in r)
                and 0 <= r[0] < r[1] <= len(text)):
            errs.append(f"bad text_loss range {r}")
    norm = rec.get("normalization")
    if not isinstance(norm, dict):
        errs.append("missing normalization block")
    elif (norm.get("num_spans") != len(ts) or not isinstance(norm.get("spans"), list)
          or len(norm["spans"]) != len(ts)):
        errs.append("normalization span count mismatch")
    return errs
