"""Peer emitters from canonical examples; format conversion is never the primary path."""
from __future__ import annotations

from core import chatml
from core.governance import apply_governance
from core.ir import from_record, to_record
from core.schema import TextToTs, TsToText, _prov


CPT_SCHEMA_VERSION = "cpt-world-knowledge@1"
FREQ_TO_CPT = {"daily": "1d", "business_daily": "1d", "weekly": "1w",
               "monthly": "1M", "quarterly": "1q", "yearly": "1y",
               "annual": "1y", "hourly": "1h", "6-hourly": "6h"}


class UnsupportedFormat(ValueError):
    """The canonical example is valid but the requested target format cannot express it."""


def emit_chatml(example):
    if isinstance(example, TsToText):
        record = chatml.build_text_desc(
            example.user_intro, example.series, example.answer, _prov(example, example.meta))
    elif isinstance(example, TextToTs):
        from core.match_score import match_score
        meta = _prov(example, example.meta)
        cause = example.user_text
        for cut in ("\n(history=", "Given this background", " Given this", " Given the"):
            cause = cause.split(cut)[0]
        score, label, _ = match_score(
            cause.strip(), example.meta.get("event_date") or example.meta.get("spike_date"))
        meta["text_ts_match"], meta["text_ts_match_score"] = label, score
        record = chatml.build_ts_forecast(
            example.user_text, example.history, example.future, example.series_name,
            example.unit, example.freq, meta, covariates=example.covariates)
    else:
        raise TypeError(f"unknown canonical example type: {type(example)}")
    return apply_governance(record)


def emit_cpt(example):
    if not isinstance(example, TsToText):
        raise UnsupportedFormat("CPT world_knowledge can represent only ts_to_text examples")
    channels = [
        {"values": list(series["values"]), "unit": series.get("unit"),
         "freq": FREQ_TO_CPT.get(str(series.get("freq")), str(series.get("freq")))}
        for series in example.series
    ]
    record = {
        "schema_version": CPT_SCHEMA_VERSION,
        "text": example.answer.rstrip(". ") + ". " + " ".join("<ts></ts>" for _ in channels),
        "timeseries": channels,
        "task_type": "world_knowledge",
        "alignment": "describes",
        **_prov(example, example.meta),
    }
    record["text_quality"] = "real" if example.is_generated == "real" else "generated"
    if len(channels) > 1:
        record["multi_series"] = True
    return apply_governance(record)


def emit_ir(example):
    record = to_record(example)
    payload = dict(record["payload"])
    payload["meta"] = apply_governance(payload.get("meta", {}))
    record["payload"] = payload
    return record


def emit(example, output_format="chatml"):
    try:
        emitter = {"chatml": emit_chatml, "cpt": emit_cpt, "ir": emit_ir}[output_format]
    except KeyError as exc:
        raise ValueError(f"unknown output format {output_format!r}") from exc
    return emitter(example)


def validate_output(record, output_format):
    if output_format == "chatml":
        return chatml.validate(record)
    if output_format == "ir":
        try:
            from_record(record)
        except (TypeError, ValueError) as exc:
            return [str(exc)]
        return []
    if output_format == "cpt":
        errors = []
        spans = record.get("timeseries")
        text = record.get("text", "")
        if record.get("task_type") != "world_knowledge":
            errors.append("CPT task_type must be world_knowledge")
        if not isinstance(spans, list) or not spans:
            errors.append("CPT timeseries missing/empty")
            spans = []
        if text.count("<ts></ts>") != len(spans):
            errors.append("CPT placeholder/span count mismatch")
        return errors
    return [f"unknown output format {output_format!r}"]
