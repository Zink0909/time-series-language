"""Versioned, lossless serialization for adapter-level canonical pair examples."""
from __future__ import annotations

import math
from dataclasses import asdict

from core.schema import PAIR_SCHEMA_VERSION, TextToTs, TsToText


KINDS = {"ts_to_text": TsToText, "text_to_ts": TextToTs}


def _finite(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in values):
        raise ValueError(f"{label} must contain only finite numbers")


def validate_example(example):
    if isinstance(example, TsToText):
        if not example.answer.strip() or not example.series:
            raise ValueError("ts_to_text requires answer and at least one series")
        for i, series in enumerate(example.series):
            _finite(series.get("values"), f"series[{i}].values")
        return
    if isinstance(example, TextToTs):
        _finite(example.history, "history")
        _finite(example.future, "future")
        if len(example.history) < 2:
            raise ValueError("text_to_ts history must contain at least two values")
        for i, covariate in enumerate(example.covariates):
            _finite(covariate.get("values"), f"covariates[{i}].values")
        return
    raise TypeError(f"unknown canonical example type: {type(example)}")


def to_record(example):
    validate_example(example)
    kind = "ts_to_text" if isinstance(example, TsToText) else "text_to_ts"
    return {"schema_version": PAIR_SCHEMA_VERSION, "kind": kind, "payload": asdict(example)}


def from_record(record):
    if record.get("schema_version") != PAIR_SCHEMA_VERSION:
        raise ValueError(f"unsupported pair schema {record.get('schema_version')!r}")
    cls = KINDS.get(record.get("kind"))
    if cls is None or not isinstance(record.get("payload"), dict):
        raise ValueError("canonical pair requires a known kind and object payload")
    try:
        example = cls(**record["payload"])
    except TypeError as exc:
        raise ValueError(f"invalid canonical pair payload: {exc}") from exc
    validate_example(example)
    return example
