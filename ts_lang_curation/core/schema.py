# core/schema.py — the adapter-facing contract. Adapters yield one of two task examples;
# core/chatml.py turns them into full Data-Schema training records (ChatML + z-score spans +
# loss masks). Adapters never format ChatML or normalize directly.
from dataclasses import dataclass, field
from typing import Union
from core import chatml

BUILDER = "ts_lang_curation@0.5"
PAIR_SCHEMA_VERSION = "pair@1"


@dataclass
class TsToText:
    """ts -> text  (task_type text_desc). Series are user-side context; assistant writes `answer`."""
    user_intro: str                 # user text introducing the series (before each <stats><ts>)
    series: list                    # [{"name","values"(raw),"unit","freq"}]
    answer: str                     # assistant description (the supervised text)
    meta: dict = field(default_factory=dict)
    text_source: str = ""
    is_generated: str = "real"


@dataclass
class TextToTs:
    """text -> ts  (task_type ts_forecast[_covariates]). Conditioning text + history -> future."""
    user_text: str                  # world-knowledge conditioning text
    history: list                   # raw history (context span)
    future: list                    # raw future (supervised suffix of the target span)
    series_name: str
    unit: str
    freq: str
    covariates: list = field(default_factory=list)   # [{"name","values","unit","freq"}]
    meta: dict = field(default_factory=dict)
    text_source: str = ""
    is_generated: str = "real"
    knowledge_time: str = ""


Pair = Union[TsToText, TextToTs]


def _prov(ex, base):
    m = dict(base)
    m["builder"] = BUILDER
    m["pair_schema_version"] = PAIR_SCHEMA_VERSION
    m["text_source"] = ex.text_source
    m["text_quality"] = "real" if ex.is_generated == "real" else "derived"
    m["is_generated_text"] = ex.is_generated
    if getattr(ex, "knowledge_time", ""):
        m["knowledge_time"] = ex.knowledge_time
    return m


def build_record(ex):
    """Backward-compatible ChatML entry point; new code should use core.emitters.emit."""
    from core.emitters import emit_chatml
    return emit_chatml(ex)


def validate(rec):
    return chatml.validate(rec)
