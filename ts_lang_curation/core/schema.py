# core/schema.py — the adapter-facing contract. Adapters yield one of two task examples;
# core/chatml.py turns them into full Data-Schema training records (ChatML + z-score spans +
# loss masks). Adapters never format ChatML or normalize directly.
from dataclasses import dataclass, field
from core import chatml

BUILDER = "ts_lang_curation@0.2"   # 0.2 = full Data-Schema (ChatML) output


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


def _prov(ex, base):
    m = dict(base)
    m["builder"] = BUILDER
    m["text_source"] = ex.text_source
    m["text_quality"] = "real" if ex.is_generated == "real" else "derived"
    m["is_generated_text"] = ex.is_generated
    if getattr(ex, "knowledge_time", ""):
        m["knowledge_time"] = ex.knowledge_time
    return m


def build_record(ex):
    if isinstance(ex, TsToText):
        return chatml.build_text_desc(ex.user_intro, ex.series, ex.answer, _prov(ex, ex.meta))
    if isinstance(ex, TextToTs):
        # text<->series coupling (world-knowledge quality dim) — stamped CENTRALLY so every text->ts
        # source carries it: a standing bio scores 'weak', an event-grounded cause 'coupled'.
        from core.match_score import match_score                       # local import: avoid any cycle
        meta = _prov(ex, ex.meta)
        cause = ex.user_text
        for cut in ("\n(history=", "Given this background", " Given this", " Given the"):
            cause = cause.split(cut)[0]
        ms, ml, _ = match_score(cause.strip(), ex.meta.get("event_date") or ex.meta.get("spike_date"))
        meta["text_ts_match"], meta["text_ts_match_score"] = ml, ms
        return chatml.build_ts_forecast(ex.user_text, ex.history, ex.future, ex.series_name,
                                        ex.unit, ex.freq, meta, covariates=ex.covariates)
    raise TypeError(f"unknown example type: {type(ex)}")


def validate(rec):
    return chatml.validate(rec)
