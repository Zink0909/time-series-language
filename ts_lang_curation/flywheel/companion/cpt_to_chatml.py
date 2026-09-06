#!/usr/bin/env python3
"""cpt_to_chatml.py — convert CPT world-knowledge (understanding) records into the ChatML
`text_desc` format, so the two producers' UNDERSTANDING data lands in one training format.

This is the WRITE half of the format converter (the READ half lives in verify_any.detect+parse).
Understanding data = "text about a series"; the target training format is ChatML text_desc
(series on the user side as observed spans, the prose as the assistant answer with loss on it).

Mapping (CPT -> ChatML text_desc):
  * each CPT channel {values(raw),unit,freq}  -> a user-side observed series (build_text_desc
    handles the z-normalisation + <stats>/<ts> assembly)
  * the CPT prose (its `<ts></ts>` splice point removed + lightly cleaned) -> the assistant answer
  * a generic understanding intro on the user side
  * provenance carried into meta (series_id, dataset, source, license, alignment, ...)

HONEST CAVEAT (surfaced by building this): CPT prose is authored to splice the series INTO the
sentence (usually a closing "... : <ts>."). When <ts> is at the end this cleans up fine; when it
is mid-sentence the answer reads slightly off. So this auto-conversion is structurally valid but
not semantically perfect — for the cleanest understanding data, prose authored for the standalone
text_desc answer is better. Flagged for the team.

Usage:
    python cpt_to_chatml.py CPT.jsonl [--out chatml.jsonl] [--limit N]
"""
from __future__ import annotations
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # ts_lang_curation (for core)
sys.path.insert(0, HERE)
import verify_cpt as vcpt
from core import chatml  # noqa: E402

TS = "<ts></ts>"
UNDERSTAND_INTRO = "Describe the following series and what it reflects."

# CPT freq token -> a human freq label used on the ChatML side (best-effort; unify as needed).
FREQ_MAP = {"1d": "daily", "1w": "weekly", "1W": "weekly", "1M": "monthly", "1m": "monthly",
            "1q": "quarterly", "1y": "yearly", "6h": "6-hourly", "1h": "hourly", "100ms": "100ms"}


def _clean_answer(text):
    """Remove the <ts> splice point and tidy the seam left behind."""
    a = text.replace(TS, " ").strip()
    a = re.sub(r"\s+([.,;:])", r"\1", a)         # " ." -> "."
    a = re.sub(r"[:\-–—]\s*\.", ".", a)          # "week: ." -> "week."
    a = re.sub(r"\s{2,}", " ", a)                # collapse double spaces
    a = re.sub(r"\bat\s*\.", ".", a)             # "was low at ." -> "was low."
    return a.strip()


def convert(rec):
    """CPT record -> ChatML text_desc record. Returns (rec, note) or (None, reason)."""
    text = rec.get("text", "")
    ts = rec.get("timeseries") or []
    if not text or not ts:
        return None, "empty text/timeseries"
    series = []
    for ch in ts:
        if not isinstance(ch, dict) or not isinstance(ch.get("values"), list) or not ch["values"]:
            return None, "bad channel"
        fr = str(ch.get("freq"))
        series.append({"name": ch.get("unit", "series"), "values": ch["values"],
                       "unit": ch.get("unit"), "freq": FREQ_MAP.get(fr, fr)})
    answer = _clean_answer(text)
    if len(answer) < 10:
        return None, "answer too short after cleaning"
    meta = {k: rec[k] for k in ("series_id", "dataset", "source", "license", "license_status",
                                "license_terms_url", "attribution", "license_reason",
                                "governance_registry_version", "governance_reviewed_at",
                                "governance_canonical_dataset", "alignment",
                                "text_source", "domain", "region", "period_start", "period_end",
                                "text_quality") if k in rec}
    meta["converted_from"] = "cpt_world_knowledge"
    out = chatml.build_text_desc(UNDERSTAND_INTRO, series, answer, meta)
    note = "ts_mid_sentence" if TS in text and not re.search(r"[:.]\s*" + re.escape(TS), text) else "ok"
    return out, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    recs = vcpt.load_records(a.path)
    if a.limit:
        recs = recs[:a.limit]
    ok, skip, notes = [], 0, {}
    for r in recs:
        out, note = convert(r)
        if out is None:
            skip += 1
        else:
            ok.append(out)
            notes[note] = notes.get(note, 0) + 1
    print(f"converted {len(ok)}/{len(recs)}  (skipped {skip})  notes={notes}")
    # validate every converted record with the authoritative ChatML validator (不自证 of the converter)
    bad = sum(1 for r in ok if chatml.validate(r))
    print(f"ChatML schema validation: {len(ok)-bad}/{len(ok)} pass, {bad} fail")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in ok:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
