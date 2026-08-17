#!/usr/bin/env python3
"""chatml_to_cpt.py — convert ChatML `text_desc` (understanding) records into the CPT
world-knowledge (instruction-free) format. This is the PRIMARY merge direction after the team
chose CPT as the unified understanding format (both converters exist; CPT is canonical first).

Mapping (ChatML text_desc -> CPT world_knowledge):
  * assistant description text (the loss-masked span)  -> the CPT prose
  * each ChatML observed span (z-normalised)           -> RAW values via raw = z*std + mean
    (using the record's own normalization block), kept as a CPT channel {values,unit,freq}
  * append one <ts></ts> as a closing reference (or one per channel when multi_series)
  * task_type -> "world_knowledge"; text_quality mapped to the CPT enum; provenance carried

LOSSY BY DESIGN (flagged): CPT is instruction-free, so the ChatML task framing is dropped ---
the user prompt ("Describe the trend."), the loss masks, and the <think> scaffold do not survive.
That is expected (that is what "instruction-free" means), but it means a round-trip is not
bit-identical. The prose + series + provenance are preserved.

Usage:
    python chatml_to_cpt.py ChatML.jsonl [--out cpt.jsonl] [--limit N]
"""
from __future__ import annotations
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import verify_cpt as vcpt  # reuse loader + the CPT checker (不自证 of the converter)

TS = "<ts></ts>"
# ChatML freq label -> CPT compact token.
FREQ_MAP = {"daily": "1d", "business_daily": "1d", "weekly": "1w", "monthly": "1M",
            "quarterly": "1q", "yearly": "1y", "annual": "1y", "hourly": "1h", "6-hourly": "6h"}
TQ_MAP = {"real": "real", "derived": "generated", "generated": "generated",
          "derived_generated": "generated", "template": "generated"}


def _description(rec):
    """The assistant answer = the loss-masked range (falls back to parsing the assistant turn)."""
    rngs = rec.get("text_loss_char_ranges")
    text = rec.get("text", "")
    if rngs:
        return " ".join(text[a:b] for a, b in rngs).strip()
    m = re.search(r"<\|im_start\|>assistant\s*(?:<think>.*?</think>)?\s*(.*?)<\|im_end\|>", text, re.S)
    return (m.group(1).strip() if m else "")


def _denorm(rec):
    """Return [{values(raw),unit,freq}] by de-normalising each span with its own mean/std."""
    spans = rec.get("timeseries") or []
    norm = {s.get("span_idx", i): s for i, s in enumerate(rec.get("normalization", {}).get("spans", []))}
    out = []
    for i, sp in enumerate(spans):
        vals = sp.get("values") or []
        ns = norm.get(i, {})
        m, sd = ns.get("mean"), ns.get("std")
        if m is None or sd is None:      # no norm info -> keep as-is (rare)
            raw = list(vals)
        else:
            raw = [round(v * sd + m, 6) for v in vals]
        fr = str(sp.get("freq"))
        out.append({"values": raw, "unit": sp.get("unit"), "freq": FREQ_MAP.get(fr, fr)})
    return out


def convert(rec):
    """ChatML text_desc -> CPT world_knowledge. (rec, note) or (None, reason)."""
    if rec.get("task_type") != "text_desc":
        return None, f"not text_desc ({rec.get('task_type')})"
    desc = _description(rec)
    if len(desc) < 10:
        return None, "empty/short description"
    chans = _denorm(rec)
    if not chans or any(not c["values"] for c in chans):
        return None, "no series values"
    freqs = {c["freq"] for c in chans}
    multi = len(freqs) > 1 or len(chans) > 1 and False  # group same-freq channels under one <ts>
    if len(freqs) > 1:
        text = desc.rstrip(". ") + ". " + " ".join(TS for _ in chans)
        multi = True
    else:
        text = desc.rstrip(". ") + ". " + TS
        multi = False
    out = {
        "text": text,
        "timeseries": chans,
        "task_type": "world_knowledge",
        "text_quality": TQ_MAP.get(rec.get("text_quality"), "generated"),
    }
    if multi:
        out["multi_series"] = True
    for k_src, k_dst in [("series_id", "series_id"), ("dataset", "dataset"), ("source", "source"),
                         ("license", "license"), ("knowledge_time", "knowledge_time")]:
        if k_src in rec:
            out[k_dst] = rec[k_src]
    out["alignment"] = rec.get("alignment", "describes")   # a text_desc is a describe by construction
    out["text_source"] = rec.get("text_source", "converted_from_chatml")
    out["converted_from"] = "chatml_text_desc"
    return out, ("multi_series" if multi else "ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    recs = vcpt.load_records(a.path)
    if a.limit:
        recs = recs[:a.limit]
    src_desc = sum(1 for r in recs if r.get("task_type") == "text_desc")
    ok, skip, notes = [], 0, {}
    for r in recs:
        out, note = convert(r)
        if out is None:
            skip += 1
        else:
            ok.append(out); notes[note] = notes.get(note, 0) + 1
    print(f"text_desc in input: {src_desc}/{len(recs)}  ->  converted {len(ok)}  (skipped {skip})  notes={notes}")
    # independent check: every converted record must pass the CPT contract (不自证)
    bad = sum(1 for r in ok if vcpt.check_record(r, 2)[0])
    print(f"CPT contract validation: {len(ok)-bad}/{len(ok)} pass, {bad} fail")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in ok:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
