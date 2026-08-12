#!/usr/bin/env python3
"""Export the held-out test events (3 arms + anonymized text) to a self-contained JSON that the
Colab notebook embeds — so the rigorous base-model probe runs with NO file upload. Deterministic.
  micromamba run -n ts-language python flywheel/companion/export_probe_data.py
"""
import os, sys, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import data, arms                                                  # noqa: E402
from core.faithfulness import extract_entities                     # noqa: E402

OUT = os.path.join(HERE, "_probe_export.json")


def anonymize(text):
    for e in sorted(set(extract_entities(text)), key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(e) + r"\b", "a major entity", text)
    return text


def main():
    items = data.load(flywheel_only=True)
    _, test, _ = data.time_split(items, "2024-01-01", "2023-10-01")
    the_arms = arms.make_arms(test)
    exp = []
    for a, b, c in zip(the_arms["A_no_text"], the_arms["B_flywheel_text"],
                       the_arms["C_shuffled_text"]):
        exp.append({"series_id": b["series_id"], "dataset": b["dataset"], "origin": b["origin"],
                    "history": [round(v, 4) for v in b["history"]],
                    "future": [round(v, 4) for v in b["future"]],
                    "text": b["text"], "text_shuffled": c["text"], "text_anon": anonymize(b["text"])})
    json.dump({"cutoff": "2024-01-01", "base_cutoff": "2023-10-01", "n": len(exp), "events": exp},
              open(OUT, "w"), ensure_ascii=False)
    print(f"exported {len(exp)} test events -> {OUT}  ({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
