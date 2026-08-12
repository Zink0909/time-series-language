#!/usr/bin/env python3
"""Export train+test splits (3 TRAINING arms) for the ③ FINE-TUNE experiment — the CORRECT test of
flywheel data (its value is in TRAINING/continued-pretraining, not zero-shot inference).

Time-split (no leakage): train = event origin < cutoff, test = origin >= cutoff. Three training
arms differ only in the conditioning TEXT the model is trained on:
  A_no_text   — series only (text stripped)
  B_flywheel  — the real flywheel text
  C_shuffled  — text permuted onto the wrong series
All three fine-tune a LoRA, then are evaluated on the SAME held-out test set (history+real text) →
if B beats A and C downstream, the flywheel text taught the model to use text. Deterministic.
  micromamba run -n ts-language python flywheel/companion/export_finetune_data.py
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import data, arms                                                  # noqa: E402

OUT = os.path.join(HERE, "_finetune_export.json")


def _pack(x):
    return {"history": [round(v, 4) for v in x["history"]],
            "future": [round(v, 4) for v in x["future"]], "text": x["text"]}


def main():
    items = data.load(flywheel_only=True)
    train, test, problems = data.time_split(items, "2024-01-01", "2023-10-01")
    tr = arms.make_arms(train)                                     # A_no_text / B_flywheel_text / C_shuffled_text
    export = {
        "cutoff": "2024-01-01", "leakage_ok": not problems,
        "train": {"A_no_text": [_pack(x) for x in tr["A_no_text"]],
                  "B_flywheel": [_pack(x) for x in tr["B_flywheel_text"]],
                  "C_shuffled": [_pack(x) for x in tr["C_shuffled_text"]]},
        "test": [{"series_id": x["series_id"], "dataset": x["dataset"], **_pack(x)} for x in test],
    }
    json.dump(export, open(OUT, "w"), ensure_ascii=False)
    print(f"train {len(train)} (x3 arms) + test {len(test)} events -> {OUT} "
          f"({os.path.getsize(OUT) // 1024} KB), leakage_ok={not problems}")


if __name__ == "__main__":
    main()
