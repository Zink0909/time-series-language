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
OUT_ENTITY_OOD = os.path.join(HERE, "_finetune_export_entity_ood.json")


def _pack(x):
    return {"history": [round(v, 4) for v in x["history"]],
            "future": [round(v, 4) for v in x["future"]], "text": x["text"],
            "series_group": x.get("series_group")}


def _export(items, group_disjoint):
    train, test, problems = data.time_split(
        items, "2024-01-01", "2023-10-01", group_disjoint=group_disjoint)
    tr = arms.make_arms(train, strata=("dataset",))                # within-source shuffled control
    train_groups = {x["series_group"] for x in train}
    test_groups = {x["series_group"] for x in test}
    export = {
        "cutoff": "2024-01-01", "leakage_ok": not problems,
        "split_policy": ("temporal_entity_disjoint" if group_disjoint
                         else "temporal_same_entity_allowed"),
        "group_overlap": len(train_groups & test_groups),
        "shuffle_policy": "within_dataset_strict_derangement",
        "train": {"A_no_text": [_pack(x) for x in tr["A_no_text"]],
                  "B_flywheel": [_pack(x) for x in tr["B_flywheel_text"]],
                  "C_shuffled": [_pack(x) for x in tr["C_shuffled_text"]]},
        "test": [{"series_id": x["series_id"], "dataset": x["dataset"], **_pack(x)} for x in test],
    }
    return export, len(train), len(test), problems


def main():
    items = data.load(flywheel_only=True)
    for out, group_disjoint in ((OUT, False), (OUT_ENTITY_OOD, True)):
        export, n_train, n_test, problems = _export(items, group_disjoint)
        json.dump(export, open(out, "w"), ensure_ascii=False)
        print(f"{export['split_policy']}: train {n_train} (x3 arms) + test {n_test} -> {out} "
              f"({os.path.getsize(out) // 1024} KB), group_overlap={export['group_overlap']}, "
              f"leakage_ok={not problems}")


if __name__ == "__main__":
    main()
