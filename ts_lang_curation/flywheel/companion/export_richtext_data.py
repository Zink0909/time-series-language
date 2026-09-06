#!/usr/bin/env python3
"""Controlled THIN-vs-RICH commodity export for the ③ line-B test.

The MiGAS verdict (Time-MMD AND TimesX, both external) was: text is hugely useful, but correct
text↔series PAIRING has no generalization value — shuffled C wins. The open question (line B): is that
because our text was LOW-QUALITY (single-sentence causes / standing wiki bios), not because pairing is
worthless? TimesX's own finding — swapping low- for high-quality context gives +7.3% — says quality
matters. So here we hold EVERYTHING fixed except text quality:

  * same commodity events (the intersection of the thin `flywheel_commodity` and the rich
    `flywheel_commodity_rich` sets — only events that passed the gates in BOTH),
  * same time-split, same shuffle seed,
  * ONLY the conditioning text differs: thin single-sentence cause vs rich TimesX-style numbered
    multi-event context.

Emit two exports the MiGAS scaffold runs via --finetune-file; comparing B-vs-C on the external
generalization set (TimesX price subset) across the two tells us whether higher-quality event text
makes correct pairing valuable.
  micromamba run -n ts-language python flywheel/companion/export_richtext_data.py
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data, arms                                                    # noqa: E402

OUT_THIN = os.path.join(HERE, "_richtext_thin.json")
OUT_RICH = os.path.join(HERE, "_richtext_rich.json")
CUTOFF, PRETRAIN = "2024-01-01", "2023-10-01"


def _pack(x):
    return {"history": [round(v, 4) for v in x["history"]],
            "future": [round(v, 4) for v in x["future"]], "text": x["text"],
            "series_group": x.get("series_group")}


def _key(x):
    """Event identity independent of text style: strip the style suffix from series_id."""
    return x["series_id"].replace("_flywheel_rich_t2s", "").replace("_flywheel_t2s", "")


def build(items, out):
    train, test, problems = data.time_split(items, CUTOFF, PRETRAIN, group_disjoint=False)
    tr = arms.make_arms(train, strata=("dataset",))
    export = {
        "cutoff": CUTOFF, "leakage_ok": not problems,
        "split_policy": "temporal_same_entity_allowed",
        "group_overlap": len({x["series_group"] for x in train} &
                             {x["series_group"] for x in test}),
        "shuffle_policy": "within_dataset_strict_derangement",
        "train": {"A_no_text": [_pack(x) for x in tr["A_no_text"]],
                  "B_flywheel": [_pack(x) for x in tr["B_flywheel_text"]],
                  "C_shuffled": [_pack(x) for x in tr["C_shuffled_text"]]},
        "test": [{"series_id": x["series_id"], "dataset": x["dataset"], **_pack(x)} for x in test],
    }
    json.dump(export, open(out, "w"), ensure_ascii=False)
    return len(train), len(test), not problems


def main():
    thin = data.load(datasets=["flywheel_commodity"])
    rich = data.load(datasets=["flywheel_commodity_rich"])
    tmap = {_key(x): x for x in thin}
    rmap = {_key(x): x for x in rich}
    common = sorted(set(tmap) & set(rmap))
    thin_c = [tmap[k] for k in common]
    rich_c = [rmap[k] for k in common]                              # SAME events, richer text
    print(f"thin {len(thin)} · rich {len(rich)} · common {len(common)} events (controlled)")
    tn = build(thin_c, OUT_THIN)
    rn = build(rich_c, OUT_RICH)
    print(f"thin export: train {tn[0]} (x3 arms) + test {tn[1]}  leakage_ok={tn[2]} -> {OUT_THIN}")
    print(f"rich export: train {rn[0]} (x3 arms) + test {rn[1]}  leakage_ok={rn[2]} -> {OUT_RICH}")
    # sanity: same event set, so identical train/test counts
    assert tn[:2] == rn[:2], "thin/rich splits diverged — event alignment bug"
    print("OK: thin and rich cover identical events (text-only difference)")


if __name__ == "__main__":
    main()
