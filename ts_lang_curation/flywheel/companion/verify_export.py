#!/usr/bin/env python3
"""VERIFICATION PIPELINE (step 1 / export): for ONE source, export a 3-arm train/test split so we can
check whether TRAINING on that source's data helps forecasting on its OWN held-out (in-distribution).

The question (from the 2026-07-17 meeting): before merging a source into production, prove it is useful
by fine-tuning MiGAS on its train split and seeing if it beats the not-trained base on its test split. If
even ID doesn't improve, the source/domain is useless -> drop it, do not merge.

Chronological 80/20 split (train = earliest 80% by forecast origin, test = latest 20%; leakage-safe).
Arms: A no-text / B correct-text / C shuffled-text. B>baseline = training helps; B>A = the TEXT helps.

  micromamba run -n ts-language python flywheel/companion/verify_export.py <source> [train_frac]
then on the box:
  cd /data/jiaju/synthefy-migas && CUDA_VISIBLE_DEVICES=N uv run python /data/jiaju/migas_finetune.py \
      --migas-dir /data/jiaju/synthefy-migas --finetune-file /data/jiaju/_verify_<source>.json \
      --gen-test timemmd --seeds 3 --epochs 6 --embedder-device cuda:0
and read the "held-out" block (baseline vs A vs B) = the ID verdict.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import data, arms                                                    # noqa: E402


def _pack(x):
    return {"history": [round(v, 4) for v in x["history"]],
            "future": [round(v, 4) for v in x["future"]], "text": x["text"]}


def main():
    src = sys.argv[1]
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    items = [x for x in data.load(datasets=[src], flywheel_only=False) if x["origin"]]
    items.sort(key=lambda x: x["origin"])
    n = len(items)
    if n < 20:
        print(f"WARNING {src}: only {n} forecast items — verification will be noisy")
    cutoff = items[int(n * frac)]["origin"]
    train = [x for x in items if x["origin"] < cutoff]
    test = [x for x in items if x["origin"] >= cutoff]
    # leakage check: every train origin strictly before cutoff <= every test origin
    assert (not train or max(x["origin"] for x in train) < cutoff) and \
           (not test or min(x["origin"] for x in test) >= cutoff), "time split leak"
    tr = arms.make_arms(train)
    export = {
        "source": src, "cutoff": cutoff, "n_train": len(train), "n_test": len(test),
        "train": {"A_no_text": [_pack(x) for x in tr["A_no_text"]],
                  "B_flywheel": [_pack(x) for x in tr["B_flywheel_text"]],
                  "C_shuffled": [_pack(x) for x in tr["C_shuffled_text"]]},
        "test": [{"series_id": x["series_id"], "dataset": x["dataset"], **_pack(x)} for x in test],
    }
    out = os.path.join(HERE, f"_verify_{src}.json")
    json.dump(export, open(out, "w"), ensure_ascii=False)
    hl = [len(x["history"]) for x in items]
    fl = [len(x["future"]) for x in items]
    print(f"{src}: train {len(train)}x3 arms + test {len(test)} (cutoff {cutoff})")
    print(f"  history len min/median/max = {min(hl)}/{sorted(hl)[len(hl)//2]}/{max(hl)} · "
          f"future = {min(fl)}/{sorted(fl)[len(fl)//2]}/{max(fl)}")
    print(f"  -> {out}   (MiGAS clips to history>=10, future>=5; short-window sources lose records)")


if __name__ == "__main__":
    main()
