#!/usr/bin/env python3
"""W6 companion-study scaffold — the executable pipeline that will answer "does the manufactured
text actually help downstream forecasting?". It runs end-to-end TODAY on the flywheel data with
NUMERIC baselines (no GPU): time-split -> 3 arms (A no-text / B flywheel-text / C shuffled-text)
-> forecast -> MASE, with a leakage assertion. Numeric baselines are text-blind so A=B=C by
construction — that establishes the no-text floor and proves the plumbing; the TEXT gain (B > A
and B > C) needs a semantic base model + GPU, wired via models.TextConditionedStub (see README).

  micromamba run -n ts-language python flywheel/companion/run.py [--cutoff YYYY-MM-DD]
"""
import os, sys, json, argparse
from statistics import mean, median

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))         # ts_lang_curation (for core)
sys.path.insert(0, HERE)                                            # companion modules
import data, arms, models                                          # noqa: E402

REPORT = os.path.join(HERE, "_companion_result.json")


def mase(pred, actual, history):
    scale = mean(abs(history[i] - history[i - 1]) for i in range(1, len(history))) or 1e-6
    H = min(len(pred), len(actual))
    return mean(abs(pred[i] - actual[i]) for i in range(H)) / scale


def eval_arm(arm_items, model):
    # median MASE: robust to the long tail of near-zero-baseline spike events (a few flywheel_wiki
    # series blow up in z-space), which would otherwise dominate a mean.
    scores = []
    for x in arm_items:
        H = len(x["future"])
        pred = model.predict(x["history"], x["text"], H)
        scores.append(mase(pred, x["future"], x["history"]))
    return median(scores) if scores else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2024-01-01", help="train origin < cutoff <= test origin")
    ap.add_argument("--base-cutoff", default="2023-10-01",
                    help="base-model pretrain cutoff; test events after it can't be memorization")
    ap.add_argument("--llm", action="store_true",
                    help="also run the LLMTime text arm via the team vLLM (zero-GPU probe)")
    ap.add_argument("--limit", type=int, default=0, help="only first N test events (quick probe)")
    a = ap.parse_args()

    items = data.load(flywheel_only=True)
    train, test, problems = data.time_split(items, a.cutoff, a.base_cutoff)
    after = [x for x in test if x.get("after_base_cutoff")]
    print(f"== W6 companion scaffold · {len(items)} flywheel text->ts "
          f"(train {len(train)} / test {len(test)}; {len(after)} test events after base cutoff) ==\n")

    print("leakage assertion (cut time before windowing):",
          "CLEAN ✓" if not problems else "VIOLATIONS ✗")
    for p in problems:
        print("   ", p)
    if problems:
        sys.exit(1)

    if a.limit:
        test = test[:a.limit]
    the_arms = arms.make_arms(test)
    print(f"\narms built on {len(test)} test events: "
          + ", ".join(f"{k}={len(v)}" for k, v in the_arms.items()))

    # numeric baselines (text-blind): fill the no-text floor; A=B=C expected.
    print("\nmedian MASE by arm × numeric baseline (lower=better; A=B=C since numeric ignores text):")
    header = "  {:<18}".format("model") + "".join(f"{k:>18}" for k in the_arms)
    print(header)
    table = {}
    for model in models.NUMERIC_BASELINES:
        row = {arm: round(eval_arm(items_, model), 4) for arm, items_ in the_arms.items()}
        table[model.name] = row
        print("  {:<18}".format(model.name) + "".join(f"{row[k]:>18}" for k in the_arms))

    # zero-GPU text arm: LLMTime via the team vLLM (real numbers, not a faked gain).
    if a.llm:
        llm = models.LLMTimeForecaster(os.path.join(HERE, "_llmtime_cache.json"))
        print(f"\nmedian MASE — LLMTime text arm (zero-shot via team vLLM, temp=0, {len(test)} events):")
        print(header)
        lrow = {arm: round(eval_arm(items_, llm), 4) for arm, items_ in the_arms.items()}
        llm.save()
        print("  {:<18}".format(llm.name) + "".join(f"{lrow[k]:>18}" for k in the_arms))
        A_, B_, C_ = lrow["A_no_text"], lrow["B_flywheel_text"], lrow["C_shuffled_text"]
        met = B_ < A_ and B_ < C_
        print(f"  success criterion B<A and B<C: {'MET ✓' if met else 'not met'}  "
              f"(B={B_} vs A={A_}, C={C_})  — first signal, not the final verdict")
        # per-event win rate — medians can hide a few dominating items.
        sc = {arm: [mase(llm.predict(x["history"], x["text"], len(x["future"])),
                         x["future"], x["history"]) for x in items_]
              for arm, items_ in the_arms.items()}
        n = len(test)
        bwa = sum(b < a for a, b in zip(sc["A_no_text"], sc["B_flywheel_text"]))
        bwc = sum(b < c for c, b in zip(sc["C_shuffled_text"], sc["B_flywheel_text"]))
        print(f"  per-event win rate: B beats A on {bwa}/{n}, B beats C on {bwc}/{n}")
        print(f"  caveats: n={n}, GENERAL LLM (not a TS base model), possible implicit lookahead "
              "(may have seen these events) — a directional signal, not proof.")
        table["llmtime"] = lrow
        table["llmtime_winrate"] = {"B_beats_A": bwa, "B_beats_C": bwc, "n": n}

    # text-conditioned arm: pending a real multimodal TS base model (do not fake a gain).
    stub = models.TextConditionedStub()
    try:
        eval_arm(the_arms["B_flywheel_text"], stub)
        text_status = "ran"
    except NotImplementedError as e:
        text_status = f"PENDING — {e}"
    print(f"\ntext-conditioned base ({stub.name}): {text_status}")
    print("  -> success criterion when wired: B_flywheel_text < A_no_text AND < C_shuffled_text,")
    print("     strongest on the {} test events after the base-model cutoff.".format(len(after)))

    json.dump({"cutoff": a.cutoff, "base_cutoff": a.base_cutoff, "n_train": len(train),
               "n_test": len(test), "n_after_base_cutoff": len(after),
               "leakage_clean": not problems, "numeric_mase": table,
               "text_arm": text_status}, open(REPORT, "w"), indent=1)
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
