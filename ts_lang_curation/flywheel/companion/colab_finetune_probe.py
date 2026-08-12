# ============================================================================================
# Colab FINE-TUNE ③ probe — paste this whole thing into ONE Colab cell and run.
#   Runtime → Change runtime type → T4 (free, fits 7B in 4-bit) or L4/A100 (faster).
#   Then upload `_finetune_export.json` (exported by companion/export_finetune_data.py:
#   train A/B/C arms + a shared held-out test set) via the Files panel or the prompt below.
#
# WHY fine-tune, not zero-shot: a general LLM zero-shot IGNORES the conditioning text (the rigorous
# zero-shot probe found A≈B≈C). The real question — "does the manufactured text TEACH a model to
# use text?" — is a TRAINING question. So we LoRA-fine-tune three copies of the same base model,
# one per arm, then evaluate all three on the SAME held-out events:
#   A_no_text   — trained (and eval'd) with NO text  → the pure-numeric floor
#   B_flywheel  — trained (and eval'd) with the REAL flywheel text
#   C_shuffled  — trained on text paired to the WRONG series, eval'd with real text (control)
# Success = B's median MASE < A AND < C.  B<A but B≈C ⇒ gain is "having a text channel", not the
# content (answers the MiGAS relabel claim).  Per-dataset breakdown shows whether the DENSE
# commodity text (real events) helps more than the generic-bio wiki text — the whole hypothesis.
#
# HONEST NOTES: (1) this is a SMALL probe (train ~339 / test ~99) on a GENERAL LLM, not the team's
# TS-language base — a directional signal, not proof; the definitive run is the team's multimodal
# base (ask Xinyue whether/how it can be fine-tuned). (2) a few near-zero-baseline scaled records
# make MASE explode → we report the ROBUST median + per-dataset medians. (3) not run on the author's
# 8.5GB Mac; the QLoRA recipe is standard — sanity-check that the first eval outputs parse as numbers.
# ============================================================================================
get_ipython().system('pip -q install -U peft transformers accelerate bitsandbytes')  # noqa
# Colab preinstalls an old torchao (0.10) that newer peft rejects during LoRA dispatch; we don't
# use torchao, so remove it to avoid the ImportError (peft then skips the torchao path cleanly).
get_ipython().system('pip -q uninstall -y torchao')  # noqa

import gc, json, re, statistics as st, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Colab Pro: L4 (24GB) → 7B or 14B in bf16;  A100 (40GB) → 14B bf16 (strongest, recommended).
# Free T4 (15GB, no bf16) → set QUANT_4BIT=True and use -7B or -3B.
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"  # A100: 14B. On L4 use -7B (or -14B if it fits). T4: -7B/-3B.
QUANT_4BIT = False                      # False = bf16 (A100/L4, cleaner); True = 4-bit QLoRA (T4)
EPOCHS, LR, MAXLEN = 3, 2e-4, 640
LIMIT_TRAIN = 0                         # 0 = all; set e.g. 150 for a faster first pass

# --- 1. data ---------------------------------------------------------------------------------
try:
    DATA = json.load(open("_finetune_export.json"))
except FileNotFoundError:
    from google.colab import files
    DATA = json.load(open(next(iter(files.upload()))))
TRAIN, TEST = DATA["train"], DATA["test"]
print(f"cutoff {DATA['cutoff']} · leakage_ok={DATA['leakage_ok']} · "
      f"train {len(TRAIN['B_flywheel'])}/arm · test {len(TEST)}")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_SYS = ("You are a forecasting engine. Given optional real-world context and recent normalized "
        "numeric values, predict the next N values. Respond with ONLY N comma-separated numbers.")


def _prompt(history, text, H, use_text):
    ctx = f"Context: {text}\n" if (use_text and text) else ""
    user = (ctx + f"Recent normalized values ({len(history)}): "
            + ", ".join(f"{v:.2f}" for v in history)
            + f"\nPredict the next {H} values (comma-separated, exactly {H} numbers):")
    return tok.apply_chat_template([{"role": "system", "content": _SYS},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def _target(future):
    return ", ".join(f"{v:.2f}" for v in future)


def _tok_example(e, use_text):
    p = tok(_prompt(e["history"], e.get("text", ""), len(e["future"]), use_text),
            add_special_tokens=False).input_ids
    c = tok(_target(e["future"]) + tok.eos_token, add_special_tokens=False).input_ids
    ids, labels = (p + c)[:MAXLEN], ([-100] * len(p) + c)[:MAXLEN]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def _train_arm(name, key, use_text, seed=0):
    print(f"\n=== [seed {seed}] training arm {name} ({key}, text={use_text}) ===", flush=True)
    torch.manual_seed(seed)
    gc.collect()
    torch.cuda.empty_cache()                             # clear the previous arm before loading
    # device_map={"":0} forces the WHOLE model onto GPU 0 — never CPU/meta offload (offload +
    # LoRA backward = "expected device meta" errors). One 14B bf16 (~28GB) fits an A100; on a
    # smaller GPU drop to -7B or set QUANT_4BIT=True.
    if QUANT_4BIT:                                       # T4 path: 4-bit QLoRA
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                    device_map={"": 0})
        base = prepare_model_for_kbit_training(base)
    else:                                                # A100/L4 path: bf16 (no unscale issues)
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                    device_map={"": 0})
    m = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    ex = TRAIN[key][:LIMIT_TRAIN] if LIMIT_TRAIN else TRAIN[key]
    ds = [_tok_example(e, use_text) for e in ex]
    args = TrainingArguments(output_dir="/tmp/" + name, per_device_train_batch_size=2,
                             gradient_accumulation_steps=4, num_train_epochs=EPOCHS,
                             learning_rate=LR, bf16=not QUANT_4BIT, fp16=QUANT_4BIT,
                             seed=seed, data_seed=seed,
                             logging_steps=25, save_strategy="no", report_to=[])
    Trainer(model=m, args=args, train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True)).train()
    m.eval()
    return m


@torch.no_grad()
def _predict(m, history, text, H, use_text):
    ids = tok(_prompt(history, text, H, use_text), return_tensors="pt").to(m.device)
    out = m.generate(**ids, max_new_tokens=16 * H + 48, do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", txt)][:H]
    return (nums + [history[-1]] * H)[:H]


def _mase(pred, actual, hist):
    scale = st.mean(abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))) or 1e-6
    H = min(len(pred), len(actual))
    return st.mean(abs(pred[i] - actual[i]) for i in range(H)) / scale


# --- 2. run: N_SEEDS × three arms (eval on the SAME test events each seed) --------------------
import random
N_SEEDS = 3                                     # 1 = quick look; 3 = enough for a bootstrap CI
ARMS = {"A": ("A_no_text", False), "B": ("B_flywheel", True), "C": ("C_shuffled", True)}
DSET = [e["dataset"] for e in TEST]
N = len(TEST)
# mase_seed[arm] = list over seeds of [per-event MASE]
mase_seed = {a: [] for a in ARMS}
for seed in range(N_SEEDS):
    for name, (key, use_text) in ARMS.items():
        m = _train_arm(name, key, use_text, seed)
        ev_text = name != "A"                   # A stays text-blind; B & C see the real test text
        per = [_mase(_predict(m, e["history"], e.get("text", ""), len(e["future"]), ev_text),
                     e["future"], e["history"]) for e in TEST]
        mase_seed[name].append(per)
        print(f"  [seed {seed}] arm {name} median MASE {st.median(per):.3f}", flush=True)
        del m                                   # m holds the only ref to its base model; gc frees both
        gc.collect()
        torch.cuda.empty_cache()

# --- 3. aggregate: average each event's MASE across seeds, then bootstrap the median delta -----
def _avg(arm):                                  # per-event mean across seeds (cuts seed noise)
    S = len(mase_seed[arm])
    return [st.mean(mase_seed[arm][s][i] for s in range(S)) for i in range(N)]


A_, B_, C_ = _avg("A"), _avg("B"), _avg("C")


def _boot(deltas, iters=3000):                  # 95% CI of the MEDIAN delta, resampling events
    obs = st.median(deltas)
    meds = []
    for _ in range(iters):
        s = [deltas[random.randrange(N)] for _ in range(N)]
        meds.append(st.median(s))
    meds.sort()
    return obs, meds[int(0.025 * iters)], meds[int(0.975 * iters)]


dBA = [B_[i] - A_[i] for i in range(N)]         # <0 = flywheel text beats no-text
dBC = [B_[i] - C_[i] for i in range(N)]         # <0 = correct pairing beats shuffled
random.seed(0)
mBA, loBA, hiBA = _boot(dBA)
mBC, loBC, hiBC = _boot(dBC)
medA, medB, medC = st.median(A_), st.median(B_), st.median(C_)
print(f"\n=== FINE-TUNE ③ probe · {MODEL_ID} · n={N} · seeds={N_SEEDS} ===")
print(f"  A no-text  {medA:.3f}   B flywheel {medB:.3f}   C shuffled {medC:.3f}  (seed-avg medians)")
print(f"  Δ(B−A) median {mBA:+.3f}  95% CI [{loBA:+.3f}, {hiBA:+.3f}]   "
      f"{'SIGNIFICANT (CI<0) ✅' if hiBA < 0 else 'not significant (CI crosses 0)'}")
print(f"  Δ(B−C) median {mBC:+.3f}  95% CI [{loBC:+.3f}, {hiBC:+.3f}]   "
      f"{'SIGNIFICANT (CI<0) ✅' if hiBC < 0 else 'not significant (CI crosses 0)'}")
print(f"  per-event B<A {sum(x < 0 for x in dBA)}/{N} · B<C {sum(x < 0 for x in dBC)}/{N}")
print("\n  per-dataset (seed-avg median MASE A → B):")
for dsname in sorted(set(DSET)):
    idx = [i for i in range(N) if DSET[i] == dsname]
    a, b = st.median(A_[i] for i in idx), st.median(B_[i] for i in idx)
    print(f"    {dsname:24} n={len(idx):<3} A={a:6.2f} → B={b:6.2f}  ({'helps' if b < a else 'no'})")
print("\n  reading: a 95% CI ENTIRELY below 0 = the effect is unlikely to be seed/sample noise.")
print("  still a general LLM + small n; the dense COMMODITY rows are the hypothesis. Definitive "
      "test = the team's TS-language multimodal base (confirm with Xinyue it can be fine-tuned).")
