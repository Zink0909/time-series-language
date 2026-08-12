# ============================================================================================
# Colab rigorous ③ probe — paste this whole thing into ONE Colab cell and run.
# Runtime → Change runtime type → T4 GPU (free).  Then upload `_probe_export.json` (79 test events,
# exported by companion/export_probe_data.py) via the left Files panel or the upload prompt below.
#
# What it does (the rigorous version of the local team-vLLM probe): runs FOUR arms with an
# OPEN, reproducible LLM instead of the team's private vLLM —
#   A no_text · B flywheel_text · C shuffled_text · B' anonymized_text
# and reports median MASE, per-event win rates, and the lookahead-retention %.
#
# WHY this is more rigorous than the local probe:
#   • open + reproducible model (anyone can re-run; not a private endpoint);
#   • KNOWN model identity → you can reason about its pretraining cutoff. For the cleanest
#     lookahead control, pick a model whose cutoff PREDATES the test window (origins ≥ 2024-01):
#     e.g. a Llama-2 (2023) checkpoint. The default below (Qwen2.5) has a ~2024 cutoff, so it
#     still overlaps — swap MODEL_ID to tighten the lookahead argument.
#
# HONEST NOTE: this cell was NOT run on the author's 8.5 GB Mac (can't fit the model) — the
# LLMTime prompt/parse/MASE logic is identical to the locally-verified companion/models.py; the
# transformers load/generate block is standard API. Verify the first few outputs parse as numbers.
# ============================================================================================
import json, re, statistics as st

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # fits fully on Colab Pro L4 (24GB) or A100 — no CPU offload.
                                        # A100 can go bigger: Qwen/Qwen2.5-14B-Instruct (stronger).
                                        # Free T4 (15GB) can only fit 3B (Qwen/Qwen2.5-3B-Instruct).
LIMIT = 0                               # 0 = all 79 events; set 40 for a faster first look

# --- 1. data ---------------------------------------------------------------------------------
try:
    DATA = json.load(open("_probe_export.json"))
except FileNotFoundError:
    from google.colab import files                     # upload prompt if not already present
    up = files.upload()
    DATA = json.load(open(next(iter(up))))
events = DATA["events"][:LIMIT] if LIMIT else DATA["events"]
print(f"{len(events)} test events (cutoff {DATA['cutoff']}, base_cutoff {DATA['base_cutoff']})")

# --- 2. model --------------------------------------------------------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID)
# .to("cuda") forces the whole 7B onto the L4 GPU (24GB fits 7B's ~14GB). device_map="auto" was
# over-cautiously offloading a few layers to CPU — that's what made it crawl. If you ever OOM here,
# use 4-bit instead: from_pretrained(MODEL_ID, load_in_4bit=True) (needs bitsandbytes).
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
model.eval()

_SYS = ("You are a forecasting engine. Given optional real-world context and a series of recent "
        "normalized numeric values, predict the next N values. Respond with ONLY N numbers "
        "separated by commas — no words, no explanation.")

@torch.no_grad()
def llmtime(history, text, H):
    user = (f"Context: {text or '(none)'}\n"
            f"Recent normalized values ({len(history)}): " + ", ".join(f"{v:.2f}" for v in history) +
            f"\nPredict the next {H} values (comma-separated, exactly {H} numbers):")
    prompt = tok.apply_chat_template([{"role": "system", "content": _SYS},
                                      {"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=16 * H + 48, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", txt)][:H]
    return (nums + [history[-1]] * H)[:H]               # pad short/garbled

# --- 3. metric -------------------------------------------------------------------------------
def mase(pred, actual, hist):
    scale = st.mean(abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))) or 1e-6
    H = min(len(pred), len(actual))
    return st.mean(abs(pred[i] - actual[i]) for i in range(H)) / scale

# --- 4. four arms ----------------------------------------------------------------------------
rows = []
for k, e in enumerate(events):
    h, f, H = e["history"], e["future"], len(e["future"])
    r = {"A": mase(llmtime(h, "", H), f, h),
         "B": mase(llmtime(h, e["text"], H), f, h),
         "C": mase(llmtime(h, e["text_shuffled"], H), f, h),
         "Ba": mase(llmtime(h, e["text_anon"], H), f, h),
         "dataset": e["dataset"]}
    rows.append(r)
    print(f"  [{k+1}/{len(events)}] {e['dataset']:22} A={r['A']:.2f} B={r['B']:.2f} "
          f"C={r['C']:.2f} Ba={r['Ba']:.2f}", flush=True)

# --- 5. report -------------------------------------------------------------------------------
def med(key): return st.median(x[key] for x in rows)
n = len(rows)
degenerate = sum(1 for x in rows if x["A"] == x["B"] == x["C"])   # model returned no numbers -> padded
mA, mB, mC, mBa = med("A"), med("B"), med("C"), med("Ba")
bwa = sum(x["B"] < x["A"] for x in rows)
bwc = sum(x["B"] < x["C"] for x in rows)
ret = ((mA - mBa) / (mA - mB) * 100) if mA != mB else 0.0
print(f"\n=== rigorous ③ probe · {MODEL_ID} · n={n} ===")
print(f"  A no-text        median MASE {mA:.3f}")
print(f"  B flywheel-text  median MASE {mB:.3f}   (B<A on {bwa}/{n})")
print(f"  C shuffled-text  median MASE {mC:.3f}   (B<C on {bwc}/{n})")
print(f"  B' anonymized    median MASE {mBa:.3f}   gain retained {ret:.0f}% (lookahead check)")
print(f"  criterion B<A and B<C on median: {'MET' if (mB<mA and mB<mC) else 'not met'}")
print(f"  degenerate (model returned no parseable numbers, padded): {degenerate}/{n}"
      f"{'  <- if high, results unreliable; tell Claude to harden the parse' if degenerate > n*0.15 else ''}")
print("  reminder: still a general LLM; the definitive test is a TS-specialized multimodal base "
      "trained by the team. This just makes the signal open + reproducible + cutoff-controllable.")
