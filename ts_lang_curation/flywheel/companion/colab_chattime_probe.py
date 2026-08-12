# ============================================================================================
# ChatTime zero-shot context-guided ③ probe — paste into ONE Colab cell.
#   Runtime → L4 / A100 (7B fp16 ~14GB); a T4 needs 4-bit. Upload `_chattime_export.json`.
#
# WHY ChatTime (vs the Qwen fine-tune probe): ChatTime (AAAI'25) is a text+series model whose
# predict(history, context) IS context-guided forecasting — it was PURPOSE-BUILT to read the text,
# so it can (hopefully) separate correct vs shuffled content, i.e. answer the B<C question that a
# general LLM left borderline. Two bonuses: (1) NO fine-tuning → no training confound, no seed
# variance (zero-shot); (2) its base is LLaMA-2-7B (pretrain cutoff ~2022-23, BEFORE most of our
# 2024+ test events) → materially weaker lookahead than Qwen2.5 (~2024).
#
# Arms (only the context differs; same history, same eval events):
#   A no-text  = predict(history, "")          B flywheel = predict(history, cause)
#   C shuffled = predict(history, wrong cause)
# Success = B < A (text helps) AND B < C (the CONTENT / correct pairing helps).
#
# HONEST ADAPTER NOTE: this was not run locally. The ChatTime load/predict below follows the repo's
# documented API — ChatTime(hist_len, pred_len, model_path).predict(history, context). VERIFY the
# import path and the predict() output shape against github.com/ForestsKing/ChatTime on the first
# few events (the loop prints per-event numbers; if they don't look like MASE values, fix the parse).
# ============================================================================================
import json, re, statistics as st, random

get_ipython().system('[ -d ChatTime ] || git clone -q https://github.com/ForestsKing/ChatTime.git')  # noqa
# ChatTime's requirements.txt pins torch==2.7.1+cu118, which Colab's pip cannot resolve — and Colab
# already ships a working torch. So drop every torch* line and install only the remaining deps.
get_ipython().system('grep -viE "^torch" ChatTime/requirements.txt > /tmp/req.txt || true')  # noqa
get_ipython().system('pip -q install -r /tmp/req.txt')                            # noqa
import sys
sys.path.append("ChatTime")

# --- 1. data ---------------------------------------------------------------------------------
try:
    DATA = json.load(open("_chattime_export.json"))
except FileNotFoundError:
    from google.colab import files
    DATA = json.load(open(next(iter(files.upload()))))
EV, H, F = DATA["events"], DATA["hist_len"], DATA["pred_len"]
print(f"{len(EV)} events · {H}->{F} · leakage_ok={DATA['leakage_ok']}")

# --- 2. model (verify import path against the repo README) -----------------------------------
import numpy as np
import transformers
transformers.logging.set_verbosity_error()             # silence the max_new_tokens warning flood
from model.model import ChatTime                        # repo module path — adjust if README differs
model = ChatTime(hist_len=H, pred_len=F, model_path="ChengsenWang/ChatTime-1-7B-Chat")

_FAILS = [0]


def predict(hist, context):
    # ChatTime.predict can raise internally when a generation is unparseable/empty — catch it and
    # fall back to last-value persistence so one bad generation doesn't kill the whole run.
    try:
        out = model.predict(np.asarray(hist, dtype=float), context or "")   # "" = no-context (arm A)
        v = [float(x) for x in np.asarray(out).ravel()][:F]
    except Exception:
        v = []
    if not v:
        _FAILS[0] += 1
    return (v + [hist[-1]] * F)[:F]                                     # pad short/garbled/failed


# --- 3. metric -------------------------------------------------------------------------------
def mase(pred, actual, hist):
    scale = st.mean(abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))) or 1e-6
    n = min(len(pred), len(actual))
    return st.mean(abs(pred[i] - actual[i]) for i in range(n)) / scale


# --- 4. three arms (zero-shot; context is the only difference) -------------------------------
rows = []
for k, e in enumerate(EV):
    h, f = e["history"], e["future"]
    r = {"A": mase(predict(h, ""), f, h),
         "B": mase(predict(h, e["text"]), f, h),
         "C": mase(predict(h, e["text_shuffled"]), f, h),
         "dataset": e["dataset"]}
    rows.append(r)
    print(f"  [{k+1}/{len(EV)}] {e['dataset']:20} A={r['A']:.2f} B={r['B']:.2f} C={r['C']:.2f}",
          flush=True)

# --- 5. bootstrap CI on per-event deltas -----------------------------------------------------
N = len(rows)
dBA = [x["B"] - x["A"] for x in rows]
dBC = [x["B"] - x["C"] for x in rows]


def boot(d, it=3000):
    obs = st.median(d)
    m = [st.median([d[random.randrange(N)] for _ in range(N)]) for _ in range(it)]
    m.sort()
    return obs, m[int(.025 * it)], m[int(.975 * it)]


random.seed(0)
mba, lba, hba = boot(dBA)
mbc, lbc, hbc = boot(dBC)
med = lambda k: st.median(x[k] for x in rows)
print(f"\n=== ChatTime zero-shot ③ probe · ChatTime-1-7B-Chat · n={N} ===")
print(f"  A no-text {med('A'):.3f}   B flywheel {med('B'):.3f}   C shuffled {med('C'):.3f}  (median MASE)")
print(f"  Δ(B−A) {mba:+.3f}  95% CI [{lba:+.3f}, {hba:+.3f}]  {'SIGNIFICANT ✅' if hba < 0 else 'ns'}")
print(f"  Δ(B−C) {mbc:+.3f}  95% CI [{lbc:+.3f}, {hbc:+.3f}]  {'SIGNIFICANT ✅' if hbc < 0 else 'ns'}")
print(f"  per-event  B<A {sum(x < 0 for x in dBA)}/{N} · B<C {sum(x < 0 for x in dBC)}/{N}")
print(f"  fallbacks (unparseable ChatTime gen, padded to last value): {_FAILS[0]}/{3*N}"
      + ("  <- HIGH, results unreliable" if _FAILS[0] > 0.15 * 3 * N else ""))
print("  per-dataset (median MASE A → B):")
for ds in sorted({x["dataset"] for x in rows}):
    a = st.median(x["A"] for x in rows if x["dataset"] == ds)
    b = st.median(x["B"] for x in rows if x["dataset"] == ds)
    n = sum(x["dataset"] == ds for x in rows)
    print(f"    {ds:22} n={n:<3} A={a:6.2f} → B={b:6.2f}  ({'helps' if b < a else 'no'})")
print("\n  ChatTime is purpose-built for context + a LLaMA-2 base (earlier cutoff) — a cleaner ③ test "
      "than a general LLM. If B<C is significant here, the flywheel CONTENT is validated zero-shot.")
