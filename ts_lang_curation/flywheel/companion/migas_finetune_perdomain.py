#!/usr/bin/env python3
"""Fine-tune MiGAS-1.5's fusion on the flywheel 3-arm data; evaluate on held-out + Time-MMD.

RUNS ON THE MACHINE (needs: the Synthefy/synthefy-migas repo on PYTHONPATH, Chronos-2 + a Qwen text
embedder + a GPU). The data-loading, arm-construction and metric code below is pure-python and runs
anywhere; only build_model/train_arm/forecast touch MiGAS and are clearly marked.

Recipe (confirmed by reading src/migaseval/model/migas15.py):
  * Migas15 is a trainable nn.Module. The univariate base (Chronos-2, called via evaluate_chronos)
    and the text embedder (encode_texts under torch.no_grad) are FROZEN; the trainable parts are the
    projection MLPs (fact_embedder/prediction_embedder/timeseries_embedder/timeseries_decoder), the
    GatedAttentionFusion, the norms and the head.
  * forward(x, text, pred_len, history_mean, history_std, summaries=..., ...) ->
    (forecast[B,pred_len,1], ts_forecast, None). Pass `summaries` (fact/preds) to skip the LLM
    summarizer — Time-MMD already provides fact/preds; for flywheel causes we either pre-summarize or
    pass `text` and let MiGAS summarize.
  * Weights load via MigasPipeline.from_pretrained("Synthefy/migas-1.5"); `.model` is the Migas15.

Experiment (answers BOTH questions):
  (i)  Xinyue's "trained vs not": pretrained MiGAS (no fine-tune) vs MiGAS fine-tuned on OUR data.
  (ii) our content control: arm B (correct text) vs A (no text) vs C (shuffled text).
Metric = median MASE on the held-out flywheel test AND the Time-MMD generalization test.

  python flywheel/companion/migas_finetune.py --migas-dir /path/to/synthefy-migas [--epochs 5]
"""
import os, sys, json, argparse, statistics as st, random

HERE = os.path.dirname(os.path.abspath(__file__))
FINETUNE = os.path.join(HERE, "_finetune_export.json")     # train A/B/C + held-out test (our data)
TIMEMMD = os.path.join(HERE, "_timemmd_test.json")         # external generalization test (Time-MMD)
TIMESX = os.path.join(HERE, "_timesx_test.json")           # external generalization test (TimesX, larger)
GEN_SETS = {"timemmd": TIMEMMD, "timesx": TIMESX}


# ============================ pure-python: data + metric (runs anywhere) ============================
def _fix_shape(e, H=10, F=5):
    """Unify to a fixed H->F so a batch tensor is rectangular (our sources have varying windows)."""
    e = dict(e)
    e["history"] = e["history"][-H:]
    e["future"] = e["future"][:F]
    return e


def load_arms(path=None, H=10, F=5):
    d = json.load(open(path or FINETUNE))
    ok = lambda e: len(e["history"]) >= H and len(e["future"]) >= F
    train = {k: [_fix_shape(e, H, F) for e in v if ok(e)] for k, v in d["train"].items()}
    test = [_fix_shape(e, H, F) for e in d["test"] if ok(e)]
    return train, test


def load_gen(path, H=10, F=5):
    """Load an external generalization test set (Time-MMD or TimesX) — same {events:[...]} shape."""
    ev = json.load(open(path))["events"] if os.path.exists(path) else []
    return [_fix_shape(e, H, F) for e in ev if len(e["history"]) >= H and len(e["future"]) >= F]


def load_timemmd(H=10, F=5):
    return load_gen(TIMEMMD, H, F)


def mase(pred, actual, hist):
    scale = st.mean(abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))) or 1e-6
    n = min(len(pred), len(actual))
    return st.mean(abs(pred[i] - actual[i]) for i in range(n)) / scale


def summaries_of(ev):
    """Return (fact, preds) for an event. Time-MMD carries them; flywheel carries a single `text`
    which we put in `fact` (MiGAS will still embed it; a pre-summarize pass can split it later)."""
    if ev.get("fact") or ev.get("preds"):
        return ev.get("fact", ""), ev.get("preds", "")
    return ev.get("text", ""), ""


# ============================ MiGAS-dependent (RUNS ON THE MACHINE) ============================
def build_model(migas_dir, device="cuda:0", embedder_device="cpu"):
    # Build at the pretrained horizon (16; forecast_head + chronos are sized to it — we slice to our
    # F in _forward_batch). The 8B text embedder goes on CPU (the shared GPUs only have ~11 GB free;
    # the box has 2 TB RAM); Chronos + the fusion stack go on the GPU. Load weights manually (instead
    # of from_pretrained) so we can place the embedder on CPU.
    sys.path.insert(0, os.path.join(migas_dir, "src"))
    import torch                                                       # noqa
    from migaseval.model.migas15 import Migas15                        # noqa
    from huggingface_hub import hf_hub_download                        # noqa
    # The released checkpoint was trained with the FinBERT embedder (768-dim), NOT the default
    # qwen8b (4096-dim) — the fact/prediction projection shapes prove it. FinBERT is also small.
    model = Migas15(pred_len=16, device=device, chronos_device=device,
                    text_embedder_name="finbert", text_embedder_device=embedder_device)
    ckpt = torch.load(hf_hub_download("Synthefy/migas-1.5", "model.pt"), map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    model.to(device)
    model.eval()
    return model


def _trainable(model):
    """Only the fusion stack learns; the base forecaster + text embedder are frozen (the latter is
    already no_grad inside forward). Returns the parameter list for the optimizer."""
    import torch.nn as nn                                             # noqa
    names = ("fact_embedder", "prediction_embedder", "timeseries_embedder", "timeseries_decoder",
             "fusion", "ts_norm", "fact_norm", "pred_norm", "forecast_head", "convex_weight_net")
    params = []
    for n, p in model.named_parameters():
        p.requires_grad = any(n.startswith(k) or ("." + k + ".") in ("." + n) for k in names)
        if p.requires_grad:
            params.append(p)
    return params


def _forward_batch(model, batch, device):
    """One forward over a list of examples -> forecast tensor (B, F). Feeds raw history + summaries."""
    import torch                                                      # noqa
    B = len(batch)
    F = len(batch[0]["future"])
    x = torch.tensor([e["history"] for e in batch], dtype=torch.float32, device=device)
    hmean = torch.tensor([st.mean(e["history"]) for e in batch], dtype=torch.float32)
    hstd = torch.tensor([st.pstdev(e["history"]) or 1.0 for e in batch], dtype=torch.float32)
    fp = [summaries_of(e) for e in batch]
    summaries = [f"FACTUAL SUMMARY:\n{fa}\n\nPREDICTIVE SIGNALS:\n{pr}" for fa, pr in fp]
    text = [[s] for s in summaries]                                   # per-sample text (fallback if no summaries)
    forecast, ts_forecast, _ = model(x, text=text, pred_len=model.pred_len, history_mean=hmean,
                                     history_std=hstd, summaries=summaries)
    return forecast[..., 0][:, :F], ts_forecast[..., 0][:, :F]        # slice pretrained 16 -> our F


def train_arm(model, train_examples, arm, epochs, lr, bs, device, eval_fn=None):
    """Fine-tune the fusion on one arm. arm 'A_no_text' trains nothing extra (report the base); 'B'/'C'
    train the fusion on correct/shuffled summaries. eval_fn(model, epoch) runs after each epoch."""
    import torch                                                      # noqa
    if arm == "A_no_text":
        return model                                                 # A = frozen base, no text fusion
    params = _trainable(model)
    opt = torch.optim.AdamW(params, lr=lr)
    lossf = torch.nn.functional.l1_loss
    model.train()
    for ep in range(epochs):
        random.shuffle(train_examples)
        tot = 0.0
        for i in range(0, len(train_examples), bs):
            batch = train_examples[i:i + bs]
            y = torch.tensor([e["future"] for e in batch], dtype=torch.float32, device=device)
            mu = y.mean(1, keepdim=True)
            sd = y.std(1, keepdim=True).clamp(min=1e-6)
            fused, _ = _forward_batch(model, batch, device)
            loss = lossf(fused, (y - mu) / sd)                       # loss in normalized space
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"    [arm {arm}] epoch {ep+1}/{epochs} loss {tot/max(1,len(train_examples)//bs):.4f}",
              flush=True)
        if eval_fn:
            model.eval(); eval_fn(model, ep + 1); model.train()
    model.eval()
    return model


def forecast_mase(model, events, arm, device):
    """Evaluate on a list of events -> list of MASE. arm 'A_no_text' uses the unimodal base output."""
    import torch                                                      # noqa
    out = []
    with torch.no_grad():
        for i in range(0, len(events), 16):
            batch = events[i:i + 16]
            fused, base = _forward_batch(model, batch, device)
            use = base if arm == "A_no_text" else fused
            sd = torch.tensor([st.pstdev(e["future"]) or 1.0 for e in batch]).view(-1, 1)
            mu = torch.tensor([st.mean(e["future"]) for e in batch]).view(-1, 1)
            pred = (use.cpu() * sd + mu).tolist()                    # de-normalize
            for e, p in zip(batch, pred):
                out.append(mase(p, e["future"], e["history"]))
    return out


# ============================ orchestration ============================
def report(tag, ms):
    ms = sorted(ms)
    print(f"  {tag:28} median MASE {ms[len(ms)//2]:.3f}  (n={len(ms)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--migas-dir", required=True, help="path to a cloned Synthefy/synthefy-migas")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=1, help="repeat train/eval with N seeds -> bootstrap CI")
    ap.add_argument("--scan", action="store_true", help="per-epoch overfit diagnostic on B and C")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--embedder-device", default="cpu", help="FinBERT embedder device (cpu or cuda:0)")
    ap.add_argument("--finetune-file", default=None,
                    help="override the train/test export path (for the rich-vs-thin controlled runs)")
    ap.add_argument("--gen-test", choices=list(GEN_SETS), default="timemmd",
                    help="external generalization test set: timemmd (default) or timesx (larger)")
    ap.add_argument("--limit-tmmd", type=int, default=0, help="cap generalization eval events (0=all)")
    ap.add_argument("--limit-train", type=int, default=0, help="cap train examples per arm (0=all)")
    a = ap.parse_args()
    glabel = {"timemmd": "Time-MMD", "timesx": "TimesX"}[a.gen_test]

    train, test = load_arms(a.finetune_file)
    tmmd = load_gen(GEN_SETS[a.gen_test])
    if a.limit_tmmd:
        tmmd = tmmd[:a.limit_tmmd]
    F = len(test[0]["future"])
    print(f"train {len(train['B_flywheel'])}/arm · held-out {len(test)} · {glabel} {len(tmmd)} · F={F}")

    import torch

    if a.scan:                                                       # per-epoch overfit diagnostic (B & C)
        tmmd_s = tmmd[:500]
        torch.manual_seed(0)
        random.seed(0)
        for arm, key in [("B", "B_flywheel"), ("C", "C_shuffled")]:
            model = build_model(a.migas_dir, a.device, a.embedder_device)

            def ev(m, ep, _arm=arm):
                h = st.median(forecast_mase(m, test, _arm, a.device))
                t = st.median(forecast_mase(m, tmmd_s, _arm, a.device)) if tmmd_s else float("nan")
                print(f"  >> {_arm} epoch {ep}: held {h:.3f} · {glabel} {t:.3f}", flush=True)

            train_arm(model, list(train[key]), arm, a.epochs, a.lr, a.bs, a.device, eval_fn=ev)
            del model
        return

    def boot(deltas, iters=3000):
        n = len(deltas)
        ms = sorted(st.median([deltas[random.randrange(n)] for _ in range(n)]) for _ in range(iters))
        return st.median(deltas), ms[int(.025 * iters)], ms[int(.975 * iters)]

    arms3 = [("A_no_text", "A_no_text"), ("B", "B_flywheel"), ("C", "C_shuffled")]
    acc = {arm: {"held": [], "tmmd": []} for arm, _ in arms3}         # per arm/split: seeds x per-event MASE
    baseline = {"held": [], "tmmd": []}

    for s in range(a.seeds):
        torch.manual_seed(s)
        random.seed(s)
        if s == 0:                                                   # not-trained baseline (deterministic)
            base = build_model(a.migas_dir, a.device, a.embedder_device)
            baseline["held"] = forecast_mase(base, test, "B", a.device)
            baseline["tmmd"] = forecast_mase(base, tmmd, "B", a.device) if tmmd else []
            print(f"  [baseline] held {st.median(baseline['held']):.3f}"
                  + (f" · {glabel} {st.median(baseline['tmmd']):.3f}" if tmmd else ""), flush=True)
            del base
        for arm, key in arms3:
            model = build_model(a.migas_dir, a.device, a.embedder_device)
            ex = list(train[key])[:a.limit_train] if a.limit_train else list(train[key])
            model = train_arm(model, ex, arm, a.epochs, a.lr, a.bs, a.device)
            acc[arm]["held"].append(forecast_mase(model, test, arm, a.device))
            if tmmd:
                acc[arm]["tmmd"].append(forecast_mase(model, tmmd, arm, a.device))
            print(f"  [seed {s}] arm {arm} held {st.median(acc[arm]['held'][-1]):.3f}"
                  + (f" · {glabel} {st.median(acc[arm]['tmmd'][-1]):.3f}" if tmmd else ""), flush=True)
            del model

    def avg(arm, split):                                             # per-event mean across seeds
        S = len(acc[arm][split]); n = len(acc[arm][split][0])
        return [st.mean(acc[arm][split][k][i] for k in range(S)) for i in range(n)]

    for split, name in [("held", "held-out"), ("tmmd", f"{glabel} generalization")]:
        if not acc["B"][split]:
            continue
        A_, B_, C_ = avg("A_no_text", split), avg("B", split), avg("C", split)
        dBA = [B_[i] - A_[i] for i in range(len(B_))]
        dBC = [B_[i] - C_[i] for i in range(len(B_))]
        random.seed(0)
        mba, lba, hba = boot(dBA)
        mbc, lbc, hbc = boot(dBC)
        bl = st.median(baseline[split]) if baseline[split] else float("nan")
        print(f"\n=== {name} · seeds={a.seeds} · n={len(B_)} ===")
        print(f"  not-trained {bl:.3f} | A {st.median(A_):.3f} | B {st.median(B_):.3f} | C {st.median(C_):.3f}")
        print(f"  Δ(B−A) {mba:+.3f}  95% CI [{lba:+.3f}, {hba:+.3f}]  "
              f"{'SIG<0: text helps' if hba < 0 else 'ns'}")
        print(f"  Δ(B−C) {mbc:+.3f}  95% CI [{lbc:+.3f}, {hbc:+.3f}]  "
              f"{'SIG<0: content helps' if hbc < 0 else 'ns'}")
        print(f"  per-event B<A {sum(x < 0 for x in dBA)}/{len(dBA)} · B<C {sum(x < 0 for x in dBC)}/{len(dBC)}")
        if split == "tmmd":                                          # per-domain breakdown (Xinyue: is shuffled>paired a mix artifact?)
            from collections import defaultdict
            dom = defaultdict(lambda: {"a": [], "c": []})
            for i in range(len(B_)):
                dd = tmmd[i].get("domain", "?")
                dom[dd]["a"].append(dBA[i]); dom[dd]["c"].append(dBC[i])
            print(f"  --- per-domain ({glabel}); medΔ(B-A)<0 = text helps, medΔ(B-C)<0 = correct-pairing helps ---")
            for dd in sorted(dom, key=lambda k: -len(dom[k]['c'])):
                aa, cc = dom[dd]["a"], dom[dd]["c"]
                mba, mbc = st.median(aa), st.median(cc)
                print(f"  {dd[:32]:32} n={len(cc):5}  medD(B-A) {mba:+.3f}  medD(B-C) {mbc:+.3f}  "
                      f"B<C {sum(x<0 for x in cc):4}/{len(cc):<4}  "
                      f"{'TEXT-HELPS' if mba<0 else 'text-noop/hurts'}")


if __name__ == "__main__":
    main()
