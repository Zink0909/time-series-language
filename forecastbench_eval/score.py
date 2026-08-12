#!/usr/bin/env python3
"""Three-way comparison on ForecastBench FRED+Yahoo dataset questions (2024-07-21 round):
Chronos-2 (pure TS) vs superforecasters vs LLMs. All scored on the SAME 213 (id, resolution_date)
points that Chronos covers (horizon < ~2y). Metric = Brier (lower better); persistence(0.5) is the floor.

  micromamba run -n ts-language python forecastbench_eval/score.py
"""
import json, os, glob, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
fc = json.load(open(os.path.join(HERE, "ts_forecast.json")))
pts = {(o["id"], o["resolution_date"]): o for o in fc}
ALL = list(pts)
FRED = [k for k in ALL if pts[k]["source"] == "fred"]
YF = [k for k in ALL if pts[k]["source"] == "yfinance"]


def brier(preds, keys):
    v = [(preds[k] - pts[k]["resolved_to"]) ** 2 for k in keys if k in preds]
    return sum(v) / len(v) if v else float("nan")


chronos = {k: o["p_up"] for k, o in pts.items()}
sup = {k: o["human_super"] for k, o in pts.items() if o["human_super"] is not None}

# every file's `organization` is "ForecastBench" (the org that ran the round); the naive / human
# reference baselines are identified by their `model` NAME, so skip those and keep only real LLMs.
NAIVE = {"Always 0", "Always 0.5", "Always 1", "Imputed Forecaster", "Naive Forecaster",
         "Random Uniform", "Public median forecast", "Superforecaster median forecast"}

llm = []
for f in glob.glob(os.path.join(HERE, "raw", "llm", "*.json")):
    d = json.load(open(f))
    model = d.get("model", "")
    base = model.split(" (")[0]                                     # strip the "(prompt variant)" suffix
    if base in NAIVE:
        continue
    preds = {}
    for x in d.get("forecasts", []):
        if x.get("source") in ("fred", "yfinance") and isinstance(x.get("id"), str):
            try:
                preds[(x["id"], x["resolution_date"][:10])] = float(x["forecast"])
            except (ValueError, TypeError):
                continue
    if sum(1 for k in ALL if k in preds) >= 200:                    # only models covering the point set
        llm.append((model, brier(preds, ALL), brier(preds, FRED), brier(preds, YF)))
llm.sort(key=lambda t: t[1])


def row(name, preds):
    print(f"{name:30} ALL {brier(preds,ALL):.3f}   FRED {brier(preds,FRED):.3f}   Yahoo {brier(preds,YF):.3f}")


print(f"=== Brier (lower=better) · {len(ALL)} points · base rate {sum(pts[k]['resolved_to'] for k in ALL)/len(ALL):.2f} ===")
row("Chronos-2 (pure TS)", chronos)
row("Superforecasters (human)", sup)
if llm:
    b = llm[0]
    print(f"{'best LLM: '+b[0]:30} ALL {b[1]:.3f}   FRED {b[2]:.3f}   Yahoo {b[3]:.3f}")
    print(f"{'median of '+str(len(llm))+' LLM configs':30} ALL {st.median([t[1] for t in llm]):.3f}"
          f"   FRED {st.median([t[2] for t in llm]):.3f}   Yahoo {st.median([t[3] for t in llm]):.3f}")
print(f"{'persistence (0.5)':30} ALL 0.250   FRED 0.250   Yahoo 0.250")
print(f"\ntop 6 LLM configs (by ALL Brier):")
for m, b, bf, by in llm[:6]:
    print(f"  ALL {b:.3f}  FRED {bf:.3f}  Yahoo {by:.3f}   {m[:52]}")
