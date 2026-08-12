#!/usr/bin/env python3
"""Pull the ForecastBench 2024-07-21 round (the only round with baseline forecasts) and extract the
FRED + Yahoo dataset questions — the time-series-driven subset a pure TS model (Chronos) CAN answer.

Xinyue asked how pure TS models do on ForecastBench. Its dataset questions are auto-generated from real
numeric series (FRED macro, Yahoo stock close, Wikipedia, ACLED, DBnomics) with the template "will the
value be higher at {resolution_date} than at {forecast_due_date}?". FRED and Yahoo are the cleanest: the
question `id` IS the series id / ticker and the rule is "future > current". A TS model can forecast the
underlying series and read P(future > threshold) directly.

This script downloads (a) the question set, (b) the resolution set with the ground-truth 0/1, and
(c) the human forecasts (super + public — the only baselines on GitHub; LLM forecasts live in the big
tarball / leaderboard). It keeps only FRED + Yahoo questions and joins them to their resolutions and
human forecasts, writing a single self-contained file the evaluator consumes.

  micromamba run -n ts-language python forecastbench_eval/fetch_bench.py
"""
import os, json, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"
ROUND = "2024-07-21"
SOURCES = ("fred", "yfinance")                       # cleanest TS-driven subset for the MVP


def _get(url):
    cache = os.path.join(RAW, url.split("/")[-1])
    if os.path.exists(cache):
        return json.load(open(cache))
    req = urllib.request.Request(url, headers={"User-Agent": "ts-language research"})
    data = json.loads(urllib.request.urlopen(req, timeout=60).read())
    json.dump(data, open(cache, "w"))
    return data


def main():
    os.makedirs(RAW, exist_ok=True)
    qset = _get(f"{BASE}/question_sets/{ROUND}-human.json")   # human round = matches the human baselines
    rset = _get(f"{BASE}/resolution_sets/{ROUND}_resolution_set.json")
    human_super = _get(f"{BASE}/forecast_sets/{ROUND}/{ROUND}.ForecastBench.human_super_individual.json")
    try:                                                             # public file may be LFS-gated on raw;
        human_public = _get(f"{BASE}/forecast_sets/{ROUND}/{ROUND}.ForecastBench.human_public_individual.json")
    except Exception:                                               # superforecasters are the stronger baseline anyway
        human_public = {}

    due = qset["forecast_due_date"]
    questions = [q for q in qset["questions"] if q.get("source") in SOURCES]

    # resolutions keyed on (id, resolution_date) -> resolved_to (0/1)
    res = {}
    for r in rset["resolutions"]:
        rid, rd = r.get("id"), r.get("resolution_date")
        if (r.get("source") in SOURCES and r.get("resolved")
                and isinstance(rid, str) and isinstance(rd, str)):     # skip combination questions (list ids)
            res[(rid, rd)] = r["resolved_to"]

    # human forecasts: average the individual forecasts per (id, resolution_date) into a crowd forecast
    def crowd(hf):
        agg = {}
        for f in hf.get("forecasts", hf if isinstance(hf, list) else []):
            fid, frd = f.get("id"), f.get("resolution_date")
            if (f.get("source") in SOURCES and isinstance(f.get("forecast"), (int, float))
                    and isinstance(fid, str) and isinstance(frd, str)):
                agg.setdefault((fid, frd), []).append(f["forecast"])
        return {k: sum(v) / len(v) for k, v in agg.items()}

    hs, hp = crowd(human_super), crowd(human_public)

    out = {"round": ROUND, "forecast_due_date": due, "sources": list(SOURCES),
           "n_questions": len(questions), "questions": []}
    n_res = 0
    for q in questions:
        rows = []
        for rd in q["resolution_dates"]:
            key = (q["id"], rd)
            if key in res:
                rows.append({"resolution_date": rd, "resolved_to": res[key],
                             "human_super": hs.get(key), "human_public": hp.get(key)})
                n_res += 1
        if rows:
            out["questions"].append({
                "id": q["id"], "source": q["source"], "question": q["question"],
                "freeze_value": q["freeze_datetime_value"],
                "freeze_explanation": q["freeze_datetime_value_explanation"],
                "resolution_rows": rows})

    path = os.path.join(HERE, "bench_fred_yahoo.json")
    json.dump(out, open(path, "w"), indent=1)
    fred = sum(1 for q in out["questions"] if q["source"] == "fred")
    yf = sum(1 for q in out["questions"] if q["source"] == "yfinance")
    print(f"forecast_due_date={due}")
    print(f"FRED questions {fred} · Yahoo questions {yf} · with-resolution {len(out['questions'])}")
    print(f"resolved (id,date) rows: {n_res} -> {path}")
    if out["questions"]:
        q = out["questions"][0]
        print(f"sample [{q['source']}] {q['id']}: freeze={q['freeze_value']} · rows={len(q['resolution_rows'])}")
        print(f"  {q['resolution_rows'][0]}")


if __name__ == "__main__":
    main()
