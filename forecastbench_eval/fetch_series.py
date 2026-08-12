#!/usr/bin/env python3
"""Pull the underlying FRED + Yahoo series histories UP TO the forecast_due_date (2024-07-21) — the
input a TS model needs to forecast each ForecastBench dataset question. Leakage-safe: nothing after the
due date is fetched. Pure urllib, so no torch/pandas/yfinance is needed locally; Chronos runs on the box.

  micromamba run -n ts-language python forecastbench_eval/fetch_series.py
"""
import os, json, csv, io, time, urllib.request, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench_fred_yahoo.json")
OUT = os.path.join(HERE, "series.json")
DUE = "2024-07-21"
START = "2014-01-01"
UA = "ts-language research (majiaju89@gmail.com)"


def _fred(sid):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={START}&coed={DUE}"
    txt = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}),
                                 timeout=45).read().decode("utf-8", "ignore")
    dates, vals = [], []
    for r in csv.reader(io.StringIO(txt)):
        if not r or not r[0][:4].isdigit() or len(r) < 2 or r[1] in (".", ""):
            continue
        dates.append(r[0]); vals.append(float(r[1]))
    return dates, vals


def _yahoo(tk):
    p1 = int(dt.datetime(2014, 1, 1).timestamp())
    p2 = int(dt.datetime(2024, 7, 22).timestamp())                 # exclusive-ish upper bound = due date
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{tk}"
           f"?period1={p1}&period2={p2}&interval=1d")
    j = json.loads(urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=45).read())
    res = j["chart"]["result"][0]
    ts, close = res["timestamp"], res["indicators"]["quote"][0]["close"]
    dv = [(dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"), v)
          for t, v in zip(ts, close) if v is not None and dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d") <= DUE]
    return [d for d, _ in dv], [v for _, v in dv]


def main():
    bench = json.load(open(BENCH))
    out, ok, fail = {}, 0, 0
    for q in bench["questions"]:
        sid, src = q["id"], q["source"]
        try:
            dates, vals = _fred(sid) if src == "fred" else _yahoo(sid)
            if len(vals) < 30:
                raise ValueError(f"too few points ({len(vals)})")
            out[sid] = {"source": src, "dates": dates, "values": vals, "freeze": q["freeze_value"]}
            ok += 1
            print(f"  {src:8} {sid:12} {len(vals):5} pts · last={vals[-1]:.3f} (freeze={q['freeze_value']})")
        except Exception as e:
            fail += 1
            print(f"  FAIL {src} {sid}: {repr(e)[:60]}")
        time.sleep(0.25)
    json.dump(out, open(OUT, "w"))
    print(f"\n{ok} series ok · {fail} failed -> {OUT}")


if __name__ == "__main__":
    main()
