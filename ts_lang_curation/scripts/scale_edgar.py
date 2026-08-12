#!/usr/bin/env python3
"""Background scale-up for sec_edgar: pull more companies' XBRL revenue/net income, then build.
Run: micromamba run -n ts-language python scripts/scale_edgar.py [N_to_add]
Uses the SEC ticker universe; keeps companies with a clean >=4yr annual revenue ending >=2023."""
import subprocess, json, os, sys
from datetime import date

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw", "sec")
UA = "ts-language research (majiaju89@gmail.com)"
TARGET_ADD = int(sys.argv[1]) if len(sys.argv) > 1 else 200
REV = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
       "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"]
NI = ["NetIncomeLoss", "ProfitLoss"]


def curl(url, out, minb=200, force=False):
    # force=True: never reuse an existing file (the shared temp must always re-download,
    # otherwise a stale _try_bg.json is read back as another company's/tag's data — the bug
    # that duplicated one company's net income into ~179 revenue files).
    if not force and os.path.exists(out) and os.path.getsize(out) > minb:
        return True
    r = subprocess.run(["curl", "-s", "--max-time", "45", "-H", "User-Agent: " + UA, url, "-o", out])
    return r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > minb


def days(a, b):
    y1, m1, d1 = map(int, a.split("-")); y2, m2, d2 = map(int, b.split("-"))
    return (date(y2, m2, d2) - date(y1, m1, d1)).days


def annual(p):
    try:
        d = json.load(open(p))
    except Exception:
        return 0, 0
    if "USD" not in d.get("units", {}):
        return 0, 0
    fy = {x["end"]: 1 for x in d["units"]["USD"]
          if x.get("form") == "10-K" and x.get("fp") == "FY" and x.get("start") and x.get("end")
          and 350 <= days(x["start"], x["end"]) <= 380}
    if not fy:
        return 0, 0
    y = sorted(fy); return len(y), int(y[-1][:4])


def best(cik, tags, out):
    if os.path.exists(out) and os.path.getsize(out) > 200:
        return annual(out)
    bk, bd = (-1, -1), None
    for t in tags:
        tmp = os.path.join(RAW, "_try_bg.json")
        if curl(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{t}.json", tmp, force=True):
            # sanity: the response's cik must match the company we asked for, else skip (defensive)
            try:
                if str(json.load(open(tmp)).get("cik", "")).zfill(10) != str(cik).zfill(10):
                    continue
            except Exception:
                continue
            n, yr = annual(tmp)
            if (n, yr) > bk and n >= 4:
                bk, bd = (n, yr), open(tmp).read()
    if bd:
        open(out, "w").write(bd); return bk
    return (0, 0)


def main():
    tk2cik = {v["ticker"]: str(v["cik_str"]).zfill(10)
              for v in json.load(open(os.path.join(RAW, "_tickers.json"))).values()}
    cik = json.load(open(os.path.join(RAW, "_cik.json")))
    added = 0
    for tk, c in tk2cik.items():
        if added >= TARGET_ADD:
            break
        if tk in cik:
            continue
        n, yr = best(c, REV, os.path.join(RAW, f"{tk}_revenue.json"))
        if n >= 4 and yr >= 2023:
            best(c, NI, os.path.join(RAW, f"{tk}_netincome.json"))
            cik[tk] = c; added += 1
            if added % 25 == 0:
                json.dump(cik, open(os.path.join(RAW, "_cik.json"), "w"))
                print(f"  ...{added} added", flush=True)
    json.dump(cik, open(os.path.join(RAW, "_cik.json"), "w"))
    print(f"fetched {added} new companies; total {len(cik)}. Building...", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "build.py"), "--source", "sec_edgar"], cwd=HERE)


if __name__ == "__main__":
    main()
