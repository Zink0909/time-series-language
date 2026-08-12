import json, os, sys, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import scale_edgar as SE
RAW = SE.RAW
tk2cik = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in json.load(open(os.path.join(RAW,"_tickers.json"))).values()}
cik_map = json.load(open(os.path.join(RAW,"_cik.json")))
tickers = sorted(os.path.basename(p).replace("_revenue.json","") for p in glob.glob(os.path.join(RAW,"*_revenue.json")))
print(f"re-pulling {len(tickers)} SEC tickers with fixed fetch...", flush=True)
ok=bad=0; new_cik={}
for i,tk in enumerate(tickers):
    c = tk2cik.get(tk) or cik_map.get(tk)
    if not c: bad+=1; continue
    rp=os.path.join(RAW,f"{tk}_revenue.json"); np_=os.path.join(RAW,f"{tk}_netincome.json")
    for p in (rp,np_):
        if os.path.exists(p): os.remove(p)
    n,yr = SE.best(c, SE.REV, rp)
    if n>=4:
        SE.best(c, SE.NI, np_); new_cik[tk]=c; ok+=1
    else:
        if os.path.exists(rp): os.remove(rp)
    time.sleep(0.08)
    if i%50==0: print(f"  {i}/{len(tickers)} ok={ok}", flush=True)
# clean shared temp
t=os.path.join(RAW,"_try_bg.json")
if os.path.exists(t): os.remove(t)
json.dump(new_cik, open(os.path.join(RAW,"_cik.json"),"w"))
print(f"DONE re-pull: {ok} ok / {bad} no-cik / {len(tickers)} total", flush=True)
