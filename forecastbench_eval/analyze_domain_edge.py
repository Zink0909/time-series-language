#!/usr/bin/env python3
"""7/28 会议 §一 两点深挖（离线，纯缓存，无 GPU）：
  #1  分 domain 拆 + lookback 是否非单调 + 谁主导 >1季 gap
  #2  edge case：最强方法失败的 bad case，判断"信息丢失(不可控)" vs "模型弱(可指导)"
数据全部来自已缓存的 Chronos 预测 + 官方 resolution + 人类基线。
运行：micromamba run -n ts-language python analyze_domain_edge.py
"""
import json, statistics as st, os
HERE=os.path.dirname(os.path.abspath(__file__))
def L(f): return json.load(open(os.path.join(HERE,f)))

main=L("ts_forecast.json")                                  # 512 满历史主跑
ctx={c:L(f"ts_forecast_ctx{c}.json") for c in [32,64,128,256]}; ctx[512]=main
def brier(p,y): return (p-y)**2
def dom(r): return "fred" if r["source"]=="fred" else "yfinance"
LAB={5:"≤1周",21:"≤1月",65:"≤1季",129:">1季·2季",260:">1季·1年"}
HS=[5,21,65,129,260]   # 交易日；horizon 为离散值

def med_brier(rows,key="p_up"):
    return st.median(brier(r[key],r["resolved_to"]) for r in rows)

def run():
    print("=== #1a  512满历史 · 逐域 × 精确跨度  中位Brier (C/H) [n] ===")
    for h in HS:
        line=f"{h:>3d} {LAB[h]:<8}"
        for src in ["fred","yfinance","ALL"]:
            rs=[r for r in main if r["horizon"]==h and (src=="ALL" or dom(r)==src)]
            line+= f" {med_brier(rs):.3f}/{med_brier(rs,'human_super'):.3f}[{len(rs)}]".ljust(20) if rs else " --".ljust(20)
        print(line)

    print("\n=== #1b  历史长度 × 跨度 中位Brier(Chronos) —— lookback 是否非单调 ===")
    print("  ctx " + "".join(f"{LAB[h]:>10s}" for h in HS) + "     ALL")
    for c in [32,64,128,256,512]:
        row="".join(f"{med_brier([r for r in ctx[c] if r['horizon']==h]):10.3f}" for h in HS)
        print(f"  {c:4d}{row}   {med_brier(ctx[c]):.3f}")

    print("\n=== #1c  谁主导 >1季 gap (h≥129, 512满历史) ===")
    for src in ["fred","yfinance"]:
        rs=[r for r in main if r["horizon"]>=129 and dom(r)==src]
        print(f"  {src:9s} n={len(rs):3d}  medC={med_brier(rs):.3f}  medH={med_brier(rs,'human_super'):.3f}  Δ={med_brier(rs)-med_brier(rs,'human_super'):+.3f}")

    print("\n=== #2a  Chronos(512) 最差 10 点 ===")
    for r in sorted(main,key=lambda r:-brier(r["p_up"],r["resolved_to"]))[:10]:
        print(f"  {r['id']:<12}{r['source']:<9}h={r['horizon']:<4}p_up={r['p_up']:.2f} real={r['resolved_to']:.0f} C_Br={brier(r['p_up'],r['resolved_to']):.3f}  H={r['human_super']:.2f} H_Br={brier(r['human_super'],r['resolved_to']):.3f}")

    print("\n=== #2c  两者都差(C_Br>0.4 且 H_Br>0.4 = 信息丢失/不可控) ===")
    both=[r for r in main if brier(r["p_up"],r["resolved_to"])>0.4 and brier(r["human_super"],r["resolved_to"])>0.4]
    print(f"  共 {len(both)} 个   （人类最差 Brier = {max(brier(r['human_super'],r['resolved_to']) for r in main):.3f}）")

if __name__=="__main__":
    run()
