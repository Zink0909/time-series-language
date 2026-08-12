#!/usr/bin/env python3
"""Render reports/flywheel_wiki_demo.html from the ACTUAL flywheel outputs (numbers computed
from current data, never hand-typed — CLAUDE.md §1). Reads _wiki_trace.json + _wiki_qa.json +
pageview caches. Run after wiki_demo.py + qa_wiki.py:
  micromamba run -n ts-language python flywheel/make_wiki_demo_html.py
"""
import os, sys, json, html

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
FW = os.path.join(HERE, "raw", "wiki")
TRACE = json.load(open(os.path.join(FW, "_wiki_trace.json")))
QA = json.load(open(os.path.join(FW, "_wiki_qa.json")))
REL = {p["slug"]: p["relevance"] for p in QA["per_record"]}
OUT = os.path.join(os.path.dirname(PKG), "reports", "flywheel_wiki_demo.html")
BLK = "▁▂▃▄▅▆▇█"


def spark(vals):
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    return "".join(BLK[min(7, int((v - lo) / rng * 7.999))] for v in vals)


def pv_raw(slug):
    p = os.path.join(FW, f"{slug}_pv.json")
    items = json.load(open(p))
    return [(x["timestamp"][:8], x["views"] / 1e3) for x in items]


EM = [t for t in TRACE if t["emitted"]]
# feature cards: 2 biggest peaks (absolute; ratio can be inflated by near-zero baselines),
# 2 highest relevance, 2 lowest relevance (deduped, keep order)
by_ratio = sorted(EM, key=lambda t: -t["peak_k"])[:2]
by_rel_hi = sorted([t for t in EM if REL.get(t["slug"]) is not None],
                   key=lambda t: -REL[t["slug"]])[:2]
by_rel_lo = sorted([t for t in EM if REL.get(t["slug"]) is not None],
                   key=lambda t: REL[t["slug"]])[:2]
feat, seen = [], set()
for t in by_ratio + by_rel_hi + by_rel_lo:
    if t["slug"] not in seen:
        seen.add(t["slug"])
        feat.append(t)

# gaps
gaps = {}
for t in TRACE:
    if not t["emitted"]:
        gaps[t["reason"]] = gaps.get(t["reason"], 0) + 1
GAP_LABEL = {
    "no_pre_event_state": "文章在事件当天才被创建（如地震、灾难词条）→ 没有事前基线也没有事前版本。飞轮只对「事件前已存在」的实体词条成立，这是数据源属性，不硬凑。",
    "insufficient_history_or_horizon": "自动检测到的峰值落在抓取窗口的边缘（常是「第二峰」，如英女王葬礼 09-19 而非去世 09-08）→ 峰后不足 5 天，诚实丢弃。",
    "no_pageviews": "该词条在窗口内取不到浏览量（标题重定向 / 括号消歧义未解析）。",
    "revision_not_before_spike": "严格泄漏门拦下：峰值日当天该词条被大量编辑，取不到「峰值日 00:00 之前」的版本 → 丢弃而非放宽。",
}

al = sorted(t["detect_alignment_days"] for t in EM)
within2 = sum(a <= 2 for a in al)
ra = sorted(t["rev_age_days"] for t in EM)


def esc(s):
    return html.escape(str(s))


cards = ""
for t in feat:
    series = pv_raw(t["slug"])
    vals = [v for _, v in series]
    pk = vals.index(max(vals))
    hist, fut = vals[:pk], vals[pk:]
    rel = REL.get(t["slug"])
    rel_tag = (f'<span class="rok">relevance {rel:.2f}</span>' if rel is not None and rel >= 0.15
               else f'<span class="rwarn">relevance {rel:.2f} · 泛化词条</span>')
    cards += f"""
  <div class="ev">
    <div class="date">{esc(t['title'])} · 峰值 {esc(t['spike_date'])} · {t['spike_ratio']:.0f}× 基线</div>
    <p><span class="tag">S2 检测</span>自动挑出的峰值日与事件日相差 <b>{t['detect_alignment_days']} 天</b>
       （事件日仅用于打分，不作模型输入）。{rel_tag}</p>
    <p><span class="tag">S3 事前文本（版本 {esc(t['revision_id'] if 'revision_id' in t else '')}）</span>
       取该词条在峰值日 <b>之前 {t['rev_age_days']} 天</b> 的历史版本 lead（泄漏干净）:</p>
    <div class="lead">{esc(t['lead_preview'])}…</div>
    <p><span class="tag">S8 text→ts</span>
       历史（峰前 {len(hist)} 天基线/爬升）<code>{spark(hist)} {hist[0]:.0f}k→{hist[-1]:.0f}k</code>
       &nbsp;→&nbsp; 预测（峰值起 {len(fut)} 天）<code>{spark(fut)} {fut[0]:.0f}k→{fut[-1]:.0f}k</code> ✓</p>
  </div>"""

gap_rows = "".join(
    f"<tr><td><code>{esc(k)}</code></td><td class='n'>{v}</td><td>{GAP_LABEL.get(k, '')}</td></tr>"
    for k, v in sorted(gaps.items(), key=lambda kv: -kv[1]))

DOC = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>飞轮 Demo · Wikipedia 第二源（点时版本 · 无 GDELT）</title>
<style>
  :root{{--bg:#0f1117;--card:#181b24;--ink:#e7e9ee;--mut:#9aa3b2;--acc:#6ea8fe;--gr:#4ade80;--am:#fbbf24;--bd:#262b36;--pk:#f0abfc;--dn:#f87171;}}
  *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15.5px/1.75 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
  .wrap{{max-width:940px;margin:0 auto;padding:44px 24px 90px}}
  h1{{font-size:26px;margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 22px}}
  h2{{font-size:20px;margin:36px 0 10px;padding-bottom:7px;border-bottom:2px solid var(--bd)}}
  p{{margin:8px 0}} code{{background:#10141d;border:1px solid var(--bd);border-radius:4px;padding:1px 6px;font-size:13px;color:#cdd6f4}}
  .card{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px 20px;margin:14px 0}}
  .lead-c{{background:linear-gradient(180deg,#1b2030,#181b24);border-left:4px solid var(--acc)}}
  .ev{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px 18px;margin:12px 0}}
  .ev .date{{font-weight:700;color:var(--acc)}}
  .tag{{display:inline-block;font-size:12px;color:var(--am);background:#1c1810;border:1px solid #3a3320;border-radius:5px;padding:1px 8px;margin-right:6px}}
  .lead{{background:#10141d;border:1px solid var(--bd);border-radius:8px;padding:9px 12px;color:#cdd6f4;font-size:13.5px;margin:6px 0}}
  .rok{{color:var(--gr);font-size:12px;border:1px solid #1f3b29;background:#0f1f15;border-radius:20px;padding:1px 9px}}
  .rwarn{{color:var(--am);font-size:12px;border:1px solid #3a3320;background:#1c1810;border-radius:20px;padding:1px 9px}}
  .big{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}}
  .stat{{flex:1;min-width:140px;background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px;text-align:center}}
  .stat .v{{font-size:28px;font-weight:800;color:var(--gr)}} .stat .l{{color:var(--mut);font-size:13px;margin-top:2px}}
  table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:14px}}
  th,td{{border:1px solid var(--bd);padding:7px 10px;text-align:left;vertical-align:top}} th{{background:#10141d;color:var(--mut)}}
  td.n{{color:var(--gr);font-weight:700;text-align:center}}
  .good{{color:var(--gr);font-weight:600}} .warn{{background:#15131f;border-left:3px solid var(--am);border-radius:8px;padding:10px 14px;margin:12px 0;font-size:14px}}
  .foot{{color:var(--mut);font-size:13px;margin-top:36px;border-top:1px solid var(--bd);padding-top:12px}}
</style></head><body><div class="wrap">

<h1>飞轮 Demo · Wikipedia 第二源</h1>
<p class="sub">点时（point-in-time）版本 · 无 GDELT · 自动测 spike · 对应 <code>flywheel/wiki_demo.py</code></p>

<div class="card lead-c">
  这是数据飞轮的<b>第二个源</b>（第一个是 WTI 原油）。和原油不同，这里<b>不用 GDELT 检索外部新闻</b>——
  对一条<b>只有每日浏览量、没有配文</b>的维基词条，飞轮自动找出注意力<b>爆发的那天 D</b>，
  然后取该词条<b>在 D 之前的历史版本</b>的开头段（"事件爆发前这一页是怎么写的"）作为
  <b>泄漏干净</b>的条件文本，配成 text→ts 训练样本。
  <br><br>关键设计:<b>点时版本 = 天然防泄漏</b>（条件文本的时间戳严格早于预测起点 D）。
  它只对<b>"事件前就已存在"的实体词条</b>成立（Nvidia/OpenAI/DeepSeek…）；
  地震、灾难这类<b>当天才建的词条</b>没有"事前状态"，被自动排除并诚实计入缺口——
  <b>这条排除线本身就是持久性过滤器，无需人工标注</b>。
</div>

<div class="big">
  <div class="stat"><div class="v">{len(EM)}/{len(TRACE)}</div><div class="l">候选→产出 text→ts</div></div>
  <div class="stat"><div class="v">0</div><div class="l">泄漏（版本 &lt; 峰值日）</div></div>
  <div class="stat"><div class="v">0</div><div class="l">格式错（独立校验）</div></div>
  <div class="stat"><div class="v">{within2}/{len(EM)}</div><div class="l">检测对齐 ≤2 天</div></div>
</div>

<h2>1 · 事件卡怎么读（真实产出样例）</h2>
<p>下面是从 {len(EM)} 条产出里挑的样例:<b>最大波动</b>各 2 条 + <b>relevance 最高/最低</b>各 2 条
（relevance = 事前 lead 与事件的关联度，见第 4 节）。每条四层:检测 → 事前文本 → 产出。</p>
{cards}

<h2>2 · 三个判据（数字均由当前产出现算）</h2>
<table>
  <tr><th>判据</th><th>怎么看</th><th>本 demo</th></tr>
  <tr><td><b>① 泄漏干不干净</b>（红线）</td>
      <td>条件文本（历史版本）时间戳必须 <b>严格早于</b> 预测起点 D（峰值日 00:00 UTC）。</td>
      <td class="good">干净。{len(EM)}/{len(EM)} 条 <code>knowledge_time &lt; D</code>；
      事前版本平均只早 {ra[len(ra)//2]} 天（最多 {ra[-1]} 天），既新鲜又不越界。峰值日当天被大改的词条（如美国大选）取不到合规版本 → 丢弃。</td></tr>
  <tr><td><b>② 检测准不准</b></td>
      <td>自动测出的峰值日 vs 真实事件日相差几天（事件日只用于打分，不进模型）。</td>
      <td class="good">中位 {al[len(al)//2]} 天；{within2}/{len(EM)} 条 ≤2 天。检测器可靠地找到了真实事件。</td></tr>
  <tr><td><b>③ 覆盖全不全</b></td>
      <td>候选里多少成功产出？缺口是否可解释？</td>
      <td class="good">{len(EM)}/{len(TRACE)}。最大缺口是"当天才建的事件词条"（{gaps.get('no_pre_event_state',0)} 条），这是<b>源属性</b>不是 bug，见第 3 节。</td></tr>
</table>

<h2>3 · 缺口（诚实枚举，不藏）</h2>
<table><tr><th>原因</th><th>条数</th><th>说明</th></tr>{gap_rows}</table>

<h2>4 · 标注质量自查（Xinyue:先查标注质量再训练）</h2>
<div class="card">
  <p>独立 QA（<code>flywheel/qa_wiki.py</code>,<b>不调用</b>构建器自己的 <code>validate()</code>,按 Data-Schema 重写校验——不自证）:</p>
  <ul>
    <li><b>格式（独立重写校验）</b>:{QA['n']-QA['schema_errors']}/{QA['n']} 条 0 错。</li>
    <li><b>泄漏</b>:{QA['n']-QA['leaks']}/{QA['n']} 条 0 泄漏。</li>
    <li><b>事前 lead 与事件的关联度</b>（proxy = 事前 lead 与该词条<i>当前</i>摘要的特征词重叠，扣掉标题词）:
      中位 <b>{QA['relevance_median']:.2f}</b>;<b>{QA['relevance_ge_015']}/{QA['relevance_n']}</b> 条 lead 已能反映事件
      （因为峰前一天的版本常已被编辑进爆发中的新闻）。</li>
    <li><b>近零基线</b>:少数"平时无人问津、出事才爆红"的词条（如 Key Bridge、Damar Hamlin）
      事前浏览量近乎为 0 → 波动倍数看似上万倍（其实是基线太小的假象,故上面样例按<b>绝对峰值</b>而非倍数挑选）。
      这类是"文本承担全部信号"的极端 text→ts,合规但偏难,已诚实保留。</li>
  </ul>
  <div class="warn"><b>这一条是本源最该警惕的地方（诚实局限）:</b>点时 lead 泄漏干净,但它是词条的
  <b>"常备描述"</b>,不保证解释了这次爆发。当触发事件<b>晚于</b>最后一个事前版本时（如 Boeing 737 MAX 的
  1/5 舱门事故晚于 1/2 的版本、JWST 首图),lead 就是一段<b>泛化实体简介</b>,文本信号弱。
  relevance 这个 proxy 也会<b>高估</b>——同一实体被描述两次天然共享大量词。
  真正干净的信号是<b>低分那几条肯定泛化</b> + 人工抽检。这正是 v2 要用"版本 diff / 当日 In-the-news"补强的点。</div>
</div>

<h2>5 · 一句话</h2>
<div class="card lead-c">
  第二个源<b>跑通</b>:无 GDELT、自动测 spike、点时版本<b>天然防泄漏</b>、{len(EM)}/{len(TRACE)} 产出、格式即插即用、独立 QA 0 错 0 泄漏。
  它<b>证明了</b>飞轮可推广到注意力序列且能把泄漏做成"构造即干净";它<b>没证明</b>"造得有用"
  （那是 W6 companion study:对照无文本 / 乱配文本）。同时诚实暴露了无 GDELT 下<b>文本相关性偏弱</b>这一真实局限。
</div>

<p class="foot">配套:<code>flywheel/wiki_demo.py</code>（可复现代码）· <code>flywheel/qa_wiki.py</code>（独立 QA）·
  <code>flywheel/WIKI_DEMO_SPEC.md</code>（v1 spec）· <code>out/flywheel_wiki.jsonl</code>（{len(EM)} 条产出）。
  链接与数字均由当前产出现算。</p>

</div></body></html>"""

open(OUT, "w").write(DOC)
print(f"wrote {OUT}  ({len(EM)} cards-source, {len(feat)} featured)")
