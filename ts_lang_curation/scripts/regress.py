#!/usr/bin/env python3
"""TS+Language 最低回归（CLAUDE.md §3：flywheel demo + 两个手工源 build，全部 0 invalid）。

  micromamba run -n ts-language python scripts/regress.py

离线运行（缓存优先），全过 exit 0，任一失败 exit 1。
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = []


def run(args):
    r = subprocess.run([sys.executable] + args, cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    return r.returncode, (r.stdout + r.stderr).strip()


def check(name, ok, detail):
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")


def main():
    print("== ts_lang_curation regress ==")

    for src in ["usgs_quakes", "treasury_fomc"]:
        code, out = run(["build.py", "--source", src])
        m = re.search(r"(\d+) valid records .*\((\d+) invalid, (\d+) unsupported", out)
        ok = code == 0 and m and int(m.group(2)) == 0 and int(m.group(3)) == 0 and int(m.group(1)) > 0
        check(f"build --source {src}", bool(ok),
              m.group(0) if m else f"exit {code}: {out.splitlines()[-1] if out else '(no output)'}")

    code, out = run(["flywheel/oil_demo.py", "--offline"])
    m = re.search(r"(\d+)/(\d+) events emitted", out)
    ok = code == 0 and m and int(m.group(1)) > 0            # forecast_leak may drop a soft-leak cause (7/8)
    check("flywheel oil_demo (cached replay)", bool(ok),
          out.splitlines()[-1] if out else f"exit {code}")

    code, out = run(["flywheel/wiki_demo.py"])
    m = re.search(r"(\d+) text->ts pairs .*\((\d+)/(\d+) candidates emitted", out)
    ok = code == 0 and m and int(m.group(1)) == int(m.group(2)) and int(m.group(2)) > 0
    check("flywheel wiki_demo (cached replay)", bool(ok),
          out.splitlines()[-1] if out else f"exit {code}")

    # scaled wiki flywheel (multi-peak): cached replay, deterministic. Leakage/schema covered by
    # the cross-source audit below (343 flywheel records, 0 leak) + build-time validate.
    code, out = run(["flywheel/wiki_scale.py"])
    m = re.search(r"(\d+) text->ts from", out)
    ok = code == 0 and m and int(m.group(1)) > 0
    check("flywheel wiki_scale (multi-peak cached replay)", bool(ok),
          out.splitlines()[-1] if out else f"exit {code}")

    # commodity flywheel (dense sources, BigQuery-retrieved news): frozen cached replay = no LLM,
    # deterministic. Leakage/schema/faithfulness verified by qa_commodity + the leakage audit below.
    code, out = run(["flywheel/commodity_demo.py", "--offline", "--frozen"])
    m = re.search(r"flywheel_commodity: (\d+) text", out)
    ok = code == 0 and m and int(m.group(1)) > 0
    check("flywheel commodity (frozen cached replay)", bool(ok),
          out.splitlines()[-1] if out else f"exit {code}")

    code, out = run(["flywheel/qa_commodity.py"])
    ok = code == 0 and "schema 0 err, leakage 0 leaks, 0 unfaithful" in out
    check("flywheel commodity QA (independent, 0 err/leak/unfaithful)", bool(ok),
          out.splitlines()[-1] if out else f"exit {code}")

    # 不自证: independent Data-Schema check + leakage audit on the wiki flywheel output.
    code, out = run(["flywheel/qa_wiki.py"])
    ok = code == 0 and "schema 0 err, leakage 0 leaks" in out
    check("flywheel wiki QA (independent, 0 leak/0 err)", bool(ok),
          out.splitlines()[-1] if out else f"exit {code}")

    # W3: independent faithfulness audit — every synthesized cause traces to its headlines, and
    # an injected hallucination is rejected (gate discriminates).
    code, out = run(["flywheel/audit_faithfulness.py"])
    ok = code == 0 and "negative control caught" in out
    check("flywheel faithfulness audit (entity-tracing + neg control)", bool(ok),
          out.splitlines()[-1] if out else f"exit {code}")

    # W4: independent cross-source leakage audit — knowledge_time < forecast origin (cut time
    # before windowing). Hard guarantee = flywheel sources 0-leak.
    code, out = run(["flywheel/audit_leakage.py"])
    ok = code == 0 and "flywheel 0 leaks" in out
    check("flywheel leakage audit (kt < origin, cross-source)", bool(ok),
          [l for l in out.splitlines() if l.startswith("==")][-1] if out else f"exit {code}")

    # text<->series coupling (world-knowledge quality dim): the scorer must rank event-grounded
    # commodity text clearly above standing wiki bios, else it has stopped measuring coupling (不自证).
    code, out = run(["flywheel/audit_match.py"])
    ok = code == 0 and "MATCH SCORER OK" in out
    check("flywheel match audit (coupling scorer validated on known data)", bool(ok),
          [l for l in out.splitlines() if l.startswith("MATCH")][-1] if "MATCH" in out else f"exit {code}")

    # W6: companion-study scaffold runs end-to-end (time split + arms + metric) with a clean
    # leakage assertion — proves the downstream-eval plumbing before the real base model plugs in.
    code, out = run(["flywheel/companion/run.py"])
    ok = code == 0 and "leakage assertion (cut time before windowing): CLEAN" in out
    check("companion scaffold (pipeline + leakage clean)", bool(ok),
          next((l for l in out.splitlines() if "train" in l and "test" in l), f"exit {code}"))

    # W1: multi-scale detection covers every threshold hit AND recovers multi-day moves it missed.
    code, out = run(["flywheel/audit_detection.py"])
    ok = code == 0 and "multi-day" in out
    check("detection multi-scale (covers threshold + recovers multi-day)", bool(ok),
          [l for l in out.splitlines() if l.startswith("==")][-1] if out else f"exit {code}")

    # W2: relevance ranking is correct/lossless (hard-denoise honestly deferred to v2 — semantic).
    code, out = run(["flywheel/audit_retrieval.py"])
    ok = code == 0 and "ranking correct" in out
    check("retrieval ranking (lossless; hard-denoise deferred v2)", bool(ok),
          [l for l in out.splitlines() if l.startswith("==")][-1] if out else f"exit {code}")

    n_ok = sum(RESULTS)
    print(f"== {n_ok}/{len(RESULTS)} passed ==")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
