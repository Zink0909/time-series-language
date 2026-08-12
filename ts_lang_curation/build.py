#!/usr/bin/env python3
"""Unified builder for all sources.

Add a new source = drop one file in sources/<name>.py exposing a `pairs()` generator
that yields core.schema.Pair objects. Nothing else changes.

Usage:
    python build.py --source fred_fomc
    python build.py --source fnspid --out out/fnspid.jsonl
    python build.py --list
"""
import argparse, importlib, os, sys, pkgutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core.schema import build_record, validate
from core.writer import write_jsonl


def list_sources():
    import sources
    return [m.name for m in pkgutil.iter_modules(sources.__path__) if not m.name.startswith("_")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="adapter name in sources/ (e.g. fred_fomc, fnspid)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true", help="list available sources")
    a = ap.parse_args()

    if a.list or not a.source:
        print("available sources:", ", ".join(list_sources())); return

    mod = importlib.import_module(f"sources.{a.source}")
    records, invalid = [], 0
    for p in mod.pairs():
        rec = build_record(p)
        errs = validate(rec)
        if errs:
            invalid += 1
            print(f"  [INVALID] {rec.get('series_id', '?')}: {errs}", file=sys.stderr)
            continue
        records.append(rec)
    out = a.out or os.path.join(HERE, "out", f"{a.source}.jsonl")
    n = write_jsonl(records, out)
    print(f"{a.source}: {n} valid records -> {out}  ({invalid} invalid)")


if __name__ == "__main__":
    main()
