#!/usr/bin/env python3
"""Unified builder for all sources.

Add a new source = drop one file in sources/<name>.py exposing a `pairs()` generator
that yields core.schema.Pair objects. Nothing else changes.

Usage:
    python build.py --source treasury_fomc
    python build.py --source sec_edgar --format cpt
    python build.py --source sec_edgar --format ir
    python build.py --source fnspid --out out/fnspid.jsonl
    python build.py --list
"""
import argparse, ast, datetime, hashlib, importlib, os, pkgutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core.emitters import UnsupportedFormat, emit, validate_output
from core.schema import BUILDER
from core.writer import write_json, write_jsonl


def list_sources():
    import sources
    available = []
    for module in pkgutil.iter_modules(sources.__path__):
        if module.name.startswith("_"):
            continue
        source_path = os.path.join(next(iter(sources.__path__)), module.name + ".py")
        with open(source_path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=source_path)
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "pairs"
               for node in tree.body):
            available.append(module.name)
    return available


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="adapter name in sources/ (e.g. fred_fomc, fnspid)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--format", choices=("chatml", "cpt", "ir"), default="chatml",
                    help="peer output emitted directly from the canonical pair (default: chatml)")
    ap.add_argument("--generated-policy", choices=("include", "real-only"), default="include",
                    help="real-only filters generated annotations into a separate ledger")
    ap.add_argument("--list", action="store_true", help="list available sources")
    a = ap.parse_args()

    if a.list or not a.source:
        print("available sources:", ", ".join(list_sources())); return

    mod = importlib.import_module(f"sources.{a.source}")
    records, rejected, unsupported, filtered = [], [], [], []
    for p in mod.pairs():
        if a.generated_policy == "real-only" and getattr(p, "is_generated", "real") != "real":
            filtered.append({"series_id": getattr(p, "meta", {}).get("series_id"),
                             "reason": "generated_text_excluded_by_release_policy"})
            continue
        try:
            rec = emit(p, a.format)
            errs = validate_output(rec, a.format)
        except UnsupportedFormat as exc:
            series_id = getattr(p, "meta", {}).get("series_id")
            unsupported.append({"series_id": series_id, "reason": str(exc)})
            continue
        except (TypeError, ValueError) as exc:
            rec, errs = {}, [str(exc)]
        if errs:
            series_id = rec.get("series_id") or getattr(p, "meta", {}).get("series_id")
            rejected.append({"series_id": series_id, "errors": errs})
            print(f"  [INVALID] {series_id or '?'}: {errs}", file=sys.stderr)
            continue
        records.append(rec)
    suffix = "" if a.format == "chatml" else f".{a.format}"
    out = a.out or os.path.join(HERE, "out", f"{a.source}{suffix}.jsonl")
    n = write_jsonl(records, out)
    rejected_path = out + ".rejected.jsonl"
    write_jsonl(rejected, rejected_path)
    unsupported_path = out + ".unsupported.jsonl"
    write_jsonl(unsupported, unsupported_path)
    filtered_path = out + ".filtered.jsonl"
    write_jsonl(filtered, filtered_path)
    receipt = {
        "artifact": os.path.abspath(out),
        "artifact_sha256": _sha256(out),
        "builder": BUILDER,
        "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "profile": "structural",
        "output_format": a.format,
        "generated_policy": a.generated_policy,
        "records_filtered": len(filtered),
        "records_rejected": len(rejected),
        "records_unsupported": len(unsupported),
        "records_written": n,
        "release_validated": False,
        "source": a.source,
    }
    write_json(receipt, out + ".manifest.json")
    print(f"{a.source}: {n} valid records -> {out}  "
          f"({len(rejected)} invalid, {len(unsupported)} unsupported by {a.format}, "
          f"{len(filtered)} policy-filtered)")
    print(f"  manifest -> {out}.manifest.json · rejected -> {rejected_path} · "
          f"unsupported -> {unsupported_path} · filtered -> {filtered_path}")


if __name__ == "__main__":
    main()
