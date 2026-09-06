#!/usr/bin/env python3
"""Create a cryptographically bound release catalog after all gates pass.

The catalog is written only when every JSONL artifact has a matching build manifest, matching
SHA-256/counts, zero rejected/unsupported records, and zero release-profile errors or warnings.
Policy-filtered ledgers are also counted and hash-bound; artifacts left empty by policy are recorded
as exclusions instead of being presented as releasable datasets.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMPANION = os.path.join(HERE, "flywheel", "companion")
sys.path.insert(0, HERE)
sys.path.insert(0, COMPANION)

from core.schema import BUILDER  # noqa: E402
from core.writer import write_json  # noqa: E402
import verify_any  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_record(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    return {}


def _jsonl_count(path):
    count = 0
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            count += 1
    return count


def build_catalog(directory):
    artifacts = verify_any.discover_files([directory])
    errors, entries, exclusions = [], [], []
    for path in artifacts:
        if not path.endswith(".jsonl"):
            continue
        manifest_path = path + ".manifest.json"
        if not os.path.isfile(manifest_path):
            errors.append(f"{path}: missing build manifest")
            continue
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        filtered_path = path + ".filtered.jsonl"
        filtered_exists = os.path.isfile(filtered_path)
        try:
            filtered_count = _jsonl_count(filtered_path) if filtered_exists else None
        except ValueError as exc:
            errors.append(str(exc))
            continue
        actual_hash = sha256(path)
        audit = verify_any.audit_file(path, min_window=4, release=True)
        warning_count = sum(audit["warns"].values())
        error_count = sum(audit["errors"].values())
        checks = {
            "artifact_sha256": manifest.get("artifact_sha256") == actual_hash,
            "builder": manifest.get("builder") == BUILDER,
            "record_count": manifest.get("records_written") == audit["n"],
            "records_rejected": manifest.get("records_rejected") == 0,
            "records_unsupported": manifest.get("records_unsupported") == 0,
            "generated_policy": manifest.get("generated_policy") == "real-only",
            "filtered_ledger": filtered_exists,
            "filtered_count": filtered_count == manifest.get("records_filtered"),
            "release_errors": error_count == 0,
            "release_warnings": warning_count == 0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            errors.append(f"{path}: failed {', '.join(failed)}")
            continue
        ledger_hash = sha256(filtered_path)
        if audit["n"] == 0:
            if not filtered_count:
                errors.append(f"{path}: empty artifact has no policy-filtered records")
                continue
            exclusions.append({
                "artifact": os.path.abspath(path),
                "dataset": manifest.get("source"),
                "reason": "empty_after_real_only_policy",
                "records_filtered": filtered_count,
                "filtered_ledger_sha256": ledger_hash,
                "source_manifest_sha256": sha256(manifest_path),
            })
            continue
        first = _first_record(path)
        entries.append({
            "artifact": os.path.abspath(path),
            "artifact_sha256": actual_hash,
            "dataset": first.get("dataset"),
            "format": audit["format"],
            "license": first.get("license"),
            "license_status": first.get("license_status"),
            "records": audit["n"],
            "records_filtered": filtered_count,
            "filtered_ledger_sha256": ledger_hash,
            "source_manifest_sha256": sha256(manifest_path),
        })
    if not artifacts:
        errors.append(f"{directory}: no JSONL artifacts")
    if not entries:
        errors.append(f"{directory}: no non-empty releasable artifacts")
    if errors:
        raise ValueError("release catalog refused:\n- " + "\n- ".join(errors))
    entries.sort(key=lambda entry: (entry["dataset"] or "", entry["artifact"]))
    exclusions.sort(key=lambda entry: (entry["dataset"] or "", entry["artifact"]))
    bound_hashes = []
    for entry in entries:
        bound_hashes.extend((entry["artifact_sha256"], entry["source_manifest_sha256"],
                             entry["filtered_ledger_sha256"]))
    for entry in exclusions:
        bound_hashes.extend((entry["source_manifest_sha256"],
                             entry["filtered_ledger_sha256"]))
    aggregate = hashlib.sha256("\n".join(bound_hashes).encode("ascii")).hexdigest()
    return {
        "schema_version": "release-catalog@2",
        "builder": BUILDER,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "release_validated": True,
        "strict": True,
        "artifact_count": len(entries),
        "record_count": sum(entry["records"] for entry in entries),
        "records_filtered": sum(entry["records_filtered"] for entry in entries)
                            + sum(entry["records_filtered"] for entry in exclusions),
        "excluded_artifact_count": len(exclusions),
        "aggregate_sha256": aggregate,
        "artifacts": entries,
        "excluded_artifacts": exclusions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out")
    args = parser.parse_args()
    output = args.out or os.path.join(args.directory, "release_catalog.json")
    try:
        catalog = build_catalog(args.directory)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
    write_json(catalog, output)
    print(f"release catalog: {catalog['artifact_count']} artifacts · "
          f"{catalog['record_count']} records · {catalog['aggregate_sha256']}")
    print(f"-> {output}")


if __name__ == "__main__":
    main()
