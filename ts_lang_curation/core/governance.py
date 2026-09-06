"""Source-license registry and fail-closed release policy.

This module records evidence, not legal conclusions. Unknown or unresolved sources are
``review_required`` and cannot pass the release profile.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "governance" / "source_licenses.json"
RELEASABLE_STATUSES = frozenset({"approved", "conditional"})
KNOWN_STATUSES = RELEASABLE_STATUSES | {"review_required", "blocked"}


def load_registry(path=REGISTRY_PATH):
    with open(path, encoding="utf-8") as handle:
        registry = json.load(handle)
    if not isinstance(registry.get("sources"), dict):
        raise ValueError("governance registry must contain a sources object")
    if not isinstance(registry.get("aliases", {}), dict):
        raise ValueError("governance registry aliases must be an object")
    return registry


def decision_for(dataset, registry=None):
    registry = registry or load_registry()
    canonical = registry.get("aliases", {}).get(dataset, dataset)
    decision = deepcopy(registry["sources"].get(canonical))
    if decision is None:
        decision = {
            "status": "review_required",
            "license": "NOASSERTION",
            "terms_url": "",
            "attribution": "",
            "reason": f"Dataset {dataset!r} is absent from the source-license registry.",
        }
    status = decision.get("status")
    if status not in KNOWN_STATUSES:
        raise ValueError(f"invalid governance status {status!r} for dataset {dataset!r}")
    decision["registry_version"] = registry.get("registry_version")
    decision["reviewed_at"] = registry.get("reviewed_at")
    decision["canonical_dataset"] = canonical
    return decision


def apply_governance(record, registry=None):
    """Return a copy stamped from the central registry; registry policy wins over adapters."""
    out = dict(record)
    decision = decision_for(out.get("dataset"), registry=registry)
    out["license"] = decision["license"]
    out["license_status"] = decision["status"]
    out["license_terms_url"] = decision["terms_url"]
    out["attribution"] = decision["attribution"]
    out["license_reason"] = decision["reason"]
    out["governance_registry_version"] = decision["registry_version"]
    out["governance_reviewed_at"] = decision["reviewed_at"]
    out["governance_canonical_dataset"] = decision["canonical_dataset"]
    return out


def release_errors(record):
    """Return stable error codes/messages for governance release gating."""
    errors = []
    status = record.get("license_status")
    if status not in KNOWN_STATUSES:
        errors.append(("license_status_missing", "release record requires a known license_status"))
    elif status not in RELEASABLE_STATUSES:
        errors.append(("license_not_releasable", f"license_status={status!r}"))
    if not record.get("license") or record.get("license") == "NOASSERTION":
        errors.append(("license_missing", "release record requires an asserted license"))
    if not record.get("license_terms_url"):
        errors.append(("license_terms_url_missing", "release record requires license evidence URL"))
    if status == "conditional" and not record.get("attribution"):
        errors.append(("attribution_missing", "conditional release requires attribution text"))
    return errors
