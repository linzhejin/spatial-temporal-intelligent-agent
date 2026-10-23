"""Fail closed when a purported same-model Agent comparison is not matched."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


CONDITIONS = {"luojia", "ordinary_tool_agent", "deepseek_harness"}
MATCHED_FIELDS = (
    "resolved_model_id", "tool_schema_sha256", "data_snapshot_id",
    "event_snapshot_sha256", "event_clock_epoch", "temperature",
    "max_tokens_per_request", "max_model_rounds_per_turn", "time_budget_s",
)


def audit_same_model_conditions(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Check paired declarations against immutable local request/tool evidence."""
    blockers = []
    labels = [row.get("condition") for row in manifests if isinstance(row, dict)]
    if Counter(labels) != Counter({name: 1 for name in CONDITIONS}):
        blockers.append("condition_set_incomplete_or_duplicated")
    if len(manifests) != 3 or any(not isinstance(row, dict) for row in manifests):
        return {"formal_ready": False, "blockers": blockers or ["invalid_manifest"]}
    for field in MATCHED_FIELDS:
        values = [row.get(field) for row in manifests]
        if any(value is None for value in values):
            blockers.append(f"missing:{field}")
        elif len({str(value) for value in values}) != 1:
            blockers.append(f"mismatch:{field}")
    names = [row.get("enabled_tool_names") for row in manifests]
    if (any(not isinstance(value, list) or not value for value in names)
            or len({tuple(sorted(value)) for value in names if isinstance(value, list)}) != 1):
        blockers.append("tool_permissions_differ")
    if any(row.get("extra_tool_names") for row in manifests):
        blockers.append("extra_tools_enabled")
    if any(row.get("pilot_status") != "passed" for row in manifests):
        blockers.append("pilot_not_passed")
    if any(not row.get("implementation_revision") for row in manifests):
        blockers.append("implementation_revision_missing")
    if any(not isinstance(row.get("measured_usage"), dict)
           or type(row["measured_usage"].get("requests")) is not int
           or row["measured_usage"]["requests"] < 0 for row in manifests):
        blockers.append("measured_usage_missing")
    elif not any(row["measured_usage"]["requests"] for row in manifests):
        blockers.append("no_model_requests_observed")
    for row in manifests:
        condition = str(row.get("condition"))
        evidence_path = row.get("run_evidence_path")
        expected_digest = row.get("run_evidence_sha256")
        if not evidence_path or not expected_digest:
            blockers.append("run_evidence_missing")
            continue
        try:
            path = Path(evidence_path)
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("evidence file exceeds size limit")
            raw = path.read_bytes()
            records = json.loads(raw)
        except (OSError, ValueError, TypeError):
            blockers.append(f"run_evidence_unreadable:{condition}")
            continue
        if hashlib.sha256(raw).hexdigest() != expected_digest:
            blockers.append(f"run_evidence_hash_mismatch:{condition}")
            continue
        if not isinstance(records, list) or not records or not isinstance(records[0], dict):
            blockers.append(f"run_evidence_invalid:{condition}")
            continue
        header = records[0]
        if header.get("kind") != "run" or header.get("condition") != condition:
            blockers.append(f"run_evidence_header_invalid:{condition}")
            continue
        for field in ("data_snapshot_id", "event_snapshot_sha256",
                      "event_clock_epoch", "enabled_tool_names",
                      "extra_tool_names", "max_model_rounds_per_turn",
                      "time_budget_s"):
            if header.get(field) != row.get(field):
                blockers.append(f"run_evidence_mismatch:{condition}:{field}")
        requests = 0
        input_tokens = output_tokens = 0
        enabled = set(row.get("enabled_tool_names") or [])
        for record in records[1:]:
            if not isinstance(record, dict):
                blockers.append(f"run_evidence_invalid:{condition}")
                continue
            kind = record.get("kind")
            if kind == "request":
                requests += 1
                if (record.get("resolved_model_id") != row.get("resolved_model_id")
                        or record.get("tool_schema_sha256") != row.get("tool_schema_sha256")
                        or record.get("temperature") != row.get("temperature")
                        or record.get("max_tokens") != row.get("max_tokens_per_request")):
                    blockers.append(f"observed_request_mismatch:{condition}")
                for field in ("input_tokens", "output_tokens"):
                    value = record.get(field)
                    if type(value) is not int or value < 0:
                        blockers.append(f"run_evidence_usage_invalid:{condition}")
                input_tokens += record.get("input_tokens", 0) if type(record.get("input_tokens")) is int else 0
                output_tokens += record.get("output_tokens", 0) if type(record.get("output_tokens")) is int else 0
            elif kind == "tool":
                name = record.get("name")
                if name not in enabled:
                    blockers.append(f"observed_extra_tool:{condition}:{name}")
                if record.get("outcome") not in {"success", "tool_error", "invalid_args"}:
                    blockers.append(f"run_evidence_tool_invalid:{condition}")
            else:
                blockers.append(f"run_evidence_kind_invalid:{condition}")
        usage = row.get("measured_usage") or {}
        if (usage.get("requests") != requests or usage.get("input_tokens") != input_tokens
                or usage.get("output_tokens") != output_tokens):
            blockers.append(f"run_evidence_usage_mismatch:{condition}")
    return {"formal_ready": not blockers, "blockers": sorted(set(blockers)),
            "conditions": sorted(str(label) for label in labels)}
