"""Separate source inventory for simulated-event development cases.

Keeping this profile separate avoids changing already frozen route and
recommendation manifests. It claims no real campus event evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.controlled_source_manifest import CONTENT_ENCODING, _file, _sha_file


FORMAT = "controlled-event-source-v1"
EVENT_PILOT_INPUTS = frozenset({
    "graph", "poi", "boundary", "anchors", "event_snapshot",
    "oracle_mode", "oracle_event", "oracle_route", "oracle_existence",
    "generator", "templates", "coordinate_transform", "anchor_builder",
    "boundary_source", "spatial_snapshot_builder", "dataset_builder",
    "gold_contract", "source_manifest_builder", "scenario_evaluator",
    "route_scorer", "event_scorer", "collector", "route_case_helpers",
    "source_hash_helpers", "packet_helpers", "observation_validator",
})


def _validated_metadata(metadata: Any) -> dict[str, Any]:
    if (not isinstance(metadata, dict) or set(metadata) != {
            "graph_crs", "boundary_crs", "event_scope", "campus_truth"}
            or metadata["graph_crs"] != "EPSG:4326"
            or metadata["boundary_crs"] != metadata["graph_crs"]
            or metadata["event_scope"] != "synthetic_development_only"
            or metadata["campus_truth"] is not False):
        raise ValueError("event scope must be synthetic_development_only with campus_truth=false")
    return dict(metadata)


def _effective_hash(inputs: dict[str, Any], metadata: dict[str, Any]) -> str:
    encoded = json.dumps({"format": FORMAT, "profile": "event_pilot",
                          "inputs": inputs, "metadata": metadata},
                         sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_event_manifest(root: Path, inputs: dict[str, str],
                        metadata: dict[str, Any]) -> dict[str, Any]:
    """Hash every explicit source after UTF-8 LF canonicalization."""
    if not isinstance(inputs, dict) or set(inputs) != EVENT_PILOT_INPUTS:
        raise ValueError("event source inputs incomplete")
    clean_metadata = _validated_metadata(metadata)
    records = {}
    for name in sorted(EVENT_PILOT_INPUTS):
        relative = inputs[name]
        target = _file(Path(root), relative)
        records[name] = {"path": Path(relative).as_posix(),
                         "content_encoding": CONTENT_ENCODING,
                         "sha256": _sha_file(target)}
    return {"format": FORMAT, "scope": "controlled_graph_input",
            "profile": "event_pilot", "inputs": records,
            "metadata": clean_metadata,
            "effective_inputs_sha256": _effective_hash(records, clean_metadata)}


def verify_event_manifest(manifest: dict[str, Any], root: Path) -> str:
    """Reject changed paths, content, schema or effective source digest."""
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
            or manifest.get("scope") != "controlled_graph_input"
            or manifest.get("profile") != "event_pilot"
            or not isinstance(manifest.get("inputs"), dict)
            or set(manifest["inputs"]) != EVENT_PILOT_INPUTS):
        raise ValueError("invalid event source manifest")
    metadata = _validated_metadata(manifest.get("metadata"))
    for name in sorted(EVENT_PILOT_INPUTS):
        record = manifest["inputs"][name]
        if (not isinstance(record, dict) or set(record) != {
                "path", "content_encoding", "sha256"}
                or record.get("content_encoding") != CONTENT_ENCODING):
            raise ValueError(f"{name} source record invalid")
        if _sha_file(_file(Path(root), record["path"])) != record["sha256"]:
            raise ValueError(f"{name} sha256 mismatch")
    expected = _effective_hash(manifest["inputs"], metadata)
    if manifest.get("effective_inputs_sha256") != expected:
        raise ValueError("effective_inputs_sha256 mismatch")
    return expected
