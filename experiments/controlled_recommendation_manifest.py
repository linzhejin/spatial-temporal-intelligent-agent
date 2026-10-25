"""Versioned source inventory for the development-only recommendation packet.

This is separate from the frozen route-pilot v3 manifest so that adding
recommendation inputs cannot silently change its existing source fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


FORMAT = "controlled-recommendation-source-v1"
RECOMMENDATION_INPUTS = frozenset({
    "graph", "poi", "boundary", "anchors", "boundary_source",
    "coordinate_transform", "spatial_snapshot_builder", "anchor_builder",
    "oracle_mode", "oracle_route", "oracle_existence",
    "recommendation_gold", "generator", "templates", "dataset_builder",
    "gold_contract", "scenario_evaluator", "controlled_source_manifest",
    "requirements", "source_manifest_builder",
})
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_METADATA = {
    "scope": "development_only",
    "graph_crs": "EPSG:4326",
    "poi_crs": "GCJ-02",
    "boundary_crs": "EPSG:4326",
    "activity_rule": "source_supported_with_nonempty_activity_basis",
    "anchor_scope": "walk_nodes_within_20m_last_meter_unverified",
    "distance_rule": "shortest_permitted_walk_graph_length_m",
    "event_scope": "not_applicable",
}


def _source(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("recommendation source path outside root")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError("recommendation source path outside root or missing")
    return target


def _sha_file(path: Path) -> str:
    content = path.read_bytes()
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"recommendation source is not UTF-8: {path.name}") from exc
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def _effective_hash(inputs: dict[str, Any]) -> str:
    payload = {"format": FORMAT, "inputs": inputs, "metadata": _METADATA}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_recommendation_manifest(root: Path,
                                 paths: dict[str, str]) -> dict[str, Any]:
    """Freeze the exact source files used to derive the recommendation gold."""
    root = Path(root).resolve()
    if not isinstance(paths, dict) or set(paths) != RECOMMENDATION_INPUTS:
        raise ValueError("recommendation source inputs incomplete")
    inputs = {}
    for name in sorted(RECOMMENDATION_INPUTS):
        relative = paths[name]
        inputs[name] = {"path": Path(relative).as_posix(),
                        "content_encoding": "utf8_lf_canonical_v1",
                        "sha256": _sha_file(_source(root, relative))}
    return {"format": FORMAT, "profile": "recommendation_pilot",
            "inputs": inputs, "metadata": dict(_METADATA),
            "effective_inputs_sha256": _effective_hash(inputs)}


def verify_recommendation_manifest(manifest: dict[str, Any], root: Path) -> str:
    """Reject a changed source, missing input or altered metadata."""
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
            or manifest.get("profile") != "recommendation_pilot"
            or manifest.get("metadata") != _METADATA):
        raise ValueError("invalid recommendation source manifest")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != RECOMMENDATION_INPUTS:
        raise ValueError("recommendation source inputs incomplete")
    root = Path(root).resolve()
    for name in sorted(RECOMMENDATION_INPUTS):
        record = inputs[name]
        if (not isinstance(record, dict) or set(record) !=
                {"path", "content_encoding", "sha256"}
                or record["content_encoding"] != "utf8_lf_canonical_v1"
                or not isinstance(record["sha256"], str)
                or not _DIGEST.fullmatch(record["sha256"])):
            raise ValueError(f"{name} recommendation source record invalid")
        if _sha_file(_source(root, record["path"])) != record["sha256"]:
            raise ValueError(f"{name} recommendation source sha256 mismatch")
    expected = _effective_hash(inputs)
    if manifest.get("effective_inputs_sha256") != expected:
        raise ValueError("recommendation source effective hash mismatch")
    return expected
