"""Versioned source inventory for the development-only multi-turn packet.

A separate inventory leaves the older frozen route and recommendation source
fingerprints intact while including the helper and delegated scorer imports.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from experiments.controlled_source_manifest import ROUTE_PILOT_INPUTS


FORMAT = "controlled-multi-turn-source-v1"
MULTI_TURN_INPUTS = ROUTE_PILOT_INPUTS | {
    "route_case_helpers", "route_scorer", "observation_projection",
    "controlled_source_manifest", "scenario_evaluator", "requirements",
}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_METADATA = {
    "scope": "development_only", "graph_crs": "EPSG:4326",
    "boundary_crs": "EPSG:4326", "event_scope": "not_applicable",
    "revision_rule": "retain_origin_avoid_steps_shortest_change_mode_destination",
    "anchor_scope": "walk_nodes_within_20m_last_meter_unverified",
}


def _source(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("multi-turn source path outside root")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError("multi-turn source path outside root or missing")
    return target


def _sha_file(path: Path) -> str:
    content = path.read_bytes()
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"multi-turn source is not UTF-8: {path.name}") from exc
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def _effective_hash(inputs: dict[str, Any]) -> str:
    payload = {"format": FORMAT, "inputs": inputs, "metadata": _METADATA}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_multi_turn_manifest(root: Path, paths: dict[str, str]) -> dict[str, Any]:
    """Freeze all source files read by the case builder and offline scorer."""
    root = Path(root).resolve()
    if not isinstance(paths, dict) or set(paths) != MULTI_TURN_INPUTS:
        raise ValueError("multi-turn source inputs incomplete")
    inputs = {}
    for name in sorted(MULTI_TURN_INPUTS):
        relative = paths[name]
        inputs[name] = {"path": Path(relative).as_posix(),
                        "content_encoding": "utf8_lf_canonical_v1",
                        "sha256": _sha_file(_source(root, relative))}
    return {"format": FORMAT, "profile": "multi_turn_pilot",
            "inputs": inputs, "metadata": dict(_METADATA),
            "effective_inputs_sha256": _effective_hash(inputs)}


def verify_multi_turn_manifest(manifest: dict[str, Any], root: Path) -> str:
    """Reject drift in any source byte, record, or declared rule."""
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
            or manifest.get("profile") != "multi_turn_pilot"
            or manifest.get("metadata") != _METADATA):
        raise ValueError("invalid multi-turn source manifest")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != MULTI_TURN_INPUTS:
        raise ValueError("multi-turn source inputs incomplete")
    root = Path(root).resolve()
    for name in sorted(MULTI_TURN_INPUTS):
        record = inputs[name]
        if (not isinstance(record, dict) or set(record) !=
                {"path", "content_encoding", "sha256"}
                or record["content_encoding"] != "utf8_lf_canonical_v1"
                or not isinstance(record["sha256"], str)
                or not _DIGEST.fullmatch(record["sha256"])):
            raise ValueError(f"{name} multi-turn source record invalid")
        if _sha_file(_source(root, record["path"])) != record["sha256"]:
            raise ValueError(f"{name} multi-turn source sha256 mismatch")
    expected = _effective_hash(inputs)
    if manifest.get("effective_inputs_sha256") != expected:
        raise ValueError("multi-turn source effective hash mismatch")
    return expected
