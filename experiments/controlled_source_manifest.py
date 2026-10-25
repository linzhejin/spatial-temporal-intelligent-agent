"""Content-addressed input inventory for controlled route experiments.

The inventory freezes UTF-8 content after explicit CRLF-to-LF normalization,
so Windows and Linux checkouts share a fingerprint. It does not establish field
passability or certify that a generated label is semantically correct.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


FORMAT = "controlled-source-v3"
CONTENT_ENCODING = "utf8_lf_canonical_v1"
REQUIRED_INPUTS = frozenset({
    "graph", "poi", "boundary", "event_snapshot", "oracle_mode",
    "oracle_route", "oracle_existence", "generator", "templates", "information_snapshot",
    "coordinate_transform", "poi_taxonomy", "attribute_master",
    "anchor_builder", "anchors", "boundary_source", "spatial_snapshot_builder",
    "dataset_builder", "gold_contract", "source_manifest_builder",
})
ROUTE_PILOT_INPUTS = REQUIRED_INPUTS - {
    "event_snapshot", "information_snapshot", "poi_taxonomy", "attribute_master",
}
_PROFILES = {"full": REQUIRED_INPUTS, "route_pilot": ROUTE_PILOT_INPUTS}
REQUIRED_METADATA = frozenset({
    "graph_crs", "boundary_crs", "boundary_provenance", "event_scope",
})
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _sha_file(path: Path) -> str:
    source = path.read_bytes()
    try:
        source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"source file is not UTF-8 text: {path.name}") from exc
    return hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest()


def _file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("source path outside root")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("source path outside root")
    if not target.is_file():
        raise ValueError(f"source file missing: {relative}")
    return target


def _validate_metadata(metadata: Any, profile: str) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("source metadata required")
    for name in REQUIRED_METADATA:
        if not isinstance(metadata.get(name), str) or not metadata[name].strip():
            raise ValueError(f"source metadata {name} required")
    if metadata["graph_crs"] != metadata["boundary_crs"]:
        raise ValueError("graph_crs and boundary_crs differ")
    if profile == "route_pilot" and metadata["event_scope"] != "not_applicable":
        raise ValueError("route pilot event_scope must be not_applicable")
    if profile == "full" and metadata["event_scope"] == "not_applicable":
        raise ValueError("full source profile requires event_scope")


def _effective_hash(inputs: dict[str, Any], metadata: dict[str, str], profile: str) -> str:
    payload = {"format": FORMAT, "profile": profile, "inputs": inputs,
               "metadata": metadata}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_source_manifest(root: Path, inputs: dict[str, str],
                         metadata: dict[str, str], *, profile: str = "full") -> dict[str, Any]:
    """Build a deterministic inventory from explicitly selected local files."""
    root = Path(root).resolve()
    if profile not in _PROFILES:
        raise ValueError("unknown source profile")
    required = _PROFILES[profile]
    if not isinstance(inputs, dict) or set(inputs) != required:
        missing = sorted(required - set(inputs or {}))
        raise ValueError(f"source inputs incomplete: {', '.join(missing)}")
    _validate_metadata(metadata, profile)
    records = {}
    for name in sorted(required):
        relative = inputs[name]
        target = _file(root, relative)
        records[name] = {"path": Path(relative).as_posix(),
                         "content_encoding": CONTENT_ENCODING,
                         "sha256": _sha_file(target)}
    clean_metadata = {name: metadata[name] for name in sorted(REQUIRED_METADATA)}
    return {
        "format": FORMAT, "scope": "controlled_graph_input", "profile": profile,
        "inputs": records, "metadata": clean_metadata,
        "effective_inputs_sha256": _effective_hash(records, clean_metadata, profile),
    }


def verify_source_manifest(manifest: dict[str, Any], root: Path) -> str:
    """Reject source drift, missing files and malformed or fake digests."""
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
            or manifest.get("scope") != "controlled_graph_input"):
        raise ValueError("invalid controlled source manifest")
    profile = manifest.get("profile")
    if profile not in _PROFILES:
        raise ValueError("unknown source profile")
    required = _PROFILES[profile]
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise ValueError("source inputs incomplete")
    metadata = manifest.get("metadata")
    _validate_metadata(metadata, profile)
    root = Path(root).resolve()
    for name in sorted(required):
        record = inputs[name]
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"{name} source record invalid")
        if record.get("content_encoding") != CONTENT_ENCODING:
            raise ValueError(f"{name} content encoding invalid")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError(f"{name} sha256 invalid")
        if _sha_file(_file(root, record["path"])) != digest:
            raise ValueError(f"{name} sha256 mismatch")
    expected = _effective_hash(inputs, metadata, profile)
    if manifest.get("effective_inputs_sha256") != expected:
        raise ValueError("effective_inputs_sha256 mismatch")
    return expected
