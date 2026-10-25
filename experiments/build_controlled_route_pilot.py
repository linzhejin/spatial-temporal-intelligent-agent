"""Build a replayable, development-only route packet from frozen local sources.

This packet does not satisfy the six-category formal test gate or establish
field passability. It never calls an LLM or writes to the production database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import networkx as nx

from experiments.build_controlled_spatial_snapshot import (
    build_spatial_snapshot, write_spatial_snapshot,
)
from experiments.controlled_gold_contract import validate_controlled_dataset
from experiments.controlled_route_cases import make_route_case
from experiments.controlled_source_manifest import (
    ROUTE_PILOT_INPUTS, make_source_manifest, verify_source_manifest,
)
from experiments.independent_source_snapshot import mode_edge_evidence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("experiments/datasets/controlled_route_pilot_v1")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build_route_pilot_dataset(
    graph: Any, anchors: dict[str, dict[str, Any]], spec: dict[str, Any],
    manifest: dict[str, Any], manifest_sha256: str,
    evidence_by_mode: dict[str, dict[str, set[tuple[Any, Any, Any]]]],
) -> dict[str, Any]:
    """Generate each case from one immutable specification; never skip failures."""
    if (not isinstance(spec, dict) or spec.get("format") != "controlled-route-pilot-spec-v1"
            or spec.get("scope") != "development_only"
            or not isinstance(spec.get("templates"), dict)
            or not isinstance(spec.get("cases"), list) or not spec["cases"]):
        raise ValueError("invalid route pilot specification")
    if (not isinstance(manifest, dict) or manifest.get("profile") != "route_pilot"
            or not _DIGEST.fullmatch(str(manifest.get("effective_inputs_sha256", "")))
            or not _DIGEST.fullmatch(manifest_sha256)):
        raise ValueError("verified route pilot source manifest required")
    inputs = manifest.get("inputs")
    if (not isinstance(inputs, dict) or any(
            not isinstance(inputs.get(name), dict)
            or not _DIGEST.fullmatch(str(inputs[name].get("sha256", "")))
            for name in ("generator", "templates"))):
        raise ValueError("generator and template source digests required")
    seen_ids = set()
    cases = []
    for entry in spec["cases"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("invalid route pilot case specification")
        if entry["id"] in seen_ids:
            raise ValueError("duplicate route pilot case id")
        seen_ids.add(entry["id"])
        template_id = entry.get("template_id")
        template = spec["templates"].get(template_id)
        if not isinstance(template, dict) or not isinstance(template.get("text"), str):
            raise ValueError("missing route pilot template")
        parameters = entry.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("invalid route pilot parameters")
        mode = parameters.get("mode")
        evidence = evidence_by_mode.get(mode)
        if not isinstance(evidence, dict):
            raise ValueError("mode evidence missing for route pilot case")
        cases.append(make_route_case(
            graph, anchors, permitted_edge_ids=evidence["permitted"],
            unknown_edge_ids=evidence["unknown"], closed_edge_ids=set(),
            templates={template_id: template["text"]}, template_id=template_id,
            case_id=entry["id"], category=template["category"],
            template_group=template["template_group"], od_group=entry["od_group"],
            snapshot_id=manifest["effective_inputs_sha256"],
            generator_sha256=inputs["generator"]["sha256"],
            template_sha256=inputs["templates"]["sha256"],
            parameters=parameters,
        ))
    return {"protocol_version": "scenario-v2", "gold_layer": "controlled_graph",
            "split": "development", "repeats": 1,
            "source_manifest_sha256": manifest_sha256,
            "snapshot_id": manifest["effective_inputs_sha256"],
            "cases": cases}


def verify_route_pilot_dataset(
    dataset: dict[str, Any], graph: Any, anchors: dict[str, dict[str, Any]],
    spec: dict[str, Any], manifest: dict[str, Any], manifest_sha256: str,
    evidence_by_mode: dict[str, dict[str, set[tuple[Any, Any, Any]]]],
) -> None:
    """Replay the full packet so group, prompt and answer tampering is detected."""
    expected = build_route_pilot_dataset(graph, anchors, spec, manifest,
                                         manifest_sha256, evidence_by_mode)
    if dataset != expected:
        raise ValueError("route pilot packet differs from case specification")
    validate_controlled_dataset(dataset, formal=False)


def _write_new_or_same(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"frozen pilot output changed; choose a new version: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    spec_path = out_dir / "route_pilot_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    from config import CAMPUS_POLYS_GCJ
    from spatial.coord_transform import gcj02_to_wgs84

    graph = nx.read_graphml(root / "data/whu_road_network.graphml",
                            node_type=int, force_multigraph=True)
    poi_data = json.loads((root / "data/pois.json").read_text(encoding="utf-8"))
    if not isinstance(poi_data, dict) or not isinstance(poi_data.get("pois"), list):
        raise ValueError("POI source lacks pois list")
    boundary_wkt, anchor_data = build_spatial_snapshot(
        graph, poi_data["pois"], CAMPUS_POLYS_GCJ, to_wgs84=gcj02_to_wgs84,
    )
    spatial_dir = root / "experiments/datasets/controlled_spatial_v1"
    write_spatial_snapshot(spatial_dir / "boundary_wgs84.wkt",
                           spatial_dir / "walk_anchors.json", boundary_wkt, anchor_data)
    anchors = {item["poi_id"]: item for item in anchor_data["anchors"]}
    if len(anchors) != len(anchor_data["anchors"]):
        raise ValueError("duplicate anchor id")
    source_paths = {
        "graph": "data/whu_road_network.graphml",
        "poi": "data/pois.json",
        "boundary": "experiments/datasets/controlled_spatial_v1/boundary_wgs84.wkt",
        "oracle_mode": "experiments/independent_source_snapshot.py",
        "oracle_route": "experiments/independent_route_oracle.py",
        "oracle_existence": "experiments/controlled_route_gold.py",
        "generator": "experiments/controlled_route_cases.py",
        "templates": spec_path.resolve().relative_to(root).as_posix(),
        "coordinate_transform": "spatial/coord_transform.py",
        "anchor_builder": "experiments/controlled_anchor_snapshot.py",
        "anchors": "experiments/datasets/controlled_spatial_v1/walk_anchors.json",
        "boundary_source": "config.py",
        "spatial_snapshot_builder": "experiments/build_controlled_spatial_snapshot.py",
        "dataset_builder": "experiments/build_controlled_route_pilot.py",
        "gold_contract": "experiments/controlled_gold_contract.py",
        "source_manifest_builder": "experiments/controlled_source_manifest.py",
    }
    if set(source_paths) != ROUTE_PILOT_INPUTS:
        raise ValueError("pilot source input list differs from contract")
    manifest = make_source_manifest(root, source_paths, {
        "graph_crs": "EPSG:4326", "boundary_crs": "EPSG:4326",
        "boundary_provenance": "product config.CAMPUS_POLYS_GCJ via gcj02_to_wgs84",
        "event_scope": "not_applicable",
    }, profile="route_pilot")
    if verify_source_manifest(manifest, root) != manifest["effective_inputs_sha256"]:
        raise ValueError("source manifest failed verification")
    manifest_bytes = _json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    modes = {item["parameters"]["mode"] for item in spec["cases"]}
    evidence = {
        mode: mode_edge_evidence(graph, mode, require_boundary=True,
                                 boundary_wkt=boundary_wkt)
        for mode in modes
    }
    dataset = build_route_pilot_dataset(graph, anchors, spec, manifest,
                                        manifest_sha, evidence)
    verify_route_pilot_dataset(dataset, graph, anchors, spec, manifest,
                               manifest_sha, evidence)
    dataset_bytes = _json_bytes(dataset)
    manifest_path = out_dir / "source_manifest_route_pilot_dev_v1.json"
    dataset_path = out_dir / "scenarios_route_pilot_dev_v2.json"
    for path, content in ((manifest_path, manifest_bytes), (dataset_path, dataset_bytes)):
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"frozen pilot output changed; choose a new version: {path}")
    _write_new_or_same(manifest_path, manifest_bytes)
    _write_new_or_same(dataset_path, dataset_bytes)
    print(json.dumps({"split": "development", "cases": len(dataset["cases"]),
                      "snapshot_id": dataset["snapshot_id"],
                      "formal_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
