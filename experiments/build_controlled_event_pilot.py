"""Build two simulated-event development tasks from frozen local sources.

No production event is created. The packet is not a blind or formal test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import networkx as nx

from experiments.build_controlled_route_pilot import _json_bytes, _write_new_or_same
from experiments.build_controlled_spatial_snapshot import (
    build_spatial_snapshot, write_spatial_snapshot,
)
from experiments.controlled_event_cases import make_event_case
from experiments.controlled_event_manifest import (
    EVENT_PILOT_INPUTS, make_event_manifest, verify_event_manifest,
)
from experiments.controlled_gold_contract import validate_controlled_dataset
from experiments.independent_source_snapshot import mode_edge_evidence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("experiments/datasets/controlled_event_pilot_v1")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _snapshots(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if (not isinstance(source, dict)
            or source.get("format") != "controlled-simulated-events-v1"
            or source.get("scope") != "synthetic_development_only"
            or source.get("campus_truth") is not False
            or not isinstance(source.get("snapshots"), list)
            or not source["snapshots"]):
        raise ValueError("invalid simulated event source")
    snapshots = {item.get("id"): item for item in source["snapshots"]
                 if isinstance(item, dict)}
    if len(snapshots) != len(source["snapshots"]) or None in snapshots:
        raise ValueError("duplicate or invalid simulated event snapshot id")
    return snapshots


def build_event_pilot_dataset(
    graph: Any, anchors: dict[str, dict[str, Any]],
    evidence_by_mode: dict[str, dict[str, set[tuple[Any, Any, Any]]]],
    spec: dict[str, Any], source: dict[str, Any], manifest: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    """Generate every case; reject ambiguous, stale or non-effective events."""
    if (not isinstance(spec, dict)
            or spec.get("format") != "controlled-event-pilot-spec-v1"
            or spec.get("scope") != "synthetic_development_only"
            or not isinstance(spec.get("templates"), dict)
            or not isinstance(spec.get("cases"), list) or not spec["cases"]):
        raise ValueError("invalid event pilot specification")
    if (not isinstance(manifest, dict) or manifest.get("profile") != "event_pilot"
            or not _DIGEST.fullmatch(str(manifest.get("effective_inputs_sha256", "")))
            or not _DIGEST.fullmatch(manifest_sha256)):
        raise ValueError("verified event source manifest required")
    inputs = manifest.get("inputs")
    if (not isinstance(inputs, dict) or any(
            not _DIGEST.fullmatch(str((inputs.get(name) or {}).get("sha256", "")))
            for name in ("generator", "templates"))):
        raise ValueError("event generator and template digests required")
    snapshots = _snapshots(source)
    cases = []
    seen_ids = set()
    for entry in spec["cases"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("invalid event case specification")
        if entry["id"] in seen_ids:
            raise ValueError("duplicate event case id")
        seen_ids.add(entry["id"])
        template = spec["templates"].get(entry.get("template_id"))
        parameters = entry.get("parameters")
        if (not isinstance(template, dict)
                or not isinstance(template.get("text"), str)
                or not isinstance(parameters, dict)):
            raise ValueError("event case template or parameters missing")
        event_snapshot = snapshots.get(parameters.get("event_snapshot_id"))
        evidence = evidence_by_mode.get(parameters.get("mode"))
        if event_snapshot is None or evidence is None:
            raise ValueError("event snapshot or mode evidence missing")
        cases.append(make_event_case(
            graph, anchors, evidence, event_snapshot,
            template=template["text"], case_id=entry["id"],
            category=template["category"],
            template_group=template["template_group"],
            od_group=entry["od_group"],
            snapshot_id=manifest["effective_inputs_sha256"],
            generator_sha256=inputs["generator"]["sha256"],
            template_sha256=inputs["templates"]["sha256"],
            parameters=parameters,
            template_id=entry["template_id"],
        ))
    return {"protocol_version": "scenario-v2", "gold_layer": "controlled_graph",
            "split": "development", "repeats": 1,
            "source_manifest_sha256": manifest_sha256,
            "snapshot_id": manifest["effective_inputs_sha256"], "cases": cases}


def verify_event_pilot_dataset(
    dataset: dict[str, Any], graph: Any, anchors: dict[str, dict[str, Any]],
    evidence_by_mode: dict[str, dict[str, set[tuple[Any, Any, Any]]]],
    spec: dict[str, Any], source: dict[str, Any], manifest: dict[str, Any],
    manifest_sha256: str,
) -> None:
    """Rebuild all questions, event effects and labels before scoring."""
    expected = build_event_pilot_dataset(graph, anchors, evidence_by_mode, spec,
                                         source, manifest, manifest_sha256)
    if dataset != expected:
        raise ValueError("event pilot packet differs from case specification")
    validate_controlled_dataset(dataset, formal=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != ROOT.resolve():
        raise ValueError("event pilot root must match its imported code checkout")
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    if not out_dir.resolve().is_relative_to(root):
        raise ValueError("event pilot output must stay inside project root")
    spec_path = out_dir / "event_pilot_spec.json"
    event_path = out_dir / "event_snapshots_dev_v1.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source = json.loads(event_path.read_text(encoding="utf-8"))

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
        "graph": "data/whu_road_network.graphml", "poi": "data/pois.json",
        "boundary": "experiments/datasets/controlled_spatial_v1/boundary_wgs84.wkt",
        "anchors": "experiments/datasets/controlled_spatial_v1/walk_anchors.json",
        "event_snapshot": event_path.resolve().relative_to(root).as_posix(),
        "oracle_mode": "experiments/independent_source_snapshot.py",
        "oracle_event": "experiments/independent_source_snapshot.py",
        "oracle_route": "experiments/independent_route_oracle.py",
        "oracle_existence": "experiments/controlled_route_gold.py",
        "generator": "experiments/controlled_event_cases.py",
        "templates": spec_path.resolve().relative_to(root).as_posix(),
        "coordinate_transform": "spatial/coord_transform.py",
        "anchor_builder": "experiments/controlled_anchor_snapshot.py",
        "boundary_source": "config.py",
        "spatial_snapshot_builder": "experiments/build_controlled_spatial_snapshot.py",
        "dataset_builder": "experiments/build_controlled_event_pilot.py",
        "gold_contract": "experiments/controlled_gold_contract.py",
        "source_manifest_builder": "experiments/controlled_event_manifest.py",
        "scenario_evaluator": "experiments/scenario_evaluator.py",
        "route_scorer": "experiments/controlled_route_replay.py",
        "event_scorer": "experiments/controlled_event_replay.py",
        "collector": "experiments/scenario_live_adapter.py",
        "route_case_helpers": "experiments/controlled_route_cases.py",
        "source_hash_helpers": "experiments/controlled_source_manifest.py",
        "packet_helpers": "experiments/build_controlled_route_pilot.py",
        "observation_validator": "experiments/scenario_runner.py",
    }
    if set(source_paths) != EVENT_PILOT_INPUTS:
        raise ValueError("event source input list differs from contract")
    manifest = make_event_manifest(root, source_paths, {
        "graph_crs": "EPSG:4326", "boundary_crs": "EPSG:4326",
        "event_scope": "synthetic_development_only", "campus_truth": False,
    })
    if verify_event_manifest(manifest, root) != manifest["effective_inputs_sha256"]:
        raise ValueError("event manifest failed verification")
    manifest_bytes = _json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    modes = {entry["parameters"]["mode"] for entry in spec["cases"]}
    evidence = {
        mode: mode_edge_evidence(graph, mode, require_boundary=True,
                                 boundary_wkt=boundary_wkt)
        for mode in modes
    }
    dataset = build_event_pilot_dataset(graph, anchors, evidence, spec, source,
                                        manifest, manifest_sha)
    verify_event_pilot_dataset(dataset, graph, anchors, evidence, spec, source,
                               manifest, manifest_sha)
    dataset_bytes = _json_bytes(dataset)
    manifest_path = out_dir / "source_manifest_event_pilot_dev_v1.json"
    dataset_path = out_dir / "scenarios_event_pilot_dev_v2.json"
    for path, content in ((manifest_path, manifest_bytes), (dataset_path, dataset_bytes)):
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"frozen event output changed; choose a new version: {path}")
    _write_new_or_same(manifest_path, manifest_bytes)
    _write_new_or_same(dataset_path, dataset_bytes)
    print(json.dumps({"split": "development", "cases": len(dataset["cases"]),
                      "snapshot_id": dataset["snapshot_id"],
                      "formal_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
