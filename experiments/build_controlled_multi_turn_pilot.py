"""Build a source-frozen two-turn route revision packet for development only.

This does not call an LLM or a production service and is not formal test data.
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
from experiments.controlled_multi_turn_cases import make_multi_turn_route_case
from experiments.controlled_multi_turn_manifest import (
    MULTI_TURN_INPUTS, make_multi_turn_manifest, verify_multi_turn_manifest,
)
from experiments.independent_source_snapshot import mode_edge_evidence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("experiments/datasets/controlled_multi_turn_pilot_v2")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build_multi_turn_pilot_dataset(
    graph: Any, anchors: dict[str, dict[str, Any]], spec: dict[str, Any],
    manifest: dict[str, Any], manifest_sha256: str,
    evidence_by_mode: dict[str, dict[str, set[tuple[Any, Any, Any]]]],
) -> dict[str, Any]:
    """Regenerate every case, failing rather than skipping uncertain routes."""
    if (not isinstance(spec, dict)
            or spec.get("format") != "controlled-multi-turn-pilot-spec-v2"
            or spec.get("scope") != "development_only"
            or not isinstance(spec.get("templates"), dict)
            or not isinstance(spec.get("cases"), list) or not spec["cases"]):
        raise ValueError("invalid multi-turn pilot specification")
    if (not isinstance(manifest, dict) or manifest.get("profile") != "multi_turn_pilot"
            or not _DIGEST.fullmatch(str(manifest.get("effective_inputs_sha256", "")))
            or not _DIGEST.fullmatch(manifest_sha256)):
        raise ValueError("verified route pilot source manifest required")
    inputs = manifest.get("inputs")
    if (not isinstance(inputs, dict) or any(
            not isinstance(inputs.get(name), dict)
            or not _DIGEST.fullmatch(str(inputs[name].get("sha256", "")))
            for name in ("generator", "templates"))):
        raise ValueError("generator and template source digests required")
    seen_ids: set[str] = set()
    cases = []
    for entry in spec["cases"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("id"), str)
                or not entry["id"].strip() or not isinstance(entry.get("od_group"), str)
                or not entry["od_group"].strip()
                or not isinstance(entry.get("parameters"), dict)):
            raise ValueError("invalid multi-turn pilot case specification")
        if entry["id"] in seen_ids:
            raise ValueError("duplicate multi-turn pilot case id")
        seen_ids.add(entry["id"])
        cases.append(make_multi_turn_route_case(
            graph, anchors, evidence_by_mode, templates=spec["templates"],
            case_id=entry["id"], template_group="route_revision",
            od_group=entry["od_group"],
            snapshot_id=manifest["effective_inputs_sha256"],
            generator_sha256=inputs["generator"]["sha256"],
            template_sha256=inputs["templates"]["sha256"],
            parameters=entry["parameters"],
        ))
    return {"protocol_version": "scenario-v2", "gold_layer": "controlled_graph",
            "split": "development", "repeats": 1,
            "source_manifest_sha256": manifest_sha256,
            "snapshot_id": manifest["effective_inputs_sha256"], "cases": cases}


def verify_multi_turn_pilot_dataset(
    dataset: dict[str, Any], graph: Any, anchors: dict[str, dict[str, Any]],
    spec: dict[str, Any], manifest: dict[str, Any], manifest_sha256: str,
    evidence_by_mode: dict[str, dict[str, set[tuple[Any, Any, Any]]]],
) -> None:
    """Reject any edit to prompts, groups, constraints, or either turn's gold."""
    expected = build_multi_turn_pilot_dataset(
        graph, anchors, spec, manifest, manifest_sha256, evidence_by_mode,
    )
    if dataset != expected:
        raise ValueError("multi-turn pilot packet differs from case specification")
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
    spec_path = out_dir / "multi_turn_pilot_spec.json"
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
        "route_case_helpers": "experiments/controlled_route_cases.py",
        "route_scorer": "experiments/controlled_route_replay.py",
        "observation_projection": "experiments/scenario_runner.py",
        "controlled_source_manifest": "experiments/controlled_source_manifest.py",
        "scenario_evaluator": "experiments/scenario_evaluator.py",
        "requirements": "requirements.txt",
        "generator": "experiments/controlled_multi_turn_cases.py",
        "templates": spec_path.resolve().relative_to(root).as_posix(),
        "coordinate_transform": "spatial/coord_transform.py",
        "anchor_builder": "experiments/controlled_anchor_snapshot.py",
        "anchors": "experiments/datasets/controlled_spatial_v1/walk_anchors.json",
        "boundary_source": "config.py",
        "spatial_snapshot_builder": "experiments/build_controlled_spatial_snapshot.py",
        "dataset_builder": "experiments/build_controlled_multi_turn_pilot.py",
        "gold_contract": "experiments/controlled_gold_contract.py",
        "source_manifest_builder": "experiments/controlled_multi_turn_manifest.py",
    }
    if set(source_paths) != MULTI_TURN_INPUTS:
        raise ValueError("pilot source input list differs from contract")
    manifest = make_multi_turn_manifest(root, source_paths)
    if verify_multi_turn_manifest(manifest, root) != manifest["effective_inputs_sha256"]:
        raise ValueError("source manifest failed verification")
    manifest_bytes = _json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    modes = {entry["parameters"][name] for entry in spec["cases"]
             for name in ("initial_mode", "revised_mode")}
    evidence = {mode: mode_edge_evidence(
        graph, mode, require_boundary=True, boundary_wkt=boundary_wkt,
    ) for mode in modes}
    dataset = build_multi_turn_pilot_dataset(
        graph, anchors, spec, manifest, manifest_sha, evidence,
    )
    verify_multi_turn_pilot_dataset(
        dataset, graph, anchors, spec, manifest, manifest_sha, evidence,
    )
    outputs = (
        (out_dir / "source_manifest_multi_turn_dev_v1.json", manifest_bytes),
        (out_dir / "scenarios_multi_turn_dev_v2.json", _json_bytes(dataset)),
    )
    if any(path.exists() and path.read_bytes() != content for path, content in outputs):
        raise ValueError("frozen pilot output changed; choose a new version")
    for path, content in outputs:
        _write_new_or_same(path, content)
    print(json.dumps({"split": "development", "cases": len(dataset["cases"]),
                      "snapshot_id": dataset["snapshot_id"],
                      "formal_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
