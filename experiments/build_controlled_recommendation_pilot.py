"""Build reproducible, development-only POI recommendation cases.

No model, production database or external API is called. Graph ranking does
not certify opening hours, venue access, or final metres from a road node.
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
from experiments.controlled_recommendation_cases import make_recommendation_case
from experiments.controlled_recommendation_manifest import (
    RECOMMENDATION_INPUTS, make_recommendation_manifest,
    verify_recommendation_manifest,
)
from experiments.independent_source_snapshot import mode_edge_evidence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("experiments/datasets/controlled_recommendation_pilot_v2")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) +
            "\n").encode("utf-8")


def build_recommendation_pilot_dataset(
    graph: Any, anchors: dict[str, dict[str, Any]], pois: list[dict[str, Any]],
    spec: dict[str, Any], manifest: dict[str, Any], *,
    permitted_edge_ids: set[tuple[Any, Any, Any]],
    unknown_edge_ids: set[tuple[Any, Any, Any]],
) -> dict[str, Any]:
    """Derive every case from an explicit specification and complete POI pool."""
    if (not isinstance(spec, dict)
            or spec.get("format") != "controlled-recommendation-pilot-spec-v1"
            or spec.get("scope") != "development_only"
            or not isinstance(spec.get("templates"), dict)
            or not isinstance(spec.get("cases"), list) or not spec["cases"]):
        raise ValueError("invalid recommendation pilot specification")
    if (not isinstance(manifest, dict)
            or manifest.get("profile") != "recommendation_pilot"
            or not _DIGEST.fullmatch(str(manifest.get("effective_inputs_sha256", "")))):
        raise ValueError("recommendation source manifest required")
    inputs = manifest.get("inputs")
    if (not isinstance(inputs, dict)
            or any(not isinstance(inputs.get(key), dict)
                   or not _DIGEST.fullmatch(str(inputs[key].get("sha256", "")))
                   for key in ("generator", "templates"))):
        raise ValueError("recommendation generator and template digests required")
    seen = set()
    cases = []
    for entry in spec["cases"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("id"), str)
                or entry["id"] in seen):
            raise ValueError("invalid or duplicate recommendation case id")
        seen.add(entry["id"])
        template_id = entry.get("template_id")
        template = spec["templates"].get(template_id)
        if (not isinstance(template, dict)
                or not isinstance(template.get("text"), str)
                or not isinstance(template.get("template_group"), str)
                or not isinstance(entry.get("parameters"), dict)):
            raise ValueError("invalid recommendation template or parameters")
        cases.append(make_recommendation_case(
            graph, pois, anchors,
            permitted_edge_ids=permitted_edge_ids,
            unknown_edge_ids=unknown_edge_ids, closed_edge_ids=set(),
            templates={template_id: template["text"]}, template_id=template_id,
            case_id=entry["id"], template_group=template["template_group"],
            od_group=entry["od_group"],
            snapshot_id=manifest["effective_inputs_sha256"],
            generator_sha256=inputs["generator"]["sha256"],
            template_sha256=inputs["templates"]["sha256"],
            parameters=entry["parameters"],
        ))
    manifest_sha = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    return {"protocol_version": "scenario-v2", "gold_layer": "controlled_graph",
            "split": "development", "repeats": 1,
            "source_manifest_sha256": manifest_sha,
            "snapshot_id": manifest["effective_inputs_sha256"], "cases": cases}


def verify_recommendation_pilot_dataset(
    dataset: dict[str, Any], graph: Any,
    anchors: dict[str, dict[str, Any]], pois: list[dict[str, Any]],
    spec: dict[str, Any], manifest: dict[str, Any], *,
    permitted_edge_ids: set[tuple[Any, Any, Any]],
    unknown_edge_ids: set[tuple[Any, Any, Any]],
) -> None:
    """Rebuild all fields; neither a prompt nor gold rank may drift alone."""
    expected = build_recommendation_pilot_dataset(
        graph, anchors, pois, spec, manifest,
        permitted_edge_ids=permitted_edge_ids,
        unknown_edge_ids=unknown_edge_ids,
    )
    if dataset != expected:
        raise ValueError("recommendation pilot packet differs from case specification")
    validate_controlled_dataset(dataset, formal=False)


def _write_new_or_same(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"frozen recommendation output changed; use a new version: {path}")
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
    spec_path = out_dir / "recommendation_pilot_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    from config import CAMPUS_POLYS_GCJ
    from spatial.coord_transform import gcj02_to_wgs84

    graph = nx.read_graphml(root / "data/whu_road_network.graphml",
                            node_type=int, force_multigraph=True)
    poi_data = json.loads((root / "data/pois.json").read_text(encoding="utf-8"))
    if not isinstance(poi_data, dict) or not isinstance(poi_data.get("pois"), list):
        raise ValueError("POI source lacks pois list")
    pois = poi_data["pois"]
    boundary_wkt, anchor_data = build_spatial_snapshot(
        graph, pois, CAMPUS_POLYS_GCJ, to_wgs84=gcj02_to_wgs84,
    )
    spatial_dir = root / "experiments/datasets/controlled_spatial_v1"
    write_spatial_snapshot(spatial_dir / "boundary_wgs84.wkt",
                           spatial_dir / "walk_anchors.json", boundary_wkt, anchor_data)
    anchors = {item["poi_id"]: item for item in anchor_data["anchors"]}
    if len(anchors) != len(anchor_data["anchors"]):
        raise ValueError("duplicate spatial anchor")
    source_paths = {
        "graph": "data/whu_road_network.graphml", "poi": "data/pois.json",
        "boundary": "experiments/datasets/controlled_spatial_v1/boundary_wgs84.wkt",
        "anchors": "experiments/datasets/controlled_spatial_v1/walk_anchors.json",
        "boundary_source": "config.py",
        "coordinate_transform": "spatial/coord_transform.py",
        "spatial_snapshot_builder": "experiments/build_controlled_spatial_snapshot.py",
        "anchor_builder": "experiments/controlled_anchor_snapshot.py",
        "oracle_mode": "experiments/independent_source_snapshot.py",
        "oracle_route": "experiments/independent_route_oracle.py",
        "oracle_existence": "experiments/controlled_route_gold.py",
        "recommendation_gold": "experiments/controlled_recommendation_gold.py",
        "generator": "experiments/controlled_recommendation_cases.py",
        "templates": spec_path.resolve().relative_to(root).as_posix(),
        "dataset_builder": "experiments/build_controlled_recommendation_pilot.py",
        "gold_contract": "experiments/controlled_gold_contract.py",
        "scenario_evaluator": "experiments/scenario_evaluator.py",
        "controlled_source_manifest": "experiments/controlled_source_manifest.py",
        "requirements": "requirements.txt",
        "source_manifest_builder": "experiments/controlled_recommendation_manifest.py",
    }
    if set(source_paths) != RECOMMENDATION_INPUTS:
        raise ValueError("recommendation source input list differs from contract")
    manifest = make_recommendation_manifest(root, source_paths)
    verify_recommendation_manifest(manifest, root)
    evidence = mode_edge_evidence(graph, "walk", require_boundary=True,
                                  boundary_wkt=boundary_wkt)
    dataset = build_recommendation_pilot_dataset(
        graph, anchors, pois, spec, manifest,
        permitted_edge_ids=evidence["permitted"],
        unknown_edge_ids=evidence["unknown"],
    )
    verify_recommendation_pilot_dataset(
        dataset, graph, anchors, pois, spec, manifest,
        permitted_edge_ids=evidence["permitted"],
        unknown_edge_ids=evidence["unknown"],
    )
    outputs = (
        (out_dir / "source_manifest_recommendation_dev_v1.json", _json_bytes(manifest)),
        (out_dir / "scenarios_recommendation_dev_v2.json", _json_bytes(dataset)),
    )
    for path, content in outputs:
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"frozen recommendation output changed; use a new version: {path}")
    for path, content in outputs:
        _write_new_or_same(path, content)
    print(json.dumps({"split": "development", "cases": len(dataset["cases"]),
                      "snapshot_id": dataset["snapshot_id"],
                      "formal_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
