"""Audit edge attribute coverage, provenance and preference evidence readiness."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.network.build_edge_attribute_master import graph_fingerprint, geometry_hash  # noqa: E402
from spatial.edge_attributes import edge_key, source_manifest_fingerprint, validate_master  # noqa: E402


VERIFIED_STATUSES = {"field_verified", "institution_verified"}
ATTRIBUTE_FIELDS = {
    "terrain": ("grade_abs_pct", "grade_signed_pct", "slope_level", "elevation_gain_m"),
    "scenery": ("scenery_score", "scenery_level", "greenery", "shade", "water", "heritage"),
}


def _attribute_summary(records: list[dict], name: str) -> dict:
    values = [record[name] for record in records]
    statuses = Counter(value.get("verification_status", "source_only") for value in values)
    confidences = Counter(value.get("confidence", "unknown") for value in values)
    values_available = sum(
        any(value.get(field) is not None for field in ATTRIBUTE_FIELDS[name])
        for value in values
    )
    verified_usable = sum(
        value.get("verification_status") in VERIFIED_STATUSES
        and any(value.get(field) is not None for field in ATTRIBUTE_FIELDS[name])
        for value in values
    )
    return {
        "records": len(values),
        "values_available_records": values_available,
        "verified_usable_edges": verified_usable,
        "unverified_or_unknown_edges": len(values) - verified_usable,
        "verification_status_counts": dict(sorted(statuses.items())),
        "confidence_counts": dict(sorted(confidences.items())),
    }


def audit_attribute_master(master: dict, graph: nx.MultiDiGraph, sources: dict) -> dict:
    """Return a deterministic audit; never upgrades source-only values to verified."""
    validate_master(master)
    current_network = graph_fingerprint(graph)
    current_manifest = source_manifest_fingerprint(sources)
    graph_edges = {
        edge_key([int(u), int(v), int(key)]): (u, v, key)
        for u, v, key in graph.edges(keys=True)
    }

    records = master["records"]
    bound_edges: dict[str, str] = {}
    unknown_bindings = 0
    duplicate_bindings = 0
    stale_geometry = 0
    for record in records:
        for binding in record["network_bindings"]:
            edge_id = edge_key(binding["edge_id"])
            edge = graph_edges.get(edge_id)
            if edge is None:
                unknown_bindings += 1
                continue
            if edge_id in bound_edges:
                duplicate_bindings += 1
            bound_edges[edge_id] = record["segment_id"]
            u, v, key = edge
            if geometry_hash(graph, u, v, key) != binding["geometry_hash"]:
                stale_geometry += 1

    missing_edges = sorted(set(graph_edges) - set(bound_edges))
    terrain = _attribute_summary(records, "terrain")
    scenery = _attribute_summary(records, "scenery")
    blockers = []

    def add(code: str, count: int = 1) -> None:
        if count:
            blockers.append({"code": code, "count": count})

    network_matches = master.get("network_version") == current_network
    manifest_matches = master.get("source_manifest_fingerprint") == current_manifest
    add("network_version_mismatch", int(not network_matches))
    add("source_manifest_mismatch", int(not manifest_matches))
    add("unknown_edge_bindings", unknown_bindings)
    add("duplicate_edge_bindings", duplicate_bindings)
    add("stale_geometry_bindings", stale_geometry)
    add("unbound_graph_edges", len(missing_edges))
    add("terrain_preference_evidence_missing", int(terrain["verified_usable_edges"] == 0))
    add("scenery_preference_evidence_missing", int(scenery["verified_usable_edges"] == 0))
    add("terrain_attribute_gaps", terrain["unverified_or_unknown_edges"])
    add("scenery_attribute_gaps", scenery["unverified_or_unknown_edges"])

    structural_codes = {
        "network_version_mismatch", "source_manifest_mismatch", "unknown_edge_bindings",
        "duplicate_edge_bindings", "stale_geometry_bindings", "unbound_graph_edges",
    }
    structural_errors = sum(
        blocker["count"] for blocker in blockers if blocker["code"] in structural_codes
    )
    quality_gaps = sum(
        blocker["count"] for blocker in blockers if blocker["code"] not in structural_codes
    )
    return {
        "schema_version": 1,
        "report_type": "attribute_master_data_readiness",
        "ready": structural_errors == 0,
        "degraded": quality_gaps > 0,
        "blockers": blockers,
        "source_consistency": {
            "network_version_matches": network_matches,
            "source_manifest_matches": manifest_matches,
            "network_version": current_network,
            "master_network_version": master.get("network_version"),
            "source_manifest_fingerprint": current_manifest,
            "master_source_manifest_fingerprint": master.get("source_manifest_fingerprint"),
        },
        "coverage": {
            "graph_directed_edges": len(graph_edges),
            "master_records": len(records),
            "bound_edges": len(bound_edges),
            "missing_edges": len(missing_edges),
            "unknown_bindings": unknown_bindings,
            "duplicate_bindings": duplicate_bindings,
            "stale_geometry_bindings": stale_geometry,
            "edge_coverage_ratio": round(len(bound_edges) / len(graph_edges), 6)
            if graph_edges else 0.0,
        },
        "attributes": {"terrain": terrain, "scenery": scenery},
        "interpretation": (
            "ready checks structural currency and edge bindings only; degraded means the "
            "available preference attributes are not sufficiently verified for a formal study."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit route preference edge-attribute evidence")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/whu_road_network.graphml")
    parser.add_argument("--master", type=Path, default=ROOT / "data/edge_attribute_master.json")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/edge_attribute_sources.json")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "scripts/audit_output/attribute_master_data_readiness.json",
    )
    args = parser.parse_args(argv)

    graph = nx.read_graphml(args.graph, node_type=int)
    master = json.loads(args.master.read_text(encoding="utf-8"))
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    report = audit_attribute_master(master, graph, sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "ready": report["ready"],
        "degraded": report["degraded"],
        "coverage": report["coverage"],
        "terrain_verified_edges": report["attributes"]["terrain"]["verified_usable_edges"],
        "scenery_verified_edges": report["attributes"]["scenery"]["verified_usable_edges"],
        "blockers": report["blockers"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
