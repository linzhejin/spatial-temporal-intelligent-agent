"""Build the corrected simulated-event packet without changing frozen v1/v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx

from experiments.build_controlled_event_pilot import (
    build_event_pilot_dataset, verify_event_pilot_dataset,
)
from experiments.build_controlled_route_pilot import _json_bytes, _write_new_or_same
from experiments.controlled_event_manifest import (
    make_event_manifest, verify_event_manifest,
)
from experiments.independent_source_snapshot import mode_edge_evidence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/datasets/controlled_event_pilot_v2"
DEFAULT_OUT = ROOT / "experiments/datasets/controlled_event_pilot_v3"


def build_v3_packet(*, root: Path = ROOT, out_dir: Path = DEFAULT_OUT) -> dict:
    """Recompute source hashes and gold; refuse changes to frozen outputs."""
    root = Path(root).resolve()
    if root != ROOT.resolve():
        raise ValueError("event packet must use its imported code checkout")
    out_dir = Path(out_dir).resolve()
    if not out_dir.is_relative_to(root):
        raise ValueError("event packet output must remain in the checkout")
    base_manifest = json.loads((BASE / "source_manifest_event_pilot_dev_v1.json")
                               .read_text(encoding="utf-8"))
    verify_event_manifest(base_manifest, root)
    spec_path = out_dir / "event_pilot_spec.json"
    snapshot_path = out_dir / "event_snapshots_dev_v1.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source = json.loads(snapshot_path.read_text(encoding="utf-8"))
    source_paths = {name: record["path"]
                    for name, record in base_manifest["inputs"].items()}
    source_paths.update({
        "templates": spec_path.relative_to(root).as_posix(),
        "event_snapshot": snapshot_path.relative_to(root).as_posix(),
        "dataset_builder": "experiments/build_controlled_event_pilot_v3.py",
        "event_scorer": "experiments/controlled_event_replay_v2.py",
        "collector": "experiments/event_live_replay.py",
    })
    manifest = make_event_manifest(root, source_paths, {
        "graph_crs": "EPSG:4326", "boundary_crs": "EPSG:4326",
        "event_scope": "synthetic_development_only", "campus_truth": False,
    })
    verify_event_manifest(manifest, root)
    manifest_bytes = _json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    inputs = manifest["inputs"]
    graph = nx.read_graphml(root / inputs["graph"]["path"],
                            node_type=int, force_multigraph=True)
    anchor_data = json.loads((root / inputs["anchors"]["path"])
                             .read_text(encoding="utf-8"))
    anchors = {item["poi_id"]: item for item in anchor_data["anchors"]}
    if len(anchors) != len(anchor_data["anchors"]):
        raise ValueError("duplicate frozen anchor id")
    boundary = (root / inputs["boundary"]["path"]).read_text(encoding="utf-8")
    evidence = {
        mode: mode_edge_evidence(graph, mode, require_boundary=True,
                                 boundary_wkt=boundary)
        for mode in {item["parameters"]["mode"] for item in spec["cases"]}
    }
    dataset = build_event_pilot_dataset(graph, anchors, evidence, spec, source,
                                        manifest, manifest_sha)
    verify_event_pilot_dataset(dataset, graph, anchors, evidence, spec, source,
                               manifest, manifest_sha)
    outputs = (
        (out_dir / "source_manifest_event_pilot_dev_v1.json", manifest_bytes),
        (out_dir / "scenarios_event_pilot_dev_v3.json", _json_bytes(dataset)),
    )
    for target, content in outputs:
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"frozen event output changed: {target}")
    for target, content in outputs:
        _write_new_or_same(target, content)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    dataset = build_v3_packet(out_dir=args.out_dir)
    print(json.dumps({"split": "development", "cases": len(dataset["cases"]),
                      "snapshot_id": dataset["snapshot_id"],
                      "formal_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
