"""Keep simulated event gold aligned with the actual road binding semantics."""

import json
from pathlib import Path

import networkx as nx

from experiments.independent_source_snapshot import active_closed_edges
from experiments.controlled_event_manifest import verify_event_manifest
from spatial.road_conditions import resolve_condition_effects


ROOT = Path(__file__).resolve().parents[1]


def _closed_sets(version, snapshot):
    graph = nx.read_graphml(ROOT / "data/whu_road_network.graphml",
                            node_type=int, force_multigraph=True)
    source = json.loads((ROOT / "experiments/datasets" / version /
                         "event_snapshots_dev_v1.json").read_text(encoding="utf-8"))
    case = next(item for item in source["snapshots"] if item["id"] == snapshot)
    independent = active_closed_edges(graph, case, "walk")
    conditions = [dict(item, type="closure") for item in case["conditions"]]
    _, product, _ = resolve_condition_effects(graph, conditions, "walk", strict=True)
    return independent, product


def test_historical_detour_fixture_closes_more_product_edges_than_gold():
    independent, product = _closed_sets("controlled_event_pilot_v1",
                                        "simulated_detour_001")
    assert len(independent) == 1
    assert len(product) == 12
    assert independent < product


def test_corrected_event_snapshots_match_product_road_closures():
    for snapshot in ("simulated_detour_001", "simulated_blocked_001"):
        independent, product = _closed_sets("controlled_event_pilot_v2", snapshot)
        assert len(independent) == 2
        assert product == independent


def test_v3_packet_versions_scorer_and_collector_without_rewriting_v2():
    v3 = ROOT / "experiments/datasets/controlled_event_pilot_v3"
    manifest = json.loads((v3 / "source_manifest_event_pilot_dev_v1.json")
                          .read_text(encoding="utf-8"))
    dataset = json.loads((v3 / "scenarios_event_pilot_dev_v3.json")
                         .read_text(encoding="utf-8"))
    assert manifest["inputs"]["event_scorer"]["path"] == (
        "experiments/controlled_event_replay_v2.py"
    )
    assert manifest["inputs"]["collector"]["path"] == (
        "experiments/event_live_replay.py"
    )
    assert manifest["inputs"]["dataset_builder"]["path"] == (
        "experiments/build_controlled_event_pilot_v3.py"
    )
    assert verify_event_manifest(manifest, ROOT) == dataset["snapshot_id"]
    assert dataset["snapshot_id"] != json.loads((
        ROOT / "experiments/datasets/controlled_event_pilot_v2/"
        "scenarios_event_pilot_dev_v2.json"
    ).read_text(encoding="utf-8"))["snapshot_id"]
