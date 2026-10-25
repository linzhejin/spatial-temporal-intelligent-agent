"""Keep simulated event gold aligned with the actual road binding semantics."""

import json
from pathlib import Path

import networkx as nx

from experiments.independent_source_snapshot import active_closed_edges
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
