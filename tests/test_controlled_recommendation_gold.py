"""Location-ranked recommendation gold must come from the full eligible pool."""

import networkx as nx
import pytest

from experiments.controlled_recommendation_gold import rank_reachable_activity_pois


def _fixture():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="footway", length=10)
    graph.add_edge(1, 3, key=0, highway="footway", length=25)
    graph.add_edge(1, 4, key=0, highway="footway", length=5)
    graph.add_edge(4, 3, key=0, highway="footway", length=7)
    graph.add_node(5)
    anchors = {
        name: {"poi_id": name, "name": name, "node_id": node,
               "access_status": "last_meter_unverified"}
        for name, node in {"start": 1, "far": 3, "near": 2,
                           "other": 4, "unreachable": 5}.items()
    }
    pois = [
        {"id": "far", "name": "Far", "activities": ["canteen_meal"],
         "activity_review_status": "source_supported", "activity_basis": "canteen"},
        {"id": "near", "name": "Near", "activities": ["canteen_meal"],
         "activity_review_status": "source_supported", "activity_basis": "canteen"},
        {"id": "other", "name": "Other", "activities": ["coffee"],
         "activity_review_status": "source_supported", "activity_basis": "coffee"},
        {"id": "unreachable", "name": "Unreachable", "activities": ["canteen_meal"],
         "activity_review_status": "source_supported", "activity_basis": "canteen"},
    ]
    permitted = {(u, v, key) for u, v, key in graph.edges(keys=True)}
    return graph, anchors, pois, permitted


def test_full_pool_rank_uses_graph_distance_before_display_limit():
    graph, anchors, pois, permitted = _fixture()
    ranked = rank_reachable_activity_pois(
        graph, pois, anchors, start_poi_id="start", activity="canteen_meal",
        mode="walk", limit=1, permitted_edge_ids=permitted,
        unknown_edge_ids=set(), closed_edge_ids=set(),
    )
    assert ranked["candidate_poi_ids"] == ["near"]
    assert ranked["all_reachable_poi_ids"] == ["near", "far"]
    assert ranked["graph_distance_m"] == {"near": 10.0, "far": 12.0}
    assert ranked["excluded"] == [{"poi_id": "unreachable", "reason": "unreachable"}]


def test_activity_filter_does_not_mix_drink_with_meal():
    graph, anchors, pois, permitted = _fixture()
    ranked = rank_reachable_activity_pois(
        graph, pois, anchors, start_poi_id="start", activity="coffee",
        mode="walk", limit=2, permitted_edge_ids=permitted,
        unknown_edge_ids=set(), closed_edge_ids=set(),
    )
    assert ranked["candidate_poi_ids"] == ["other"]


def test_unreviewed_activity_and_unknown_edge_fail_closed():
    graph, anchors, pois, permitted = _fixture()
    pois[0]["activity_review_status"] = "unknown"
    with pytest.raises(ValueError, match="activity evidence"):
        rank_reachable_activity_pois(
            graph, pois, anchors, start_poi_id="start", activity="canteen_meal",
            mode="walk", limit=2, permitted_edge_ids=permitted,
            unknown_edge_ids=set(), closed_edge_ids=set(),
        )
    pois[0]["activity_review_status"] = "source_supported"
    permitted.remove((1, 2, 0))
    with pytest.raises(ValueError, match="unknown edge"):
        rank_reachable_activity_pois(
            graph, pois, anchors, start_poi_id="start", activity="canteen_meal",
            mode="walk", limit=2, permitted_edge_ids=permitted,
            unknown_edge_ids={(1, 2, 0)}, closed_edge_ids=set(),
        )
