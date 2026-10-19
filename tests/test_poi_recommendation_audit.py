import networkx as nx

from experiments.audit_poi_recommendation_matrix import (
    _activity_corpus_counts,
    independent_reachable_candidate_rank,
    ranked_candidates_match,
)
from spatial.coord_transform import wgs84_to_gcj02


def _poi(poi_id, name, lng, *, lat=30.0, activity="basketball"):
    gcj_lng, gcj_lat = wgs84_to_gcj02(lng, lat)
    return {
        "id": poi_id,
        "name": name,
        "type": "sports",
        "subcategory": "basketball_court",
        "coordinates": {"lng": gcj_lng, "lat": gcj_lat},
        "activities": [activity],
    }


def _graph():
    graph = nx.MultiDiGraph()
    graph.graph["routing_boundary_wkt"] = (
        "POLYGON ((113.99 29.99, 114.01 29.99, 114.01 30.01, "
        "113.99 30.01, 113.99 29.99))"
    )
    for node, lng in ((1, 114.0), (2, 114.001), (3, 114.002), (4, 114.003)):
        graph.add_node(node, x=lng, y=30.0)
    for u, v in ((1, 2), (2, 3), (3, 4)):
        graph.add_edge(u, v, key=0, length=96.0, highway="footway")
        graph.add_edge(v, u, key=0, length=96.0, highway="footway")
    return graph


def test_independent_rank_uses_all_eligible_pois_not_input_importance_order():
    graph = _graph()
    origin = _poi("origin", "起点", 114.0)
    pois = [
        _poi("far", "远处热门球场", 114.003),
        _poi("near", "附近球场", 114.001),
        _poi("middle", "中间球场", 114.002),
    ]

    ranked = independent_reachable_candidate_rank(
        graph, origin, pois, max_distance_m=1000, max_access_snap_m=80, limit=2,
    )

    assert [item["poi_id"] for item in ranked["candidates"]] == ["near", "middle"]
    assert ranked["reachable_count"] == 3
    # A production response truncated to a popular far-away POI must fail this check.
    assert not ranked_candidates_match(
        ranked,
        [{"poi_id": "far", "network_distance_m": ranked["candidates"][0]["distance_m"]}],
        total_count=3,
    )


def test_independent_rank_excludes_unreachable_and_long_access_pois():
    graph = _graph()
    origin = _poi("origin", "起点", 114.0)
    pois = [
        _poi("near", "附近球场", 114.001),
        _poi("far_access", "远离道路的球场", 114.002, lat=30.001),
    ]

    ranked = independent_reachable_candidate_rank(
        graph, origin, pois, max_distance_m=1000, max_access_snap_m=80, limit=4,
    )

    assert [item["poi_id"] for item in ranked["candidates"]] == ["near"]
    assert ranked["reachable_count"] == 1


def test_rank_comparison_allows_only_one_meter_of_independent_rounding_delta():
    expected = {
        "reachable_count": 2,
        "candidates": [
            {"poi_id": "near", "distance_m": 101},
            {"poi_id": "next", "distance_m": 202},
        ],
    }

    assert ranked_candidates_match(
        expected,
        [
            {"poi_id": "near", "network_distance_m": 100},
            {"poi_id": "next", "network_distance_m": 203},
        ],
        total_count=2,
    )
    assert not ranked_candidates_match(
        expected,
        [
            {"poi_id": "near", "network_distance_m": 99},
            {"poi_id": "next", "network_distance_m": 202},
        ],
        total_count=2,
    )


def test_activity_corpus_uses_both_activity_and_allowed_subcategory():
    matching = _poi("court", "校内篮球场", 114.001)
    wrong_subcategory = {
        **_poi("gym", "普通体育馆", 114.002),
        "subcategory": "gym",
    }
    case = {"activities": ["basketball"],
            "subcategories": ["basketball_court"]}

    assert _activity_corpus_counts(
        [matching, wrong_subcategory], [case],
    ) == {"basketball": 1}
