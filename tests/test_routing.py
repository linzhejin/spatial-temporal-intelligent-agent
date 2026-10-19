import inspect
import pytest
import networkx as nx
import random

import spatial.routing as routing_module
from spatial.routing import (
    resolve_weights,
    _normalize_lengths,
    _compute_edge_cost,
    _filter_by_constraints,
    _budgeted_ordered_path,
    compute_budgeted_multimodal_route,
    compute_budgeted_ordered_route,
    compute_route,
)


def _set_campus_coords(G):
    """给测试图所有节点设置校园内坐标（珞珈山附近），避免被校外边过滤误删。"""
    for n in G.nodes():
        G.nodes[n]["x"] = 114.360
        G.nodes[n]["y"] = 30.535
    return G


def _build_mock_graph():
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(6))
    _set_campus_coords(G)
    edges = [
        (0, 1, {"length": 100.0, "slope_level": 2, "scenery_level": 3}),
        (1, 2, {"length": 200.0, "slope_level": 4, "scenery_level": 5}),
        (2, 3, {"length": 150.0, "slope_level": 5, "scenery_level": 4}),
        (3, 4, {"length": 180.0, "slope_level": 1, "scenery_level": 5}),
        (4, 5, {"length": 120.0, "slope_level": 3, "scenery_level": 2}),
        (0, 2, {"length": 350.0, "slope_level": 2, "scenery_level": 3}),
        (1, 4, {"length": 400.0, "slope_level": 3, "scenery_level": 4}),
    ]
    for u, v, data in edges:
        G.add_edge(u, v, **data)
        G.add_edge(v, u, **data)
    return G


@pytest.fixture
def mock_graph():
    return _build_mock_graph()


@pytest.fixture
def detour_graph():
    graph = nx.MultiDiGraph()
    for node in (1, 2, 4):
        graph.add_node(node, x=114.360, y=30.535)
    graph.add_edge(1, 4, length=100.0, slope_level=5, scenery_level=1)
    graph.add_edge(1, 2, length=80.0, slope_level=2, scenery_level=5)
    graph.add_edge(2, 4, length=80.0, slope_level=2, scenery_level=5)
    return graph


def test_shortest_strategy_runs_one_search_and_returns_identical_paths(mock_graph):
    result = compute_route(
        mock_graph, 0, 5, strategy_name="shortest", road_conditions=[]
    )

    assert result["applied_weights"] == {
        "distance": 1.0,
        "slope": 0.0,
        "scenery": 0.0,
    }
    assert result["recommended_edges"] == result["shortest_edges"]
    assert result["recommended_length_m"] == result["shortest_length_m"]
    assert result["dijkstra_runs"] == 1


def test_route_calculation_fails_closed_on_unmatched_active_road_event(mock_graph):
    from spatial.road_conditions import RoadConditionBindingError

    stale_closure = {
        "id": "stale-closure", "type": "closure", "start_time": 0, "end_time": 0,
        "edge": {"u": 999001, "v": 999002, "key": 0,
                 "snap": {"lng": 114.37, "lat": 30.54}},
    }

    with pytest.raises(RoadConditionBindingError):
        compute_route(mock_graph, 0, 5, strategy_name="shortest",
                      road_conditions=[stale_closure])


def test_weighted_strategy_never_returns_route_beyond_cap(detour_graph, monkeypatch):
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    result = compute_route(
        detour_graph,
        1,
        4,
        strategy_name="scenery",
        weights={"distance": 0.5, "slope": 0.1, "scenery": 0.4},
        detour_cap=1.25,
        road_conditions=[],
    )

    assert result["recommended_length_m"] <= result["shortest_length_m"] * 1.25 + 0.1
    assert result["length_capped"] is True


def test_hard_distance_budget_uses_feasible_shortest_when_scenic_detour_exceeds_it(
    detour_graph, monkeypatch,
):
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    result = compute_route(
        detour_graph, 1, 4,
        constraints={"max_distance_m": 120},
        weights={"distance": 0.3, "slope": 0.1, "scenery": 0.6},
        strategy_name="scenery", detour_cap=2.0, road_conditions=[],
    )

    assert result["shortest_length_m"] == 100.0
    assert result["recommended_length_m"] == 100.0
    assert result["recommended_edges"] == result["shortest_edges"]
    assert result["hard_budget_fallback"] is True


def test_time_budget_does_not_offer_slower_shortest_path_as_compliant(
    detour_graph, monkeypatch,
):
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    result = compute_route(
        detour_graph, 1, 4,
        constraints={"max_duration_s": 150},
        weights={"distance": 0.3, "slope": 0.1, "scenery": 0.6},
        strategy_name="scenery", detour_cap=2.0, road_conditions=[],
    )

    assert result["recommended_length_m"] == 160.0
    assert result["duration_min"] * 60 <= 153
    assert result["shortest_duration_min"] * 60 > 153
    assert result["shortest_baseline_feasible"] is False


def test_budget_search_finds_third_route_when_preferred_and_shortest_both_fail(
    detour_graph, monkeypatch,
):
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    detour_graph.add_node(3, x=114.360, y=30.535)
    detour_graph.add_edge(1, 3, length=65.0, slope_level=2, scenery_level=1)
    detour_graph.add_edge(3, 4, length=65.0, slope_level=2, scenery_level=1)

    result = compute_route(
        detour_graph, 1, 4,
        constraints={"max_distance_m": 150, "max_duration_s": 120},
        weights={"distance": 0.3, "slope": 0.1, "scenery": 0.6},
        strategy_name="scenery", detour_cap=2.0, road_conditions=[],
    )

    assert result["recommended"] == [1, 3, 4]
    assert result["recommended_length_m"] == 130.0
    assert result["duration_min"] * 60 <= 123
    assert result["hard_budget_fallback"] is True


def test_ordered_budget_search_applies_time_budget_to_whole_waypoint_chain():
    graph = nx.MultiDiGraph()
    for node in range(5):
        graph.add_node(node, x=114.360, y=30.535)
    graph.add_edge(0, 1, length=100.0, slope_level=1, highway="steps")
    graph.add_edge(0, 2, length=120.0, slope_level=1, highway="path")
    graph.add_edge(2, 1, length=120.0, slope_level=1, highway="path")
    graph.add_edge(1, 3, length=100.0, slope_level=1, highway="path")

    result = _budgeted_ordered_path(
        graph, [0, 1, 3], mode="walk", edge_weight=routing_module._length_weight,
        duration_penalty_map={}, max_distance_m=350.0, max_duration_s=270.0,
    )

    assert result is not None
    assert [leg["nodes"] for leg in result["legs"]] == [[0, 2, 1], [1, 3]]
    assert result["distance_m"] == 340.0
    assert result["duration_min"] * 60 <= 288.0


def test_ordered_budget_search_returns_none_only_when_global_budgets_are_infeasible():
    graph = nx.MultiDiGraph()
    for node in range(4):
        graph.add_node(node, x=114.360, y=30.535)
    graph.add_edge(0, 1, length=100.0, slope_level=1, highway="steps")
    graph.add_edge(0, 2, length=120.0, slope_level=1, highway="path")
    graph.add_edge(2, 1, length=120.0, slope_level=1, highway="path")
    graph.add_edge(1, 3, length=100.0, slope_level=1, highway="path")

    result = _budgeted_ordered_path(
        graph, [0, 1, 3], mode="walk", edge_weight=routing_module._length_weight,
        duration_penalty_map={}, max_distance_m=300.0, max_duration_s=100.0,
    )

    assert result is None


def test_ordered_budget_route_uses_filtered_graph_and_shared_eta_rules(monkeypatch):
    graph = nx.MultiDiGraph()
    for node in range(4):
        graph.add_node(node, x=114.360, y=30.535)
    graph.add_edge(0, 1, length=100.0, slope_level=1, highway="steps")
    graph.add_edge(0, 2, length=120.0, slope_level=1, highway="path")
    graph.add_edge(2, 1, length=120.0, slope_level=1, highway="path")
    graph.add_edge(1, 3, length=100.0, slope_level=1, highway="path")
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)

    result = compute_budgeted_ordered_route(
        graph, [0, 1, 3], strategy_name="shortest", mode="walk",
        constraints={"max_distance_m": 350, "max_duration_s": 270},
        road_conditions=[], weather_info=None,
    )

    assert result is not None
    assert [leg["nodes"] for leg in result["legs"]] == [[0, 2, 1], [1, 3]]
    assert result["distance_m"] == 340.0
    assert result["duration_min"] == 4.5
    assert result["legs"][0]["slope_avg_recommended"] == 1.0
    assert result["legs"][0]["degraded"] is True


def test_multimodal_budget_search_optimizes_all_modes_under_whole_trip_time():
    graph = nx.MultiDiGraph()
    for node in range(4):
        graph.add_node(node, x=114.360 + node * 0.0001, y=30.535)
    graph.add_edge(0, 1, key=0, length=100.0, highway="steps", slope_level=1)
    graph.add_edge(0, 2, key=0, length=70.0, highway="path", slope_level=1)
    graph.add_edge(2, 1, key=0, length=70.0, highway="path", slope_level=1)
    graph.add_edge(1, 3, key=0, length=100.0, highway="residential", slope_level=5)

    result = compute_budgeted_multimodal_route(
        graph,
        [{"mode": "walk", "start_node": 0, "end_node": 1},
         {"mode": "bike", "start_node": 1, "end_node": 3}],
        constraints={"max_distance_m": 250.0, "max_duration_s": 240.0},
        strategy_name="shortest", road_conditions=[],
    )

    assert result is not None
    assert [leg["nodes"] for leg in result["legs"]] == [[0, 2, 1], [1, 3]]
    assert sum(leg["distance_m"] for leg in result["legs"]) == 240.0
    assert sum(leg["duration_min"] for leg in result["legs"]) * 60 <= 243.0


def test_multimodal_budget_search_returns_none_when_combined_distance_is_impossible():
    graph = nx.MultiDiGraph()
    for node in range(3):
        graph.add_node(node, x=114.360 + node * 0.0001, y=30.535)
    graph.add_edge(0, 1, key=0, length=100.0, highway="path", slope_level=1)
    graph.add_edge(1, 2, key=0, length=100.0, highway="residential", slope_level=1)

    result = compute_budgeted_multimodal_route(
        graph,
        [{"mode": "walk", "start_node": 0, "end_node": 1},
         {"mode": "bike", "start_node": 1, "end_node": 2}],
        constraints={"max_distance_m": 150.0},
        strategy_name="shortest", road_conditions=[],
    )

    assert result is None


def test_budget_search_retains_exact_parallel_edge_for_fastest_feasible_choice(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.360, y=30.535)
    graph.add_node(2, x=114.361, y=30.535)
    graph.add_edge(1, 2, key=0, length=100.0, slope_level=5, scenery_level=1)
    graph.add_edge(1, 2, key=1, length=130.0, slope_level=2, scenery_level=1)
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)

    result = compute_route(
        graph, 1, 2, strategy_name="shortest",
        constraints={"max_duration_s": 120}, road_conditions=[],
    )

    assert result["shortest_edges"] == [(1, 2, 0)]
    assert result["recommended_edges"] == [(1, 2, 1)]
    assert result["recommended_length_m"] == 130.0
    assert result["shortest_baseline_feasible"] is False


def test_budget_search_matches_public_rounded_duration_at_boundary(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.360, y=30.535)
    graph.add_node(2, x=114.361, y=30.535)
    graph.add_edge(1, 2, key=0, length=100.0, slope_level=5, scenery_level=1)
    # 158.75 m at 4.5 km/h takes 127 s, displayed as 2.1 min (126 s).
    graph.add_edge(1, 2, key=1, length=158.75, slope_level=2, scenery_level=1)
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)

    result = compute_route(
        graph, 1, 2, strategy_name="shortest",
        constraints={"max_duration_s": 123}, road_conditions=[],
    )

    assert result["recommended_edges"] == [(1, 2, 1)]
    assert result["duration_min"] == 2.1
    assert result["duration_min"] * 60 <= 123 + 3


def test_budget_search_matches_exhaustive_small_graph_oracle():
    randomizer = random.Random(20261017)

    def weight(_u, _v, edge_data):
        return next(iter(edge_data.values()))["soft_cost"]

    for _ in range(30):
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(range(5))
        for u in range(4):
            for v in range(u + 1, 5):
                if v == u + 1 or randomizer.random() < 0.6:
                    for _key in range(1 + (randomizer.random() < 0.2)):
                        graph.add_edge(
                            u, v, length=randomizer.randint(20, 100),
                            slope_level=randomizer.choice([2, 4, 5]),
                            soft_cost=randomizer.randint(1, 15),
                        )
        distance_limit = randomizer.randint(100, 320)
        duration_limit = randomizer.randint(100, 400)
        exhaustive = []
        for edges in nx.all_simple_edge_paths(graph, 0, 4):
            distance = sum(graph[u][v][key]["length"] for u, v, key in edges)
            seconds = sum(
                routing_module._edge_duration_min(graph[u][v][key], "walk") * 60
                for u, v, key in edges
            )
            if (distance <= distance_limit + 0.15
                    and round(seconds / 60, 1) * 60 <= duration_limit + 3):
                exhaustive.append(sum(graph[u][v][key]["soft_cost"] for u, v, key in edges))

        result = routing_module._budgeted_path(
            graph, 0, 4, mode="walk", penalty_map={}, edge_weight=weight,
            max_distance_m=distance_limit, max_duration_s=duration_limit,
        )
        if not exhaustive:
            assert result is None
            continue
        assert result is not None
        _, selected_edges = result
        selected_cost = sum(graph[u][v][key]["soft_cost"] for u, v, key in selected_edges)
        assert selected_cost == min(exhaustive)


@pytest.mark.parametrize(
    ("mode", "highway", "slope", "speed_factor"),
    [("walk", "steps", 3, 0.4), ("bike", "residential", 5, 0.4)],
)
def test_eta_applies_physical_mode_slowdown_once(mode, highway, slope, speed_factor, monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.36, y=30.53)
    graph.add_node(2, x=114.361, y=30.53)
    graph.add_edge(1, 2, key=0, length=100.0, highway=highway, slope_level=slope)
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)

    result = compute_route(graph, 1, 2, mode=mode, strategy_name="shortest", road_conditions=[])

    expected = round(
        100.0 / (routing_module.MODE_SPEEDS_KMH[mode] * 1000 / 60 * speed_factor), 1,
    )
    assert result["duration_min"] == expected
    assert result["shortest_duration_min"] == expected


def test_budget_search_limit_is_unknown_not_a_false_no_route(detour_graph, monkeypatch):
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    monkeypatch.setattr(routing_module, "_MAX_CONSTRAINED_LABELS", 1)
    detour_graph.add_node(3, x=114.360, y=30.535)
    detour_graph.add_edge(1, 3, length=65.0, slope_level=2, scenery_level=1)
    detour_graph.add_edge(3, 4, length=65.0, slope_level=2, scenery_level=1)

    with pytest.raises(routing_module.BudgetSearchIncomplete):
        compute_route(
            detour_graph, 1, 4,
            constraints={"max_distance_m": 150, "max_duration_s": 120},
            weights={"distance": 0.3, "slope": 0.1, "scenery": 0.6},
            strategy_name="scenery", detour_cap=2.0, road_conditions=[],
        )


def test_shortest_still_obeys_hard_closure(detour_graph):
    result = compute_route(
        detour_graph,
        1,
        4,
        strategy_name="shortest",
        constraints={"slope": "avoid"},
        road_conditions=[],
    )

    assert (1, 4, 0) not in result["recommended_edges"]


def test_certified_exact_shortest_path_reuse_skips_dijkstra_but_rebuilds_route_metrics(
    monkeypatch,
):
    assert "reuse_exact_shortest_edges" in inspect.signature(compute_route).parameters
    graph = nx.MultiDiGraph()
    for node, x in ((1, 114.360), (2, 114.361), (3, 114.362)):
        graph.add_node(node, x=x, y=30.530)
    graph.add_edge(1, 2, key=0, length=10.0, highway="footway", slope_level=2)
    graph.add_edge(2, 3, key=0, length=12.0, highway="footway", slope_level=3)

    def unexpected_search(*_args, **_kwargs):
        raise AssertionError("certified path reuse should skip Dijkstra")

    monkeypatch.setattr(routing_module, "_run_path", unexpected_search)
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)

    result = compute_route(
        graph, 1, 3, strategy_name="shortest", road_conditions=[],
        reuse_exact_shortest_edges=[(1, 2, 0), (2, 3, 0)],
    )

    assert result["recommended_edges"] == [(1, 2, 0), (2, 3, 0)]
    assert result["shortest_edges"] == [(1, 2, 0), (2, 3, 0)]
    assert result["recommended_length_m"] == 22.0
    assert result["dijkstra_runs"] == 0
    assert result["duration_min"] > 0


def test_exact_edge_exclusion_uses_other_parallel_key():
    graph = nx.MultiDiGraph()
    for node in (1, 2):
        graph.add_node(node, x=114.36, y=30.535)
    graph.add_edge(1, 2, key=0, length=50, highway="footway")
    graph.add_edge(1, 2, key=1, length=80, highway="footway")

    result = compute_route(
        graph, 1, 2, strategy_name="shortest", road_conditions=[],
        constraints={"excluded_edge_ids": ["[1,2,0]"]},
    )

    assert result["recommended_edges"] == [(1, 2, 1)]


class TestNormalizeWeights:
    def test_resolve_weights_default(self):
        result = resolve_weights(None)
        assert result == {"distance": 0.90, "slope": 0.05, "scenery": 0.05}
        assert abs(sum(result.values()) - 1.0) < 1e-9

    @pytest.mark.parametrize("raw,expected_min", [
        ({"distance": 0.001, "slope": 0.001, "scenery": 0.001}, 0.05),
        ({"distance": 0.99, "slope": 0.5, "scenery": 0.5}, 0.8),
    ])
    def test_resolve_weights_bounds(self, raw, expected_min):
        result = resolve_weights(raw)
        for v in result.values():
            assert v >= 0.05 - 1e-9
            assert v <= 0.9 + 1e-9
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_resolve_weights_boundary_normalized(self):
        raw = {"distance": 0.8, "slope": 0.8, "scenery": 0.8}
        result = resolve_weights(raw)
        for k in ("distance", "slope", "scenery"):
            assert abs(result[k] - 1 / 3) < 1e-9
        assert abs(sum(result.values()) - 1.0) < 1e-9


class TestThreeFactorNormalization:
    def test_normalize_lengths(self, mock_graph):
        max_len, norm_map = _normalize_lengths(mock_graph)
        assert max_len == 400.0
        assert len(norm_map) > 0
        for (u, v, k), val in norm_map.items():
            assert 0.0 <= val <= 1.0

    def test_compute_edge_cost_three_components(self, mock_graph):
        weights = {"distance": 0.5, "slope": 0.2, "scenery": 0.3}
        data = {"length": 200.0, "slope_level": 2, "scenery_level": 4}
        norm_length = 200.0 / 400.0
        cost, degraded = _compute_edge_cost(data, norm_length, weights)
        expected = 0.5 * 0.5 + 0.2 * (2 / 5) + 0.3 * (1 - 4 / 5)
        assert abs(cost - expected) < 1e-9
        assert degraded is False


class TestFilterConstraints:
    def test_filter_slope_avoid_removes_level5(self, mock_graph):
        G_filtered, status, penalty = _filter_by_constraints(
            mock_graph, {"slope": "avoid"}
        )
        assert status in ("filtered", "degraded_slope")
        for u, v, k, data in G_filtered.edges(keys=True, data=True):
            if data.get("slope_level") == 5 and status == "filtered":
                pytest.fail("slope_level=5 路段未被过滤")

    def test_filter_no_filter_when_slope_normal(self, mock_graph):
        G_filtered, status, penalty = _filter_by_constraints(
            mock_graph, {"slope": "normal"}
        )
        assert status == "no_filter"
        assert penalty == {}

    def test_avoid_steps_is_a_hard_filter(self):
        graph = nx.MultiDiGraph()
        for node in (1, 2, 3):
            graph.add_node(node, x=114.360, y=30.535)
        graph.add_edge(1, 3, length=100.0, highway="steps", slope_level=2, scenery_level=3)
        graph.add_edge(1, 2, length=80.0, highway="footway", slope_level=2, scenery_level=3)
        graph.add_edge(2, 3, length=80.0, highway="footway", slope_level=2, scenery_level=3)

        result = compute_route(
            graph, 1, 3, constraints={"avoid_steps": True}, strategy_name="shortest",
            road_conditions=[],
        )

        assert result["recommended_edges"] == [(1, 2, 0), (2, 3, 0)]
        assert "steps_avoid" in result["filter_status"]
        assert "steps_unverified" in result["filter_status"]


class TestDegradeMechanism:
    def test_degraded_flag_on_missing_attrs(self):
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, length=100.0)
        weights = {"distance": 0.5, "slope": 0.2, "scenery": 0.3}
        data = {"length": 100.0}
        cost, degraded = _compute_edge_cost(data, 0.1, weights)
        assert degraded is True
        assert cost > 0


class TestComputeRouteErrors:
    def test_no_path_raises_value_error(self):
        G = nx.MultiDiGraph()
        G.add_node(0)
        G.add_node(1)
        G.add_edge(0, 0, length=10.0)
        with pytest.raises(ValueError, match="校内"):
            compute_route(G, 0, 1)

    def test_compute_route_success(self, mock_graph):
        result = compute_route(mock_graph, 0, 5)
        assert "recommended" in result
        assert "shortest" in result
        assert result["recommended"][0] == 0
        assert result["recommended"][-1] == 5
        assert result["recommended_length_m"] > 0
        assert result["shortest_length_m"] > 0
        assert "applied_weights" in result
        assert 0.0 <= result["overlap_rate"] <= 1.0


# ---------------------------------------------------------------------------
# 出行方式（walk / bike / drive）测试
# ---------------------------------------------------------------------------

from spatial.routing import (  # noqa: E402
    DEFAULT_WEIGHTS,
    TRAVEL_MODES,
    MODE_SPEEDS_KMH,
    MODE_DEFAULT_WEIGHTS,
    MODE_OUTSIDE_ROAD_PENALTY,
    normalize_mode,
    _edge_highway_tags,
    filter_graph_for_mode,
    estimate_duration_min,
    _estimate_route_duration_min,
    nearest_in_mode_node,
    compute_route_with_annotations,
)


def _build_mode_mock_graph():
    """含 footway/steps/service/residential/corridor/path/混合标签 的小图（双向边）。"""
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(10))
    _set_campus_coords(G)
    edges = [
        # (u, v, highway, length, slope_level)
        (0, 1, "footway", 100.0, 2),
        (1, 2, "steps", 100.0, 3),                     # 纯台阶：bike/drive 均移除
        (2, 3, "['steps', 'footway']", 100.0, 3),      # 多标签台阶：bike/drive 均移除
        (3, 4, "service", 100.0, 2),
        (4, 5, "residential", 100.0, 5),               # slope5：bike penalty 3.0
        (5, 6, "corridor", 100.0, 3),                  # 纯走廊：bike/drive 均移除
        (6, 7, "path", 100.0, 3),                      # 步行小径：bike 保留、drive 移除
        (0, 7, "['service', 'footway']", 500.0, 2),    # 人车共存：bike/drive 均保留
        (1, 6, "primary", 600.0, 2),
        (2, 6, "service", 250.0, 4),                   # slope4：bike penalty 1.5
        (3, 8, "['service', 'steps']", 90.0, 3),       # 线上漏网原型：drive 旧逻辑保留，新逻辑移除
        (4, 9, "['residential', 'steps']", 90.0, 3),   # 同上：车行道标签+台阶也必须否决
        (8, 9, "['corridor', 'footway']", 80.0, 3),    # 楼内连廊：bike/drive 均移除
    ]
    for u, v, highway, length, slope in edges:
        data = {
            "length": length,
            "slope_level": slope,
            "scenery_level": 3,
            "highway": highway,
        }
        G.add_edge(u, v, **data)
        G.add_edge(v, u, **data)
    return G


@pytest.fixture
def mode_graph():
    return _build_mode_mock_graph()


class TestTravelModeConstants:
    def test_travel_modes_tuple(self):
        assert TRAVEL_MODES == ("walk", "bike", "drive")

    def test_walk_default_weights_unchanged(self):
        assert MODE_DEFAULT_WEIGHTS["walk"] == DEFAULT_WEIGHTS
        assert MODE_DEFAULT_WEIGHTS["walk"] == {
            "distance": 0.90, "slope": 0.05, "scenery": 0.05
        }

    def test_walk_outside_penalty_unchanged(self):
        assert MODE_OUTSIDE_ROAD_PENALTY["walk"] == 10.0

    def test_mode_speeds(self):
        assert MODE_SPEEDS_KMH == {"walk": 4.5, "bike": 14.0, "drive": 25.0}


class TestNormalizeMode:
    @pytest.mark.parametrize("raw,expected", [
        ("walk", "walk"),
        ("bike", "bike"),
        ("drive", "drive"),
        (None, "walk"),
        ("", "walk"),
        ("fly", "walk"),
        ("WALK", "walk"),
        (123, "walk"),
    ])
    def test_normalize_mode(self, raw, expected):
        assert normalize_mode(raw) == expected


class TestEstimateDuration:
    def test_durations_by_mode(self):
        # 1000m / (14km/h) = 4.286min → 4.3
        assert estimate_duration_min(1000.0, "bike") == 4.3
        # 1000m / (4.5km/h) = 13.333min → 13.3
        assert estimate_duration_min(1000.0, "walk") == 13.3
        # 1000m / (25km/h) = 2.4min
        assert estimate_duration_min(1000.0, "drive") == 2.4

    def test_zero_and_none_length(self):
        assert estimate_duration_min(0, "bike") == 0.0
        assert estimate_duration_min(None, "drive") == 0.0

    def test_invalid_mode_falls_back_to_walk(self):
        # 4500m / 4.5km/h = 60min
        assert estimate_duration_min(4500.0, None) == 60.0


class TestEdgeHighwayTags:
    def test_parse_various_forms(self):
        assert _edge_highway_tags({"highway": "steps"}) == ["steps"]
        assert _edge_highway_tags({"highway": "['steps', 'footway']"}) == ["steps", "footway"]
        assert _edge_highway_tags({"highway": ["service", "footway"]}) == ["service", "footway"]
        assert _edge_highway_tags({}) == []


class TestFilterGraphForMode:
    def test_walk_no_filter_returns_original(self, mode_graph):
        # A mapped corridor is not itself an access prohibition.
        original_edges = mode_graph.number_of_edges()
        Gf, status, penalty = filter_graph_for_mode(mode_graph, "walk")
        assert status == "no_filter"  # mock 图无校外边
        assert Gf.number_of_edges() == original_edges
        assert Gf.has_edge(5, 6, 0)
        assert Gf.has_edge(6, 5, 0)
        assert Gf.has_edge(8, 9, 0)
        assert Gf.has_edge(9, 8, 0)

        # 台阶软惩罚仍然施加（混合标签台阶也惩罚）
        from spatial.routing import _WALK_STEPS_PENALTY
        assert penalty[(1, 2, 0)] == _WALK_STEPS_PENALTY
        assert penalty[(2, 1, 0)] == _WALK_STEPS_PENALTY
        assert penalty[(2, 3, 0)] == _WALK_STEPS_PENALTY
        assert penalty[(3, 2, 0)] == _WALK_STEPS_PENALTY
        assert (0, 1, 0) not in penalty
        assert (3, 4, 0) not in penalty

    def test_bike_removes_all_steps_and_corridors(self, mode_graph):
        Gf, status, penalty = filter_graph_for_mode(mode_graph, "bike")
        assert status == "mode_bike"

        # 纯台阶边双向移除
        assert not Gf.has_edge(1, 2)
        assert not Gf.has_edge(2, 1)
        # 多标签台阶边同样移除（['steps','footway'] / ['service','steps']
        # / ['residential','steps']）——杜绝"车上台阶"，含台阶标签一律否决
        assert not Gf.has_edge(2, 3)
        assert not Gf.has_edge(3, 2)
        assert not Gf.has_edge(3, 8)
        assert not Gf.has_edge(8, 3)
        assert not Gf.has_edge(4, 9)
        assert not Gf.has_edge(9, 4)
        # 楼内走廊/连廊骑行不可用（即使混了 footway 标签）
        assert not Gf.has_edge(5, 6)
        assert not Gf.has_edge(8, 9)
        # 普通边保留
        assert Gf.has_edge(0, 1)   # footway
        assert Gf.has_edge(3, 4)   # service
        assert Gf.has_edge(6, 7)   # path
        assert Gf.has_edge(0, 7)   # ['service','footway'] 人车共存

        # 陡坡软惩罚（双向）
        assert penalty[(4, 5, 0)] == 3.0
        assert penalty[(5, 4, 0)] == 3.0
        assert penalty[(2, 6, 0)] == 1.5
        assert penalty[(6, 2, 0)] == 1.5

        # 原图不被修改
        assert mode_graph.has_edge(1, 2)
        assert mode_graph.has_edge(5, 6)

    def test_drive_keeps_only_road_edges(self, mode_graph):
        Gf, status, penalty = filter_graph_for_mode(mode_graph, "drive")
        assert status == "mode_drive"
        assert penalty == {}

        # 保留：service / residential / primary / ['service','footway']
        assert Gf.has_edge(3, 4)
        assert Gf.has_edge(4, 5)
        assert Gf.has_edge(1, 6)
        assert Gf.has_edge(0, 7)
        # 移除：footway / 纯steps / ['steps','footway'] / corridor / path
        assert not Gf.has_edge(0, 1)
        assert not Gf.has_edge(1, 2)
        assert not Gf.has_edge(2, 3)
        assert not Gf.has_edge(5, 6)
        assert not Gf.has_edge(6, 7)
        # 多标签台阶即使带车行道标签也必须移除（线上"车上台阶"回归保护）
        assert not Gf.has_edge(3, 8)
        assert not Gf.has_edge(8, 3)
        assert not Gf.has_edge(4, 9)
        assert not Gf.has_edge(9, 4)
        assert not Gf.has_edge(8, 9)

        # 原图不被修改
        assert mode_graph.has_edge(0, 1)
        assert mode_graph.has_edge(5, 6)


class TestComputeRouteModes:
    _EXISTING_KEYS = (
        "recommended", "shortest", "filter_status", "overlap_rate",
        "degraded", "degraded_count", "recommended_length_m",
        "shortest_length_m", "applied_weights", "length_capped",
        "max_len", "_annotation_degraded",
    )

    def test_shortest_time_includes_slowdown_on_its_selected_edges(self, monkeypatch):
        graph = nx.MultiDiGraph()
        graph.add_nodes_from((1, 2))
        _set_campus_coords(graph)
        graph.add_edge(1, 2, key=0, length=100.0, highway="footway",
                       slope_level=3, scenery_level=3)
        monkeypatch.setattr(
            "spatial.road_conditions.apply_conditions_to_graph",
            lambda candidate, events, mode, strict: (
                candidate, {(1, 2, 0): 2.0}, set(), 1,
            ),
        )

        result = compute_route(
            graph, 1, 2, mode="walk", strategy_name="shortest",
            road_conditions=[{"id": "slowdown"}],
        )
        assert result["duration_min"] == 2.7
        assert result["shortest_duration_min"] == 2.7

    def test_bike_route(self, mode_graph):
        result = compute_route(mode_graph, 0, 7, mode="bike")
        assert result["mode"] == "bike"
        assert result["speed_kmh"] == 14.0
        assert result["duration_min"] > 0
        assert result["shortest_duration_min"] > 0
        for key in self._EXISTING_KEYS:
            assert key in result
        assert result["filter_status"].startswith("mode_bike")
        # 推荐路径不得经过纯台阶边 (1,2) / (2,1)
        path = result["recommended"]
        for i in range(len(path) - 1):
            assert (path[i], path[i + 1]) not in ((1, 2), (2, 1))

    def test_drive_route(self, mode_graph):
        result = compute_route(mode_graph, 0, 7, mode="drive")
        assert result["mode"] == "drive"
        assert result["speed_kmh"] == 25.0
        assert result["duration_min"] > 0
        assert result["shortest_duration_min"] > 0
        assert result["filter_status"].startswith("mode_drive")

    def test_drive_no_path_steps_only_raises(self):
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(4))
        _set_campus_coords(G)
        # 0-1 仅台阶连通（驾车不可达）
        for u, v in ((0, 1), (1, 0)):
            G.add_edge(
                u, v, length=100.0, slope_level=3, scenery_level=3,
                highway="steps",
            )
        # 另有车行道边，保证方式过滤后图非空（不触发全局 _degraded 回退）
        for u, v in ((2, 3), (3, 2)):
            G.add_edge(
                u, v, length=100.0, slope_level=2, scenery_level=3,
                highway="service",
            )
        with pytest.raises(ValueError, match="驾车"):
            compute_route(G, 0, 1, mode="drive")

    def test_walk_default_unchanged(self, mode_graph):
        result = compute_route(mode_graph, 0, 7)
        assert result["mode"] == "walk"
        assert result["speed_kmh"] == 4.5
        # walk 不过滤，filter_status 不以 mode_ 开头
        assert not result["filter_status"].startswith("mode_")
        assert result["applied_weights"] == DEFAULT_WEIGHTS  # 通勤默认 distance 主导

    def test_annotations_passthrough_mode(self, mode_graph):
        result = compute_route_with_annotations(mode_graph, 0, 7, mode="bike")
        assert result["mode"] == "bike"
        assert result["speed_kmh"] == 14.0


# ---------------------------------------------------------------------------
# 途经点（via）与游览环线（tour）测试
# ---------------------------------------------------------------------------

from spatial.routing import (  # noqa: E402
    VIA_MAX_DETOUR_RATIO,
    rank_via_candidates,
    compute_via_route,
    compute_tour_route,
)


class TestViaRoute:
    """mock_graph 拓扑：0→3 最短 450m（0-1-2-3）；
    节点 1 在路上（ratio=1.0），节点 5 显著绕路（ratio≈2.04）。"""

    def test_rank_via_on_the_way(self, mock_graph):
        cands = [({"name": "途经点A"}, 1), ({"name": "绕路点B"}, 5)]
        on, off = rank_via_candidates(mock_graph, 0, 3, cands)
        assert [p["name"] for p, _, _ in on] == ["途经点A"]
        assert on[0][2] == pytest.approx(1.0, abs=0.01)
        assert [p["name"] for p, _, _ in off] == ["绕路点B"]
        assert off[0][2] > VIA_MAX_DETOUR_RATIO

    def test_rank_via_unreachable_dropped(self, mock_graph):
        mock_graph.add_node(99)  # 孤立点：不可达候选直接丢弃
        mock_graph.nodes[99]["x"] = 114.360
        mock_graph.nodes[99]["y"] = 30.535
        on, off = rank_via_candidates(mock_graph, 0, 3, [({"name": "孤岛"}, 99)])
        assert on == [] and off == []

    def test_compute_via_route_two_legs(self, mock_graph):
        result = compute_via_route(mock_graph, 0, 1, 3)
        assert result["leg1"]["recommended"][0] == 0
        assert result["leg1"]["recommended"][-1] == 1
        assert result["leg2"]["recommended"][-1] == 3
        # detour_ratio 基于 recommended 软成本口径，权重变化会导致比例微调；
        # 核心是路径拼接正确、total_length_m 正常、ratio 在合理范围即可。
        assert 0.8 <= result["detour_ratio"] <= 1.5
        assert result["total_length_m"] > 0

    def test_compute_via_route_prepares_mode_graph_once(self, mock_graph, monkeypatch):
        from spatial import routing_index

        routing_index.invalidate_routing_index()
        calls = 0
        original = routing_index.build_mode_index

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(routing_index, "build_mode_index", counted)
        compute_via_route(mock_graph, 0, 1, 3, road_conditions=[])

        assert calls == 1


class TestTourRoute:
    def _pois(self):
        return [
            ({"name": "A", "importance": 1.0}, 1),
            ({"name": "B", "importance": 2.0}, 3),
            ({"name": "C", "importance": 3.0}, 5),
        ]

    def test_tour_open_order_and_legs(self, mock_graph):
        """开放游览：从 0 出发按最近邻 0→1→3→5，三段 leg。"""
        result = compute_tour_route(mock_graph, self._pois(), start_node=0, loop=False)
        assert [p["name"] for p in result["ordered_pois"]] == ["A", "B", "C"]
        assert len(result["legs"]) == 3
        assert result["dropped"] == []
        assert result["total_length_m"] == pytest.approx(
            sum(leg["recommended_length_m"] for leg in result["legs"]), abs=0.1
        )

    def test_tour_loop_returns_to_start(self, mock_graph):
        result = compute_tour_route(mock_graph, self._pois(), start_node=0, loop=True)
        assert len(result["legs"]) == 4  # 多一段返回起点
        first = result["legs"][0]["recommended"][0]
        last = result["legs"][-1]["recommended"][-1]
        assert first == last

    def test_two_stop_tour_loop_includes_return_leg(self, mock_graph):
        result = compute_tour_route(mock_graph, self._pois()[:2], loop=True)

        assert result["loop"] is True
        assert len(result["legs"]) == 2
        assert result["legs"][0]["recommended"][0] == result["legs"][-1]["recommended"][-1]

    def test_tour_drops_unreachable(self, mock_graph):
        mock_graph.add_node(99)
        mock_graph.nodes[99]["x"] = 114.360
        mock_graph.nodes[99]["y"] = 30.535
        pois = self._pois() + [({"name": "孤岛", "importance": 9.0}, 99)]
        result = compute_tour_route(mock_graph, pois, start_node=0, loop=False)
        assert [p["name"] for p in result["dropped"]] == ["孤岛"]
        assert "孤岛" not in [p["name"] for p in result["ordered_pois"]]

    def test_tour_trims_lowest_importance_when_over_cap(self, mock_graph):
        """总长上限 200m：依次剔 importance 最低的 A、B，直到只剩终点 C。"""
        result = compute_tour_route(
            mock_graph, self._pois(), start_node=0, loop=False, max_total_m=200.0
        )
        assert [p["name"] for p in result["ordered_pois"]] == ["C"]
        assert sorted(p["name"] for p in result["dropped"]) == ["A", "B"]

    def test_tour_empty_pois(self, mock_graph):
        result = compute_tour_route(mock_graph, [], start_node=0)
        assert result["ordered_pois"] == [] and result["legs"] == []


# ---------------------------------------------------------------------------
# 时间估算按边累加 + 多模态换乘 测试
# ---------------------------------------------------------------------------

class TestEstimateDurationEdgeAccumulator:
    """新签名 estimate_duration_min(G, route_nodes, mode, penalty_map)
    按边累加时长，速度受 slope_level/steps 标签/penalty_map 影响。"""

    def test_old_signature_compat_length_number(self):
        # 旧签名：length_m 是 number → 走旧分支，与 TestEstimateDuration 一致
        assert estimate_duration_min(1000.0, "walk") == 13.3
        assert estimate_duration_min(1000.0, "bike") == 4.3
        assert estimate_duration_min(1000.0, "drive") == 2.4

    def test_flat_walk_path(self, mock_graph):
        # 0→1 (100m, slope=2) walk → 1.3min（slope 2 不打折：100/75=1.33min）
        result = _estimate_route_duration_min(mock_graph, [0, 1], "walk")
        assert result == 1.3

    def test_slope5_walk_path(self, mock_graph):
        # 2→3 (150m, slope=5) walk → factor=0.5 → 速度 2.25km/h → 4min
        # 150m / (4.5*0.5 km/h * 1000/60 m/min) = 150 / 37.5 = 4.0min
        result = _estimate_route_duration_min(mock_graph, [2, 3], "walk")
        assert result == 4.0

    def test_slope4_bike_path(self, mock_graph):
        # 1→2 (200m, slope=4) bike → factor=0.6 → 速度 8.4km/h → 200m / 140 m/min ≈ 1.4min
        # 200 / (14*0.6 * 1000/60) = 200 / 140 = 1.43 → round to 1.4
        result = _estimate_route_duration_min(mock_graph, [1, 2], "bike")
        assert result == 1.4

    def test_slope_ignored_for_drive(self, mock_graph):
        # drive 不受小坡影响：2→3 (150m, slope=5) → 速度 25km/h → 0.36min → 0.4
        # 150 / (25 * 1000/60) = 150 / 416.67 = 0.36 → round to 0.4
        result = _estimate_route_duration_min(mock_graph, [2, 3], "drive")
        assert result == 0.4

    def test_road_penalty_divides_speed(self, mock_graph):
        # 0→1 (100m, walk, slope=2) + penalty 1.5 → 速度 ÷ 1.5 = 3km/h
        # 100 / (4.5/1.5 * 1000/60) = 100 / 50 = 2.0min
        # penalty_map 用 (u, v, k)；mock_graph 默认 key=0
        penalty = {(0, 1, 0): 1.5}
        result = _estimate_route_duration_min(mock_graph, [0, 1], "walk",
                                              penalty_map=penalty)
        assert result == 2.0

    def test_empty_or_short_route(self, mock_graph):
        assert _estimate_route_duration_min(mock_graph, [], "walk") == 0.0
        assert _estimate_route_duration_min(mock_graph, [0], "walk") == 0.0

    def test_steps_tag_slows_walk(self, mode_graph):
        # mode_graph: 1→2 highway=steps, length=100, slope=3
        # steps_factor=0.4 → 速度 1.8km/h → 100m / 30 m/min = 3.33 → 3.3min
        result = _estimate_route_duration_min(mode_graph, [1, 2], "walk")
        assert result == 3.3


class TestNearestInModeNode:
    """nearest_in_mode_node：bike/drive 起终点 POI 在 walk-only 节点时
    找 G_mode 中最近节点（限半径 50m）。"""

    def test_returns_none_when_no_node_in_radius(self):
        # 空图
        G = nx.MultiDiGraph()
        result, dist = nearest_in_mode_node(G, G, 114.36, 30.54, 50.0)
        assert result is None and dist == float("inf")

    def test_returns_nearest_within_radius(self, mode_graph):
        # mode_graph 所有节点坐标都是 (114.360, 30.535)（_set_campus_coords）
        # 在该坐标 50m 内找 bike 过滤图最近节点
        G_bike, _, _ = filter_graph_for_mode(mode_graph, "bike")
        node, dist = nearest_in_mode_node(mode_graph, G_bike, 114.360, 30.535, 50.0)
        assert node is not None
        assert dist <= 50.0

    def test_returns_none_when_far(self, mode_graph):
        # 远距离坐标（10km 外）应返回 None
        G_bike, _, _ = filter_graph_for_mode(mode_graph, "bike")
        node, dist = nearest_in_mode_node(mode_graph, G_bike, 114.5, 30.6, 50.0)
        assert node is None and dist == float("inf")


class TestToolPlanMultimodalRoute:
    """_tool_plan_multimodal_route 拼接多段路径为单个 route artifact。"""

    def test_invalid_legs_too_few(self):
        from agents.tools import _tool_plan_multimodal_route
        args = {"start": {"name": "X"}, "legs": [{"mode": "walk", "end": {"name": "Y"}}]}
        result, artifact = _tool_plan_multimodal_route(args, None)
        assert "error" in result and result["error"] == "invalid_legs"

    def test_invalid_legs_too_many(self):
        from agents.tools import _tool_plan_multimodal_route
        legs = [{"mode": "walk", "end": {"name": f"P{i}"}} for i in range(5)]
        args = {"start": {"name": "X"}, "legs": legs}
        result, artifact = _tool_plan_multimodal_route(args, None)
        assert "error" in result and result["error"] == "too_many_legs"

    def test_pois_not_found_returns_error(self):
        from agents.tools import _tool_plan_multimodal_route
        # 校外不存在的 POI
        args = {
            "start": {"name": "不存在的地点"},
            "legs": [
                {"mode": "walk", "end": {"name": "武汉大学教1楼"}},
                {"mode": "bike", "end": {"name": "武汉大学图书馆(总馆)"}},
            ],
        }
        result, artifact = _tool_plan_multimodal_route(args, None)
        assert "error" in result and result["error"] == "poi_not_found"


class TestRoutingReliability:
    """通勤参数与硬约束必须在实际寻路结果上成立。"""

    @pytest.fixture(autouse=True)
    def isolate_annotations(self, monkeypatch):
        monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)

    @pytest.mark.parametrize("mode", ["walk", "bike", "drive"])
    @pytest.mark.parametrize("weights", [None, {"distance": 0.90, "slope": 0.05, "scenery": 0.05}])
    def test_commute_weights_survive_hot_weather(self, mode_graph, mode, weights):
        result = compute_route(mode_graph, 0, 7, mode=mode, weights=weights,
                               road_conditions=[], weather_info={"temperature": 38, "weather": "晴"})
        assert result["applied_weights"] == pytest.approx(
            {"distance": 0.90, "slope": 0.05, "scenery": 0.05})

    def test_explicit_preferences_survive_hot_weather(self, mode_graph):
        weights = {"distance": 0.2, "slope": 0.6, "scenery": 0.2}
        result = compute_route(mode_graph, 0, 7, weights=weights, road_conditions=[],
                               weather_info={"temperature": 38, "weather": "晴"})
        assert result["applied_weights"] == pytest.approx(weights)

    def test_parallel_edge_length_and_geometry_match_selected_edge(self):
        from api.routes import _path_to_coords

        graph = nx.MultiDiGraph()
        graph.add_node(0, x=114.36, y=30.54)
        graph.add_node(1, x=114.361, y=30.54)
        common = {"highway": "residential", "name": "校内路", "scenery_level": 3}
        graph.add_edge(0, 1, key=0, length=80.0, slope_level=5,
                       geometry="LINESTRING (114.36 30.54, 114.3605 30.5402, 114.361 30.54)",
                       **common)
        graph.add_edge(0, 1, key=1, length=120.0, slope_level=1,
                       geometry="LINESTRING (114.36 30.54, 114.3605 30.5395, 114.361 30.54)",
                       **common)

        result = compute_route(graph, 0, 1,
                               weights={"distance": 0.1, "slope": 0.9, "scenery": 0.0},
                               detour_cap=1.5,
                               road_conditions=[])
        assert result["recommended_edges"] == [(0, 1, 1)]
        assert result["shortest_edges"] == [(0, 1, 0)]
        assert result["recommended_length_m"] == 120.0
        assert result["shortest_length_m"] == 80.0
        coords = _path_to_coords(graph, result["recommended"], result["recommended_edges"])
        assert coords[1] == {"lng": 114.3605, "lat": 30.5395}
        from scripts.validate_route_quality import path_edges
        from scripts.validate_route_quality import reachability_regressions
        from scripts.verify_road_condition_effects import edge_on_path
        assert path_edges(graph, result["recommended_edges"])[0][2] == 1
        assert edge_on_path(result["recommended_edges"], 0, 1, 1)
        assert not edge_on_path(result["recommended_edges"], 0, 1, 0)
        base = {mode: {"fail": 0} for mode in ("walk", "bike", "drive")}
        fixed = {mode: {"fail": int(mode == "bike")} for mode in base}
        assert reachability_regressions(base, fixed) == {"bike": (0, 1)}

    @staticmethod
    def _chain():
        graph = nx.MultiDiGraph()
        for u, v in ((0, 1), (1, 2), (2, 3)):
            graph.add_edge(u, v, length=100.0, highway="residential",
                           slope_level=2, scenery_level=3)
            graph.add_edge(v, u, length=100.0, highway="residential",
                           slope_level=2, scenery_level=3)
        return _set_campus_coords(graph)

    def test_closure_disconnect_must_not_restore_closed_edge(self):
        graph = self._chain()
        conditions = [{"id": "blocked", "type": "closure", "edge": {"u": 1, "v": 2, "key": 0}}]
        with pytest.raises(ValueError, match="封闭|无法通行"):
            compute_route(graph, 0, 3, road_conditions=conditions)

    @pytest.mark.parametrize("all_steep", [False, True])
    def test_slope_disconnect_must_not_relax_constraint(self, all_steep):
        graph = self._chain()
        for u, v, data in graph.edges(data=True):
            if all_steep or u == 1:
                data["slope_level"] = 5
        with pytest.raises(ValueError, match="陡坡|避坡"):
            compute_route(graph, 0, 3, constraints={"slope": "avoid"}, road_conditions=[])

    def test_shortest_comparison_also_respects_slope_constraint(self):
        graph = self._chain()
        graph.add_edge(0, 3, length=10.0, highway="residential", slope_level=5, scenery_level=3)
        result = compute_route(graph, 0, 3, constraints={"slope": "avoid"}, road_conditions=[])
        assert result["recommended"] == [0, 1, 2, 3]
        assert result["shortest"] == [0, 1, 2, 3]

    def test_tour_must_not_fallback_through_closed_edge(self):
        graph = self._chain()
        conditions = [{"id": "blocked", "type": "closure", "edge": {"u": 1, "v": 2, "key": 0}}]
        with pytest.raises(ValueError, match="封闭|无法通行"):
            compute_tour_route(graph, [({"name": "B"}, 3)], start_node=0,
                               loop=False, road_conditions=conditions)

    def test_tour_must_not_fallback_through_steep_edge(self):
        graph = self._chain()
        graph[1][2][0]["slope_level"] = 5
        with pytest.raises(ValueError, match="陡坡|避坡"):
            compute_tour_route(graph, [({"name": "B"}, 3)], start_node=0,
                               loop=False, constraints={"slope": "avoid"}, road_conditions=[])
