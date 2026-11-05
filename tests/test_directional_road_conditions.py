"""Directed road events must not close the opposite direction or crossing roads."""

from copy import deepcopy

import networkx as nx
import pytest

from spatial import road_conditions as rc
from spatial.coord_transform import wgs84_to_gcj02


@pytest.fixture
def directional_case():
    graph = nx.MultiDiGraph()
    points = {
        1: (114.3600, 30.5300), 2: (114.3610, 30.5300),
        3: (114.3605, 30.5298), 4: (114.3605, 30.5302),
        5: (114.3600, 30.53005), 6: (114.3610, 30.53005),
    }
    for node, (x, y) in points.items():
        graph.add_node(node, x=x, y=y)
    forward_wkt = "LINESTRING (114.3600 30.5300, 114.3610 30.5300)"
    for key in (0, 1):
        # Reversed graph edges may share the same stored WKT orientation.
        for u, v in ((1, 2), (2, 1)):
            graph.add_edge(u, v, key, length=96, geometry=forward_wkt)
    for u, v in ((5, 6), (6, 5), (3, 4), (4, 3)):
        graph.add_edge(u, v, 0, length=96)
    geometry = [list(wgs84_to_gcj02(*points[node])) for node in (1, 2)]
    event = {
        "id": "directed", "type": "closure", "name": "模拟封闭",
        "action": "closure", "affected_modes": ["walk"],
        "direction": "forward", "start_time": 10, "end_time": 20,
        "edge": {"u": 1, "v": 2, "key": 0, "edges": [[1, 2, 0]],
                 "geometry_gcj": geometry},
    }
    return graph, event


@pytest.mark.parametrize(
    ("direction", "closed", "open_edges"),
    [
        ("forward", {(1, 2, 0), (1, 2, 1), (5, 6, 0)},
         {(2, 1, 0), (2, 1, 1), (6, 5, 0)}),
        ("reverse", {(2, 1, 0), (2, 1, 1), (6, 5, 0)},
         {(1, 2, 0), (1, 2, 1), (5, 6, 0)}),
    ],
)
def test_direction_closes_only_matching_directed_corridor(directional_case, direction, closed, open_edges):
    graph, event = directional_case
    event["direction"] = direction
    _, actual, applied = rc.resolve_condition_effects(graph, [event], "walk", strict=True)
    assert applied == 1
    assert actual == closed
    assert not actual & open_edges
    assert not actual & {(3, 4, 0), (4, 3, 0)}
    filtered, _, _, _ = rc.apply_conditions_to_graph(graph, [event], "walk", strict=True)
    source, target = (1, 2) if direction == "forward" else (2, 1)
    assert not nx.has_path(filtered, source, target)
    assert nx.has_path(filtered, target, source)


def test_legacy_event_remains_bidirectional_and_other_modes_are_unaffected(directional_case):
    graph, event = directional_case
    event.pop("direction")
    _, closed, applied = rc.resolve_condition_effects(graph, [event], "walk", strict=True)
    assert applied == 1
    assert {(1, 2, 0), (2, 1, 0)} <= closed
    _, other_closed, other_applied = rc.resolve_condition_effects(graph, [event], "bike", strict=True)
    assert not other_closed and other_applied == 0


def test_directional_slowdown_penalizes_only_selected_direction(directional_case):
    graph, event = directional_case
    event.update(action="slowdown", cost_multiplier=2.0)
    penalties, closed, _ = rc.resolve_condition_effects(graph, [event], "walk", strict=True)
    assert not closed
    assert penalties[(1, 2, 0)] == 2.0
    assert (2, 1, 0) not in penalties


def test_agent_and_route_notice_disclose_direction(directional_case, monkeypatch):
    from agents import tools

    graph, event = directional_case
    event["action"] = "notice"
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [event])
    result, _ = tools.execute_tool("list_road_conditions", {}, {})
    assert result["conditions"][0]["direction"] == "forward"
    assert "A → B" in result["conditions"][0]["direction_label"]
    assert result["conditions"][0]["route_effect"]["direction"] == "forward"
    notices, _, overview = rc.route_condition_notices(
        graph, [event], "walk", [[1, 2, 0]], include_overview=True)
    assert notices[0]["direction"] == "forward"
    assert overview[0]["direction"] == "forward"
    opposite, _ = rc.route_condition_notices(graph, [event], "walk", [[2, 1, 0]])
    assert opposite == []


@pytest.mark.parametrize(("direction", "label"), [("forward", "地图 A → B"), ("reverse", "地图 B → A")])
def test_spoken_road_summary_and_route_feedback_preserve_direction(directional_case, monkeypatch, direction, label):
    from agents import planner, tools

    _, event = directional_case
    event["direction"] = direction
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [event])
    result, _ = tools.execute_tool("list_road_conditions", {}, {})
    summary = planner._vision_condition_summary(result, "先查路况，说明封闭方向")
    assert label in summary
    route = {"mode": "walk", "road_conditions_applied": 1,
             "road_condition_overview": [{"id": event["id"], "name": event["name"],
                                           "action": "closure", "direction": direction}]}
    feedback = planner._road_condition_feedback(route, "查路况和方向")
    assert label in feedback
    assert "不一定影响" in feedback


def test_unrecognized_direction_in_summary_remains_unverified():
    from agents import planner

    event = {"name": "模拟事件", "status": "confirmed_active", "action_label": "禁止通行",
             "affected_modes": ["walk"], "direction": "unexpected"}
    summary = planner._vision_condition_summary({"conditions": [event]}, "查路况和方向")
    assert "方向" in summary and ("未知" in summary or "未核实" in summary)
    assert "地图 A → B" not in summary and "地图 B → A" not in summary


def test_direction_survives_graph_id_rebuild_without_closing_crossing(directional_case):
    graph, event = directional_case
    new_graph = nx.relabel_nodes(graph, {node: node + 100 for node in graph}, copy=True)
    _, closed, _ = rc.resolve_condition_effects(new_graph, [event], "walk", strict=True)
    assert closed == {(101, 102, 0), (101, 102, 1), (105, 106, 0)}
    assert not closed & {(103, 104, 0), (104, 103, 0)}


def test_direction_requires_unambiguous_geometry(directional_case):
    graph, event = directional_case
    event["edge"]["geometry_gcj"] = []
    with pytest.raises(rc.RoadConditionsUnavailableError):
        rc.resolve_condition_effects(graph, [event], "walk", strict=True)


def test_direction_rejects_self_crossing_geometry(directional_case):
    graph, event = directional_case
    crossing = [(114.3600, 30.5300), (114.3610, 30.5310),
                (114.3600, 30.5310), (114.3610, 30.5300)]
    event["edge"]["geometry_gcj"] = [list(wgs84_to_gcj02(*point)) for point in crossing]
    with pytest.raises(rc.RoadConditionsUnavailableError):
        rc.resolve_condition_effects(graph, [event], "walk", strict=True)


def test_unavailable_direction_has_no_effect_when_reverse_corridor_is_complete(directional_case):
    graph, event = directional_case
    graph.remove_edges_from([(1, 2, 0), (1, 2, 1), (5, 6, 0)])
    penalties, closed, applied = rc.resolve_condition_effects(graph, [event], "walk", strict=True)
    assert not penalties and not closed and applied == 1
    assert graph.has_edge(2, 1, 0)


def test_directional_record_is_stored_and_vision_rebind_requires_new_confirmation(
        tmp_path, monkeypatch, directional_case):
    graph, event = directional_case
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(tmp_path / "events.json"))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    monkeypatch.setattr(rc, "_cache_error", None)
    middle = wgs84_to_gcj02(114.3605, 30.5300)
    snap = rc.snap_to_edge(graph, *middle)
    stored = rc.add_condition("closure", "方向核对", snap, action="closure",
                              affected_modes=["walk"], direction="forward",
                              source={"kind": "vision_job", "job_id": "vision-test"})
    assert rc.list_conditions(strict=True)[0]["direction"] == "forward"
    before = deepcopy(rc.list_conditions(include_inactive=True, strict=True))
    with pytest.raises(ValueError, match="方向"):
        rc.rebind_vision_condition_edge(stored["id"], snap, field_confirmation="已人工核对道路位置")
    assert rc.list_conditions(include_inactive=True, strict=True) == before


def test_directional_event_lifecycle_keeps_expiry_and_direction(tmp_path, monkeypatch, directional_case):
    graph, event = directional_case
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(tmp_path / "events.json"))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    monkeypatch.setattr(rc, "_cache_error", None)
    rc._save_conditions([event])
    monkeypatch.setattr(rc.time, "time", lambda: 15)
    active = rc.list_conditions(strict=True)
    assert len(active) == 1 and active[0]["direction"] == "forward"
    assert (1, 2, 0) in rc.resolve_condition_effects(graph, active, "walk", strict=True)[1]
    monkeypatch.setattr(rc.time, "time", lambda: 20)
    assert rc.list_conditions(strict=True) == []
