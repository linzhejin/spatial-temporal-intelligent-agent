"""Offline regression for ordered and clarified trips on the actual campus graph."""

import shutil
from pathlib import Path

import pytest
import networkx as nx

from agents import planner, tools
from spatial import network


@pytest.fixture
def offline_graph(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "data" / "whu_road_network.graphml"
    cache = tmp_path / "graph.graphml"
    shutil.copyfile(source, cache)
    events = tmp_path / "conditions.json"
    events.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(events))
    monkeypatch.setenv("AMAP_POI_LOOKUP_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setattr(network, "_G", None)
    monkeypatch.setattr(network, "_cache_path", lambda: str(cache))
    monkeypatch.setattr(network, "_download_network", lambda *_: pytest.fail("network requested"))
    monkeypatch.setattr(planner, "_make_client", lambda: pytest.fail("model requested"))
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: {})
    network.load_or_download_network()


def test_ordered_trip_uses_real_graph_and_retains_all_stops(offline_graph):
    result = planner.run_agent("从玉兰2门先到卓尔体育馆，再到桂园食堂，最后樱顶。")
    route = result["route"]
    assert result["response_kind"] == "route"
    assert route["start_name"] == "玉兰2门"
    assert [p["name"] for p in route["via"]["points"]] == ["卓尔体育馆", "桂园食堂"]
    assert route["end_name"] == "武汉大学老斋舍"
    assert route["recommended_edge_ids"]


def test_clarification_and_both_selections_build_complete_trip_on_real_graph(offline_graph):
    first = planner.run_agent("我在总图，先找一个球场然后去吃饭，不走台阶")
    assert first["response_kind"] == "clarify"
    sports = planner.run_agent("篮球", context={"place_request": first["place_request"]})
    assert sports["response_kind"] == "candidates"
    assert sports["place_request"]["constraints"]["avoid_steps"] is True
    sport = sports["candidates"][0]
    next_step = planner.run_agent("去这个篮球场", context={
        "place_request": sports["place_request"], "selected_poi_id": sport["poi_id"],
    })
    assert next_step["response_kind"] == "route"
    assert next_step["needs_input"] is True
    assert next_step["candidates"]
    meal = next_step["candidates"][0]
    final = planner.run_agent("去这个吃饭的地方", context={
        "place_request": next_step["place_request"], "selected_poi_id": meal["poi_id"],
    })
    assert final["response_kind"] == "route"
    assert final["place_request"] is None
    route = final["route"]
    assert route["start_name"] == "武汉大学图书馆(总馆)"
    assert route["end_name"] == meal["name"]
    assert len(route["legs"]) == 2
    assert route["legs"][0]["end_name"] == sport["name"]
    assert route["legs"][1]["start_name"] == sport["name"]
    assert route["route_state"]["hard_constraints"]["avoid_steps"] is True
    assert route["recommended_edge_ids"]
    edges = route["recommended_edge_ids"]
    assert all(a[1] == b[0] for a, b in zip(edges, edges[1:]))
    graph = network.get_network()
    for u, v, key in edges:
        data = graph.get_edge_data(u, v, key)
        assert data is not None
        highway = data.get("highway", "")
        tags = highway if isinstance(highway, (list, tuple)) else str(highway).split(";")
        assert "steps" not in [str(value).strip().lower() for value in tags]


def test_requested_detour_is_verified_against_direct_real_graph_route(offline_graph):
    query = "从桂园食堂到樱顶，途经WHU1893咖啡馆，告诉我绕行多少距离"
    args = {"start": {"name": "桂园食堂"}, "end": {"name": "樱顶"},
            "via_name": "WHU1893咖啡馆", "mode": "walk", "constraints": {}}
    route, artifact = tools._tool_plan_via_route(args, {"query": query})
    assert artifact["route"] is route
    assert route["direct_reference_verified"] is True
    graph = tools.get_routing_index(network.get_network()).for_mode("walk").graph
    nodes = [tools._resolve_endpoint({"name": name}, graph)[0]
             for name in ("桂园食堂", "WHU1893咖啡馆", "樱顶")]
    direct = nx.shortest_path_length(graph, nodes[0], nodes[2], weight="length")
    via = (nx.shortest_path_length(graph, nodes[0], nodes[1], weight="length")
           + nx.shortest_path_length(graph, nodes[1], nodes[2], weight="length"))
    assert route["direct_reference_length_m"] == pytest.approx(direct, abs=0.1)
    assert route["recommended_length_m"] == pytest.approx(via, abs=0.1)
    assert "多走约 144 米" in planner._build_route_message(route)


def test_detour_question_survives_clarification_and_two_place_choices(offline_graph):
    query = "我在总图，先找一个球场然后去吃饭，不走台阶，告诉我绕行多少距离"
    first = planner.run_agent(query)
    sports = planner.run_agent("篮球", context={"place_request": first["place_request"]})
    sport = sports["candidates"][0]
    next_step = planner.run_agent("去这个篮球场", context={
        "place_request": sports["place_request"], "selected_poi_id": sport["poi_id"],
    })
    meal = next_step["candidates"][0]
    final = planner.run_agent("去这个吃饭的地方", context={
        "place_request": next_step["place_request"], "selected_poi_id": meal["poi_id"],
    })
    assert final["response_kind"] == "route"
    assert "相比同条件下不经途经点的路线" in final["message"]
    assert "无法核实" not in final["message"]
    assert final["route"]["direct_reference_verified"] is True
    assert final["route"]["route_state"]["hard_constraints"]["avoid_steps"] is True


def test_reference_failure_keeps_valid_via_route_without_inventing_distance(offline_graph, monkeypatch):
    original = tools._audit_tool_route

    def audit(graph, payload, context, nodes, kind):
        if kind == "direct":
            raise RuntimeError("simulated direct-reference verifier failure")
        return original(graph, payload, context, nodes, kind)

    monkeypatch.setattr(tools, "_audit_tool_route", audit)
    query = "从桂园食堂到樱顶，途经WHU1893咖啡馆，告诉我绕行多少距离"
    args = {"start": {"name": "桂园食堂"}, "end": {"name": "樱顶"},
            "via_name": "WHU1893咖啡馆", "mode": "walk", "constraints": {}}
    route, artifact = tools._tool_plan_via_route(args, {"query": query})
    assert artifact["route"] is route
    assert route["route_verification"]["status"] == "pass"
    assert "direct_reference_verified" not in route
    assert "无法核实" in planner._build_route_message(route)


def test_changed_snapshot_replans_full_trip_and_answers_detour_once(offline_graph, monkeypatch):
    query = "我在总图，先找一个球场然后去吃饭，不走台阶，告诉我绕行多少距离"
    first = planner.run_agent(query)
    sports = planner.run_agent("篮球", context={"place_request": first["place_request"]})
    sport = sports["candidates"][0]
    next_step = planner.run_agent("去这个篮球场", context={
        "place_request": sports["place_request"], "selected_poi_id": sport["poi_id"],
    })
    meal = next_step["candidates"][0]
    monkeypatch.setattr(planner, "_place_route_snapshot_issue", lambda _legs: "当前路况已更新")
    final = planner.run_agent("去这个吃饭的地方", context={
        "place_request": next_step["place_request"], "selected_poi_id": meal["poi_id"],
    })
    assert final["response_kind"] == "route"
    assert final["route"]["direct_reference_verified"] is True
    assert final["message"].count("相比同条件下不经途经点的路线") == 1
    assert "路况发生变化" in final["message"]
