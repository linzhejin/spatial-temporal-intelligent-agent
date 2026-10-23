from types import SimpleNamespace

import pytest

from scripts.analysis.replay_synthetic_road_events import replay_bank


def _bank():
    event = {"action": "closure", "affected_modes": ["walk"],
             "start_time": 1000, "end_time": 2000,
             "edge": {"u": "1", "v": "2", "key": "0"},
             "review_status": "synthetic_fixture_only"}
    base = {"source_video_id": "foreign-video-1", "source_link_id": "road-1",
            "source_spatial_truth": "Songdo, not Wuhan",
            "campus_road_assignment": "hypothetical_unpublished",
            "mode": "walk", "start_node": "1", "end_node": "2", "event": event}
    return {"format": "synthetic-road-event-bank-v1", "split": "development",
            "cases": [
                {**base, "id": "active", "as_of_epoch": 1500,
                 "expected_graph_effect": {"detour_edge_ids": [
                     ["1", "3", "0"], ["3", "2", "0"]]}},
                {**base, "id": "expired", "as_of_epoch": 2500,
                 "expected_graph_effect": {"direct_edge_id": ["1", "2", "0"]}},
            ]}


def test_active_and_expired_event_replay_uses_distinct_tool_snapshots():
    seen = []
    graph = SimpleNamespace(nodes={
        1: {"x": 114.36, "y": 30.53}, 2: {"x": 114.37, "y": 30.53}})

    def execute(name, args, context):
        assert name == "plan_route"
        assert args["strategy"] == "shortest"
        seen.append(context["_route_conditions_snapshot"])
        edges = [(1, 3, 0), (3, 2, 0)] if seen[-1] else [(1, 2, 0)]
        return {"recommended_edge_ids": edges,
                "route_verification": {"status": "pass"}}, None

    report = replay_bank(_bank(), graph, execute)
    assert report["tool_route_pass_count"] == 2
    assert seen[0][0]["review_status"] == "synthetic_fixture_only"
    assert seen[1] == []
    assert report["status"] == "development_tool_replay_only_not_agent_or_vision_score"


def test_replay_does_not_mark_mismatched_product_route_as_pass():
    graph = SimpleNamespace(nodes={
        1: {"x": 114.36, "y": 30.53}, 2: {"x": 114.37, "y": 30.53}})
    report = replay_bank(_bank(), graph, lambda *_args: (
        {"recommended_edge_ids": [(1, 2, 0)],
         "route_verification": {"status": "pass"}}, None))
    assert report["tool_route_pass_count"] == 1
    assert not report["cases"][0]["graph_reference_match"]


def test_replay_accepts_another_detour_that_excludes_closed_road():
    graph = SimpleNamespace(nodes={
        1: {"x": 114.36, "y": 30.53}, 2: {"x": 114.37, "y": 30.53}})

    def execute(_name, _args, context):
        edges = ([(1, 4, 0), (4, 2, 0)]
                 if context["_route_conditions_snapshot"] else [(1, 2, 0)])
        return {"recommended_edge_ids": edges,
                "route_verification": {"status": "pass"}}, None

    report = replay_bank(_bank(), graph, execute)
    assert report["tool_route_pass_count"] == 2
    assert not report["cases"][0]["graph_reference_match"]


def test_replay_rejects_non_development_bank():
    bank = _bank()
    bank["split"] = "test"
    with pytest.raises(ValueError, match="development"):
        replay_bank(bank, None, None)
