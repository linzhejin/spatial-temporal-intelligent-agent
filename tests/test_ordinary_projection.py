from experiments.baselines.ordinary_projection import project_ordinary_turn
from experiments.scenario_live_adapter import build_poi_lookup


def _lookup():
    return build_poi_lookup([
        {"id": "poi_start", "name": "玉兰2门", "aliases": ["玉兰二门"]},
        {"id": "poi_end", "name": "樱顶", "aliases": []},
        {"id": "poi_food", "name": "星湖园食堂", "aliases": []},
    ])


def test_project_ordinary_route_from_actual_tool_artifact():
    turn = {"status": "completed", "artifacts": [{"route": {
        "route_state": {"start": {"name": "玉兰二门"}, "end": {"name": "樱顶"},
                        "travel_mode": "walk", "strategy": {"name": "shortest"},
                        "hard_constraints": {"avoid_steps": True}},
        "recommended_edge_ids": [[1, 2, 0]],
    }}], "tool_trace": [{"tool_name": "plan_route", "argument_keys": ["start", "end"],
                        "outcome": "success", "artifact_kind": "route"}]}
    result = project_ordinary_turn(turn, _lookup())
    assert result["kind"] == "route"
    assert result["start_poi_id"] == "poi_start"
    assert result["end_poi_id"] == "poi_end"
    assert result["route_edge_ids"] == [[1, 2, 0]]
    assert result["tool_trace"][0]["tool_name"] == "plan_route"


def test_project_ordinary_candidates_keeps_tool_selected_origin():
    turn = {"status": "completed", "selected_start_name": "玉兰2门",
            "selected_mode": "walk", "artifacts": [{"candidates": [
                {"poi_id": "poi_food", "name": "星湖园食堂"},
            ]}]}
    result = project_ordinary_turn(turn, _lookup())
    assert result["kind"] == "candidates"
    assert result["start_poi_id"] == "poi_start"
    assert result["candidate_poi_ids"] == ["poi_food"]


def test_project_ordinary_text_only_is_not_spatial_proof():
    result = project_ordinary_turn(
        {"status": "completed", "message": "路线已经规划好了。", "artifacts": []},
        _lookup(),
    )
    assert result["kind"] == "information"
    assert result["route_edge_ids"] == []


def test_project_ordinary_clarification_does_not_invent_missing_slot():
    turn = {"status": "completed", "artifacts": [{"clarify": {
        "question": "想打什么球？", "options": ["篮球", "羽毛球"],
    }}]}
    result = project_ordinary_turn(turn, _lookup())
    assert result["kind"] == "clarify"
    assert result["status"] == "needs_input"
    assert result["clarification_slot"] is None


def test_ordinary_weather_requires_matching_final_reply_and_tool_observation():
    turn = {"status": "completed", "artifacts": [{"route": {
        "start_name": "玉兰2门", "end_name": "樱顶",
        "recommended_edge_ids": [[1, 2, 0]],
    }}], "weather_observation": {"weather": "晴", "temperature": 22},
            "message": "路线已规划。天气晴，22 度。"}
    assert "weather" in project_ordinary_turn(turn, _lookup())["outputs"]
    turn["message"] = "路线已规划。"
    assert "weather" not in project_ordinary_turn(turn, _lookup())["outputs"]
