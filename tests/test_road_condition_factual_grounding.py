"""Manager events must be grounded in the actual active road records."""
from types import SimpleNamespace
from unittest.mock import patch

from agents import planner
from tests.test_vision_condition_feedback import _call, _event, _message, _run


def test_manager_closure_cannot_be_reported_as_open():
    event = _event(source_kind="manager_record", name="施工封闭", vision_evidence=None)
    result = _run([event], final="科技路没有封闭，所有道路畅通。", query="科技路现在能步行吗？")
    assert "科技路" in result["message"]
    assert "禁止通行" in result["message"] and "步行" in result["message"]
    assert "没有封闭" not in result["message"] and "畅通" not in result["message"]


def test_manager_slowdown_cannot_be_reported_as_clear():
    event = _event(source_kind="manager_record", action="slowdown", action_label="缓行",
                   affected_modes=["drive"], vision_evidence=None)
    result = _run([event], final="科技路没有管制，驾车畅通。", query="科技路现在驾车路况如何？")
    assert "缓行" in result["message"] and "驾车" in result["message"]
    assert "畅通" not in result["message"]


def test_named_road_generic_status_question_uses_active_record():
    event = _event(source_kind="manager_record", vision_evidence=None)
    result = _run([event], final="科技路现在很通畅。", query="科技路现在怎么样？")
    assert "禁止通行" in result["message"] and "通畅" not in result["message"]


def test_other_road_unknown_is_not_inferred_clear():
    event = _event(source_kind="manager_record", vision_evidence=None)
    result = _run([event], final="樱园路可以畅通行驶。", query="樱园路现在路况如何？")
    assert "畅通" not in result["message"]
    assert "未覆盖" in result["message"] or "无法确认" in result["message"]


def test_uncovered_named_road_generic_status_does_not_claim_clear():
    event = _event(source_kind="manager_record", vision_evidence=None)
    result = _run([event], final="樱园路现在畅通，可以放心通行。", query="樱园路现在怎么样？")
    assert "畅通" not in result["message"]
    assert "未覆盖" in result["message"] or "无法确认" in result["message"]


def test_empty_records_named_road_generic_status_does_not_claim_clear():
    result = _run([], final="樱园路现在畅通。", query="樱园路现在怎么样？")
    assert "畅通" not in result["message"]
    assert "现场" in result["message"]


def test_unrelated_question_does_not_rewrite_model_answer():
    event = _event(source_kind="manager_record", vision_evidence=None)
    result = _run([event], final="总图是武汉大学的图书馆。", query="介绍一下总图。")
    assert result["message"] == "总图是武汉大学的图书馆。"


def test_manager_record_with_route_and_weather_preserves_both_evidences():
    replies = [_message(calls=[_call("list_road_conditions"), _call("plan_route"), _call("get_weather")]),
               _message(content="路线已生成，科技路畅通，天气晴。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    event = _event(source_kind="manager_record", vision_evidence=None)
    route = {"start_name": "总图", "end_name": "桂园", "mode": "walk",
             "recommended_length_m": 500, "duration_min": 8}

    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [event], "count": 1}, None
        if name == "plan_route":
            return route, {"route": route}
        return {"weather": "晴", "temperature": 20}, None

    with patch.object(planner.config, "DEEPSEEK_API_KEY", "fixture"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("规划路线，查询路况和天气。")
    assert result["route"] is route
    assert "禁止通行" in result["message"] and "晴" in result["message"]
    assert "畅通" not in result["message"]
