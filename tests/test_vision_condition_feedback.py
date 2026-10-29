"""Vision evidence must not be narrated as an automatically published fact."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import agents.planner as planner
import pytest


def _message(content=None, calls=None):
    message = SimpleNamespace(content=content, tool_calls=calls or [])
    message.model_dump = lambda exclude_none=False: {
        "role": "assistant", "tool_calls": [
            {"id": call.id, "type": "function", "function": {
                "name": call.function.name, "arguments": call.function.arguments,
            }} for call in (calls or [])
        ],
    }
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _call(name, args=None):
    return SimpleNamespace(id=f"call_{name}", function=SimpleNamespace(
        name=name, arguments=json.dumps(args or {}, ensure_ascii=False)))


def _run(conditions, final="视觉识别自动发布并确认发生了事故。", query="现在有什么路况，来源是什么？"):
    replies = [_message(calls=[_call("list_road_conditions")]), _message(content=final)]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", return_value=({"conditions": conditions, "count": len(conditions)}, None)):
        return planner.run_agent(query)


def _event(**overrides):
    event = {"id": "e1", "name": "实验路况", "road_name": "科技路", "action": "closure",
             "action_label": "禁止通行", "affected_modes": ["walk"], "status": "confirmed_active",
             "source_kind": "vision_job", "vision_evidence": {"candidate_index": 0,
             "candidate_kind": "possible_congestion", "reviewed_at": "2026-10-29T08:00:00+08:00",
             "reviewed_by": "token"}}
    event.update(overrides)
    return event


def test_confirmed_vision_event_reports_review_and_action_without_inventing_incident():
    response = _run([_event()])
    text = response["message"]
    assert "科技路" in text and "禁止通行" in text and "步行" in text
    assert "复核" in text and "发布" in text
    assert "自动" not in text and "事故" not in text
    assert "现场" in text and "已现场核实" not in text


def test_unreviewed_candidate_never_becomes_confirmed_road_event():
    response = _run([_event(status="pending_review", vision_evidence={"candidate_index": 0})])
    text = response["message"]
    assert "复核" in text and ("未知" in text or "尚未" in text or "无法确认" in text)
    assert "复核后发布" not in text and "自动" not in text


def test_missing_review_evidence_does_not_claim_review():
    response = _run([_event(vision_evidence={"candidate_index": 0})])
    assert "复核" in response["message"] and ("未知" in response["message"] or "未记录" in response["message"])
    assert "复核后发布" not in response["message"]


def test_unknown_nonvision_source_is_not_treated_as_reviewed():
    response = _run([_event(), _event(id="e2", name="其他事件", source_kind="unknown", vision_evidence=None)])
    assert "其他事件" in response["message"] and "来源" in response["message"]
    assert response["message"].count("复核后发布") == 1


def test_nonvision_only_road_info_keeps_existing_model_answer():
    event = _event(source_kind="manager_record", vision_evidence=None)
    response = _run([event], final="科技路禁止步行，请绕行。")
    assert response["message"] == "科技路禁止步行，请绕行。"


def test_unknown_source_alone_does_not_acquire_invented_origin():
    response = _run([_event(source_kind="unknown", vision_evidence=None)], final="这条事件由无人机自动确认。")
    assert "来源" in response["message"] and ("未知" in response["message"] or "无法确认" in response["message"])
    assert "自动" not in response["message"]


def test_composite_weather_and_road_conditions_both_remain_visible():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_weather")]),
               _message(content="视觉识别自动发布，天气晴。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        assert name == "get_weather"
        return {"weather": "晴", "temperature": 20}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("现在路况和天气如何？")
    assert "复核后发布" in result["message"] and "晴" in result["message"] and "20°C" in result["message"]
    assert "自动" not in result["message"]


def test_mixed_poi_and_road_info_preserves_other_answer_without_automatic_publish_claim():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_place_details")]),
               _message(content="视觉识别自动生成并发布了管制。桂园食堂位于桂园。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        assert name == "get_place_details"
        return {"name": "桂园食堂", "location": "桂园"}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("当前路况的来源是什么？同时介绍一下这个建筑。")
    assert "自动" not in result["message"]
    assert "复核后发布" in result["message"] and "桂园食堂" in result["message"]


def test_mixed_info_keeps_non_event_clause_within_same_sentence():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_place_details")]),
               _message(content="桂园食堂位于桂园，影像自动生成并发布了管制。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        return {"name": "桂园食堂", "location": "桂园"}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("当前路况的来源是什么？同时介绍一下这个建筑。")
    assert "桂园食堂" in result["message"] and "位置：桂园" in result["message"]
    assert "自动生成" not in result["message"]
    assert "复核后发布" in result["message"]


def test_named_road_is_not_hidden_behind_first_five_events():
    events = [_event(id=f"e{i}", name=f"事件{i}", road_name=f"道路{i}") for i in range(6)]
    events[-1]["road_name"] = "科技路"
    response = _run(events, final="科技路有封闭。", query="科技路有什么路况？")
    assert "科技路" in response["message"]
    assert "禁止通行" in response["message"]


def test_mixed_route_and_vision_conditions_keep_route_artifact():
    replies = [_message(calls=[_call("list_road_conditions"), _call("plan_route")]),
               _message(content="路线已生成。视觉识别自动发布了管制。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    route = {"start_name": "总图", "end_name": "桂园", "mode": "walk",
             "recommended_length_m": 500, "duration_min": 8}
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        assert name == "plan_route"
        return route, {"route": route}
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("查询路况并规划这条路线。")
    assert result["route"] is route


def test_explicit_road_conditions_query_with_empty_results_is_answered():
    replies = [_message(calls=[_call("list_road_conditions"), _call("plan_route")])]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    route = {"start_name": "总图", "end_name": "桂园", "mode": "walk",
             "recommended_length_m": 500, "duration_min": 8,
             "road_conditions_applied": 0, "road_condition_overview": []}

    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [], "count": 0}, None
        return route, {"route": route}

    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("先查当前路况，再从总图到桂园。")
    assert result["route"] is route
    assert "没有已生效的路况记录" in result["message"]
    assert "现场" in result["message"]


def test_empty_road_conditions_require_a_consistent_count_and_explicit_question():
    assert planner._vision_condition_summary({"conditions": [], "count": 2}, "查看路况") is None
    assert planner._vision_condition_summary({"conditions": [], "count": False}, "查看路况") is None
    assert planner._vision_condition_summary({"conditions": [], "count": 0}, "从总图到桂园") is None


def test_empty_query_does_not_override_route_with_new_active_conditions():
    replies = [_message(calls=[_call("list_road_conditions"), _call("plan_route")])]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    route = {"start_name": "总图", "end_name": "桂园", "mode": "walk",
             "recommended_length_m": 500, "road_conditions_applied": 1,
             "road_condition_overview": [{"name": "施工", "action": "closure", "road_name": "科技路"}]}
    def execute(name, _args, _context):
        return ({"conditions": [], "count": 0}, None) if name == "list_road_conditions" else (route, {"route": route})
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("先查当前路况，再从总图到桂园。")
    assert "没有已生效的路况记录" not in result["message"]
    assert "路况" in result["message"] and "不一致" in result["message"]


def test_empty_query_does_not_claim_current_state_when_route_snapshot_is_unknown():
    replies = [_message(calls=[_call("list_road_conditions"), _call("plan_route")])]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    route = {"start_name": "总图", "end_name": "桂园", "mode": "walk"}
    def execute(name, _args, _context):
        return ({"conditions": [], "count": 0}, None) if name == "list_road_conditions" else (route, {"route": route})
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("先查当前路况，再从总图到桂园。")
    assert "没有已生效的路况记录" not in result["message"]
    assert "无法核实" in result["message"]


def test_mixed_route_conditions_and_candidates_keep_the_candidate_names():
    replies = [_message(calls=[_call("list_road_conditions"), _call("plan_route"),
                              _call("search_poi_candidates")]),
               _message(content="路线和候选已经生成。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    route = {"start_name": "总图", "end_name": "桂园", "mode": "walk",
             "recommended_length_m": 500, "duration_min": 8}
    candidates = [{"poi_id": "poi_1", "name": "甲食堂", "distance_m": 100},
                  {"poi_id": "poi_2", "name": "乙食堂", "distance_m": 200}]

    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        if name == "plan_route":
            return route, {"route": route}
        return {"candidates": candidates}, {"candidates": candidates}

    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("规划这条路线，同时查询路况和备选地点。")
    assert result["route"] is route
    assert "甲食堂" in result["message"] and "乙食堂" in result["message"]
    assert "复核后发布" in result["message"]
    assert "自动发布" not in result["message"]


def test_multiple_route_artifacts_do_not_silently_replace_first_route():
    calls = [_call("plan_route"), _call("plan_route")]
    replies = [_message(calls=calls)]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    routes = [
        {"start_name": "总图", "end_name": "桂园", "mode": "walk", "recommended_length_m": 500},
        {"start_name": "桂园", "end_name": "星湖", "mode": "walk", "recommended_length_m": 700},
    ]

    def execute(_name, _args, _context):
        route = routes.pop(0)
        return route, {"route": route}

    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("先到桂园，再去星湖。")
    assert result.get("route") is None
    assert result["response_kind"] != "route"
    assert "完整" in result["message"]


def test_later_route_failure_does_not_present_first_leg_as_full_route():
    replies = [_message(calls=[_call("plan_route"), _call("plan_route")])]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    results = [
        ({"start_name": "总图", "end_name": "桂园"},
         {"route": {"start_name": "总图", "end_name": "桂园", "mode": "walk"}}),
        ({"error": "route_not_found", "reason_code": "graph_no_path"}, None),
    ]
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=lambda *_: results.pop(0)):
        result = planner.run_agent("先到桂园，再去星湖。")
    assert result.get("route") is None
    assert result["response_kind"] != "route"
    assert "完整" in result["message"]


def test_road_end_time_remains_visible():
    response = _run([_event(end_time="2026-10-29T10:00:00+08:00")],
                    final="科技路预计十点结束。", query="科技路什么时候解封？")
    assert "10:00" in response["message"]


def test_specific_event_type_is_not_hidden_behind_unrelated_records():
    events = [_event(id=f"e{i}", name=f"事件{i}", road_name=f"道路{i}", type_label="施工") for i in range(5)]
    events.append(_event(id="water", name="六号事件", road_name="科技路", type_label="积水"))
    response = _run(events, query="目前哪里有积水？")
    assert "积水" in response["message"] and "科技路" in response["message"]


def test_unrelated_automatic_menu_statement_is_not_removed_from_mixed_answer():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_place_details")]),
               _message(content="食堂已经自动发布了今日菜单，位于桂园。无人机识别后直接封路。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        return (({"conditions": [_event()], "count": 1}, None) if name == "list_road_conditions" else
                ({"found": True, "place": {"name": "桂园食堂", "campus": "桂园",
                                             "description": "食堂已经自动发布了今日菜单"}}, None))
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("当前路况的来源是什么？同时介绍一下这个建筑。")
    assert "食堂已经自动发布了今日菜单" in result["message"]
    assert "无人机识别后直接封路" not in result["message"]
    assert "复核后发布" in result["message"]


def test_mixed_answer_does_not_confirm_an_unverified_incident():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_place_details")]),
               _message(content="图像已经确认发生事故。桂园食堂位于桂园。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        return (({"conditions": [_event()], "count": 1}, None) if name == "list_road_conditions" else
                ({"found": True, "place": {"name": "桂园食堂", "location": "桂园"}}, None))
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("当前路况的来源是什么？同时介绍一下这个建筑。")
    assert "已经确认发生事故" not in result["message"]
    assert "桂园食堂" in result["message"] and "位置：桂园" in result["message"]


def test_later_road_condition_failure_does_not_reuse_earlier_success():
    replies = [_message(calls=[_call("list_road_conditions")]),
               _message(calls=[_call("list_road_conditions")]),
               _message(content="路况查询失败，请稍后重试。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    outcomes = [({"conditions": [_event()], "count": 1}, None),
                ({"error": "road_conditions_unavailable", "message": "路况查询失败"}, None)]
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=outcomes):
        result = planner.run_agent("现在有什么路况？")
    assert "失败" in result["message"]
    assert "实验路况" not in result["message"]


def test_road_failure_preserves_weather_result_and_temperature():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_weather")]),
               _message(content="路况暂不可用，天气晴。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"error": "road_conditions_unavailable", "message": "路况查询失败"}, None
        return {"weather": "晴", "temperature": 20}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("现在路况和天气如何？")
    assert "路况查询失败" in result["message"] and "晴" in result["message"]
    assert "20°C" in result["message"]


def test_road_and_weather_failure_are_both_explained():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_weather")]),
               _message(content="路况和天气查询失败。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        return {"error": "unavailable", "message": f"{name} failed"}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("现在路况和天气如何？")
    assert "路况查询失败" in result["message"] and "天气暂时获取不到" in result["message"]


def test_road_question_gives_unknown_end_time_when_not_recorded():
    response = _run([_event(end_time=None)], query="科技路什么时候解封？")
    assert "结束时间未记录" in response["message"]


def test_end_time_with_timezone_is_converted_to_china_time():
    response = _run([_event(end_time="2026-10-29T02:00:00Z")], query="科技路什么时候解封？")
    assert "10:00" in response["message"]


def test_road_and_weather_keeps_tool_advice():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_weather")]),
               _message(content="建议带伞。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        return {"weather": "小雨", "temperature": 20, "advice": "建议带伞"}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("现在路况如何？今天要带伞吗？")
    assert "建议带伞" in result["message"]


@pytest.mark.parametrize("model_answer", [
    "这条管制根据影像识别结果直接生效，无需管理人员干预。桂园食堂在桂园。",
    "系统依据监控画面完成封路发布。桂园食堂在桂园。",
    "桂园食堂在桂园且无人机图像已自动发布管制。",
    "无需人工复核，该信息是无人机识别结果。桂园食堂在桂园。",
])
def test_mixed_vision_publication_claims_use_tool_evidence(model_answer):
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_place_details")]),
               _message(content=model_answer)]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        return {"found": True, "place": {"name": "桂园食堂", "campus": "桂园"}}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("当前路况来源是什么？同时介绍一下这个建筑。")
    text = result["message"]
    assert "桂园食堂" in text and "桂园" in text and "复核后发布" in text
    assert not any(claim in text for claim in ("直接生效", "无需管理人员干预", "完成封路发布", "自动发布", "无需人工复核"))


def test_simulated_source_is_not_described_as_unknown_or_field_truth():
    response = _run([_event(source_kind="research_simulated", vision_evidence=None)])
    assert "实验模拟" in response["message"] and "不是现场事实" in response["message"]


def test_naive_end_time_does_not_assume_a_timezone():
    response = _run([_event(end_time="2026-10-29T10:00:00")], query="科技路什么时候解封？")
    assert "10:00" in response["message"] and "未注明时区" in response["message"]


def test_mixed_latest_road_failure_does_not_keep_old_road_assertion():
    replies = [_message(calls=[_call("list_road_conditions")]),
               _message(calls=[_call("list_road_conditions"), _call("get_place_details")]),
               _message(content="科技路目前仍处于封闭，桂园食堂在桂园。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "get_place_details":
            return {"found": True, "place": {"name": "桂园食堂", "location": "桂园"}}, None
        execute.count += 1
        return (({"conditions": [_event()], "count": 1}, None) if execute.count == 1 else
                ({"error": "road_conditions_unavailable", "message": "路况查询失败"}, None))
    execute.count = 0
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("当前路况来源是什么？同时介绍一下这个建筑。")
    assert "路况查询失败" in result["message"]
    assert "科技路目前仍处于封闭" not in result["message"]
    assert "桂园食堂" in result["message"] and "位置：桂园" in result["message"]


@pytest.mark.parametrize("model_answer", [
    "封闭的是科技路，桂园食堂在桂园。",
    "科技路现在不能走，桂园食堂在桂园。",
    "科技路现已解封，桂园食堂在桂园。",
    "科技门目前封了，桂园食堂在桂园。",
    "桂园食堂在桂园并且路况由无人机自动发布。",
])
def test_mixed_failure_uses_other_tool_evidence_without_unverified_road_claim(model_answer):
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_place_details")]),
               _message(content=model_answer)]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"error": "road_conditions_unavailable", "message": "路况查询失败"}, None
        return {"found": True, "place": {"name": "桂园食堂", "campus": "桂园",
                                         "description": "位于桂园"}}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("当前路况来源是什么？同时介绍一下这个建筑。")
    text = result["message"]
    assert "路况查询失败" in text and "桂园食堂" in text and "桂园" in text
    assert not any(claim in text for claim in ("封闭的是科技路", "现在不能走", "现已解封", "目前封了", "自动发布"))


def test_mixed_vision_event_uses_place_evidence_instead_of_contradictory_model_text():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_place_details")]),
               _message(content="桂园食堂在桂园并且路况由无人机自动发布。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        return {"found": True, "place": {"name": "桂园食堂", "campus": "桂园",
                                         "description": "位于桂园"}}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("当前路况来源是什么？同时介绍一下这个建筑。")
    assert "桂园食堂" in result["message"] and "桂园" in result["message"]
    assert "自动发布" not in result["message"] and "复核后发布" in result["message"]


def test_mixed_place_accessibility_and_unknown_opening_hours_are_not_lost():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_place_details")]),
               _message(content="无障碍情况未知。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        return {"found": True, "place": {"name": "星湖园食堂", "accessibility": "unknown",
                    "opening_hours": {"status": "known", "value": "unknown"}}}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("路况来源是什么？这个建筑的无障碍和开放时间如何？")
    assert "无障碍" in result["message"] and "未" in result["message"]
    assert "开放时间未登记" in result["message"] and "开放时间：unknown" not in result["message"]


def test_mixed_resolved_poi_is_not_silently_lost():
    replies = [_message(calls=[_call("list_road_conditions"), _call("resolve_poi")]),
               _message(content="找到了星湖园食堂。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        return {"found": True, "poi": {"name": "星湖园食堂"}}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("路况来源是什么？同时确认这个地点。")
    assert "星湖园食堂" in result["message"] and "复核后发布" in result["message"]


def test_mixed_route_tool_failure_is_not_silently_lost():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_place_details"), _call("plan_route")]),
               _message(content="路线没能生成。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        if name == "get_place_details":
            return {"found": True, "place": {"name": "桂园食堂"}}, None
        return {"error": "network_unavailable", "message": "校园路网暂不可用"}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("路况来源是什么？同时介绍这个建筑并规划路线。")
    assert "桂园食堂" in result["message"] and "路网暂不可用" in result["message"]


def test_latest_weather_failure_does_not_reuse_previous_weather():
    replies = [_message(calls=[_call("get_weather")]),
               _message(calls=[_call("get_weather"), _call("list_road_conditions")]),
               _message(content="天气查询失败。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        execute.count += 1
        return ({"weather": "晴", "temperature": 20}, None) if execute.count == 1 else (
            {"error": "weather_unavailable", "message": "天气暂时获取不到"}, None)
    execute.count = 0
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("路况和天气如何？")
    assert "天气暂时获取不到" in result["message"] and "晴" not in result["message"]


def test_poi_tool_does_not_treat_unknown_opening_time_as_known():
    result, _ = planner.agent_tools._tool_get_place_details({"name": "星湖园食堂"}, {})
    assert result["found"] is True
    assert result["place"]["opening_hours"]["status"] == "unknown"
    assert result["place"]["opening_hours"]["value"] is None


def test_route_failure_remains_visible_when_candidates_succeeded():
    replies = [_message(calls=[_call("list_road_conditions"), _call("search_poi_candidates"), _call("plan_route")]),
               _message(content="候选已找到，但路线失败。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    candidate = {"poi_id": "p1", "name": "桂园食堂"}
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        if name == "search_poi_candidates":
            return {"candidates": [candidate]}, {"candidates": [candidate]}
        return {"error": "network_unavailable", "message": "校园路网暂不可用"}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("路况来源是什么？同时处理其他地点和路线。")
    assert result["candidates"] == [candidate]
    assert "路网暂不可用" in result["message"]


def test_distinct_resolved_place_is_not_hidden_by_another_place_detail():
    replies = [_message(calls=[_call("list_road_conditions"), _call("resolve_poi"), _call("get_place_details")]),
               _message(content="已查询两个地点。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        if name == "resolve_poi":
            return {"found": True, "poi": {"name": "星湖园食堂"}}, None
        return {"found": True, "place": {"name": "桂园食堂", "campus": "桂园"}}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("路况来源是什么？同时确认两个地点。")
    assert "星湖园食堂" in result["message"] and "桂园食堂" in result["message"]


def test_retried_place_details_uses_latest_result_without_hiding_other_failure():
    first = _call("get_place_details", {"name": "桂园食堂"})
    second = _call("get_place_details", {"name": "桂园食堂"})
    other = _call("get_place_details", {"name": "星湖园食堂"})
    replies = [_message(calls=[_call("list_road_conditions"), first]),
               _message(calls=[second, other]), _message(content="已查询地点。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        if args.get("name") == "星湖园食堂":
            return {"error": "unavailable", "message": "星湖园食堂详情不可用"}, None
        execute.count += 1
        return ({"error": "unavailable", "message": "桂园食堂详情不可用"}, None) if execute.count == 1 else (
            {"found": True, "place": {"name": "桂园食堂", "campus": "桂园"}}, None)
    execute.count = 0
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("路况来源是什么？同时确认两个地点的详情。")
    assert "桂园食堂" in result["message"] and "桂园食堂详情不可用" not in result["message"]
    assert "星湖园食堂详情不可用" in result["message"]


def test_same_name_different_poi_ids_are_not_collapsed():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_place_details"), _call("resolve_poi")]),
               _message(content="已查询两个地点。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        if name == "get_place_details":
            return {"found": True, "place": {"poi_id": "a", "name": "篮球场", "campus": "工学部"}}, None
        return {"found": True, "poi": {"poi_id": "b", "name": "篮球场", "campus": "信息学部"}}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("路况来源是什么？同时确认两个地点。")
    assert "工学部" in result["message"] and "信息学部" in result["message"]


def test_different_route_failure_is_not_hidden_by_a_successful_route():
    replies = [_message(calls=[_call("list_road_conditions"),
                               _call("plan_route", {"start": {"name": "甲"}, "end": {"name": "乙"}}),
                               _call("plan_route", {"start": {"name": "丙"}, "end": {"name": "丁"}})]),
               _message(content="部分路线已规划。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    route = {"start_name": "甲", "end_name": "乙", "mode": "walk", "recommended_length_m": 500,
             "duration_min": 8}
    def execute(name, args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        if args.get("start", {}).get("name") == "甲":
            return route, {"route": route}
        return {"error": "network_unavailable", "message": "丙到丁的路网暂不可用"}, None
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("查询路况来源并分别处理两段路线。")
    assert result["route"] is route
    assert "丙到丁的路网暂不可用" in result["message"]


def test_route_vision_and_weather_preserve_the_weather_artifact():
    replies = [_message(calls=[_call("list_road_conditions"), _call("get_weather"), _call("plan_route")]),
               _message(content="路线和天气已查询。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    route = {"start_name": "甲", "end_name": "乙", "mode": "walk", "recommended_length_m": 500,
             "duration_min": 8}
    weather = {"weather": "晴", "temperature": 20}
    def execute(name, _args, _context):
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        if name == "get_weather":
            return weather, None
        return route, {"route": route}
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("查询路况来源和天气，同时规划路线。")
    assert result["route"] is route and result["weather"] == weather
    assert "晴" in result["message"] and result["message"].count("天气参考") == 1
