"""Planner（Agent 循环）单元测试：全部 mock LLM 客户端与工具执行，不依赖网络和真实路网。"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agents.planner as planner


def _fake_msg(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    msg.model_dump = lambda exclude_none=False: {
        "role": "assistant",
        **({"content": content} if content else {}),
        **({"tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (tool_calls or [])
        ]} if tool_calls else {}),
    }
    return msg


def _fake_response(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=_fake_msg(content, tool_calls))])


def _fake_tool_call(name, args):
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
    )


class FakeClient:
    """按脚本依次返回响应的伪 LLM 客户端。"""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs.get("messages") or [])})
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", "test-key")


def _run(client, **kwargs):
    with patch.object(planner, "_make_client", return_value=client):
        return planner.run_agent("测试 query", **kwargs)


def test_model_client_requests_supported_response_encoding():
    with patch("openai.OpenAI") as client_constructor:
        planner._make_client()
    assert client_constructor.call_args.kwargs["default_headers"]["Accept-Encoding"] == "gzip, deflate"
    assert client_constructor.call_args.kwargs["max_retries"] == 0


def test_route_summary_names_the_ordered_intermediate_stop():
    route = {
        "start_name": "信息学部", "end_name": "星湖园食堂", "mode": "walk",
        "recommended_length_m": 684, "duration_min": 9.1,
        "legs": [
            {"end_name": "武汉大学信息学部篮球场"},
            {"end_name": "星湖园食堂"},
        ],
    }
    message = planner._build_route_message(route)
    assert "从信息学部经武汉大学信息学部篮球场到星湖园食堂" in message
    assert "684 米" in message


class TestAgentLoop:
    def test_requested_road_condition_feedback_describes_records_without_claiming_route_impact(self):
        route = {"mode": "walk", "road_conditions_applied": 2,
                 "road_condition_overview": [
                     {"name": "施工管制", "road_name": "桂园路", "action": "closure"},
                     {"name": "临时提醒", "road_name": "", "action": "notice"},
                 ]}
        text = planner._road_condition_feedback(route, "先查当前路况再规划路线")
        assert "桂园路" in text and "封闭" in text
        assert "位置未标注" in text and "提醒" in text
        assert "不一定影响" in text
        assert "因此绕行" not in text

    def test_road_feedback_prioritizes_verified_on_route_notice_and_reports_omissions(self):
        overview = [{"id": str(i), "name": f"事件{i}", "road_name": "", "action": "notice"}
                    for i in range(4)]
        route = {"mode": "walk", "road_conditions_applied": 4, "road_condition_overview": overview,
                 "route_events": [{"id": "3", "action": "notice"}]}
        for query in ("为什么绕行？", "路上有积水吗？", "当前路况如何？"):
            message = planner._road_condition_feedback(route, query)
            assert "事件3" in message and "沿途" in message
            assert "另有 1 条" in message
            assert "不一定影响" in message

    def test_no_path_reply_describes_user_step_constraint(self):
        args = {"start": {"name": "梅园综合楼"}, "end": {"name": "图书馆"},
                "mode": "walk", "constraints": {"avoid_steps": True}}
        client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("plan_route", args)])])
        failure = {"error": "route_not_found", "reason_code": "graph_no_path",
                   "route_request": {"start": args["start"], "end": args["end"],
                                     "travel_mode": "walk", "hard_constraints": args["constraints"]}}
        with patch.object(planner.agent_tools, "execute_tool", return_value=(failure, None)):
            result = _run(client)
        assert result["response_kind"] == "no_route"
        assert "台阶" in result["message"]
        assert "现场" in result["message"]

    def test_no_path_reports_current_closure_without_claiming_its_cause(self):
        args = {"start": {"name": "梅园综合楼"}, "end": {"name": "图书馆"}, "mode": "walk"}
        client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("plan_route", args)])])
        failure = {"error": "route_not_found", "reason_code": "graph_no_path",
                   "route_request": {"start": args["start"], "end": args["end"],
                                     "travel_mode": "walk", "hard_constraints": {}},
                   "road_conditions_applied": 1,
                   "road_condition_overview": [{"name": "施工管制", "road_name": "桂园路", "action": "closure"}]}
        with patch.object(planner.agent_tools, "execute_tool", return_value=(failure, None)), \
                patch.object(planner, "_make_client", return_value=client):
            result = planner.run_agent("请先查当前路况，从梅园综合楼到图书馆", history=[])
        assert result["response_kind"] == "no_route"
        assert "桂园路" in result["message"] and "封闭" in result["message"]
        assert "关系尚未确认" in result["message"]
        assert "因此无法" not in result["message"]

    def test_typed_graph_no_path_returns_complete_refusal_without_more_model_calls(self):
        args = {"start": {"name": "梅园综合楼"},
                "end": {"name": "武汉大学图书馆(总馆)"}, "mode": "walk"}
        client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("plan_route", args)])])
        failure = {
            "error": "route_not_found", "reason_code": "graph_no_path",
            "message": "graph has no route", "request_digest": "a" * 64,
            "route_request": {
                "start": args["start"], "end": args["end"],
                "travel_mode": "walk", "strategy": {"name": "shortest"},
                "hard_constraints": {},
            },
        }
        trace = []
        with patch.object(planner.agent_tools, "execute_tool",
                          return_value=(failure, None)):
            result = _run(client, tool_trace_callback=trace.append)

        assert len(client.calls) == 1
        assert result["response_kind"] == "no_route"
        assert result["route"] is None
        assert result["route_state"]["start"] == args["start"]
        assert result["route_state"]["end"] == args["end"]
        assert "当前路网" in result["message"]
        assert trace[0]["error_code"] == "graph_no_path"

    @pytest.mark.parametrize("no_path_first", [False, True])
    def test_successful_route_in_same_tool_batch_takes_precedence_over_no_path(
            self, no_path_first):
        route = {"start_name": "梅园综合楼", "end_name": "图书馆", "mode": "walk",
                 "recommended_length_m": 900, "duration_min": 12}
        failure = {"error": "route_not_found", "reason_code": "graph_no_path",
                   "route_request": {"start": {"name": "梅园综合楼"},
                                     "end": {"name": "图书馆"},
                                     "travel_mode": "walk"}}
        calls = [_fake_tool_call("plan_route", {"start": {"name": "梅园综合楼"},
                                                "end": {"name": "图书馆"}})
                 for _ in range(2)]
        calls[0].id, calls[1].id = "first", "second"
        client = FakeClient([_fake_response(tool_calls=calls)])
        outcomes = [(failure, None), (route, {"route": route})]
        if not no_path_first:
            outcomes.reverse()
        with patch.object(planner.agent_tools, "execute_tool", side_effect=outcomes) as execute:
            result = _run(client)
        assert execute.call_count == 2
        assert result["response_kind"] == "route"
        assert result["route"] is route

    @pytest.mark.parametrize("terminal_kind", ["clarify", "route_comparison"])
    @pytest.mark.parametrize("no_path_first", [False, True])
    def test_terminal_artifact_in_same_batch_takes_precedence_over_no_path(
            self, terminal_kind, no_path_first):
        failure = {"error": "route_not_found", "reason_code": "graph_no_path",
                   "route_request": {"start": {"name": "起点"},
                                     "end": {"name": "终点"},
                                     "travel_mode": "walk"}}
        if terminal_kind == "clarify":
            terminal = ({"question": "请选目的地"},
                        {"clarify": {"question": "请选目的地", "options": []}})
            tool_name = "ask_user"
        else:
            comparison = {"status": "partial", "alternatives": [], "unavailable": []}
            terminal = (comparison, {"route_comparison": comparison})
            tool_name = "compare_routes"
        calls = [_fake_tool_call("plan_route", {}), _fake_tool_call(tool_name, {})]
        outcomes = [(failure, None), terminal]
        if not no_path_first:
            calls.reverse()
            outcomes.reverse()
        client = FakeClient([_fake_response(tool_calls=calls)])
        with patch.object(planner.agent_tools, "execute_tool", side_effect=outcomes):
            result = _run(client)
        assert result["response_kind"] == terminal_kind

    def test_weather_snapshot_is_shared_across_tools_only_within_one_run(self):
        snapshot = {"live": {"weather": "晴"}, "impact": {}}
        seen = []

        def execute(name, _args, tool_context):
            seen.append((name, tool_context.get("_route_weather_snapshot")))
            if name == "get_weather":
                tool_context["_route_weather_snapshot"] = snapshot
            return {"ok": True}, None

        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("get_weather", {})]),
            _fake_response(tool_calls=[_fake_tool_call("plan_route", {})]),
            _fake_response(content="完成"),
        ])
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
            planner.run_agent("先看天气再规划路线")
        assert seen == [("get_weather", None), ("plan_route", snapshot)]

        another = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("get_weather", {})]),
            _fake_response(content="完成"),
        ])
        with patch.object(planner, "_make_client", return_value=another), \
             patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
            planner.run_agent("另一轮天气")
        assert seen[-1] == ("get_weather", None)

    def test_compound_weather_route_response_includes_observed_weather(self):
        route = {"start_name": "玉兰2门", "end_name": "老斋舍", "mode": "walk",
                 "recommended_length_m": 1000, "duration_min": 15}
        weather = {"weather": "晴", "temperature": 22, "windpower": "≤3"}
        client = FakeClient([_fake_response(tool_calls=[
            _fake_tool_call("get_weather", {}),
            _fake_tool_call("plan_route", {"start": {"name": "玉兰2门"},
                                           "end": {"name": "老斋舍"}}),
        ])])

        def execute(name, _args, _context):
            if name == "get_weather":
                return weather, None
            return route, {"route": route, "route_kind": "direct"}

        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
            result = planner.run_agent("先看天气，再规划从玉兰2门到老斋舍的路线")
        assert result["response_kind"] == "route"
        assert "晴" in result["message"]
        assert "22" in result["message"]
        assert result["weather"] == weather

    @pytest.mark.parametrize("applied,expected", [
        (1, "纳入 1 条适用于步行的生效路况记录"),
        (0, "未应用生效路况记录"),
    ])
    def test_route_feedback_answers_road_condition_question_from_route_snapshot(
            self, applied, expected):
        route = {"start_name": "玉兰2门", "end_name": "老斋舍", "mode": "walk",
                 "road_conditions_applied": applied}
        client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("plan_route", {})])])
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route})):
            result = planner.run_agent("从玉兰2门到老斋舍，当前有无道路管制？")
        assert result["response_kind"] == "route"
        assert expected in result["message"]

    def test_named_destination_waiting_for_origin_is_structured_task(self):
        destination = {"id": "poi_141", "name": "凌波门东湖观景点"}
        client = FakeClient([_fake_response(content="你从哪里出发？")])
        with patch.object(planner.agent_tools, "find_poi_ambiguous",
                          return_value=(destination, [])), \
             patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool") as execute:
            result = planner.run_agent("带我去凌波门东湖观景点，起点等我补充。")
        assert result["response_kind"] == "clarify"
        assert result["place_request"]["pending_slot"] == "start"
        assert result["place_request"]["selected_poi_id"] == "poi_141"
        assert not client.calls
        execute.assert_not_called()

    @pytest.mark.parametrize("query,expected_strategy,expected_end,expected_constraints", [
        ("刚才的路线改为平坦优先", "flat", "樱顶", {}),
        ("刚才的路线改为最短路径", "shortest", "樱顶", {}),
        ("刚才的路线终点改为星湖园食堂", "shortest", "星湖园食堂", {}),
        ("刚才的路线避开陡坡", "shortest", "樱顶", {"slope": "avoid"}),
    ])
    def test_explicit_route_edit_keeps_via_and_unchanged_slots_without_llm(
            self, query, expected_strategy, expected_end, expected_constraints):
        previous = {
            "schema_version": 1, "route_id": "route-before", "route_kind": "via",
            "original_query": "从玉兰2门经卓尔体育馆到樱顶",
            "start": {"name": "玉兰2门", "type": "poi"},
            "end": {"name": "樱顶", "type": "poi"},
            "via": {"name": "卓尔体育馆", "type": "poi"},
            "tour": None, "legs": [], "travel_mode": "walk",
            "hard_constraints": {}, "strategy": {"name": "shortest", "source": "commute_default"},
            "data_version": "data-test", "road_condition_version": "roads-test",
        }
        route = {"recommended": [[1, 2]], "start_name": "玉兰2门",
                 "end_name": expected_end, "mode": "walk"}
        with patch.object(planner, "_make_client") as client, \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route, "route_kind": "via"})) as execute:
            result = planner.run_agent(query, context={
                "previous_route_state": previous, "task_revision": 12,
            })
        client.assert_not_called()
        name, args, _context = execute.call_args.args
        assert name == "plan_via_route"
        assert _context["task_revision"] == 12
        assert args["start"]["name"] == "玉兰2门"
        assert args["end"]["name"] == expected_end
        assert args["via_name"] == "卓尔体育馆"
        assert args["strategy"] == expected_strategy
        assert args["constraints"] == expected_constraints
        assert result["response_kind"] == "route"

    def test_deterministic_route_edit_preserves_verification_failure_report(self):
        previous = {
            "schema_version": 1, "route_id": "route-before", "route_kind": "direct",
            "original_query": "从玉兰2门到樱顶",
            "start": {"name": "玉兰2门", "type": "poi"},
            "end": {"name": "樱顶", "type": "poi"},
            "via": None, "tour": None, "legs": [], "travel_mode": "walk",
            "hard_constraints": {},
            "strategy": {"name": "shortest", "source": "commute_default"},
            "data_version": "data-test", "road_condition_version": "roads-test",
        }
        report = {"status": "unknown", "unknowns": ["endpoint_unknown"]}
        failure = {"error": "route_verification_unknown", "message": "缺少独立核验依据。",
                   "route_verification": report}
        with patch.object(planner, "_make_client") as client, \
             patch.object(planner.agent_tools, "execute_tool", return_value=(failure, None)):
            result = planner.run_agent(
                "刚才的路线改为平坦优先",
                context={"previous_route_state": previous},
            )

        client.assert_not_called()
        assert result["response_kind"] == "chat"
        assert result["route"] is None
        assert result["route_verification"] == report

    def test_spoken_edit_preserves_ordered_multi_via_and_hard_constraint(self):
        previous = {
            "schema_version": 1, "route_id": "route-before", "route_kind": "via",
            "original_query": "从玉兰2门经卓尔体育馆、星湖园食堂到樱顶，不走台阶",
            "start": {"name": "玉兰2门", "type": "poi"},
            "end": {"name": "樱顶", "type": "poi"},
            "via": {"type": "multi", "points": [
                {"name": "卓尔体育馆", "type": "poi"},
                {"name": "星湖园食堂", "type": "poi"},
            ]},
            "tour": None, "legs": [], "travel_mode": "walk",
            "hard_constraints": {"avoid_steps": True},
            "strategy": {"name": "shortest", "source": "commute_default"},
            "data_version": "data-test", "road_condition_version": "roads-test",
        }
        route = {"recommended": [[1, 2]], "start_name": "玉兰2门",
                 "end_name": "樱顶", "mode": "walk"}
        with patch.object(planner, "_make_client") as client, \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route, "route_kind": "via"})) as execute:
            result = planner.run_agent(
                "把刚才的路线改为风景优先",
                context={"previous_route_state": previous},
            )
        client.assert_not_called()
        name, args, _context = execute.call_args.args
        assert name == "plan_via_route"
        assert [item["name"] for item in args["via_points"]] == ["卓尔体育馆", "星湖园食堂"]
        assert args["constraints"] == {"avoid_steps": True}
        assert args["strategy"] == "scenery"
        assert result["route_kind"] == "via"

    def test_spoken_flat_edit_after_weather_keeps_avoid_steps_without_llm(self):
        previous = {
            "schema_version": 1, "route_id": "route-before", "route_kind": "direct",
            "original_query": "从玉兰2门到樱顶，别走已标记台阶",
            "start": {"name": "玉兰2门", "type": "poi"},
            "end": {"name": "樱顶", "type": "poi"},
            "via": None, "tour": None, "legs": [], "travel_mode": "walk",
            "hard_constraints": {"avoid_steps": True},
            "strategy": {"name": "shortest", "source": "explicit_nl"},
            "data_version": "data-test", "road_condition_version": "roads-test",
        }
        route = {"recommended": [[1, 2]], "start_name": "玉兰2门",
                 "end_name": "樱顶", "mode": "walk"}
        with patch.object(planner, "_make_client") as client, \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route, "route_kind": "direct"})) as execute:
            result = planner.run_agent(
                "刚才路线改为平坦优先，别走台阶仍然保留",
                context={"previous_route_state": previous,
                         "history": [{"role": "user", "content": "顺便查天气"}]},
            )
        client.assert_not_called()
        name, args, _ = execute.call_args.args
        assert name == "plan_route"
        assert args["strategy"] == "flat"
        assert args["constraints"] == {"avoid_steps": True}
        assert result["response_kind"] == "route"

    def test_explicit_mode_followup_replans_saved_via_route_without_llm(self):
        previous = {
            "schema_version": 1, "route_id": "route-before", "route_kind": "via",
            "original_query": "从玉兰2门经卓尔体育馆到樱顶，不走台阶",
            "start": {"name": "玉兰2门", "type": "poi"},
            "end": {"name": "樱顶", "type": "poi"},
            "via": {"name": "卓尔体育馆", "type": "poi"},
            "tour": None, "legs": [], "travel_mode": "walk",
            "hard_constraints": {"avoid_steps": True},
            "strategy": {"name": "flat", "source": "explicit_nl"},
            "data_version": "data-test", "road_condition_version": "roads-test",
        }
        route = {"recommended": [[1, 2]], "start_name": "玉兰2门",
                 "end_name": "樱顶", "mode": "bike"}
        with patch.object(planner, "_make_client") as client, \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route, "route_kind": "via"})) as execute:
            result = planner.run_agent(
                "刚才的路线改成骑行",
                context={"previous_route_state": previous}, travel_mode="walk",
            )
        client.assert_not_called()
        name, args, ctx = execute.call_args.args
        assert name == "plan_via_route"
        assert args["start"]["name"] == "玉兰2门"
        assert args["end"]["name"] == "樱顶"
        assert args["via_name"] == "卓尔体育馆"
        assert args["constraints"] == {"avoid_steps": True}
        assert args["strategy"] == "flat"
        assert args["mode"] == "bike"
        assert ctx["query"] == previous["original_query"]
        assert result["response_kind"] == "route"

    def test_mode_and_endpoint_edit_keeps_other_route_conditions(self):
        previous = {
            "schema_version": 1, "route_id": "route-before", "route_kind": "via",
            "original_query": "从玉兰2门经卓尔体育馆到樱顶，不走台阶",
            "start": {"name": "玉兰2门", "type": "poi"},
            "end": {"name": "樱顶", "type": "poi"},
            "via": {"name": "卓尔体育馆", "type": "poi"},
            "tour": None, "legs": [], "travel_mode": "walk",
            "hard_constraints": {"avoid_steps": True},
            "strategy": {"name": "shortest", "source": "explicit_nl"},
            "data_version": "data-test", "road_condition_version": "roads-test",
        }
        route = {"recommended": [[1, 2]], "start_name": "玉兰2门",
                 "end_name": "星湖园食堂", "mode": "bike"}
        with patch.object(planner, "_make_client") as client, \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route, "route_kind": "via"})) as execute:
            planner.run_agent(
                "改成骑行，终点换成星湖园食堂；起点、原有约束和最短路径要求保持不变。",
                context={"previous_route_state": previous}, travel_mode="walk",
            )
        client.assert_not_called()
        _, args, _ = execute.call_args.args
        assert args["start"]["name"] == "玉兰2门"
        assert args["end"]["name"] == "星湖园食堂"
        assert args["via_name"] == "卓尔体育馆"
        assert args["constraints"] == {"avoid_steps": True}
        assert args["strategy"] == "shortest"
        assert args["mode"] == "bike"

    def test_mode_and_both_endpoint_edits_do_not_drop_first_endpoint(self):
        previous = {
            "schema_version": 1, "route_id": "route-before", "route_kind": "direct",
            "original_query": "从玉兰2门到樱顶，不走台阶",
            "start": {"name": "玉兰2门", "type": "poi"},
            "end": {"name": "樱顶", "type": "poi"}, "via": None,
            "tour": None, "legs": [], "travel_mode": "walk",
            "hard_constraints": {"avoid_steps": True},
            "strategy": {"name": "shortest", "source": "explicit_nl"},
            "data_version": "data-test", "road_condition_version": "roads-test",
        }
        route = {"recommended": [[1, 2]], "start_name": "总图",
                 "end_name": "星湖园食堂", "mode": "bike"}
        with patch.object(planner, "_make_client") as client, \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route})) as execute:
            planner.run_agent("改成骑行，把起点换成总图，终点换成星湖园食堂。",
                              context={"previous_route_state": previous}, travel_mode="walk")
        client.assert_not_called()
        _, args, _ = execute.call_args.args
        assert args["start"]["name"] == "总图"
        assert args["end"]["name"] == "星湖园食堂"
        assert args["constraints"] == {"avoid_steps": True}
        assert args["mode"] == "bike"

    @pytest.mark.parametrize("weather", [
        {"weather": "小雨", "temperature": 18},
        {"error": "weather_unavailable", "message": "天气暂时获取不到"},
    ])
    def test_mode_followup_also_delivers_requested_weather(self, weather):
        previous = {
            "schema_version": 1, "route_id": "route-before", "route_kind": "via",
            "original_query": "从玉兰2门经卓尔体育馆到樱顶，不走台阶",
            "start": {"name": "玉兰2门", "type": "poi"},
            "end": {"name": "樱顶", "type": "poi"},
            "via": {"name": "卓尔体育馆", "type": "poi"},
            "tour": None, "legs": [], "travel_mode": "walk",
            "hard_constraints": {"avoid_steps": True},
            "strategy": {"name": "shortest", "source": "explicit_nl"},
            "data_version": "data-test", "road_condition_version": "roads-test",
        }
        route = {"recommended": [[1, 2]], "start_name": "玉兰2门",
                 "end_name": "樱顶", "mode": "bike"}
        def execute(name, args, _context):
            if name == "get_weather":
                return weather, None
            return route, {"route": route, "route_kind": "via"}
        with patch.object(planner.agent_tools, "execute_tool", side_effect=execute) as tool:
            result = planner.run_agent("刚才路线改成骑行，顺便查天气",
                                       context={"previous_route_state": previous})
        assert [call.args[0] for call in tool.call_args_list] == ["get_weather", "plan_via_route"]
        assert tool.call_args.args[1]["via_name"] == "卓尔体育馆"
        assert tool.call_args.args[1]["constraints"] == {"avoid_steps": True}
        assert result["response_kind"] == "route"
        if "error" in weather:
            assert "天气暂时获取不到" in result["message"]
        else:
            assert result["weather"] == weather
            assert "小雨" in result["message"] and "18" in result["message"]

    def test_named_via_route_also_delivers_requested_weather(self):
        weather = {"weather": "晴", "temperature": 22}
        route = {"recommended": [[1, 2]], "start_name": "玉兰2门",
                 "end_name": "樱顶", "mode": "walk"}
        def execute(name, args, _context):
            if name == "get_weather":
                return weather, None
            return route, {"route": route, "route_kind": "via"}
        with patch.object(planner.agent_tools, "execute_tool", side_effect=execute) as tool:
            result = planner.run_agent("从玉兰2门经卓尔体育馆到樱顶，顺便查天气")
        assert [call.args[0] for call in tool.call_args_list] == ["get_weather", "plan_via_route"]
        assert result["weather"] == weather
        assert "晴" in result["message"] and "22" in result["message"]

    def test_named_via_failure_keeps_weather_in_visible_clarification(self):
        weather = {"weather": "晴", "temperature": 22}
        def execute(name, args, _context):
            if name == "get_weather":
                return weather, None
            return {"error": "not_found", "message": "途经地点未找到"}, None
        with patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
            result = planner.run_agent("从玉兰2门经卓尔体育馆到樱顶，顺便查天气")
        assert result["response_kind"] == "clarify"
        assert "途经地点未找到" in result["clarify"]["question"]
        assert "晴" in result["clarify"]["question"] and "22" in result["clarify"]["question"]

    def test_nearest_place_route_delivers_requested_weather(self):
        start = {"lng": 114.31, "lat": 30.51, "name": "地图起点"}
        weather = {"weather": "晴", "temperature": 22}
        candidate = {"name": "星湖园食堂", "distance_m": 320, "subcategory": "canteen"}
        route = {"recommended": [[114.31, 30.51]], "start_name": "地图起点",
                 "end_name": "星湖园食堂", "mode": "walk"}
        def execute(name, args, _context):
            if name == "search_poi_candidates":
                return {"start_name": "地图起点"}, {"candidates": [candidate]}
            if name == "get_weather":
                return weather, None
            return route, {"route": route}
        with patch.object(planner.agent_tools, "execute_tool", side_effect=execute) as tool:
            result = planner.run_agent("带我去最近的食堂，顺便查天气", coord_start=start)
        assert [call.args[0] for call in tool.call_args_list] == [
            "search_poi_candidates", "get_weather", "plan_route",
        ]
        assert result["weather"] == weather
        assert "晴" in result["message"] and "22" in result["message"]

    def test_via_first_request_without_start_asks_before_planning(self):
        with patch.object(planner, "_make_client") as client, \
             patch.object(planner.agent_tools, "execute_tool") as execute:
            result = planner.run_agent("经卓尔体育馆去樱顶，不走台阶，顺便查天气")
        assert result["response_kind"] == "clarify"
        assert "从哪里出发" in result["clarify"]["question"]
        client.assert_not_called()
        execute.assert_not_called()

    def test_same_task_named_destination_is_used_for_deictic_weather_followup(self):
        destination = {"id": "poi_070", "name": "桂园食堂"}
        route = {"recommended": [[114.36, 30.53], [114.37, 30.53]],
                 "start_name": "武汉大学科技门", "end_name": "桂园食堂", "mode": "walk"}
        for origin in ("武汉大学科技门", "武汉大学茶港门", "玉兰2门"):
            with patch.object(planner.agent_tools, "find_poi_ambiguous",
                              return_value=(destination, [])), \
                 patch.object(planner.agent_tools, "execute_tool",
                              return_value=(route, {"route": route})) as execute, \
                 patch.object(planner, "_make_client") as client:
                result = planner.run_agent(
                    f"我在{origin}，查天气后再带我过去",
                    context={"active_task_request": "带我去桂园食堂"},
                )
            assert result["response_kind"] == "route"
            assert execute.call_args.args[0] == "plan_route"
            assert execute.call_args.args[1]["start"]["name"] == origin
            assert execute.call_args.args[1]["end"]["poi_id"] == "poi_070"
            client.assert_not_called()

    def test_deictic_weather_request_without_same_task_destination_asks_once(self):
        with patch.object(planner, "_make_client") as client:
            first = planner.run_agent("我在武汉大学科技门，查天气后再带我过去")
        assert first["response_kind"] == "clarify"
        assert "目的地" in first["message"]
        assert first["place_request"]["start"]["name"] == "武汉大学科技门"
        client.assert_not_called()
        destination = {"id": "poi_070", "name": "桂园食堂"}
        route = {"recommended": [[114.36, 30.53], [114.37, 30.53]],
                 "start_name": "武汉大学科技门", "end_name": "桂园食堂", "mode": "walk"}
        with patch.object(planner.agent_tools, "find_poi_ambiguous",
                          return_value=(destination, [])), \
             patch.object(planner.agent_tools, "load_pois", return_value=[destination]), \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route})) as execute:
            resumed = planner.run_agent("桂园食堂", context={"place_request": first["place_request"]})
        assert resumed["response_kind"] == "route"
        assert execute.call_args.args[1]["start"]["name"] == "武汉大学科技门"

    def test_deictic_weather_followup_does_not_drop_an_original_named_stop(self):
        route = {"recommended": [[114.36, 30.53], [114.37, 30.53]],
                 "start_name": "玉兰2门", "end_name": "樱顶", "mode": "walk"}
        with patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route})) as execute, \
             patch.object(planner, "_make_client") as client:
            result = planner.run_agent(
                "我在玉兰2门，查天气后再带我过去",
                context={"active_task_request": "从科技门经卓尔体育馆到樱顶"},
            )
        assert result["response_kind"] == "route", result
        assert result["route_kind"] == "via"
        assert execute.call_args.args[0] == "plan_via_route"
        assert execute.call_args.args[1]["via_points"][0]["name"] == "卓尔体育馆"
        client.assert_not_called()

    def test_via_first_clarification_plans_original_stop_and_constraints(self):
        original = "经卓尔体育馆去樱顶，不走台阶，顺便查天气"
        route = {"recommended": [[1, 2]], "start_name": "玉兰2门",
                 "end_name": "樱顶", "mode": "walk"}
        with patch.object(planner, "_make_client") as client, \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route, "route_kind": "via"})) as execute:
            result = planner.run_agent(
                "从玉兰2门出发", context={"active_task_request": original},
                travel_mode="walk",
            )
        client.assert_not_called()
        name, args, tool_context = execute.call_args.args
        assert name == "plan_via_route"
        assert args["start"]["name"] == "玉兰2门"
        assert args["end"]["name"] == "樱顶"
        assert [point["name"] for point in args["via_points"]] == ["卓尔体育馆"]
        assert args["constraints"] == {"avoid_steps": True}
        assert original in tool_context["query"]
        assert result["route_kind"] == "via"

    def test_clarification_keeps_original_constraints_in_tool_call(self):
        original = "从玉兰2门到樱顶，经过卓尔体育馆，不走台阶，避开陡坡"
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("plan_via_route", {
                "start": {"name": "信息学部"}, "end": {"name": "樱顶"},
                "via_points": [{"name": "卓尔体育馆"}],
            })]),
        ])
        route = {"recommended": [], "distance_m": 900,
                 "start_name": "信息学部", "end_name": "樱顶", "mode": "walk"}
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route})) as execute:
            planner.run_agent(
                "从信息学部出发",
                context={"active_task_request": original,
                         "history": [{"role": "user", "content": original},
                                     {"role": "assistant", "content": "你从哪里出发？"}]},
            )
        tool_name, args, tool_context = execute.call_args.args
        assert tool_name == "plan_via_route"
        assert args["constraints"] == {"slope": "avoid", "avoid_steps": True}
        assert original in tool_context["query"]
        assert any(original in str(message["content"]) for message in client.calls[0]["messages"])

    def test_explicit_clarification_correction_replaces_original_route_preferences(self):
        original = "从玉兰2门到樱顶，风景优先，不走台阶，避开陡坡"
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("plan_route", {
                "start": {"name": "玉兰2门"}, "end": {"name": "樱顶"},
                "weights": {"distance": 0.2, "slope": 0.1, "scenery": 0.7},
            })]),
        ])
        route = {"recommended": [], "distance_m": 900,
                 "start_name": "玉兰2门", "end_name": "樱顶", "mode": "walk"}
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route})) as execute:
            planner.run_agent(
                "改为最短路线，可以走台阶，不用避开陡坡了",
                context={"active_task_request": original},
            )
        _name, args, _context = execute.call_args.args
        assert "weights" not in args
        assert args.get("constraints", {}).get("avoid_steps") is not True
        assert args.get("constraints", {}).get("slope") != "avoid"

    def test_direct_answer_no_tool(self):
        client = FakeClient([_fake_response(content="今天适合散步，要去樱顶吗？")])
        resp = _run(client)
        assert resp["response_kind"] == "chat"
        assert resp["message"] == "今天适合散步，要去樱顶吗？"
        assert resp["turns"] == 1

    def test_tool_call_then_answer(self):
        route_payload = {
            "recommended": [{"lng": 1, "lat": 2}], "distance_m": 800,
            "start_name": "牌坊", "end_name": "樱顶", "mode": "walk",
        }
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("plan_route", {"start": {"name": "牌坊"}, "end": {"name": "樱顶"}})]),
        ])
        with patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route_payload, {"route": route_payload})) as mock_exec:
            resp = _run(client)
        assert mock_exec.call_count == 1
        assert resp["response_kind"] == "route"
        assert resp["route"] == route_payload
        assert resp["route_kind"] == "direct"
        # 路线产出后立即收尾：不再多调一轮 LLM（防 DSML 泄漏），消息由结构化数据拼出
        assert resp["message"] == "已为你规划好从牌坊到樱顶的步行路线，约 800 米。"
        assert len(client.calls) == 1

    def test_tool_trace_callback_records_safe_tool_metadata_only(self):
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("get_weather", {"location": "武汉大学"})]),
            _fake_response(content="当前天气晴。"),
        ])
        trace = []
        with patch.object(planner.agent_tools, "execute_tool",
                          return_value=({"weather": "晴", "temperature": 22}, None)):
            result = _run(client, tool_trace_callback=trace.append)

        assert trace == [{
            "tool_name": "get_weather",
            "argument_keys": ["location"],
            "outcome": "success",
            "artifact_kind": None,
        }]
        assert "武汉大学" not in json.dumps(trace, ensure_ascii=False)
        assert "晴" not in json.dumps(trace, ensure_ascii=False)
        assert result["message"] == "当前天气晴。"

    def test_road_event_trace_ids_come_only_from_successful_tool_result(self):
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("list_road_conditions", {})]),
            _fake_response(content="路况已查询。"),
        ])
        trace = []
        with patch.object(planner.agent_tools, "execute_tool", return_value=({
            "conditions": [{"id": "simulated_2", "description": "private detail"},
                           {"id": "simulated_1", "description": "another detail"}],
            "count": 2,
        }, None)):
            _run(client, tool_trace_callback=trace.append)
        assert trace == [{
            "tool_name": "list_road_conditions", "argument_keys": [],
            "outcome": "success", "artifact_kind": None,
            "road_event_ids": ["simulated_1", "simulated_2"],
        }]
        assert "private detail" not in json.dumps(trace)

        failed = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("list_road_conditions", {})]),
            _fake_response(content="路况服务不可用。"),
        ])
        with patch.object(planner.agent_tools, "execute_tool", return_value=({
            "error": "road_conditions_unavailable", "message": "private failure",
        }, None)):
            trace.clear()
            _run(failed, tool_trace_callback=trace.append)
        assert trace[0]["outcome"] == "tool_error"
        assert "road_event_ids" not in trace[0]

    def test_typed_no_path_trace_keeps_only_allowlisted_error_code(self):
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("plan_route", {
                "start": {"name": "起点"}, "end": {"name": "终点"},
            })]),
            _fake_response(content="当前无法到达。"),
        ])
        trace = []
        with patch.object(planner.agent_tools, "execute_tool", return_value=({
            "error": "route_not_found", "reason_code": "graph_no_path",
            "request_digest": "a" * 64, "message": "private graph detail",
        }, None)):
            _run(client, tool_trace_callback=trace.append)
        assert trace[0]["error_code"] == "graph_no_path"
        assert trace[0]["request_digest"] == "a" * 64
        assert "private graph detail" not in json.dumps(trace)

        generic_trace = []
        generic_client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("plan_route", {
                "start": {"name": "起点"}, "end": {"name": "终点"},
            })]),
            _fake_response(content="路线参数有误。"),
        ])
        with patch.object(planner.agent_tools, "execute_tool", return_value=({
            "error": "route_not_found", "message": "invalid budget",
        }, None)):
            _run(generic_client, tool_trace_callback=generic_trace.append)
        assert "error_code" not in generic_trace[0]

    def test_manual_map_start_is_used_when_natural_language_omits_the_origin(self):
        selected_start = {"lng": 114.31, "lat": 30.51, "name": "地图起点"}
        client = FakeClient([])
        trace = []
        candidate = {"name": "星湖园食堂", "distance_m": 320, "subcategory": "canteen"}
        route = {"recommended": [[114.31, 30.51]], "start_name": "地图起点",
                 "end_name": "星湖园食堂", "mode": "walk"}
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool",
                          side_effect=[
                              ({"start_name": "地图起点", "candidates": [candidate]},
                               {"candidates": [candidate]}),
                              (route, {"route": route}),
                          ]) as execute:
            result = planner.run_agent(
                "带我去最近的食堂", coord_start=selected_start,
                tool_trace_callback=trace.append,
            )

        search_args = execute.call_args_list[0].args[1]
        assert search_args["start"] == {
            "name": "地图起点", "type": "coord", "lng": 114.31, "lat": 30.51,
        }
        route_args = execute.call_args_list[1].args[1]
        assert route_args["end"] == {"name": "星湖园食堂", "type": "poi"}
        assert result["response_kind"] == "route"
        assert client.calls == []
        assert trace == [
            {
                "tool_name": "search_poi_candidates",
                "argument_keys": sorted(execute.call_args_list[0].args[1]),
                "outcome": "success",
                "artifact_kind": "candidates",
            },
            {
                "tool_name": "plan_route",
                "argument_keys": sorted(route_args),
                "outcome": "success",
                "artifact_kind": "route",
            },
        ]
        assert "星湖园" not in json.dumps(trace, ensure_ascii=False)
        assert "114.31" not in json.dumps(trace, ensure_ascii=False)

    def test_agent_cannot_ask_for_start_again_after_map_selection(self):
        selected_start = {"lng": 114.31, "lat": 30.51, "name": "地图起点"}
        client = FakeClient([])
        candidate = {"name": "星湖园食堂", "distance_m": 320, "subcategory": "canteen"}
        route = {"recommended": [[114.31, 30.51]], "start_name": "地图起点",
                 "end_name": "星湖园食堂", "mode": "walk"}
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool",
                          side_effect=[
                              ({"start_name": "地图起点", "candidates": [candidate]},
                               {"candidates": [candidate]}),
                              (route, {"route": route}),
                          ]) as execute:
            result = planner.run_agent("带我去最近的食堂", coord_start=selected_start)

        assert result["response_kind"] == "route"
        assert execute.call_count == 2
        assert execute.call_args_list[1].args[1]["start"]["type"] == "coord"
        assert client.calls == []

    def test_explicit_named_start_in_text_overrides_the_map_selection(self):
        selected_start = {"lng": 114.31, "lat": 30.51, "name": "地图起点"}
        client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("plan_route", {
            "start": {"name": "教五", "type": "poi"},
            "end": {"name": "图书馆", "type": "poi"},
        })])])
        route = {"recommended": [[114.31, 30.51]], "start_name": "教五",
                 "end_name": "图书馆", "mode": "walk"}
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route})) as execute:
            planner.run_agent("从教五到图书馆", coord_start=selected_start)

        assert execute.call_args.args[1]["start"] == {"name": "教五", "type": "poi"}

    def test_conversational_named_start_overrides_the_map_selection(self):
        selected_start = {"lng": 114.31, "lat": 30.51, "name": "地图起点"}
        client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("plan_route", {
            "start": {"name": "教五", "type": "poi"},
            "end": {"name": "图书馆", "type": "poi"},
        })])])
        route = {"recommended": [[114.31, 30.51]], "start_name": "教五",
                 "end_name": "图书馆", "mode": "walk"}
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route})) as execute:
            planner.run_agent("我在教五这边，帮我导航到图书馆",
                              coord_start=selected_start)

        assert execute.call_args.args[1]["start"] == {"name": "教五", "type": "poi"}

    def test_original_named_start_survives_a_follow_up_turn(self):
        selected_start = {"lng": 114.31, "lat": 30.51, "name": "地图起点"}
        original = "从教五到图书馆"
        client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("plan_route", {
            "start": {"name": "教五", "type": "poi"},
            "end": {"name": "图书馆", "type": "poi"},
        })])])
        route = {"recommended": [[114.31, 30.51]], "start_name": "教五",
                 "end_name": "图书馆", "mode": "walk"}
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route, {"route": route})) as execute:
            planner.run_agent("不走楼梯", context={"active_task_request": original},
                              coord_start=selected_start)

        assert execute.call_args.args[1]["start"] == {"name": "教五", "type": "poi"}

    def test_candidates_artifact(self):
        cands = [{"name": "桂园食堂"}, {"name": "梅园食堂"}]
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("search_poi_candidates", {"subcategory": "canteen"})]),
            _fake_response(content="找到两个食堂，你选哪个？"),
        ])
        with patch.object(planner.agent_tools, "execute_tool",
                          return_value=({"candidates": cands}, {"candidates": cands})):
            resp = _run(client)
        assert resp["response_kind"] == "candidates"
        assert resp["candidates"] == cands

    def test_route_comparison_artifact_is_returned_without_model_rephrasing(self):
        comparison = {
            "status": "partial",
            "start_name": "星湖园食堂",
            "end_name": "卓尔体育馆",
            "mode": "walk",
            "alternatives": [
                {"strategy": "shortest", "label": "最短路径", "distance_m": 500,
                 "duration_min": 7.0, "distance_delta_from_shortest_m": 0},
                {"strategy": "flat", "label": "平坦优先", "distance_m": 540,
                 "duration_min": 7.8, "distance_delta_from_shortest_m": 40},
            ],
            "unavailable": [
                {"strategy": "scenery", "label": "风景优先", "reason": "缺少景观数据"},
            ],
            "comparison_basis": "相同起终点、方式、约束与数据快照。",
            "selection_note": "比较不改变当前路线。",
        }
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("compare_routes", {
                "start": {"name": "星湖园食堂"}, "end": {"name": "卓尔体育馆"},
            })]),
        ])
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(comparison, {"route_comparison": comparison})):
            result = _run(client)

        assert result["response_kind"] == "route_comparison"
        assert result["route_comparison"] == comparison
        assert "比较不会改变当前地图路线" in result["message"]
        assert len(client.calls) == 1

    def test_followup_suggestions_artifact_survives_planner_chat_response(self):
        suggestions = [{"label": "查附近道路", "query": "查看附近道路管制"}]
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("suggest_followup", {
                "suggestions": suggestions,
            })]),
            _fake_response(content="附近暂无已发布的道路管制。"),
        ])
        with patch.object(planner.agent_tools, "execute_tool",
                          return_value=({"suggestions": suggestions},
                                        {"suggestions": suggestions})):
            resp = _run(client)

        assert resp["response_kind"] == "chat"
        assert resp["suggestions"] == suggestions
        tool_messages = [message for message in client.calls[1]["messages"]
                         if message.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert json.loads(tool_messages[0]["content"]) == {"suggestions": suggestions}

    def test_ask_user_terminates_loop(self):
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("ask_user", {"question": "从哪里出发？", "options": ["牌坊", "我的位置"]})]),
        ])
        with patch.object(planner.agent_tools, "execute_tool",
                          return_value=({"question": "从哪里出发？"}, {"clarify": {"question": "从哪里出发？", "options": ["牌坊"]}})):
            resp = _run(client)
        assert resp["response_kind"] == "clarify"
        assert resp["clarify"]["question"] == "从哪里出发？"
        assert len(client.calls) == 1  # 终止型工具：不再继续循环

    def test_second_route_verification_failure_stops_without_unverified_route(self):
        """修正预算用尽（首轮 + 两次定向修正）后仍失败，就不展示任何未核验路线。"""
        route_call = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "终点"},
        })
        client = FakeClient([
            _fake_response(tool_calls=[route_call]),
            _fake_response(tool_calls=[route_call]),
            _fake_response(tool_calls=[route_call]),
            _fake_response(content="预算用尽后不应再执行"),
        ])
        failure = {
            "error": "route_verification_failed",
            "message": "路线边序列与约束无法核对。",
            "route_verification": {"status": "fail", "checks": {
                "mode_and_constraints": {"status": "fail", "issues": ["mode_forbidden"]},
            }},
        }
        with patch.object(planner.agent_tools, "execute_tool",
                          side_effect=[(failure, None), (failure, None),
                                       (failure, None)]) as execute:
            result = _run(client)

        # 首轮 1 次 + 两次定向修正，正好用满 MAX_REFLECTION_ATTEMPTS
        assert execute.call_count == planner.MAX_REFLECTION_ATTEMPTS + 1
        assert len(client.calls) == planner.MAX_REFLECTION_ATTEMPTS + 1
        assert result["response_kind"] == "chat"
        assert result["route"] is None
        assert "路线" in result["message"] and "核验" in result["message"]

    def test_route_verification_failure_without_reflection_budget_stops_at_once(self):
        """剩余时间不够一次 LLM 往返时，核验失败直接收尾，不再发起定向修正调用。"""
        route_call = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "终点"},
        })
        client = FakeClient([
            _fake_response(tool_calls=[route_call]),
            _fake_response(content="预算不足时不应被调用"),
        ])
        failure = {
            "error": "route_verification_failed",
            "message": "路线边序列与约束无法核对。",
            "route_verification": {"status": "fail", "checks": {
                "mode_and_constraints": {"status": "fail", "issues": ["mode_forbidden"]},
            }},
        }
        with patch.object(planner, "TIME_BUDGET_S", 1.0), \
             patch.object(planner.agent_tools, "execute_tool",
                          return_value=(failure, None)) as execute:
            result = _run(client)

        assert execute.call_count == 1
        assert len(client.calls) == 1  # 没有第二次 LLM 调用
        assert result["response_kind"] == "chat"
        assert result["route"] is None
        assert "核验" in result["message"]

    def test_one_route_verification_repair_can_succeed(self):
        route_call = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "终点"},
        })
        client = FakeClient([
            _fake_response(tool_calls=[route_call]),
            _fake_response(tool_calls=[route_call]),
        ])
        failure = {"error": "route_verification_failed", "message": "边序列不符"}
        route = {"recommended": [[1, 2]], "recommended_length_m": 10}
        with patch.object(planner.agent_tools, "execute_tool", side_effect=[
                (failure, None), (route, {"route": route})]) as execute, \
             patch.object(planner, "_build_route_message", return_value="路线已核验"):
            result = _run(client)

        assert execute.call_count == 2
        assert len(client.calls) == 2
        assert result["response_kind"] == "route"
        assert result["route"] is route
        assert result["message"] == "路线已核验"

    def test_last_planner_turn_route_verification_failure_fails_closed(self):
        route_call = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "终点"},
        })
        client = FakeClient([_fake_response(tool_calls=[route_call])])
        failure = {
            "error": "route_verification_unknown",
            "message": "缺少独立核验依据。",
            "route_verification": {"status": "unknown", "unknowns": ["endpoint_unknown"]},
        }
        with patch.object(planner, "MAX_TURNS", 1), \
             patch.object(planner.agent_tools, "execute_tool", return_value=(failure, None)) as execute:
            result = _run(client)

        assert execute.call_count == 1
        assert len(client.calls) == 1
        assert result["response_kind"] == "chat"
        assert result["route"] is None
        assert "核验" in result["message"]

    def test_route_verification_failure_skips_other_calls_in_same_model_batch(self):
        failed_call = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "终点A"},
        })
        failed_call.id = "call_route_failed"
        same_batch_call = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "终点B"},
        })
        same_batch_call.id = "call_route_must_wait"
        repair_call = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "终点C"},
        })
        repair_call.id = "call_route_repair"
        client = FakeClient([
            _fake_response(tool_calls=[failed_call, same_batch_call]),
            _fake_response(tool_calls=[repair_call]),
        ])
        failure = {"error": "route_verification_failed", "message": "边序列不符"}
        route = {"recommended": [[114.36, 30.53]], "recommended_length_m": 10}
        with patch.object(planner.agent_tools, "execute_tool", side_effect=[
                (failure, None), (route, {"route": route})]) as execute, \
             patch.object(planner, "_build_route_message", return_value="路线已核验"):
            result = _run(client)

        assert execute.call_count == 2
        assert len(client.calls) == 2
        first_turn_tools = [message for message in client.calls[1]["messages"]
                            if message.get("role") == "tool"]
        assert [message["tool_call_id"] for message in first_turn_tools] == [
            "call_route_failed", "call_route_must_wait",
        ]
        assert "未执行" in first_turn_tools[1]["content"]
        assert result["response_kind"] == "route"
        assert result["route"] is route

    def test_repair_llm_error_fails_closed_without_legacy_fallback(self):
        route_call = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "终点"},
        })
        client = FakeClient([
            _fake_response(tool_calls=[route_call]),
            RuntimeError("repair request unavailable"),
        ])
        report = {"status": "unknown", "unknowns": ["endpoint_unknown"]}
        failure = {"error": "route_verification_unknown", "message": "缺少独立核验依据。",
                   "route_verification": report}
        with patch.object(planner.agent_tools, "execute_tool", return_value=(failure, None)) as execute:
            result = _run(client)

        execute.assert_called_once()
        assert result["response_kind"] == "chat"
        assert result["route"] is None
        assert result["route_verification"] == report
        assert "修正" in result["message"]

    def test_ordinary_error_during_single_repair_does_not_execute_another_route(self):
        first_route = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "终点"},
        })
        repair_route = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "终点"},
        })
        forbidden_extra_route = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "另一个终点"},
        })
        client = FakeClient([
            _fake_response(tool_calls=[first_route]),
            _fake_response(tool_calls=[repair_route, forbidden_extra_route]),
        ])
        report = {"status": "fail", "violations": ["mode_forbidden"]}
        verification_failure = {
            "error": "route_verification_failed", "message": "路线未通过核验。",
            "route_verification": report,
        }
        ordinary_failure = {"error": "route_not_found", "message": "未找到可行路线。"}
        extra_route = {"recommended": [[114.36, 30.53]], "recommended_length_m": 10}
        with patch.object(planner.agent_tools, "execute_tool", side_effect=[
                (verification_failure, None), (ordinary_failure, None),
                (extra_route, {"route": extra_route})]) as execute:
            result = _run(client)

        assert execute.call_count == 2
        assert len(client.calls) == 2
        assert result["response_kind"] == "chat"
        assert result["route"] is None
        assert result["route_verification"] == report
        assert "修正" in result["message"]

    def test_route_verification_failure_takes_precedence_over_clarification_in_same_batch(self):
        clarify_call = _fake_tool_call("ask_user", {
            "question": "从哪里出发？", "options": ["信息学部", "工学部"],
        })
        route_call = _fake_tool_call("plan_route", {
            "start": {"name": "起点"}, "end": {"name": "终点"},
        })
        client = FakeClient([
            _fake_response(tool_calls=[clarify_call, route_call]),
            _fake_response(content="这条路线无法修正。"),
        ])
        failure = {
            "error": "route_verification_unknown",
            "message": "缺少独立核验依据。",
            "route_verification": {"status": "unknown", "unknowns": ["endpoint_unknown"]},
        }
        clarify_artifact = {"clarify": {"question": "从哪里出发？", "options": ["信息学部"]}}
        with patch.object(planner.agent_tools, "execute_tool", side_effect=[
                ({"question": "从哪里出发？"}, clarify_artifact), (failure, None)]):
            result = _run(client)

        assert result["response_kind"] == "chat"
        assert result["route"] is None
        assert result["clarify"] is None
        assert result["route_verification"]["status"] == "unknown"
        assert "核验" in result["message"]

    def test_manual_map_route_verification_failure_does_not_rewrite_selected_points(self):
        selected_start = {"lng": 114.36, "lat": 30.53, "name": "地图起点"}
        selected_end = {"lng": 114.37, "lat": 30.54, "name": "地图终点"}
        selected_via = [{"lng": 114.365, "lat": 30.535, "name": "地图途经点1"}]
        client = FakeClient([])
        failure = {
            "error": "route_verification_failed",
            "message": "所选点之间的路线未通过核验。",
        }
        with patch.object(planner, "_make_client", return_value=client) as make_client, \
             patch.object(planner.agent_tools, "execute_tool", return_value=(failure, None)) as execute:
            result = planner.run_agent(
                "从地图标记起点经过地图标记途经点到地图标记终点",
                coord_start=selected_start,
                coord_end=selected_end,
                coord_waypoints=selected_via,
            )

        make_client.assert_not_called()
        execute.assert_called_once()
        assert result["response_kind"] == "chat"
        assert result["route"] is None
        assert result["clarify"] is None
        assert "核验" in result["message"]

    def test_explicit_named_via_verification_failure_is_not_reported_as_name_clarification(self):
        client = FakeClient([])
        failure = {
            "error": "route_verification_unknown",
            "message": "缺少道路证据。",
            "route_verification": {"status": "unknown", "unknowns": ["missing_edge_sequence"]},
        }
        with patch.object(planner, "_make_client", return_value=client) as make_client, \
             patch.object(planner.agent_tools, "execute_tool", return_value=(failure, None)) as execute:
            result = planner.run_agent("从桂园操场经过樱顶到星湖园食堂")

        make_client.assert_not_called()
        execute.assert_called_once()
        assert result["response_kind"] == "chat"
        assert result["route"] is None
        assert result["clarify"] is None
        assert result["route_verification"]["status"] == "unknown"
        assert "核验" in result["message"]

    def test_invalid_tool_args_reinjected(self):
        """工具参数非法 JSON → 错误回注，循环继续直至 LLM 给出答复。"""
        bad_call = SimpleNamespace(
            id="call_bad",
            function=SimpleNamespace(name="plan_route", arguments="{not valid json"),
        )
        client = FakeClient([
            _fake_response(tool_calls=[bad_call]),
            _fake_response(content="参数有误，请再说一次起点终点。"),
        ])
        with patch.object(planner.agent_tools, "execute_tool") as mock_exec:
            resp = _run(client)
        mock_exec.assert_not_called()  # 参数非法时根本不执行工具
        assert resp["response_kind"] == "chat"
        tool_msgs = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
        assert tool_msgs and "invalid_args" in tool_msgs[0]["content"]

    def test_llm_failure_raises_planner_error(self):
        client = FakeClient([ConnectionError("api down")])
        with pytest.raises(planner.PlannerError):
            _run(client)

    def test_no_api_key_raises(self, monkeypatch):
        monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", "")
        with pytest.raises(planner.PlannerError):
            planner.run_agent("query")

    def test_loop_exhaustion_with_artifact_fallback_message(self):
        """循环耗尽但有路径 artifact → 兜底文案收尾而非报错。"""
        route_payload = {"recommended": [], "distance_m": 500}
        scripted = [
            _fake_response(tool_calls=[_fake_tool_call("plan_route", {"start": {"name": "A"}, "end": {"name": "B"}})]),
        ] * planner.MAX_TURNS  # LLM 一直调工具不给结论
        client = FakeClient(scripted)
        with patch.object(planner.agent_tools, "execute_tool",
                          return_value=(route_payload, {"route": route_payload})):
            resp = _run(client)
        assert resp["response_kind"] == "route"
        assert resp["message"]  # 有兜底文案

    def test_loop_exhaustion_empty_raises(self):
        """循环耗尽且无任何产出 → PlannerError 触发路由层兜底。"""
        client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("get_weather", {})])] * planner.MAX_TURNS)
        with patch.object(planner.agent_tools, "execute_tool", return_value=({"weather": "晴"}, None)):
            with pytest.raises(planner.PlannerError):
                _run(client)


class TestChatEndpointAgentFirst:
    """/api/chat 接线：planner 成功走 Agent；planner 异常落回旧管道。"""

    @pytest.fixture
    def client(self):
        import app as app_module
        app_module.app.config["TESTING"] = True
        return app_module.app.test_client()

    def test_chat_uses_agent_response(self, client):
        fake_resp = {"response_kind": "chat", "message": "你好呀", "route": None,
                     "route_kind": None, "candidates": None, "clarify": None, "turns": 1}
        with patch("agents.planner.run_agent", return_value=fake_resp):
            r = client.post("/api/chat", json={"query": "你好"})
        assert r.status_code == 200
        data = r.get_json()["data"]
        assert data["task_type"] == "chat"
        assert data["reply"] == "你好呀"

    def test_chat_preserves_route_verification_report(self, client):
        report = {"status": "unknown", "unknowns": ["endpoint_unknown"]}
        fake_resp = {
            "response_kind": "chat", "message": "路线未通过独立核验。",
            "route": None, "route_kind": None, "candidates": None,
            "clarify": None, "turns": 1, "route_verification": report,
        }
        with patch("agents.planner.run_agent", return_value=fake_resp):
            response = client.post("/api/chat", json={"query": "从起点到终点"})

        assert response.status_code == 200
        assert response.get_json()["data"]["route_verification"] == report

    def test_chat_falls_back_when_planner_raises(self, client):
        with patch("agents.planner.run_agent", side_effect=planner.PlannerError("down")), \
             patch("api.routes.parse_query") as mock_parse, \
             patch("api.routes.generate_chat_response", return_value="兜底回复"):
            intent = SimpleNamespace(
                model_dump=lambda: {"task_type": "chat", "start": None, "end": None},
            )
            mock_parse.return_value = intent
            r = client.post("/api/chat", json={"query": "你好"})
        assert r.status_code == 200
        assert r.get_json()["data"]["task_type"] == "chat"

    def test_internal_bug_does_not_trigger_legacy_parser(self, client):
        with patch("agents.planner.run_agent", side_effect=RuntimeError("implementation bug")), \
             patch("api.routes.parse_query") as mock_parse:
            r = client.post("/api/chat", json={"query": "从教五到图书馆"})
        mock_parse.assert_not_called()
        assert r.status_code == 500
        assert r.get_json()["error"] == "internal_error"
        assert "implementation bug" not in r.get_data(as_text=True)


class TestPreferenceReliability:
    @pytest.mark.parametrize(
        "query,expected,weights",
        [
            ("从教五到图书馆", "shortest", {"distance": 1.0, "slope": 0.0, "scenery": 0.0}),
            ("带朋友逛武大，推荐路线", "recommended", {"distance": 0.5, "slope": 0.2, "scenery": 0.3}),
            ("从教五到图书馆，走风景好的路", "scenery", {"distance": 0.5, "slope": 0.1, "scenery": 0.4}),
        ],
    )
    def test_tool_policy_ignores_model_invented_numeric_weights(self, query, expected, weights):
        decision = planner.agent_tools._strategy_for_args(
            {"weights": {"distance": 0.1, "slope": 0.1, "scenery": 0.8}},
            {"query": query},
        )
        assert decision.name == expected
        assert decision.weights == weights

    def test_retry_keeps_explicit_weights(self):
        weights = {"distance": 0.2, "slope": 0.1, "scenery": 0.7}
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("plan_route", {"weights": weights})]),
            _fake_response(tool_calls=[_fake_tool_call("plan_route", {})]),
            _fake_response(content="暂未找到可行路线。"),
        ])
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool", return_value=({"error": "route_not_found"}, None)) as execute:
            planner.run_agent("从教五到图书馆，走风景好的路")
        assert execute.call_args_list[1].args[1]["weights"] == weights

    def test_slope_avoid_is_enforced_when_model_omits_it(self):
        client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("plan_route", {})]),
                             _fake_response(content="没有可行路线。")])
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool", return_value=({"error": "route_not_found"}, None)) as execute:
            planner.run_agent("从教五到图书馆，避开陡坡")
        assert execute.call_args.args[1]["constraints"]["slope"] == "avoid"

    def test_tour_exposes_all_route_preferences(self):
        schemas = {t["function"]["name"]: t["function"]["parameters"]
                   for t in planner.agent_tools.TOOL_SCHEMAS}
        properties = schemas["plan_tour"]["properties"]
        assert {"mode", "weights", "constraints", "poi_names"} <= properties.keys()
        assert properties["poi_names"]["type"] == "array"
        assert properties["poi_names"]["items"]["type"] == "string"
        assert properties["poi_names"]["minItems"] == 2
        assert properties["poi_names"]["maxItems"] == 8

    @pytest.mark.parametrize("query", ["从教五到图书馆", "从珞珈门到樱顶，赶时间", "回宿舍上课别绕路"])
    def test_commute_ignores_model_invented_scenery_weights(self, query):
        client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("plan_route", {
            "start": {"name": "教五"}, "end": {"name": "图书馆"},
            "weights": {"distance": 0.2, "slope": 0.1, "scenery": 0.7},
        })])])
        route = {"distance_m": 500, "mode": "walk"}
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool", return_value=(route, {"route": route})) as execute:
            planner.run_agent(query)
        args = execute.call_args.args[1]
        assert args.get("weights") is None or args["weights"] == {
            "distance": 0.90, "slope": 0.05, "scenery": 0.05}

    def test_explicit_preference_is_preserved(self):
        weights = {"distance": 0.2, "slope": 0.6, "scenery": 0.2}
        client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("plan_route", {
            "start": {"name": "教五"}, "end": {"name": "图书馆"},
            "weights": weights, "constraints": {"slope": "avoid"},
        })])])
        route = {"distance_m": 500, "mode": "walk"}
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool", return_value=(route, {"route": route})) as execute:
            planner.run_agent("从教五到图书馆，避开陡坡")
        assert execute.call_args.args[1]["weights"] == weights

    def test_retry_cannot_drop_slope_constraint(self):
        client = FakeClient([
            _fake_response(tool_calls=[_fake_tool_call("plan_route", {
                "start": {"name": "A"}, "end": {"name": "B"}, "constraints": {"slope": "avoid"},
            })]),
            _fake_response(tool_calls=[_fake_tool_call("plan_route", {
                "start": {"name": "A"}, "end": {"name": "B"}, "constraints": {"slope": "normal"},
            })]),
            _fake_response(content="暂无满足避坡要求的路线。"),
        ])
        with patch.object(planner, "_make_client", return_value=client), \
             patch.object(planner.agent_tools, "execute_tool", return_value=({"error": "route_not_found"}, None)) as execute:
            planner.run_agent("从 A 到 B，避开陡坡")
        assert execute.call_args_list[1].args[1]["constraints"]["slope"] == "avoid"
