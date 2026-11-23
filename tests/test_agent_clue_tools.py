"""Agent tools for road clues: query, bounded collection, field reporting."""
from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import networkx as nx
import pytest

from agents import planner, tool_skills, tools
from spatial import road_clues
from spatial import road_conditions as rc

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_clue_store(tmp_path, monkeypatch):
    monkeypatch.setattr(road_clues, "_CLUES_FILE", str(tmp_path / "clues.json"))
    conditions = tmp_path / "road_conditions.json"
    conditions.write_text("[]\n", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(conditions))
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(conditions))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    monkeypatch.setattr(rc, "_cache_error", None)
    road_clues.reset_collect_attempts()
    yield


def _graph():
    graph = nx.MultiDiGraph()
    points = {0: (114.3600, 30.5300), 1: (114.3604, 30.5300), 2: (114.3612, 30.5300)}
    for node, (x, y) in points.items():
        graph.add_node(node, x=x, y=y)
    for u, v in ((0, 1), (1, 2)):
        length = math.hypot(
            (points[u][0] - points[v][0]) * 111320 * math.cos(math.radians(30.53)),
            (points[u][1] - points[v][1]) * 111320,
        )
        attrs = {"length": length, "highway": "residential", "name": "自强大道"}
        graph.add_edge(u, v, 0, **attrs)
        graph.add_edge(v, u, 0, **attrs)
    return graph


def _sample(**overrides):
    data = {
        "kind": "ugc_report", "summary": "施工",
        "valid_from": time.time() - 60, "valid_to": time.time() + 7200,
        "draft": {"event_type": "construction"},
        "edge_candidates": [{"u": 0, "v": 1, "key": 0, "road_name": "自强大道",
                             "distance_m": 2.0,
                             "geometry_gcj": [[114.36552, 30.52767], [114.36594, 30.52767]]}],
    }
    data.update(overrides)
    return data


def _reviewing(**overrides):
    values = {"valid_from": time.time() - 60, "valid_to": time.time() + 7200}
    values.update(overrides)
    row = road_clues.create_clue(_sample(**values))
    return road_clues.update_clue(row["clue_id"], {"status": "reviewing"})


def _edge_payload():
    return {"u": 0, "v": 1, "key": 0, "road_name": "自强大道", "dist_m": 1.0,
            "snap_lng_gcj": 114.3657, "snap_lat_gcj": 30.5277,
            "geometry_gcj": [[114.36552, 30.52767], [114.36594, 30.52767]],
            "edges": [[0, 1, 0]], "chain_length_m": 40.0}


def test_registry_contains_seventeen_tools_and_skill_cards():
    assert len(tools.TOOL_NAMES) == 17
    assert {"query_road_clues", "collect_road_clues", "report_field_clue"} <= tools.TOOL_NAMES
    assert {"query_road_clues", "collect_road_clues", "report_field_clue"} <= set(tools._EXECUTORS)
    assert tool_skills.validate_skill_coverage() == []
    schema_names = {item["function"]["name"] for item in tools.TOOL_SCHEMAS}
    assert {"query_road_clues", "collect_road_clues", "report_field_clue"} <= schema_names


def test_query_defaults_to_adopted_and_reviewing_without_raw_text():
    pending = road_clues.create_clue(_sample())
    reviewing = _reviewing()
    rejected = road_clues.create_clue(_sample(summary="已排除"))
    road_clues.update_clue(rejected["clue_id"], {
        "status": "rejected", "review_note": "现场不存在该情况。",
    })
    adopted = _reviewing()
    condition, _, _ = road_clues.adopt_clues(
        adopted["clue_id"], [], edge=_edge_payload(), cond_type="construction",
        name="已采纳施工", start_time=time.time() - 1, end_time=time.time() + 1800,
        action="closure", affected_modes=["walk"],
        review_note="核对原始描述与候选路段后采纳为测试事件。",
    )
    result, artifact = tools.execute_tool("query_road_clues", {})
    assert artifact is None
    statuses = {item["status"] for item in result["clues"]}
    assert statuses == {"pending", "reviewing", "adopted"}
    assert result["count"] == 3
    assert all("raw_text" not in item for item in result["clues"])
    adopted_item = next(item for item in result["clues"] if item["status"] == "adopted")
    assert adopted_item["verified"] is True
    assert adopted_item["condition_ids"] == [condition["id"]]
    assert adopted_item["road_names"] == ["自强大道"]
    conditions, _ = tools.execute_tool("list_road_conditions", {})
    linked_event = next(item for item in conditions["conditions"]
                        if item["id"] == condition["id"])
    assert linked_event["source_label"] == "用户现场上报"
    assert linked_event["clue_evidence"]["primary_clue_id"] == adopted["clue_id"]
    pending_item = next(item for item in result["clues"] if item["status"] == "pending")
    assert pending_item["verified"] is False
    assert "尚未核实" in pending_item["verification_note"]
    assert pending["clue_id"] in {item["clue_id"] for item in result["clues"]}


def test_query_near_requires_explicit_crs_and_excludes_unbound():
    bound = road_clues.create_clue(_sample())
    unbound = road_clues.create_clue(_sample(edge_candidates=[{
        "u": 9, "v": 10, "key": 0, "road_name": "无位置道路", "distance_m": 2.0,
    }]))
    missing = tools.execute_tool("query_road_clues", {
        "near": {"lng": 114.3657, "lat": 30.5277},
    })[0]
    assert missing["error"] == "near_crs_required"
    result, _ = tools.execute_tool("query_road_clues", {
        "near": {"lng": 114.3657, "lat": 30.5277, "crs": "GCJ02", "radius_m": 50},
    })
    ids = {item["clue_id"] for item in result["clues"]}
    assert bound["clue_id"] in ids
    assert unbound["clue_id"] not in ids


def _web_candidate(summary="自强大道施工"):
    return {
        "kind": "web", "summary": summary, "raw_text": "网页正文",
        "source_url": "https://example.org/road", "source_label": "网页检索",
        "source_tags": ["web"], "search_query": "施工",
        "draft": {"event_type": "construction"}, "edge_candidates": [],
        "confidence": 0.3,
    }


def test_collect_registers_at_most_three_and_rate_limits_same_place(monkeypatch):
    monkeypatch.setattr(road_clues.registry["web"], "fetch",
                        lambda **_kwargs: [_web_candidate(), _web_candidate("第二条"),
                                           _web_candidate("第三条"), _web_candidate("第四条")])
    result, _ = tools.execute_tool("collect_road_clues", {
        "query": "自强大道施工", "place": "自强大道",
    }, {"uid": "u1"})
    assert result["count"] == 3
    assert len(road_clues.list_clues()) == 3
    limited, _ = tools.execute_tool("collect_road_clues", {
        "query": "自强大道积水", "place": "自强大道",
    }, {"uid": "u1"})
    assert limited["error"] == "clue_collect_rate_limited"
    assert limited["retry_after_seconds"] > 0


def test_collect_same_question_is_not_repeated_within_turn(monkeypatch):
    monkeypatch.setattr(road_clues, "claim_collect", lambda _key: (True, 0))
    monkeypatch.setattr(road_clues.registry["web"], "fetch",
                        lambda **_kwargs: [_web_candidate()])
    first, _ = tools.execute_tool("collect_road_clues", {
        "query": "施工", "place": "北环路",
    }, {"uid": "u2"})
    assert first["count"] == 1
    second, _ = tools.execute_tool("collect_road_clues", {
        "query": "施工", "place": "北环路",
    }, {"uid": "u2"})
    assert second["error"] == "clue_collect_already_attempted"


def test_concurrent_collect_for_same_place_allows_only_one(monkeypatch):
    monkeypatch.setattr(road_clues.registry["web"], "fetch",
                        lambda **_kwargs: [_web_candidate()])
    results = []

    def collect(_index):
        result, _ = tools.execute_tool("collect_road_clues", {
            "query": "施工", "place": "信息学部",
        }, {"uid": "u10"})
        results.append(result)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(collect, range(2)))
    allowed = [item for item in results if item.get("status") == "ok"]
    limited = [item for item in results if item.get("error") == "clue_collect_rate_limited"]
    assert len(allowed) == 1
    assert len(limited) == 1


def test_collect_timeout_and_provider_failure_write_nothing(monkeypatch):
    monkeypatch.setattr(road_clues, "claim_collect", lambda _key: (True, 0))

    def timeout(**_kwargs):
        raise TimeoutError("deadline")

    monkeypatch.setattr(road_clues.registry["web"], "fetch", timeout)
    result, _ = tools.execute_tool("collect_road_clues", {"query": "施工"},
                                    {"uid": "u3"})
    assert result["error"] == "unavailable"
    assert "暂时查不到" in result["message"]
    assert road_clues.list_clues() == []

    def unavailable(**_kwargs):
        raise road_clues.ClueSourceUnavailable("provider down")

    monkeypatch.setattr(road_clues.registry["web"], "fetch", unavailable)
    failed, _ = tools.execute_tool("collect_road_clues", {"query": "施工"},
                                    {"uid": "u4"})
    assert failed["error"] == "unavailable"
    assert road_clues.list_clues() == []


def test_collect_registers_official_notice_through_the_registered_source(monkeypatch):
    monkeypatch.setattr(road_clues, "claim_collect", lambda _key: (True, 0))
    result, _ = tools.execute_tool("collect_road_clues", {
        "source": "official_notice",
        "place": "自强大道",
        "notice_text": "通知：自强大道将于本周六全天封闭施工，请师生绕行北环路。",
        "summary": "自强大道周六封闭施工",
        "draft": {"event_type": "construction", "road_hint": "自强大道",
                  "action": "closure", "affected_modes": ["walk"]},
    }, {"uid": "notice-1"})
    assert result["status"] == "ok"
    assert result["source"] == "official_notice"
    assert result["count"] == 1
    row = road_clues.list_clues()[0]
    assert row["kind"] == "official_notice"
    assert row["trust_level"] == "A"
    assert row["status"] == "reviewing"
    assert row["need_human"] is True
    assert row["draft"]["road_hint"] == "自强大道"
    assert rc.list_conditions() == []


def test_collect_hard_deadline_holds_when_the_collector_ignores_it(monkeypatch):
    from agents.deadline import shared_deadline

    monkeypatch.setattr(road_clues, "claim_collect", lambda _key: (True, 0))

    def slow(**_kwargs):
        time.sleep(1.5)  # deliberately ignores the cooperative deadline
        return [_web_candidate()]

    monkeypatch.setattr(road_clues.registry["web"], "fetch", slow)
    started = time.monotonic()
    with shared_deadline(time.monotonic() + 0.2):
        result, _ = tools.execute_tool("collect_road_clues", {"query": "施工"},
                                        {"uid": "hard-deadline"})
    elapsed = time.monotonic() - started
    assert result["error"] == "unavailable"
    assert elapsed < 1.0
    assert road_clues.list_clues() == []


def test_collect_deadline_covers_persistence_not_just_fetch(monkeypatch):
    from agents.deadline import shared_deadline

    monkeypatch.setattr(road_clues, "claim_collect", lambda _key: (True, 0))
    monkeypatch.setattr(
        road_clues.registry["web"], "fetch",
        lambda **_kwargs: [_web_candidate("第一条"), _web_candidate("第二条")],
    )
    original_save = road_clues._save
    calls = {"count": 0}

    def staged_save(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            time.sleep(0.8)  # a slow atomic-write preparation must not commit late
        return original_save(*args, **kwargs)

    monkeypatch.setattr(road_clues, "_save", staged_save)
    started = time.monotonic()
    with shared_deadline(time.monotonic() + 0.2):
        result, _ = tools.execute_tool("collect_road_clues", {"query": "施工"},
                                        {"uid": "persist-deadline"})
    elapsed = time.monotonic() - started
    assert result["error"] == "unavailable"
    assert "超过 8 秒" in result["message"]
    assert elapsed < 0.6
    assert len(result.get("created_clue_ids", [])) == 1
    time.sleep(0.9)  # let the uninterruptible in-flight write attempt finish
    rows_after_worker_finishes = road_clues.list_clues()
    assert len(rows_after_worker_finishes) == 1
    assert rows_after_worker_finishes[0]["clue_id"] == result["created_clue_ids"][0]


def test_collect_deadline_includes_rate_limit_lock_wait(monkeypatch):
    from agents.deadline import shared_deadline

    fetch_called = threading.Event()

    def slow_claim(_key):
        time.sleep(0.8)  # simulate another process holding the rate-limit file lock
        return True, 0

    def fetch(**_kwargs):
        fetch_called.set()
        return [_web_candidate()]

    monkeypatch.setattr(road_clues, "claim_collect", slow_claim)
    monkeypatch.setattr(road_clues.registry["web"], "fetch", fetch)
    started = time.monotonic()
    with shared_deadline(started + 0.2):
        result, _ = tools.execute_tool("collect_road_clues", {"query": "施工"},
                                        {"uid": "claim-deadline"})
    elapsed = time.monotonic() - started

    assert result["error"] == "unavailable"
    assert elapsed < 0.6
    time.sleep(0.9)
    assert not fetch_called.is_set()
    assert road_clues.list_clues() == []


def test_report_field_clue_registers_pending_without_event(monkeypatch):
    graph = _graph()
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    wgs_lng, wgs_lat = graph.nodes[1]["x"], graph.nodes[1]["y"]
    result, _ = tools.execute_tool("report_field_clue", {
        "place": "自强大道", "description": "路边施工围挡，行人只能绕行。",
        "coord": {"lng": wgs_lng, "lat": wgs_lat, "crs": "WGS84"},
    }, {"uid": "u5"})
    assert result["status"] == "pending"
    assert result["candidate_count"] > 0
    assert "已登记" in result["message"]
    assert rc.list_conditions(include_inactive=True, strict=True) == []

    converted, _ = tools.execute_tool("report_field_clue", {
        "place": "自强大道", "description": "路面有积水。",
        "coord": {"lng": wgs_lng, "lat": wgs_lat, "crs": "WGS84"},
    }, {"uid": "u6"})
    assert converted["status"] == "pending"
    clue = road_clues.get_clue(converted["clue_id"])
    assert clue["location"]["lng"] != wgs_lng
    assert clue["input_crs"] == "WGS84"

    missing_crs, _ = tools.execute_tool("report_field_clue", {
        "place": "自强大道", "description": "坐标未说明坐标系。",
        "coord": {"lng": 114.36, "lat": 30.53},
    }, {"uid": "u7"})
    assert missing_crs["error"] == "coord_crs_required"

    empty, _ = tools.execute_tool("report_field_clue", {}, {"uid": "u8"})
    assert empty["error"] == "invalid_tool_request"

    far, _ = tools.execute_tool("report_field_clue", {
        "place": "东湖边", "description": "远离路网的现场描述。",
        "coord": {"lng": 114.38, "lat": 30.545, "crs": "GCJ02"},
    }, {"uid": "u9"})
    assert far["status"] == "pending"
    assert far["candidate_count"] == 0


def test_prompts_and_dialogue_schema_expose_clue_rules():
    dialogue_prompt = (ROOT / "agents" / "prompts" / "dialogue_system.txt").read_text(encoding="utf-8")
    agent_prompt = (ROOT / "agents" / "prompts" / "agent_system.txt").read_text(encoding="utf-8")
    for phrase in (
        "有线索显示", "尚未核实", "clue_id", "已登记，待核实",
        "暂时查不到可靠信息", "据{source_label}", "primary_clue_id",
    ):
        assert phrase in dialogue_prompt
    for name in ("query_road_clues", "collect_road_clues", "report_field_clue"):
        assert name in agent_prompt
        assert name in {item["function"]["name"] for item in __import__(
            "agents.dialogue", fromlist=["tool_schemas"]).tool_schemas()}


def _reply(text="", **calls):
    tool_calls = [
        SimpleNamespace(id=f"call_{name}", type="function", function=SimpleNamespace(
            name=name, arguments=json.dumps(args, ensure_ascii=False),
        )) for name, args in calls.items()
    ]
    message = SimpleNamespace(content=text, reasoning_content=None, tool_calls=tool_calls)

    def model_dump(**_kwargs):
        payload = {"role": "assistant"}
        if text:
            payload["content"] = text
        if tool_calls:
            payload["tool_calls"] = [{
                "id": call.id, "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            } for call in tool_calls]
        return payload

    message.model_dump = model_dump
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


def test_dialogue_path_executes_clue_tool_and_injects_result(monkeypatch):
    monkeypatch.setenv("AGENT_DIALOGUE_MODE", "model")
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", "unit-test-only")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())))
    monkeypatch.setattr(planner, "_make_client", lambda: client)
    calls = []

    def execute(name, args, context):
        calls.append(name)
        return {
            "clues": [{
                "clue_id": "clue-test-1", "status": "reviewing", "source_label": "官方通知",
                "summary": "自强大道施工", "road_names": ["自强大道"],
                "condition_ids": [], "verified": False,
                "verification_note": "线索尚未核实，不能当作已确认封路。",
            }],
            "count": 1, "message": "",
        }, None

    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    client.chat.completions.create.side_effect = [
        _reply(query_road_clues={"keyword": "自强大道"}),
        _reply("有线索显示自强大道正在施工，尚未核实。线索编号 clue-test-1。"),
    ]
    result = planner.run_agent("自强大道在施工吗？")
    assert calls == ["query_road_clues"]
    assert "尚未核实" in result["message"]
    second_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [message for message in second_messages if message.get("role") == "tool"]
    assert tool_messages
    assert "clue-test-1" in tool_messages[-1]["content"]


def test_legacy_planner_loop_can_execute_clue_tool(monkeypatch):
    monkeypatch.setenv("AGENT_DIALOGUE_MODE", "legacy")
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", "unit-test-only")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())))
    monkeypatch.setattr(planner, "_make_client", lambda: client)
    calls = []

    def execute(name, args, context):
        calls.append(name)
        if name == "query_road_clues":
            return {
                "clues": [{
                    "clue_id": "clue-legacy-1", "status": "reviewing",
                    "source_label": "官方通知", "summary": "自强大道施工",
                    "road_names": ["自强大道"], "condition_ids": [],
                    "verified": False,
                    "verification_note": "线索尚未核实，不能当作已确认封路。",
                }],
                "count": 1, "message": "",
            }, None
        return {"error": "unexpected_tool", "message": name}, None

    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    client.chat.completions.create.side_effect = [
        _reply(query_road_clues={"keyword": "自强大道"}),
        _reply("有线索显示自强大道正在施工，尚未核实。"),
    ]
    result = planner.run_agent("帮我看看自强大道有哪些线索")
    assert "query_road_clues" in calls
    assert "尚未核实" in result["message"]
    assert client.chat.completions.create.call_count == 2
