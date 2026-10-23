"""The experimental Harness bridge must be local, bounded, and observable."""
from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from experiments.baselines.harness_tool_bridge import open_tool_bridge


def _post(url: str, token: str, payload: dict) -> tuple[int, dict | None]:
    request = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                      headers={"Content-Type": "application/json", "X-Campus-Token": token})
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, None


def test_bridge_uses_real_tool_name_and_context_without_exposing_it() -> None:
    calls = []

    def executor(name, args, context):
        calls.append((name, args, context))
        return {"found": True, "poi": {"name": "卓尔体育馆"}}, {"candidates": [1]}

    with open_tool_bridge(executor=executor, context={"query": "去卓尔体育馆"}) as bridge:
        status, body = _post(bridge.url, bridge.token,
                             {"name": "resolve_poi", "arguments": {"name": "卓尔体育馆"}})
        assert status == 200
        assert json.loads(body["result"])["found"] is True
        assert calls == [("resolve_poi", {"name": "卓尔体育馆"}, {"query": "去卓尔体育馆"})]
        assert bridge.trace[0]["outcome"] == "success"
        assert bridge.trace[0]["artifact_kind"] == "candidates"
        assert bridge.artifacts == [{"candidates": [1]}]
        assert bridge.selection_context == [{}]
        assert bridge.token not in str(bridge.trace)


def test_bridge_keeps_only_candidate_origin_metadata() -> None:
    def executor(_name, _args, _context):
        return {"start_name": "玉兰2门", "mode": "walk", "private": "do not log"}, {
            "candidates": [{"poi_id": "poi_food"}],
        }

    with open_tool_bridge(executor=executor) as bridge:
        status, _ = _post(bridge.url, bridge.token,
                          {"name": "search_poi_candidates",
                           "arguments": {"start": {"name": "玉兰2门"}}})
        assert status == 200
        assert bridge.selection_context == [{"start_name": "玉兰2门", "mode": "walk"}]
        assert "private" not in str(bridge.trace)
        assert "private" not in str(bridge.selection_context)


def test_bridge_reuses_weather_snapshot_across_tool_calls() -> None:
    snapshot = {"live": {"weather": "晴"}, "impact": {}}
    observed = []

    def executor(name, _args, context):
        observed.append((name, context.get("_route_weather_snapshot")))
        if name == "get_weather":
            context["_route_weather_snapshot"] = snapshot
        return {"ok": True}, None

    with open_tool_bridge(executor=executor) as bridge:
        assert _post(bridge.url, bridge.token,
                     {"name": "get_weather", "arguments": {}})[0] == 200
        assert _post(bridge.url, bridge.token,
                     {"name": "plan_route", "arguments": {}})[0] == 200
        assert observed == [("get_weather", None), ("plan_route", snapshot)]
        assert "_route_weather_snapshot" not in str(bridge.trace)


def test_bridge_records_only_public_weather_observation() -> None:
    def executor(_name, _args, _context):
        return {"weather": "晴", "temperature": 22, "private": "not recorded"}, None

    with open_tool_bridge(executor=executor) as bridge:
        assert _post(bridge.url, bridge.token,
                     {"name": "get_weather", "arguments": {}})[0] == 200
        assert bridge.weather_observations == [{"weather": "晴", "temperature": 22}]
        assert "private" not in str(bridge.weather_observations)


def test_bridge_rejects_bad_token_unknown_tool_and_oversized_body() -> None:
    with open_tool_bridge(executor=lambda *_: ({"ok": True}, None)) as bridge:
        assert _post(bridge.url, "wrong", {"name": "resolve_poi", "arguments": {}})[0] == 403
        assert _post(bridge.url, bridge.token,
                     {"name": "not_a_campus_tool", "arguments": {}})[0] == 400
        assert _post(bridge.url, bridge.token,
                     {"name": "resolve_poi", "arguments": {"x": "a" * 131073}})[0] == 413
        assert bridge.trace == []


def test_bridge_records_executor_error_without_leaking_exception_text() -> None:
    def executor(*_args):
        raise RuntimeError("private detail")

    with open_tool_bridge(executor=executor) as bridge:
        status, body = _post(bridge.url, bridge.token,
                             {"name": "resolve_poi", "arguments": {"name": "a"}})
        assert status == 200
        assert json.loads(body["result"])["error"] == "tool_failure"
        assert "private detail" not in body["result"]
        assert bridge.trace[0]["error_type"] == "RuntimeError"
