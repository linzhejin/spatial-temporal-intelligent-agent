"""Durable LangGraph orchestration around the existing campus route agent."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from storage.task_repository import append_run_event, get_run_input


class AgentRunState(TypedDict, total=False):
    run_id: str
    worker_id: str
    input: dict[str, Any]
    agent_response: dict[str, Any]
    result: dict[str, Any]


class AgentRunContext(TypedDict):
    """Lease-scoped values that must never be restored from a graph checkpoint."""

    worker_id: str


def _emit(
    state: AgentRunState, runtime: Runtime[AgentRunContext], event: str, node: str, label: str,
    database_url: str | None,
) -> None:
    stored = append_run_event(
        database_url, state["run_id"], runtime.context["worker_id"], event,
        {"label": label}, node_id=node,
    )
    if stored is None:
        raise RuntimeError("run lease lost or task cancelled")


def _tool_trace_enabled() -> bool:
    return os.getenv("AGENT_TOOL_TRACE_ENABLED", "").strip().lower() in {"1", "true", "yes"}


def _emit_agent_tool_trace(state, runtime, database_url, event_type, payload):
    stored = append_run_event(
        database_url, state["run_id"], runtime.context["worker_id"],
        event_type, payload, node_id="agent_tool",
    )
    if stored is None:
        raise RuntimeError("run lease lost or task cancelled")


def _coordinate(location: dict | None) -> dict | None:
    if not isinstance(location, dict):
        return None
    point = location.get("coordinates")
    if not isinstance(point, dict):
        return None
    return {
        "lng": point.get("longitude"),
        "lat": point.get("latitude"),
        "crs": point.get("crs"),
        "name": location.get("name"),
    }


def _weather_message(weather: dict) -> str:
    parts = [f"当前天气：{weather.get('weather') or '暂缺'}"]
    if weather.get("temperature") is not None:
        parts.append(f"气温 {weather['temperature']}℃")
    if weather.get("windpower"):
        parts.append(f"风力 {weather['windpower']}")
    if weather.get("advice"):
        parts.append(str(weather["advice"]).strip())
    message = "，".join(parts).rstrip("，")
    return message if message.endswith(("。", "！", "？", "!", "?")) else message + "。"


def _run_weather_tool():
    from agents.tools import execute_tool

    result, _ = execute_tool("get_weather", {}, {})
    if not isinstance(result, dict) or result.get("error"):
        detail = result.get("message") if isinstance(result, dict) else None
        raise RuntimeError(detail or "天气暂时获取不到")
    return result


def _agent_requirement_status(response: dict[str, Any], requirement_id: str) -> str:
    if response.get("needs_input") or response.get("response_kind") in {"clarify", "candidates"}:
        return "needs_input"
    if response.get("response_kind") == "route_comparison":
        return "satisfied" if _route_comparison_satisfies(response) else "failed"
    if requirement_id != "route":
        return "satisfied"
    route = response.get("route")
    if (response.get("response_kind") == "route" and isinstance(route, dict)
            and isinstance(route.get("route_verification"), dict)
            and route["route_verification"].get("status") == "pass"
            and isinstance(route.get("recommended_edge_ids"), list)
            and isinstance(route.get("recommended"), list)):
        return "satisfied"
    return "failed"


def _route_comparison_satisfies(response: dict[str, Any]) -> bool:
    """Count a real comparison as fulfillment without requiring one chosen route."""
    comparison = response.get("route_comparison")
    if not isinstance(comparison, dict):
        return False
    expected = {"shortest", "scenery", "flat"}
    alternatives = comparison.get("alternatives")
    unavailable = comparison.get("unavailable")
    if not isinstance(alternatives, list) or not isinstance(unavailable, list):
        return False
    available_strategies = set()
    for item in alternatives:
        if not isinstance(item, dict):
            return False
        strategy = item.get("strategy")
        distance = item.get("distance_m")
        if (not isinstance(strategy, str) or strategy not in expected
                or strategy in available_strategies
                or isinstance(distance, bool) or not isinstance(distance, (int, float))):
            return False
        try:
            distance = float(distance)
        except (OverflowError, ValueError):
            return False
        if not math.isfinite(distance) or distance < 0:
            return False
        available_strategies.add(strategy)
    unavailable_strategies = set()
    for item in unavailable:
        if not isinstance(item, dict):
            return False
        strategy = item.get("strategy")
        if (not isinstance(strategy, str) or strategy not in expected
                or strategy in unavailable_strategies
                or not isinstance(item.get("reason"), str) or not item["reason"].strip()):
            return False
        unavailable_strategies.add(strategy)
    if available_strategies & unavailable_strategies:
        return False
    evidence = comparison.get("evidence")
    if (not isinstance(evidence, dict) or not evidence.get("data_version")
            or not evidence.get("road_condition_version")):
        return False
    status = comparison.get("status")
    if status == "complete":
        return available_strategies == expected and not unavailable_strategies
    return (status == "partial" and len(available_strategies) >= 2
            and available_strategies | unavailable_strategies == expected
            and bool(unavailable_strategies))


def build_agent_workflow(
    checkpointer, agent: Callable[..., dict] | None = None,
    database_url: str | None = None,
    interrupt_after: list[str] | None = None,
):
    """Compile a bounded task graph; the injected agent is for deterministic replay."""
    if agent is None:
        from agents.planner import run_agent
        agent = run_agent

    def load_task(state: AgentRunState, runtime: Runtime[AgentRunContext]):
        _emit(state, runtime, "node_started", "load_task", "正在恢复本轮对话和任务状态", database_url)
        run_input = get_run_input(database_url, state["run_id"])
        if not run_input:
            raise RuntimeError("run input is missing")
        if run_input.get("cancel_requested"):
            raise RuntimeError("task cancelled")
        _emit(state, runtime, "node_completed", "load_task", "已载入可信会话上下文", database_url)
        return {"input": run_input}

    def execute_agent(state: AgentRunState, runtime: Runtime[AgentRunContext]):
        _emit(state, runtime, "node_started", "execute_agent", "正在识别需求并调用校园空间工具", database_url)
        data = state["input"]
        task = data.get("task") or {}
        route_spec = task.get("route_spec") or {}
        start = _coordinate(route_spec.get("start"))
        end = _coordinate(route_spec.get("end"))
        stops = route_spec.get("stops") or []
        waypoints = [point for point in (_coordinate(item) for item in stops) if point]
        history = data.get("history") or []
        from agents.context_builder import build_agent_context, is_route_followup
        continuation = data.get("task_revision", 0) > 0
        origin_query = data.get("task_origin_query")
        previous_route_state = data.get("previous_route_state")
        previous_task_id = data.get("previous_route_state_task_id")
        # A clarification of a fresh task must not import a completed route
        # from another task, even though the route is still in the conversation.
        if (continuation and previous_task_id and data.get("task_id")
                and previous_task_id != data["task_id"]
                and not is_route_followup(origin_query or data["query"])):
            previous_route_state = None
        context = build_agent_context(
            data["query"], history, previous_route_state,
            same_task=continuation,
        )
        if continuation and isinstance(origin_query, str) and origin_query.strip():
            context["active_task_request"] = origin_query.strip()
        if isinstance(task.get("place_request"), dict):
            context["place_request"] = task["place_request"]
        if data.get("selected_poi_id"):
            context["selected_poi_id"] = data["selected_poi_id"]
        from api.routes import _normalize_chat_context
        context = _normalize_chat_context(context)
        context["task_revision"] = data.get("task_revision")
        route_state = task.get("route_spec") or {}
        from agents.task_planner import build_task_plan
        plan_text = (data["query"] if context.get("place_request")
                     else context.get("active_task_request") or data["query"])
        if context.get("active_task_request") and not context.get("place_request"):
            # The continuation can add an independent requirement, such as
            # weather, while the original task still supplies the destination.
            plan_text += " " + data["query"]
        plan = build_task_plan(
            plan_text,
            has_route_context=bool(start or end or waypoints or context.get("place_request")),
        )
        response = None
        weather = None
        requirement_results: dict[str, str] = {}
        errors: dict[str, str] = {}

        record_tool_call = None
        if _tool_trace_enabled():
            _emit_agent_tool_trace(
                state, runtime, database_url, "agent_tool_trace_started",
                {"source": "agent_execution"},
            )

            def record_tool_call(item):
                item = item if isinstance(item, dict) else {}
                raw_keys = item.get("argument_keys")
                argument_keys = sorted({
                    key for key in raw_keys
                    if isinstance(key, str) and key and len(key) <= 80
                }) if isinstance(raw_keys, list) else []
                artifact_kind = item.get("artifact_kind")
                if artifact_kind not in {
                        "route", "candidates", "clarify", "suggestions", "route_comparison", None}:
                    artifact_kind = None
                outcome = item.get("outcome")
                if outcome not in {
                        "success", "tool_error", "invalid_args", "exception",
                        "skipped_after_route_verification_retry",
                        "skipped_after_route_verification_failure"}:
                    outcome = "exception"
                _emit_agent_tool_trace(
                    state, runtime, database_url, "agent_tool_call", {
                        "tool_name": str(item.get("tool_name") or "unknown")[:80],
                        "argument_keys": argument_keys,
                        "outcome": outcome,
                        "artifact_kind": artifact_kind,
                    },
                )

        def run_weather_tool():
            try:
                result = _run_weather_tool()
            except Exception:
                if record_tool_call:
                    record_tool_call({
                        "tool_name": "get_weather", "argument_keys": [],
                        "outcome": "exception", "artifact_kind": None,
                    })
                raise
            if record_tool_call:
                record_tool_call({
                    "tool_name": "get_weather", "argument_keys": [],
                    "outcome": "success", "artifact_kind": None,
                })
            return result

        def run_route_agent():
            agent_kwargs = dict(
                context=context, history=history,
                coord_start=start, coord_end=end, coord_waypoints=waypoints,
                travel_mode=route_state.get("travel_mode", "walk"),
                uid=data.get("owner_id"),
                proof_identity={
                    "database_url": database_url,
                    "owner_id": data.get("owner_id"),
                    "task_id": data.get("task_id"),
                    "task_revision": data.get("task_revision"),
                },
            )
            if record_tool_call:
                agent_kwargs["tool_trace_callback"] = record_tool_call
            return agent(data["query"], **agent_kwargs)

        if plan.kind == "composite":
            # Route interpretation and live weather are independent, so neither
            # is allowed to hold the other requirement hostage.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent-subtask") as pool:
                route_future = pool.submit(run_route_agent)
                weather_future = pool.submit(run_weather_tool)
                try:
                    response = route_future.result()
                    if not isinstance(response, dict):
                        raise TypeError("agent returned a non-object response")
                    agent_requirement = plan.nodes[0].requirement_ids[0]
                    requirement_results[agent_requirement] = _agent_requirement_status(
                        response, agent_requirement,
                    )
                    if requirement_results[agent_requirement] == "failed":
                        errors[agent_requirement] = "route_output_missing_or_unverified"
                except Exception as error:
                    agent_requirement = plan.nodes[0].requirement_ids[0]
                    errors[agent_requirement] = type(error).__name__
                    requirement_results[agent_requirement] = "failed"
                try:
                    weather = weather_future.result()
                    requirement_results["weather"] = "satisfied"
                except Exception as error:
                    errors["weather"] = type(error).__name__
                    requirement_results["weather"] = "failed"
        elif plan.kind == "weather":
            try:
                weather = run_weather_tool()
                requirement_results["weather"] = "satisfied"
            except Exception as error:
                errors["weather"] = type(error).__name__
                requirement_results["weather"] = "failed"
        else:
            try:
                response = run_route_agent()
                if not isinstance(response, dict):
                    raise TypeError("agent returned a non-object response")
                agent_requirement = "route" if plan.kind == "route" else "answer"
                requirement_results[agent_requirement] = _agent_requirement_status(
                    response, agent_requirement,
                )
                if requirement_results[agent_requirement] == "failed":
                    errors[agent_requirement] = "route_output_missing_or_unverified"
            except Exception as error:
                errors["route" if plan.kind == "route" else "answer"] = type(error).__name__
                requirement_results["route" if plan.kind == "route" else "answer"] = "failed"

        clarify = bool(response and response.get("response_kind") == "clarify")
        needs_user_selection = bool(response and response.get("response_kind") == "candidates")
        succeeded = sum(value in {"satisfied", "needs_input"} for value in requirement_results.values())
        total = len(plan.requirements)
        execution_status = ("needs_input" if "needs_input" in requirement_results.values()
                            else "completed" if succeeded == total
                            else "partial" if succeeded else "failed")
        if response is None:
            response = {
                "response_kind": "chat",
                "message": "这次没有完成请求，请稍后重试。",
                "turns": 0,
            }
        elif (requirement_results.get("route") == "failed"
              and response.get("response_kind") == "route"):
            response = {
                "response_kind": "chat",
                "message": "路线核验没有通过，请调整条件后重试。",
                "turns": response.get("turns", 0),
            }
        if weather:
            weather_text = _weather_message(weather)
            if plan.kind == "weather":
                response["message"] = weather_text
            else:
                if requirement_results.get("route") == "failed":
                    route_failure = "路线暂时无法规划，请检查起终点后重试。"
                    response["message"] = route_failure + "\n" + weather_text
                else:
                    response["message"] = (response.get("message", "").strip() + "\n" + weather_text).strip()
        elif "weather" in errors:
            weather_text = "天气服务暂时不可用。"
            if plan.kind == "composite":
                response["message"] = (response.get("message", "").strip() + "\n" + weather_text).strip()
            else:
                response["message"] = weather_text
        response.update({
            "weather": weather,
            "execution_status": execution_status,
            "requirement_results": requirement_results,
            "requirement_errors": errors,
            "needs_input": bool(response.get("needs_input")) or clarify or needs_user_selection,
        })
        _emit(state, runtime, "node_completed", "execute_agent", "需求分析与空间工具执行完成", database_url)
        return {"agent_response": response}

    def compose_result(state: AgentRunState, runtime: Runtime[AgentRunContext]):
        _emit(state, runtime, "node_started", "compose_result", "正在整理路线与说明", database_url)
        from api.routes import _agent_response_to_legacy

        data = state["input"]
        route_spec = (data.get("task") or {}).get("route_spec") or {}
        start = _coordinate(route_spec.get("start"))
        end = _coordinate(route_spec.get("end"))
        result = _agent_response_to_legacy(state["agent_response"], start, end)
        response = state["agent_response"]
        from agents import provenance as provenance_mod
        provenance_mod.persist(response.get("provenance"))
        for key in ("weather", "execution_status", "requirement_results", "requirement_errors",
                    "place_request", "candidates"):
            result[key] = response.get(key)
        if response.get("needs_input"):
            result["needs_input"] = True
            result["clarify"] = response.get("clarify")
            # The composite answer may include a completed independent requirement.
            result["message"] = response.get("message", result.get("message", ""))
            if result.get("task_type") == "unknown":
                result["clarify"] = response.get("clarify")
        _emit(state, runtime, "node_completed", "compose_result", "结果已整理，可以查看", database_url)
        return {"result": result}

    graph = StateGraph(AgentRunState, context_schema=AgentRunContext)
    graph.add_node("load_task", load_task)
    graph.add_node("execute_agent", execute_agent)
    graph.add_node("compose_result", compose_result)
    graph.add_edge(START, "load_task")
    graph.add_edge("load_task", "execute_agent")
    graph.add_edge("execute_agent", "compose_result")
    graph.add_edge("compose_result", END)
    compile_options = {"checkpointer": checkpointer}
    if interrupt_after:
        compile_options["interrupt_after"] = interrupt_after
    return graph.compile(**compile_options)
