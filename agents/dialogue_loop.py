"""Run model-selected tools without classifying the user's words in advance."""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
import logging
import re
import time

import config
from agents import web_evidence
from agents.dialogue import _model_tool_content, tool_schemas

logger = logging.getLogger(__name__)


class DiagnosticCallBudgetExceeded(RuntimeError):
    """A diagnostic harness stopped before making a provider request."""


def _request_summary(messages, schemas, *, timeout):
    roles = {role: 0 for role in ("system", "user", "assistant", "tool")}
    content_chars = 0
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else None
        if role in roles:
            roles[role] += 1
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            content_chars += len(content)
    return {
        "model": re.sub(r"[^A-Za-z0-9_.:/-]", "", str(config.LLM_MODEL))[:80],
        "message_count": sum(roles.values()),
        "role_counts": roles,
        "content_chars": content_chars,
        "tool_count": len(schemas),
        "temperature": 0.2,
        "max_tokens": 2600,
        "timeout_seconds": round(timeout, 2),
    }


def _provider_error_details(error):
    response = getattr(error, "response", None)
    status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    status = str(status) if isinstance(status, int) and not isinstance(status, bool) else None
    request_id = getattr(error, "request_id", None)
    if not request_id:
        headers = getattr(response, "headers", None)
        if headers is not None:
            request_id = headers.get("x-request-id") or headers.get("request-id")
    request_id = str(request_id) if request_id is not None else None
    if request_id and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id):
        request_id = None
    code = getattr(error, "code", None)
    if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", code):
        code = "http_" + status if status else type(error).__name__
    return status, request_id, code


def _record_model_failure(dialogue, error, messages, schemas, timeout, started):
    duration_ms = round((time.monotonic() - started) * 1000)
    status, request_id, code = _provider_error_details(error)
    summary = _request_summary(messages, schemas, timeout=timeout)
    details = {
        "status": status,
        "request_id": request_id,
        "duration_ms": duration_ms,
        "error_code": code,
        "request_summary": summary,
    }
    dialogue.trace({
        "tool_name": "model_dialogue", "argument_keys": [], "outcome": "provider_error",
        "artifact_kind": None, "phase": "completed", "duration_ms": duration_ms,
        "error_code": code, "provider_status": status, "request_id": request_id,
        "request_summary": summary,
    })
    logger.warning("model_request_failed %s", json.dumps(details, ensure_ascii=True, sort_keys=True))


def _road_condition_summary(result):
    conditions = result.get("conditions") if isinstance(result, dict) else None
    if not isinstance(conditions, list):
        return ""
    messages = []
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        action = condition.get("action")
        road = str(condition.get("road_name") or condition.get("location") or "").strip()[:80]
        if not road or action not in {"closure", "slowdown", "notice"}:
            continue
        if action == "closure":
            detail = str(condition.get("name") or condition.get("description") or "").strip()[:80]
            message = road + "禁止通行" + ("（" + detail + "）" if detail and detail not in road else "")
        elif action == "slowdown":
            message = road + "有缓行"
        else:
            detail = str(condition.get("name") or condition.get("description") or "").strip()[:80]
            message = road + "有提示" + ("（" + detail + "）" if detail and detail not in road else "")
        if message not in messages:
            messages.append(message)
    return "路况：" + "；".join(messages[:3]) + "。" if messages else ""


def _failure(dialogue, message, code):
    result = dialogue.result(message, failed=True)
    result["failure_code"] = code
    return result


def _successful_terminal_reply(dialogue, tool_names):
    """Deliver verified terminal tool results without spending a second call to restate them."""
    tool_names = set(tool_names)
    result_tools = tool_names - {"update_task"}
    if "get_weather" in result_tools and result_tools <= {"get_weather"}:
        weather = dialogue.weather
        if not isinstance(weather, dict):
            return None
        parts = []
        condition = weather.get("weather")
        if isinstance(condition, str) and condition.strip():
            parts.append("武汉今天" + condition.strip() + "。")
        temperature = weather.get("temperature")
        if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
            parts.append("气温" + str(temperature) + "℃。")
        if weather.get("slippery") is True:
            parts.append("路面可能湿滑。")
        return "".join(parts) if parts else "武汉今天的天气信息已查到。"

    if "plan_route" in result_tools and result_tools <= {"plan_route"}:
        route = dialogue.route
        verification = (route or {}).get("route_verification") if isinstance(route, dict) else None
        if not isinstance(verification, dict) or verification.get("status") != "pass":
            return None
        state = route.get("route_state") or {}
        strategy = state.get("strategy") or {}
        if strategy.get("name") == "flat":
            return "已按平坦偏好规划好路线。"
        if strategy.get("name") == "shortest":
            return "已按最短路线规划好。"
        return "路线已规划好。"

    return None


def _finish_with_read_failures(dialogue, text, failures, successful_reads):
    """Keep verified partial results while reporting unresolved service failures."""
    if not failures:
        return dialogue.result(text)
    parts = []
    if dialogue.candidates:
        parts.append(text)
    elif dialogue.route:
        parts.append("路线已保留，可以查看地图和路线卡片。")
    elif dialogue.comparison:
        parts.append("路线比较结果已保留。")
    weather = successful_reads.get("get_weather")
    if isinstance(weather, dict) and weather.get("weather") is not None:
        description = str(weather["weather"])[:100]
        temperature = weather.get("temperature")
        parts.append("当前天气：" + description +
                     ("，气温 " + str(temperature)[:30] + "℃" if temperature is not None else "") + "。")
    road_summary = _road_condition_summary(successful_reads.get("list_road_conditions") or {})
    if road_summary and road_summary not in text:
        parts.append(road_summary)
    parts.extend(failures.values())
    dialogue.clarify = None
    result = dialogue.result("\n\n".join(parts))
    result["failure_code"] = next(iter(failures))
    result["execution_status"] = "partial" if (successful_reads or dialogue.route or dialogue.candidates
                                                  or dialogue.comparison) else "failed"
    return result


def run_dialogue_loop(dialogue):
    from agents.deadline import shared_deadline
    with shared_deadline(dialogue.deadline_monotonic):
        return _run_dialogue_loop(dialogue)


def _run_dialogue_loop(dialogue):
    started = time.monotonic()
    deadline = min(started + 75.0, getattr(dialogue, "deadline_monotonic", started + 75.0))
    try:
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("model_not_configured")
        client = dialogue.client_factory()
    except Exception:
        code = "model_not_configured" if not config.DEEPSEEK_API_KEY else "model_client_init_failed"
        elapsed = round((time.monotonic() - started) * 1000)
        dialogue.trace({"tool_name": "model_dialogue", "argument_keys": [],
                        "outcome": "client_error", "artifact_kind": None, "phase": "completed",
                        "duration_ms": elapsed, "error_code": code})
        logger.warning("model_client_unavailable error_code=%s duration_ms=%s", code, elapsed)
        return _failure(dialogue, "对话模型暂时不可用，已保留本轮需求，请稍后重试。", "model_unavailable")

    schemas = tool_schemas()
    names = {item["function"]["name"] for item in schemas}
    repeated_errors = {}
    undelivered_candidate_responses = 0
    read_failures, successful_reads = {}, {}
    read_errors = {
        "weather_unavailable": "天气暂时获取不到，请稍后重试。",
        "road_conditions_unavailable": "路况数据暂不可用，无法确认当前管制信息，请稍后重试。",
    }
    for _ in range(20):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _failure(dialogue, "本轮处理超时，已保留当前需求，请重试。", "deadline_exceeded")
        dialogue.turns += 1
        model_started = time.monotonic()
        dialogue.trace({"tool_name": "model_dialogue", "argument_keys": [],
                        "outcome": "success", "artifact_kind": None, "phase": "started"})
        request_timeout = min(25.0, remaining)
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL, messages=deepcopy(dialogue.messages), tools=schemas,
                temperature=0.2, max_tokens=2600, timeout=request_timeout,
            )
        except Exception as error:
            if isinstance(error, DiagnosticCallBudgetExceeded):
                elapsed = round((time.monotonic() - model_started) * 1000)
                dialogue.trace({"tool_name": "model_dialogue", "argument_keys": [],
                                "outcome": "diagnostic_budget_exhausted", "artifact_kind": None,
                                "phase": "completed", "duration_ms": elapsed,
                                "error_code": "diagnostic_call_budget_exhausted"})
                result = _failure(dialogue, "本次验收达到调用预算，已保留当前需求。",
                                  "diagnostic_call_budget_exhausted")
                result["model_status"] = "budget_exhausted"
                return result
            _record_model_failure(dialogue, error, dialogue.messages, schemas, request_timeout, model_started)
            code = str(getattr(error, "code", "") or "")
            detail = str(error).lower()
            if (code in {"context_length_exceeded", "context_length_exceeded_error"}
                    or getattr(error, "status_code", None) == 400 and
                    ("maximum context length" in detail or "context length" in detail and "exceed" in detail)):
                return _failure(dialogue, "本次会话内容超过模型可处理的长度，请新建会话后继续。", "context_length_exceeded")
            if getattr(error, "status_code", None) == 429:
                return _failure(dialogue, "对话服务请求过多，请稍后重试。", "model_rate_limited")
            if "timeout" in type(error).__name__.lower():
                return _failure(dialogue, "对话服务响应超时，已保留当前需求，请重试。", "model_timeout")
            return _failure(dialogue, "对话服务暂时不可用，已保留当前需求，请稍后重试。", "model_unavailable")
        dialogue.trace({"tool_name": "model_dialogue", "argument_keys": [], "outcome": "success",
                        "artifact_kind": None, "phase": "completed",
                        "duration_ms": round((time.monotonic() - model_started) * 1000)})
        usage = getattr(response, "usage", None)
        for field in dialogue.usage:
            amount = getattr(usage, field, 0)
            dialogue.usage[field] += int(amount) if isinstance(amount, (int, float)) and not isinstance(amount, bool) else 0
        choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or not choices or getattr(choices[0], "message", None) is None:
            return _failure(dialogue, "对话服务返回空结果，请重试。", "model_empty_response")
        msg = choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            text = str(msg.content or "").strip()
            if dialogue.place_service_error and not (dialogue.route or dialogue.comparison or dialogue.candidates):
                return _failure(dialogue, "地点服务暂时不可用，本轮没有查到地点，请稍后重试。", "place_service_unavailable")
            if dialogue.route_tool_failure and not dialogue.route and not dialogue.candidates:
                return _failure(dialogue, dialogue.route_tool_failure["message"] + " 已保留已有需求，可调整条件后重试。",
                                dialogue.route_tool_failure["code"])
            if dialogue.turn_action == "next_batch" and not dialogue.candidates:
                remaining = dialogue.browse_candidates({})
                if remaining["count"]:
                    undelivered_candidate_responses += 1
                    if undelivered_candidate_responses > 2:
                        return _failure(dialogue, "仍有候选尚未完成筛选，已保留当前需求，请重试。", "candidate_delivery_incomplete")
                    dialogue.messages.append({"role": "assistant", "content": text})
                    dialogue.messages.append({"role": "system", "content":
                        "服务器核验仍有未展示的候选记录。请调用 browse_candidates 查看具体地点，再用 present_places 交付；"
                        "如果查看后没有符合条件的地点，调用 conclude_without_places 说明原因。"})
                    continue
            if read_failures:
                return _finish_with_read_failures(dialogue, text, read_failures, successful_reads)
            return dialogue.result(text) if text else _failure(dialogue, "本轮没有形成回答，请重试。", "model_empty_response")
        dialogue.messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": call.id, "type": "function", "function": {
                "name": call.function.name, "arguments": call.function.arguments}} for call in calls
        ], **({"reasoning_content": msg.reasoning_content}
              if getattr(msg, "reasoning_content", None) is not None else {})})
        final = None
        end_turn_requested = False
        response_had_tool_error = False
        parallel = {}
        reads = [call for call in calls if call.function.name in {"get_weather", "list_road_conditions"}]
        if len(reads) == 2 and len({call.function.name for call in reads}) == 2:
            try:
                parsed = [(call, json.loads(call.function.arguments)) for call in reads]
                if not all(isinstance(args, dict) for _, args in parsed):
                    raise ValueError("invalid read arguments")
            except (ValueError, TypeError):
                parsed = []
            if parsed:
                pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dialogue-read")
                started_at = time.monotonic()
                def read_only(name, args):
                    try:
                        return dialogue.spatial(name, args)
                    except (ValueError, TypeError, KeyError) as error:
                        return {"error": "invalid_tool_request", "message": str(error)[:300]}
                try:
                    futures = []
                    for call, args in parsed:
                        dialogue.trace({"tool_name": call.function.name, "argument_keys": [], "outcome": "success",
                                        "artifact_kind": None, "phase": "started"})
                        context = copy_context()
                        futures.append((call, pool.submit(context.run, read_only, call.function.name, args)))
                    for call, future in futures:
                        parallel[call.id] = (future.result(timeout=max(0.001, deadline - time.monotonic())), started_at)
                except TimeoutError:
                    return _failure(dialogue, "本轮处理超时，已保留当前需求，请重试。", "deadline_exceeded")
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
        # A model may request state changes and spatial reads together. Apply its
        # state change first so every tool uses the accepted version.
        for call in sorted(calls, key=lambda item: item.function.name != "update_task"):
            if time.monotonic() >= deadline:
                return _failure(dialogue, "本轮处理超时，已保留当前需求，请重试。", "deadline_exceeded")
            name, args = call.function.name, {}
            reported_argument_keys = []
            end_turn_for_tool = None
            tool_started = parallel[call.id][1] if call.id in parallel else time.monotonic()
            if call.id not in parallel:
                dialogue.trace({"tool_name": name, "argument_keys": [], "outcome": "success",
                                "artifact_kind": None, "phase": "started"})
            try:
                args = json.loads(call.function.arguments)
                if not isinstance(args, dict) or name not in names:
                    raise ValueError("invalid tool or arguments")
                reported_argument_keys = sorted(args)
                if name in {"get_weather", "plan_route"}:
                    end_turn_for_tool = args.pop("end_turn_if_success", False)
                    if type(end_turn_for_tool) is not bool:
                        raise ValueError("end_turn_if_success must be a boolean")
                    end_turn_requested |= end_turn_for_tool
                if (dialogue.bind_task and dialogue.bound_task is None
                        and name not in {"update_task", "get_weather", "list_road_conditions", "search_web", "read_web_page", "read_result", "browse_candidates"}):
                    raise ValueError("task_binding_required: 先用 update_task 选择或创建任务")
                if name == "update_task":
                    answer = dialogue.update(args)
                elif name == "browse_candidates":
                    answer = dialogue.browse_candidates(args)
                elif name == "read_result":
                    answer = dialogue.read_result(**args) if dialogue.read_result else {
                        "error": "result_reference_unavailable", "message": "此环境没有可读取的历史结果。"}
                elif name == "present_places":
                    final = dialogue.present(args)
                    answer = {"accepted": True, "poi_ids": dialogue.pending["candidate_poi_ids"]}
                elif name == "conclude_without_places":
                    reason = args.get("reason")
                    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
                        raise ValueError("invalid_candidate_conclusion")
                    final = reason.strip()
                    answer = {"accepted": True}
                elif name in {"search_web", "read_web_page"}:
                    counter = "web_search_count" if name == "search_web" else "web_fetch_count"
                    limit = 4 if name == "search_web" else 3
                    if getattr(dialogue, counter) >= limit:
                        answer = {"error": "web_evidence_limit", "message": "本轮网页查询次数已达上限。"}
                    else:
                        setattr(dialogue, counter, getattr(dialogue, counter) + 1)
                        answer = web_evidence.execute(name, args, dialogue.sources, original_question=dialogue.query)
                elif call.id in parallel:
                    answer = parallel[call.id][0]
                else:
                    answer = dialogue.spatial(name, args)
            except TimeoutError:
                return _failure(dialogue, "本轮处理超时，已保留当前需求，请重试。", "deadline_exceeded")
            except (ValueError, TypeError, KeyError) as error:
                answer = {"error": "invalid_tool_request", "message": str(error)[:300]}
            trace_event = {"tool_name": name, "argument_keys": reported_argument_keys,
                           "outcome": "tool_error" if answer.get("error") else "success", "artifact_kind": None,
                           "phase": "completed", "duration_ms": round((time.monotonic() - tool_started) * 1000),
                           "error_code": answer.get("error") if isinstance(answer.get("error"), str)
                           and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", answer["error"]) else None,
                           "provider_status": answer.get("online_status") if isinstance(answer.get("online_status"), str)
                           and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", answer["online_status"]) else None}
            if end_turn_for_tool is not None:
                trace_event["end_turn_if_success"] = end_turn_for_tool
            dialogue.trace(trace_event)
            dialogue.messages.append({"role": "tool", "tool_call_id": call.id,
                                      "content": _model_tool_content(name, answer)})
            response_had_tool_error |= bool(answer.get("error"))
            if name in {"get_weather", "list_road_conditions"}:
                code = answer.get("error")
                if code in read_errors:
                    read_failures[code] = read_errors[code]
                    successful_reads.pop(name, None)
                elif not code:
                    successful_reads[name] = answer
                    expected_error = "weather_unavailable" if name == "get_weather" else "road_conditions_unavailable"
                    read_failures.pop(expected_error, None)
            if answer.get("error"):
                fingerprint = (name, str(answer["error"]), str(answer.get("message") or ""))
                repeated_errors[fingerprint] = repeated_errors.get(fingerprint, 0) + 1
                if repeated_errors[fingerprint] >= 3:
                    if answer["error"] in read_errors:
                        return _finish_with_read_failures(dialogue, "", read_failures, successful_reads)
                    return _failure(dialogue, "本轮结果反复未通过校验，已保留当前需求，请重试。", "repeated_tool_error")
            if answer.get("error") == "place_service_unavailable":
                return _failure(dialogue, "地点服务暂时不可用，本轮没有查到地点，请稍后重试。", "place_service_unavailable")
            if answer.get("error") == "place_search_limit":
                return _failure(dialogue, "本轮地点查询次数已用完，已保留当前需求，请重试。", "place_search_limit")
        if final is not None:
            return _finish_with_read_failures(dialogue, final, read_failures, successful_reads)
        if dialogue.clarify:
            return _finish_with_read_failures(dialogue, dialogue.clarify.get("question", "请补充具体要求。"),
                                              read_failures, successful_reads)
        if end_turn_requested and not response_had_tool_error:
            terminal_reply = _successful_terminal_reply(
                dialogue, {call.function.name for call in calls}
            )
            if terminal_reply is not None:
                return dialogue.result(terminal_reply)
    return _failure(dialogue, "本轮处理达到调用上限，已保留当前需求，请重试。", "tool_limit")
