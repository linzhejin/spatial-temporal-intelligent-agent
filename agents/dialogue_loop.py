"""Run model-selected tools without classifying the user's words in advance."""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
import re
import time

import config
from agents import web_evidence
from agents.dialogue import _model_tool_content, tool_schemas


def _failure(dialogue, message, code):
    result = dialogue.result(message, failed=True)
    result["failure_code"] = code
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
        return _failure(dialogue, "对话模型暂时不可用，已保留本轮需求，请稍后重试。", "model_unavailable")

    schemas = tool_schemas()
    names = {item["function"]["name"] for item in schemas}
    repeated_errors = {}
    undelivered_candidate_responses = 0
    for _ in range(20):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _failure(dialogue, "本轮处理超时，已保留当前需求，请重试。", "deadline_exceeded")
        dialogue.turns += 1
        model_started = time.monotonic()
        dialogue.trace({"tool_name": "model_dialogue", "argument_keys": [],
                        "outcome": "success", "artifact_kind": None, "phase": "started"})
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL, messages=deepcopy(dialogue.messages), tools=schemas,
                temperature=0.2, max_tokens=2600, timeout=min(25.0, remaining),
            )
        except Exception as error:
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
                        f"服务器核对：候选池仍有 {remaining['count']} 条未展示记录。请查看具体候选并用 present_places 交付；"
                        "如果已查看但没有符合条件的地点，调用 conclude_without_places 说明原因，不要声称候选池已空。"})
                    continue
            return dialogue.result(text) if text else _failure(dialogue, "本轮没有形成回答，请重试。", "model_empty_response")
        dialogue.messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": call.id, "type": "function", "function": {
                "name": call.function.name, "arguments": call.function.arguments}} for call in calls
        ], **({"reasoning_content": msg.reasoning_content}
              if getattr(msg, "reasoning_content", None) is not None else {})})
        final = None
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
            tool_started = parallel[call.id][1] if call.id in parallel else time.monotonic()
            if call.id not in parallel:
                dialogue.trace({"tool_name": name, "argument_keys": [], "outcome": "success",
                                "artifact_kind": None, "phase": "started"})
            try:
                args = json.loads(call.function.arguments)
                if not isinstance(args, dict) or name not in names:
                    raise ValueError("invalid tool or arguments")
                if (dialogue.bind_task and dialogue.bound_task is None
                        and name not in {"update_task", "get_weather", "list_road_conditions", "search_web", "read_web_page", "read_result"}):
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
                    remaining = dialogue.browse_candidates({})["count"]
                    if remaining:
                        final += f" 候选池仍有 {remaining} 条未展示记录。"
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
            dialogue.trace({"tool_name": name, "argument_keys": sorted(args) if isinstance(args, dict) else [],
                            "outcome": "tool_error" if answer.get("error") else "success", "artifact_kind": None,
                            "phase": "completed", "duration_ms": round((time.monotonic() - tool_started) * 1000),
                            "error_code": answer.get("error") if isinstance(answer.get("error"), str)
                            and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", answer["error"]) else None,
                            "provider_status": answer.get("online_status") if isinstance(answer.get("online_status"), str)
                            and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", answer["online_status"]) else None})
            dialogue.messages.append({"role": "tool", "tool_call_id": call.id,
                                      "content": _model_tool_content(name, answer)})
            if answer.get("error"):
                fingerprint = (name, str(answer["error"]), str(answer.get("message") or ""))
                repeated_errors[fingerprint] = repeated_errors.get(fingerprint, 0) + 1
                if repeated_errors[fingerprint] >= 3:
                    return _failure(dialogue, "本轮结果反复未通过校验，已保留当前需求，请重试。", "repeated_tool_error")
            if answer.get("error") == "place_service_unavailable":
                return _failure(dialogue, "地点服务暂时不可用，本轮没有查到地点，请稍后重试。", "place_service_unavailable")
            if answer.get("error") == "place_search_limit":
                return _failure(dialogue, "本轮地点查询次数已用完，已保留当前需求，请重试。", "place_search_limit")
        if final is not None:
            return dialogue.result(final)
        if dialogue.clarify:
            return dialogue.result(dialogue.clarify.get("question", "请补充具体要求。"))
    return _failure(dialogue, "本轮处理达到调用上限，已保留当前需求，请重试。", "tool_limit")
