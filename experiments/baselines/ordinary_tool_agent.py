"""Minimal transcript-only tool Agent for same-model campus comparisons.

This baseline shares the product tool schemas/executor but has no explicit
task-state machine, verified route repair, PROV memory, or cached experience.
It is a runnable condition, not a score or a substitute for Harness.
"""
from __future__ import annotations

import json
import hashlib
import time
from typing import Any, Callable

from agents import tools as campus_tools


SYSTEM_PROMPT = (
    "你是校园出行助手。根据用户当前和先前消息完成任务。校园地点、道路、天气和路况"
    "必须从提供的工具取得；不确定时提问，不编造地点或通行状态。"
)


def run_conversation(user_turns: list[str], *, client: Any, model: str,
                     tool_executor: Callable = campus_tools.execute_tool,
                     context: dict | None = None, max_rounds_per_turn: int = 6,
                     max_total_tool_calls: int | None = None,
                     time_budget_s: float = 40.0, max_tokens: int = 800,
                     data_snapshot_id: str | None = None,
                     event_snapshot_sha256: str | None = None,
                     event_clock_epoch: int | None = None) -> dict:
    """Run the same registered tools with only ordinary message-history memory."""
    if not user_turns or any(not isinstance(turn, str) or not turn.strip()
                             for turn in user_turns):
        raise ValueError("nonempty user turns required")
    if (max_rounds_per_turn < 1 or time_budget_s <= 0 or max_tokens < 1
            or max_total_tool_calls is not None and max_total_tool_calls < 1):
        raise ValueError("positive execution budgets required")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    run_started = time.monotonic()
    turns = []
    total_calls = 0
    requests = 0
    resolved_models = set()
    token_usage = {"input": 0, "output": 0}
    cache_usage = {"hit": 0, "miss": 0}
    cache_accounted = True
    schema_hash = hashlib.sha256(json.dumps(
        campus_tools.TOOL_SCHEMAS, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    tool_names = sorted(schema["function"]["name"] for schema in campus_tools.TOOL_SCHEMAS)
    evidence = [{"kind": "run", "condition": "ordinary_tool_agent",
                 "data_snapshot_id": data_snapshot_id,
                 "event_snapshot_sha256": event_snapshot_sha256,
                 "event_clock_epoch": event_clock_epoch,
                 "enabled_tool_names": tool_names, "extra_tool_names": [],
                 "max_model_rounds_per_turn": max_rounds_per_turn,
                 "time_budget_s": time_budget_s}]
    for user_text in user_turns:
        turn_started = time.monotonic()
        # Campus recommendation tools use the current utterance to infer
        # activity intent. A fixed context would silently erase that input.
        turn_context = {**(context or {}), "query": user_text}
        messages.append({"role": "user", "content": user_text})
        trace = []
        artifacts = []
        selected_start_name = None
        selected_mode = None
        final_text = ""
        status = "limit_reached"
        for _ in range(max_rounds_per_turn):
            if time.monotonic() - turn_started > time_budget_s:
                status = "time_limit"
                break
            try:
                completion = client.chat.completions.create(
                    model=model, messages=messages, tools=campus_tools.TOOL_SCHEMAS,
                    temperature=0.0, max_tokens=max_tokens,
                )
            except Exception as exc:
                requests += 1
                status = "model_error"
                trace.append({"outcome": "model_error", "error_type": type(exc).__name__})
                evidence.append({"kind": "request", "requested_model_id": model,
                                 "resolved_model_id": None,
                                 "tool_schema_sha256": schema_hash,
                                 "temperature": 0, "max_tokens": max_tokens,
                                 "input_tokens": 0, "output_tokens": 0,
                                 "outcome": "model_error"})
                break
            requests += 1
            observed_model = getattr(completion, "model", None)
            if isinstance(observed_model, str) and observed_model:
                resolved_models.add(observed_model)
            usage = getattr(completion, "usage", None)
            token_usage["input"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            token_usage["output"] += int(getattr(usage, "completion_tokens", 0) or 0)
            evidence.append({"kind": "request", "requested_model_id": model,
                             "resolved_model_id": observed_model,
                             "tool_schema_sha256": schema_hash,
                             "temperature": 0, "max_tokens": max_tokens,
                             "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                             "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                             "outcome": "success"})
            hit = getattr(usage, "prompt_cache_hit_tokens", None)
            miss = getattr(usage, "prompt_cache_miss_tokens", None)
            if hit is None or miss is None:
                cache_accounted = False
            else:
                cache_usage["hit"] += int(hit)
                cache_usage["miss"] += int(miss)
            message = completion.choices[0].message
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                final_text = (getattr(message, "content", None) or "").strip()
                messages.append({"role": "assistant", "content": final_text})
                status = "completed"
                break
            if (max_total_tool_calls is not None
                    and total_calls + len(calls) > max_total_tool_calls):
                status = "tool_call_limit"
                break
            messages.append(message.model_dump(exclude_none=True))
            for call in calls:
                total_calls += 1
                name = call.function.name
                error_type = None
                try:
                    args = json.loads(call.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    result, artifact = tool_executor(name, args, turn_context)
                    outcome = "tool_error" if isinstance(result, dict) and result.get("error") else "success"
                    if outcome == "success" and isinstance(result, dict):
                        selected_start_name = result.get("start_name") or selected_start_name
                        selected_mode = result.get("mode") or selected_mode
                except (ValueError, json.JSONDecodeError):
                    args, artifact = {}, None
                    result, outcome = {"error": "invalid_args", "message": "工具参数不是 JSON 对象"}, "invalid_args"
                except Exception as exc:
                    artifact = None
                    result, outcome = {"error": "tool_failure", "message": "工具执行失败"}, "tool_error"
                    error_type = type(exc).__name__
                artifact_kind = next((kind for kind in (
                    "route", "candidates", "clarify", "suggestions", "route_comparison"
                ) if isinstance(artifact, dict) and kind in artifact), None)
                trace.append({"tool_name": name, "argument_keys": sorted(args),
                              "outcome": outcome,
                              "error_code": result.get("error") if isinstance(result, dict) else None,
                              "error_type": error_type,
                              "artifact_kind": artifact_kind})
                evidence.append({"kind": "tool", "name": name, "outcome": outcome})
                if isinstance(artifact, dict):
                    artifacts.append(artifact)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": campus_tools.result_to_json(result)})
        turns.append({"status": status, "message": final_text,
                      "tool_trace": trace, "artifacts": artifacts,
                      "selected_start_name": selected_start_name,
                      "selected_mode": selected_mode,
                      "elapsed_ms": round((time.monotonic() - turn_started) * 1000, 2)})
    return {"condition": "ordinary_tool_agent",
            "model": model, "turns": turns, "tool_calls": total_calls,
            "resolved_model_id": next(iter(resolved_models)) if len(resolved_models) == 1 else None,
            "input_tokens": token_usage["input"],
            "output_tokens": token_usage["output"],
            "measured_usage": {"requests": requests,
                               "input_tokens": token_usage["input"],
                               "output_tokens": token_usage["output"],
                               "cache_hit_input_tokens": cache_usage["hit"] if cache_accounted else None,
                               "cache_miss_input_tokens": cache_usage["miss"] if cache_accounted else None,
                               "cache_accounted": cache_accounted,
                               "elapsed_ms": round((time.monotonic() - run_started) * 1000, 2)},
            "limits": {"rounds_per_turn": max_rounds_per_turn,
                       "total_tool_calls": max_total_tool_calls,
                       "time_budget_s": time_budget_s,
                       "max_tokens": max_tokens},
            "run_evidence": evidence}
