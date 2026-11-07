"""Compare bounded DeepSeek text and function-call behavior without exposing secrets.

Run this script once from each source revision with the same provider environment.
It follows ``agents.planner._make_client`` and the production completion params,
but the tool calls are synthetic and never reach spatial services.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values


_PROVIDER_ENV = dotenv_values(os.environ.get("LUOJIA_PROVIDER_ENV", "/nonexistent"))
for _name, _value in _PROVIDER_ENV.items():
    if _name in {"DEEPSEEK_API_KEY", "LLM_MODEL", "OPENAI_BASE_URL", "HTTP_PROXY", "HTTPS_PROXY",
                 "ALL_PROXY", "NO_PROXY"} and _value:
        os.environ[_name] = _value

from agents import planner  # noqa: E402
from agents.dialogue import tool_schemas  # noqa: E402


MAX_CALLS = 12
REQUEST_TIMEOUT_SECONDS = 25
MAX_TOKENS = 2600
TEMPERATURE = 0.2


def _origin(value):
    if not value:
        return None
    parsed = urlsplit(value)
    return f"{parsed.scheme}://{parsed.hostname or ''}" + (f":{parsed.port}" if parsed.port else "")


def _safe_id(value):
    value = str(value) if value is not None else None
    return value if value and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) else None


def _failure_fields(error):
    response = getattr(error, "response", None)
    status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    status = status if isinstance(status, int) and not isinstance(status, bool) else None
    headers = getattr(response, "headers", None)
    request_id = getattr(error, "request_id", None)
    if not request_id and headers is not None:
        request_id = headers.get("x-request-id") or headers.get("request-id")
    code = getattr(error, "code", None)
    if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", code):
        code = f"http_{status}" if status else type(error).__name__
    return {"http_status": status, "request_id": _safe_id(request_id), "error_code": code}


def _message_summary(messages):
    roles = {role: 0 for role in ("system", "user", "assistant", "tool")}
    content_chars = 0
    for item in messages:
        role = item.get("role")
        if role in roles:
            roles[role] += 1
        content = item.get("content")
        if isinstance(content, str):
            content_chars += len(content)
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return {"message_count": len(messages), "role_counts": roles, "content_chars": content_chars,
            "messages_sha256": sha256(encoded).hexdigest()}


def _schema_hash(schemas):
    encoded = json.dumps(schemas, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _assistant_tool_message(message):
    tool_calls = []
    for call in getattr(message, "tool_calls", None) or []:
        tool_calls.append({"id": call.id, "type": "function", "function": {
            "name": call.function.name, "arguments": call.function.arguments}})
    result = {"role": "assistant", "content": message.content or "", "tool_calls": tool_calls}
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is not None:
        result["reasoning_content"] = reasoning
    return result


class BoundedClient:
    def __init__(self, client, journal, label):
        self.client = client
        self.journal = journal
        self.label = label
        self.calls = 0
        self.last_fingerprint = None
        self.same_request_failures = 0

    def create(self, probe, messages, schemas=None, **extra):
        request = {"model": planner.config.LLM_MODEL, "messages": messages,
                   "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                   "timeout": REQUEST_TIMEOUT_SECONDS, **extra}
        if schemas is not None:
            request["tools"] = schemas
        fingerprint = sha256(json.dumps(request, ensure_ascii=False, sort_keys=True,
                                        default=str).encode("utf-8")).hexdigest()
        if self.calls >= MAX_CALLS:
            raise RuntimeError("diagnostic_call_budget_exhausted")
        if self.last_fingerprint == fingerprint and self.same_request_failures >= 3:
            raise RuntimeError("same_request_failed_three_times")
        self.calls += 1
        started = time.monotonic()
        row = {"label": self.label, "probe": probe, "call": self.calls, "status": "started",
               "request_sha256": fingerprint, "messages": _message_summary(messages),
               "tool_count": len(schemas or []), "tool_schema_sha256": _schema_hash(schemas or []),
               "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
               "timeout_seconds": REQUEST_TIMEOUT_SECONDS}
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as error:
            same = self.last_fingerprint == fingerprint
            self.same_request_failures = self.same_request_failures + 1 if same else 1
            self.last_fingerprint = fingerprint
            row.update(_failure_fields(error))
            row.update({"status": "failed", "duration_ms": round((time.monotonic() - started) * 1000),
                        "same_request_failure_count": self.same_request_failures})
            self._write(row)
            raise
        self.last_fingerprint = None
        self.same_request_failures = 0
        choice = (getattr(response, "choices", None) or [None])[0]
        message = getattr(choice, "message", None)
        usage = getattr(response, "usage", None)
        row.update({"status": "returned", "duration_ms": round((time.monotonic() - started) * 1000),
                    "request_id": _safe_id(getattr(response, "_request_id", None)),
                    "provider_model": getattr(response, "model", None),
                    "tool_call_count": len(getattr(message, "tool_calls", None) or []),
                    "text_chars": len(getattr(message, "content", None) or ""),
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None)})
        self._write(row)
        return message

    def _write(self, row):
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def run_probes(client, journal, label):
    bounded = BoundedClient(client, journal, label)
    outcomes = []

    text_messages = [{"role": "user", "content": "请只回复：可用"}]
    try:
        text = bounded.create("plain_text", text_messages)
        outcomes.append({"probe": "plain_text", "passed": bool(text and text.content and text.content.strip()),
                         "calls": 1})
    except Exception as error:
        outcomes.append({"probe": "plain_text", "passed": False, "calls": 1,
                         "failure": _failure_fields(error)})

    minimal = [{"type": "function", "function": {
        "name": "probe_echo", "description": "Return the supplied token.",
        "parameters": {"type": "object", "properties": {"token": {"type": "string"}},
                       "required": ["token"], "additionalProperties": False}}}]
    minimal_prompt = "调用 probe_echo，token 填写 accept。"
    minimal_messages = [{"role": "user", "content": minimal_prompt}]
    minimal_failure = None
    minimal_ok = False
    minimal_calls = 1
    try:
        first = bounded.create("minimal_function_call", minimal_messages, minimal,
                               tool_choice={"type": "function", "function": {"name": "probe_echo"}})
        calls = getattr(first, "tool_calls", None) or []
        minimal_ok = bool(calls and calls[0].function.name == "probe_echo")
        if minimal_ok:
            messages = [*minimal_messages, _assistant_tool_message(first)]
            messages.extend({"role": "tool", "tool_call_id": call.id, "content": "{\"token\":\"accept\"}"}
                            for call in calls)
            minimal_calls += 1
            final = bounded.create("minimal_function_result", messages, minimal, tool_choice="none")
            minimal_ok = bool(final and not (getattr(final, "tool_calls", None) or []))
    except Exception as error:
        minimal_failure = _failure_fields(error)
    outcomes.append({"probe": "minimal_function_roundtrip", "passed": minimal_ok,
                     "calls": minimal_calls, **({"failure": minimal_failure} if minimal_failure else {})})

    all_tools = tool_schemas()
    safe_tool = next((item["function"]["name"] for item in all_tools
                      if item.get("function", {}).get("name") in {"get_weather", "list_road_conditions"}), None)
    full_ok = False
    full_calls = 0
    full_failure = None
    if safe_tool:
        user_prompt = f"用 {safe_tool} 工具完成一次协议检查；不需要真实查询，地点使用武汉大学。"
        full_messages = [{"role": "user", "content": user_prompt}]
        try:
            first = bounded.create("full_toolset_call", full_messages, all_tools,
                                   tool_choice={"type": "function", "function": {"name": safe_tool}})
            tool_calls = getattr(first, "tool_calls", None) or []
            full_calls = 1
            full_ok = bool(tool_calls and tool_calls[0].function.name == safe_tool)
            if full_ok:
                messages = [*full_messages, _assistant_tool_message(first)]
                messages.extend({"role": "tool", "tool_call_id": call.id,
                                 "content": "{\"status\":\"protocol_probe_only\",\"items\":[]}"}
                                for call in tool_calls)
                final = bounded.create("full_toolset_result", messages, all_tools, tool_choice="none")
                full_calls += 1
                full_ok = bool(final and not (getattr(final, "tool_calls", None) or []))
        except Exception as error:
            full_failure = _failure_fields(error)
    outcomes.append({"probe": "full_toolset_roundtrip", "passed": full_ok, "calls": full_calls,
                     "tool_count": len(all_tools), "tool_schema_sha256": _schema_hash(all_tools),
                     **({"failure": full_failure} if full_failure else {})})
    return outcomes, bounded.calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=("production-baseline", "candidate"), required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not planner.config.DEEPSEEK_API_KEY:
        parser.error("provider credentials are unavailable")

    output = args.output_dir or Path(__file__).resolve().parents[1] / "output/experiments/provider_protocol_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    report_dir = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex)
    report_dir.mkdir(parents=True, exist_ok=False)
    journal = report_dir / "requests.jsonl"
    client = planner._make_client()
    try:
        http_client = getattr(client, "_client", None)
        default_timeout = getattr(client, "timeout", None)
        timeout_detail = {name: getattr(default_timeout, name, None)
                          for name in ("connect", "read", "write", "pool")}
        outcomes, count = run_probes(client, journal, args.label)
        report = {"label": args.label, "completed_at": datetime.now(timezone.utc).isoformat(),
                  "source_commit": subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], text=True,
                      cwd=Path(__file__).resolve().parents[1]).strip(),
                  "provider": {"model": planner.config.LLM_MODEL,
                              "base_url_origin": _origin(str(planner.config.OPENAI_BASE_URL)),
                              "openai_sdk_version": importlib.metadata.version("openai"),
                              "httpx_version": importlib.metadata.version("httpx"),
                              "max_retries": getattr(client, "max_retries", None),
                              "httpx_trust_env": getattr(http_client, "_trust_env", None),
                              "default_headers": {key: client.default_headers.get(key)
                                                  for key in ("Accept-Encoding",)
                                                  if client.default_headers.get(key)},
                              "proxy_origins": {name: _origin(os.environ.get(name))
                                                for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")
                                                if os.environ.get(name)},
                              "no_proxy_entry_count": len([item for item in
                                  re.split(r"[,\s]+", os.environ.get("NO_PROXY", "")) if item]),
                              "client_default_timeout_seconds": timeout_detail,
                              "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                              "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE},
                  "request_count": count, "request_limit": MAX_CALLS,
                  "outcomes": outcomes, "passed": all(row["passed"] for row in outcomes),
                  "request_journal": journal.name,
                  "request_journal_sha256": sha256(journal.read_bytes()).hexdigest() if journal.exists() else None}
        report_path = report_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(report_path), "label": args.label, "passed": report["passed"],
                          "request_count": count,
                          "probes": [{"name": row["probe"], "passed": row["passed"], "calls": row["calls"]}
                                     for row in outcomes]}, ensure_ascii=False), flush=True)
        return 0 if report["passed"] else 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
