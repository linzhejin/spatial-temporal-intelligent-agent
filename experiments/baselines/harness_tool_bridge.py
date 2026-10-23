"""Loopback-only bridge from pinned DeepSeek Harness to campus tool executor.

The model-facing schemas and Python implementation are the same objects used
by the ordinary-tool baseline. This service exists only for isolated local
experiment runs; it does not expose a production endpoint.
"""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
from typing import Callable, Iterator

from agents import tools as campus_tools


_MAX_BODY_BYTES = 128 * 1024


class ToolBridge:
    """Own one short-lived local tool service and its observable call trace."""

    def __init__(self, *, executor: Callable = campus_tools.execute_tool,
                 context: dict | None = None) -> None:
        self.executor = executor
        self.context = context or {}
        self.token = secrets.token_urlsafe(32)
        self.allowed = {row["function"]["name"] for row in campus_tools.TOOL_SCHEMAS}
        self.trace: list[dict] = []
        self.artifacts: list[dict] = []
        # Only selected, typed result fields needed for task projection. The
        # model's arguments and full tool responses stay out of the trace.
        self.selection_context: list[dict] = []
        self._lock = threading.Lock()
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if self.path != "/execute" or self.headers.get("X-Campus-Token") != bridge.token:
                    try:
                        rejected_size = int(self.headers.get("Content-Length", "-1"))
                        if 0 <= rejected_size <= _MAX_BODY_BYTES:
                            self.connection.settimeout(2)
                            self.rfile.read(rejected_size)
                    except (OSError, ValueError):
                        pass
                    self.send_error(403)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    size = -1
                if size < 0:
                    self.send_error(413)
                    return
                if size > _MAX_BODY_BYTES:
                    # Drain a small oversized request before responding. On
                    # Windows, replying while the client is still writing the
                    # body can reset the connection before it sees HTTP 413.
                    if size <= _MAX_BODY_BYTES * 2:
                        self.rfile.read(size)
                    else:
                        self.close_connection = True
                    self.send_error(413)
                    return
                try:
                    payload = json.loads(self.rfile.read(size))
                    name = payload["name"]
                    args = payload["arguments"]
                    if name not in bridge.allowed or not isinstance(args, dict):
                        raise ValueError("invalid tool name or arguments")
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    self.send_error(400)
                    return
                try:
                    result, artifact = bridge.executor(name, args, dict(bridge.context))
                    response = campus_tools.result_to_json(result)
                    outcome = "tool_error" if isinstance(result, dict) and result.get("error") else "success"
                    error_code = result.get("error") if isinstance(result, dict) else None
                except Exception as exc:  # Match the ordinary-tool baseline's failure boundary.
                    response = campus_tools.result_to_json(
                        {"error": "tool_failure", "message": "工具执行失败"})
                    artifact, outcome, error_code = None, "tool_error", "tool_failure"
                    error_type = type(exc).__name__
                else:
                    error_type = None
                with bridge._lock:
                    artifact_kind = next((kind for kind in (
                        "route", "candidates", "clarify", "suggestions", "route_comparison"
                    ) if isinstance(artifact, dict) and kind in artifact), None)
                    bridge.trace.append({"tool_name": name, "argument_keys": sorted(args),
                                         "outcome": outcome, "error_code": error_code,
                                         "error_type": error_type,
                                         "artifact_kind": artifact_kind})
                    selected = {}
                    if outcome == "success" and artifact_kind == "candidates" \
                            and isinstance(result, dict):
                        for field in ("start_name", "mode"):
                            value = result.get(field)
                            if isinstance(value, str) and value:
                                selected[field] = value
                    bridge.selection_context.append(selected)
                    if isinstance(artifact, dict):
                        bridge.artifacts.append(artifact)
                body = json.dumps({"result": response}, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/execute"

    def start(self) -> "ToolBridge":
        self.thread.start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@contextmanager
def open_tool_bridge(*, executor: Callable = campus_tools.execute_tool,
                     context: dict | None = None) -> Iterator[ToolBridge]:
    bridge = ToolBridge(executor=executor, context=context).start()
    try:
        yield bridge
    finally:
        bridge.close()
