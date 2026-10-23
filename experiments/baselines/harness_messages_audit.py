"""Loopback-only Messages relay for transport-level development evidence.

It forwards a Harness request to one fixed DeepSeek endpoint. Only HTTP status
and the model field in a provider ``message_start`` event are retained. Request
body, authentication headers and generated text are never written to disk.
"""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock, Thread
from urllib.parse import urlsplit

import requests


OFFICIAL_MESSAGES_URL = "https://api.deepseek.com/anthropic/v1/messages"
MAX_REQUEST_BYTES = 4 * 1024 * 1024
_FORWARDED_HEADERS = (
    "authorization", "x-api-key", "content-type", "accept", "anthropic-version",
    "anthropic-beta", "x-deepseek-harness-user-id", "x-deepseek-harness-session-id",
    "x-deepseek-harness-compact",
)


def _response_model(line: bytes) -> str | None:
    if not line.startswith(b"data:"):
        return None
    try:
        event = json.loads(line[5:].strip())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(event, dict) or event.get("type") != "message_start":
        return None
    message = event.get("message")
    model = message.get("model") if isinstance(message, dict) else None
    return model if isinstance(model, str) and model else None


class MessagesAudit:
    def __init__(self, endpoint: str, api_key: str):
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("invalid Messages endpoint")
        self.endpoint = endpoint
        self.api_key = api_key
        self.attempts: list[dict] = []
        self.lock = Lock()
        self.url = ""

    def append(self, row: dict) -> None:
        with self.lock:
            self.attempts.append(row)


@contextmanager
def open_messages_audit(api_key: str, *, endpoint: str = OFFICIAL_MESSAGES_URL):
    """Relay only authenticated loopback /v1/messages POSTs."""
    if not api_key:
        raise ValueError("API key required")
    audit = MessagesAudit(endpoint, api_key)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args) -> None:
            return

        def _discard_bounded_body(self) -> None:
            """Avoid a Windows TCP reset while rejecting a small POST body."""
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return
            if 0 <= length <= MAX_REQUEST_BYTES:
                self.rfile.read(length)

        def do_POST(self) -> None:
            if self.path != "/v1/messages":
                self._discard_bounded_body()
                self.send_error(404)
                return
            bearer = self.headers.get("Authorization", "")
            supplied = self.headers.get("x-api-key", "")
            if bearer != f"Bearer {audit.api_key}" and supplied != audit.api_key:
                self._discard_bounded_body()
                self.send_error(401)
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self.send_error(411)
                return
            if length < 1 or length > MAX_REQUEST_BYTES:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            headers = {name: value for name, value in self.headers.items()
                       if name.lower() in _FORWARDED_HEADERS}
            headers["Accept-Encoding"] = "identity"
            row = {"http_status": None, "response_model_id": None,
                   "stream_complete": False}
            audit.append(row)
            try:
                with requests.Session() as session:
                    session.trust_env = False
                    with session.post(audit.endpoint, data=body, headers=headers,
                                      stream=True, allow_redirects=False,
                                      timeout=(15, 60)) as response:
                        row["http_status"] = response.status_code
                        self.send_response(response.status_code)
                        self.send_header("Content-Type", response.headers.get(
                            "Content-Type", "application/octet-stream"))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        pending = b""
                        for chunk in response.iter_content(chunk_size=8192):
                            if not chunk:
                                continue
                            pending += chunk
                            while b"\n" in pending:
                                line, pending = pending.split(b"\n", 1)
                                if row["response_model_id"] is None:
                                    row["response_model_id"] = _response_model(line.rstrip(b"\r"))
                                self.wfile.write(line + b"\n")
                                self.wfile.flush()
                            if len(pending) > 1024 * 1024:
                                raise ValueError("oversize upstream SSE line")
                        if pending:
                            self.wfile.write(pending)
                            self.wfile.flush()
                        row["stream_complete"] = True
            except requests.RequestException:
                if row["http_status"] is None:
                    self.send_error(502)
                # A partly streamed upstream response is necessarily incomplete.
            except (BrokenPipeError, ConnectionResetError, ValueError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    audit.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield audit
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
