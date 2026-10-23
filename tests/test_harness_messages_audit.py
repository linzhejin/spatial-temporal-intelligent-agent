"""The local relay must capture provider identity without storing payloads."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import requests

from experiments.baselines.harness_messages_audit import (
    _response_model, open_messages_audit,
)


def test_response_model_only_reads_message_start():
    assert _response_model(b'data: {"type":"message_start","message":{"model":"m1"}}') == "m1"
    assert _response_model(b'data: {"type":"content_block_delta","text":"secret"}') is None
    assert _response_model(b'data: invalid') is None


def test_relay_captures_actual_model_and_rejects_unauthorized_client():
    seen = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            seen.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            payload = ('event: message_start\n'
                       'data: {"type":"message_start","message":{"model":"resolved-1"}}\n\n'
                       'event: content_block_delta\n'
                       'data: {"type":"content_block_delta","delta":{"text":"想打篮球"}}\n\n'
                       'event: message_stop\ndata: {"type":"message_stop"}\n\n').encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1/messages"
        with open_messages_audit("test-key", endpoint=endpoint) as audit:
            bad = requests.post(audit.url + "/v1/messages", data=b"ignored", timeout=5)
            assert bad.status_code == 401
            response = requests.post(audit.url + "/v1/messages", data=b"private prompt",
                                     headers={"Authorization": "Bearer test-key"}, timeout=5)
            assert response.status_code == 200
            assert b"resolved-1" in response.content
            assert "想打篮球" in response.content.decode("utf-8")
            assert seen == [("/v1/messages", b"private prompt")]
            assert audit.attempts == [{"http_status": 200,
                                       "response_model_id": "resolved-1",
                                       "stream_complete": True}]
            assert "private prompt" not in str(audit.attempts)
            assert "test-key" not in str(audit.attempts)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
