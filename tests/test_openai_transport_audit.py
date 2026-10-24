"""Actual HTTP attempts must be observable without retaining request secrets."""

import httpx2
from openai import OpenAI

from experiments.baselines.openai_transport_audit import OpenAITransportAudit
from scripts.analysis.probe_ordinary_tool_agent import _make_audited_client


def test_openai_retry_counts_each_http_attempt_without_request_content():
    seen = 0

    def respond(_request):
        nonlocal seen
        seen += 1
        if seen == 1:
            return httpx2.Response(429, headers={"retry-after": "0"},
                                   json={"error": {"message": "retry", "type": "rate_limit"}})
        return httpx2.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "model": "deepseek-flash",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "好"},
                         "finish_reason": "stop"}],
        })

    audit = OpenAITransportAudit()
    with httpx2.Client(transport=httpx2.MockTransport(respond),
                       event_hooks=audit.event_hooks) as http_client:
        client = OpenAI(api_key="test-secret", base_url="https://example.test/v1",
                        http_client=http_client, max_retries=1)
        response = client.chat.completions.create(
            model="deepseek-chat", messages=[{"role": "user", "content": "private query"}],
        )
    assert response.model == "deepseek-flash"
    assert audit.summary(logical_requests=1) == {
        "provider_http_attempt_count": 2,
        "provider_http_attempts": [{"http_status": 429}, {"http_status": 200}],
        "provider_transport_complete": True,
    }
    assert "test-secret" not in str(audit.summary(logical_requests=1))
    assert "private query" not in str(audit.summary(logical_requests=1))


def test_missing_response_does_not_claim_complete_transport():
    audit = OpenAITransportAudit()
    audit.request_hook(httpx2.Request("POST", "https://example.test/v1/chat/completions"))
    assert audit.summary(logical_requests=1)["provider_transport_complete"] is False


def test_ordinary_pilot_client_uses_transport_audit():
    audit = OpenAITransportAudit()
    transport = httpx2.MockTransport(lambda _request: httpx2.Response(200, json={
        "id": "chatcmpl-2", "object": "chat.completion", "model": "deepseek-flash",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "好"},
                     "finish_reason": "stop"}],
    }))
    client, http_client = _make_audited_client(
        key="private", base_url="https://example.test/v1", audit=audit,
        transport=transport,
    )
    try:
        client.chat.completions.create(model="deepseek-chat", messages=[
            {"role": "user", "content": "hello"},
        ])
    finally:
        http_client.close()
    assert audit.summary(logical_requests=1)["provider_http_attempt_count"] == 1
