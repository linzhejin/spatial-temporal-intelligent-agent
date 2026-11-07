from types import SimpleNamespace

import pytest

from scripts.provider_protocol_diagnostic import BoundedClient, _failure_fields


def test_provider_diagnostic_journal_keeps_status_and_request_id_but_not_error_body(tmp_path):
    class ProviderError(RuntimeError):
        status_code = 500
        request_id = "req-diagnostic-123"
        code = "internal_error"

        response = SimpleNamespace(status_code=500, headers={"x-request-id": "req-diagnostic-123"},
                                   text="PRIVATE_PROVIDER_BODY")

    class Completions:
        def create(self, **_kwargs):
            raise ProviderError("PRIVATE_PROVIDER_BODY")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Completions().create)))
    journal = tmp_path / "requests.jsonl"
    bounded = BoundedClient(client, journal, "candidate")

    with pytest.raises(ProviderError):
        bounded.create("plain_text", [{"role": "user", "content": "probe"}])

    content = journal.read_text(encoding="utf-8")
    assert '"http_status": 500' in content
    assert '"request_id": "req-diagnostic-123"' in content
    assert "PRIVATE_PROVIDER_BODY" not in content
    assert _failure_fields(ProviderError("PRIVATE_PROVIDER_BODY")) == {
        "http_status": 500, "request_id": "req-diagnostic-123", "error_code": "internal_error"
    }


def test_provider_diagnostic_request_budget_stops_before_network_call(tmp_path):
    calls = []

    class Completions:
        def create(self, **_kwargs):
            calls.append(1)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
                                   model="probe-model", usage=None, _request_id="req-1")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Completions().create)))
    bounded = BoundedClient(client, tmp_path / "requests.jsonl", "candidate")
    for index in range(12):
        bounded.create("plain_text", [{"role": "user", "content": f"probe {index}"}])

    with pytest.raises(RuntimeError, match="diagnostic_call_budget_exhausted"):
        bounded.create("plain_text", [{"role": "user", "content": "the thirteenth request is not sent"}])
    assert len(calls) == 12
