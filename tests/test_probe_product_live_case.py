"""The in-process development pilot must retain only bounded run evidence."""

from types import SimpleNamespace

import pytest

from scripts.analysis.probe_product_live_case import _LocalSession, main


def test_local_session_excludes_raw_model_payloads_from_pilot_evidence():
    payload = {"data": {"status": "completed", "events": [
        {"event_type": "agent_model_request", "payload": {
            "requested_model_id": "deepseek-chat", "resolved_model_id": "deepseek-flash",
            "input_tokens": 23, "output_tokens": 7,
            "raw_prompt": "private user input", "authorization": "private key",
        }},
    ]}}
    client = SimpleNamespace(get=lambda *_args, **_kwargs: SimpleNamespace(
        status_code=200, get_json=lambda: payload,
    ))
    session = _LocalSession(client, graph=None, database_url="isolated-test-database")

    session.get("http://127.0.0.1:55440/api/runs/run-1")

    assert session.terminal_runs["run-1"]["run_id"] == "run-1"
    assert session.terminal_runs["run-1"]["model_requests"][0][
        "resolved_model_id"] == "deepseek-flash"
    assert "private user input" not in str(session.terminal_runs)
    assert "private key" not in str(session.terminal_runs)


def test_live_pilot_requires_explicit_billable_call_flag(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "probe_product_live_case", "--env-file", "missing.env",
        "--test-database-url", "unused", "--case-id", "unused",
        "--output", "unused.json",
    ])
    with pytest.raises(SystemExit, match="explicit --allow-model-call"):
        main()
