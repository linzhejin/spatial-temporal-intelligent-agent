from jobs import worker


def test_unexpected_agent_failure_keeps_task_and_returns_retryable_error(monkeypatch):
    pending = {
        "request_id": "request-1", "original_query": "我在信图，想去总图附近干饭",
        "start": {"type": "poi", "name": "信息学部图书馆"},
        "search_anchor": {"type": "poi", "name": "武汉大学图书馆(总馆)"},
        "candidate_pool": [{"poi_id": "amap:MEAL1", "name": "候选餐饮"}],
    }
    monkeypatch.setattr(worker, "claim_next_run", lambda *_args: {"run_id": "run-1", "attempt": 1})
    monkeypatch.setattr(worker, "heartbeat_run", lambda *_args: True)
    monkeypatch.setattr(worker, "get_run_input", lambda *_args: {
        "task": {"place_request": pending}, "cancel_requested": False,
    })
    finished = {}

    def finish(_url, _run_id, _worker_id, **kwargs):
        finished.update(kwargs)
        return "failed"

    monkeypatch.setattr(worker, "finish_run", finish)

    class BrokenGraph:
        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("private provider detail must not reach the user")

    outcome = worker.process_next_run("postgresql://test", "worker-1", BrokenGraph())

    assert outcome["status"] == "failed"
    assert finished["status"] == "failed"
    result = finished["result"]
    assert result["response_kind"] == "chat"
    assert result["place_request"] == pending
    assert result["failure_code"] == "worker_error"
    assert "重试" in result["message"]
    assert "private provider detail" not in result["message"]
    assert "error" not in finished
