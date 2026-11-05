"""Database boundary contract for clarification memory, without a live server."""

from contextlib import contextmanager

from storage import task_repository


def test_original_request_survives_when_short_history_no_longer_contains_it(monkeypatch):
    calls = []

    class Cursor:
        def __init__(self, value):
            self.value = value

        def fetchone(self):
            return self.value

        def fetchall(self):
            return self.value

    class Connection:
        def execute(self, sql, params):
            calls.append((sql, params))
            values = [
                {"run_id": "run-1", "task_id": "task-1", "task_revision": 10,
                 "cancel_requested": False, "conversation_id": "conversation-1",
                 "revision": 10, "state": {"route_spec": {}}, "owner_id": "owner-1"},
                {"seq": 25, "content": "从信息学部出发"},
                {"content": "从玉兰2门到樱顶，经过卓尔体育馆，并查天气"},
                [{"role": "assistant", "content": "起点是哪里？"}],
                None,
                None,
            ]
            return Cursor(values[len(calls) - 1])

    @contextmanager
    def connect(_url):
        yield Connection()

    monkeypatch.setattr(task_repository.database, "connect", connect)
    result = task_repository.get_run_input("postgres://unused", "run-1")

    assert len(calls) == 6
    assert "ORDER BY seq ASC LIMIT 1" in calls[2][0]
    assert result["task_origin_query"] == "从玉兰2门到樱顶，经过卓尔体育馆，并查天气"
    assert result["query"] == "从信息学部出发"
    assert result["history"] == [{"role": "assistant", "content": "起点是哪里？"}]


def test_last_completed_route_can_be_found_past_unrelated_messages(monkeypatch):
    captured = []

    class Cursor:
        def __init__(self, value): self.value = value
        def fetchone(self): return self.value
        def fetchall(self): return self.value

    class Connection:
        def execute(self, sql, _params):
            captured.append(sql)
            values = [
                {"run_id": "run-2", "task_id": "task-2", "task_revision": 0,
                 "cancel_requested": False, "conversation_id": "conversation-1",
                 "revision": 0, "state": {"route_spec": {}}, "owner_id": "owner-1"},
                {"seq": 30, "content": "刚才的路线再短一点"},
                [{"role": "user", "content": "今天会下雨吗？"}],
                {"route_state": {"route_id": "older-route"}, "task_id": "task-1", "source_seq": 20},
                [],
                None,
            ]
            return Cursor(values[len(captured) - 1])

    @contextmanager
    def connect(_url): yield Connection()

    monkeypatch.setattr(task_repository.database, "connect", connect)
    result = task_repository.get_run_input("postgres://unused", "run-2")
    assert "NOT EXISTS" not in captured[-3]
    assert result["previous_route_state"]["route_id"] == "older-route"
    assert result["previous_route_state_task_id"] == "task-1"
