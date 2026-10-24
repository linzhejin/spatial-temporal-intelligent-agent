"""Polling must close an overdue run even when the worker is unavailable."""

from contextlib import contextmanager

from storage import task_repository


def test_get_run_invokes_deadline_cleanup_for_owned_active_run(monkeypatch):
    rows = [
        {"run_id": "run-1", "task_id": "task-1", "status": "queued"},
        {"run_id": "run-1", "task_id": "task-1", "status": "failed",
         "error": {"code": "deadline_exceeded"}},
    ]
    calls = []

    class Cursor:
        def __init__(self, row=None, items=None):
            self.row, self.items = row, items

        def fetchone(self):
            return self.row

        def fetchall(self):
            return self.items or []

    class Connection:
        def execute(self, statement, params):
            if "FROM run_event" in statement:
                return Cursor(items=[])
            return Cursor(row=rows.pop(0))

    connection = Connection()

    @contextmanager
    def connect(_url):
        yield connection

    monkeypatch.setattr(task_repository.database, "connect", connect)
    monkeypatch.setattr(task_repository, "_expire_due_runs",
                        lambda conn, run_id=None: calls.append((conn, run_id)))

    result = task_repository.get_run(None, "owner-1", "run-1")
    assert calls == [(connection, "run-1")]
    assert result["status"] == "failed"
    assert result["error"]["code"] == "deadline_exceeded"
