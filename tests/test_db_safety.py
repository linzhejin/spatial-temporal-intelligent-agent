"""Focused guard checks; none of these tests open PostgreSQL."""

from contextlib import contextmanager

import pytest

from tests import db_safety


class _Connection:
    def __init__(self, name):
        self.name = name

    def execute(self, query):
        assert query == "SELECT current_database() AS name"
        return self

    def fetchone(self):
        return {"name": self.name}


@contextmanager
def _fake_connect(name):
    yield _Connection(name)


def test_rejects_business_named_test_target_before_connect(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db_safety.database, "connect", lambda *_: pytest.fail("must not connect"))
    with pytest.raises(RuntimeError, match="dedicated test database"):
        db_safety.verify_isolated_test_database("postgresql://user:pass@localhost/campus")


def test_rejects_same_database_as_business_url_before_connect(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/campus_test")
    monkeypatch.setattr(db_safety.database, "connect", lambda *_: pytest.fail("must not connect"))
    with pytest.raises(RuntimeError, match="business database"):
        db_safety.verify_isolated_test_database("postgresql://user:pass@localhost/campus_test")


def test_checks_database_identity_on_connection(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db_safety.database, "connect", lambda *_: _fake_connect("campus_test"))
    db_safety.verify_isolated_test_database("postgresql://user:pass@localhost/campus_test")
    monkeypatch.setattr(db_safety.database, "connect", lambda *_: _fake_connect("campus"))
    with pytest.raises(RuntimeError, match="unexpected database"):
        db_safety.verify_isolated_test_database("postgresql://user:pass@localhost/campus_test")


def test_rejects_business_url_supplied_from_dotenv_before_connect(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db_safety.database, "connect", lambda *_: pytest.fail("must not connect"))
    with pytest.raises(RuntimeError, match="business database"):
        db_safety.verify_isolated_test_database(
            "postgresql://user:pass@localhost/campus_test",
            business_urls=("postgresql://owner:secret@other-host/campus_test",),
        )
