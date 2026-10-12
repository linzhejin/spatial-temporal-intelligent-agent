import io
from pathlib import Path


def _manager_client(monkeypatch, tmp_path, captured):
    import api.routes as routes
    from app import create_app
    from storage import database, vision_repository

    monkeypatch.setattr(routes.config, "ROAD_CONDITION_ADMIN_PASSWORD", "manager-secret")
    monkeypatch.setattr(routes.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(routes, "_vision_inference_readiness", lambda *_args: (True, True, True, True))
    monkeypatch.setattr(database, "initialize", lambda *_args: None)

    def create_job(_url, **values):
        captured.update(values)
        return {
            "job_id": "00000000-0000-0000-0000-000000000001",
            "status": "queued",
            "anchor_gcj": values["anchor_gcj"],
            "capture_time_source": values["capture_time_source"],
            "capture_timezone": values["capture_timezone"],
        }

    monkeypatch.setattr(vision_repository, "create_job", create_job)
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="test-secret", DATABASE_URL=None)
    client = app.test_client()
    login = client.post("/api/admin/login", json={"password": "manager-secret"})
    return client, login.get_json()["data"]["csrf_token"]


def test_manager_vision_upload_accepts_media_without_location(monkeypatch, tmp_path):
    captured = {}
    client, csrf = _manager_client(monkeypatch, tmp_path, captured)

    response = client.post(
        "/api/manager/vision-jobs",
        data={
            "media": (io.BytesIO(b"\x89PNG\r\n\x1a\nvalid"), "campus.png", "image/png"),
            "captured_at": "2026-10-09T14:00:00+08:00",
            "capture_time_source": "manual",
            "capture_timezone": "Asia/Shanghai",
        },
        headers={"X-CSRF-Token": csrf},
    )

    assert response.status_code == 202
    assert response.get_json()["data"]["job"]["anchor_gcj"] is None
    assert captured["anchor_gcj"] is None
    assert captured["capture_time_source"] == "manual"
    assert captured["capture_timezone"] == "Asia/Shanghai"
    assert response.get_json()["data"]["job"]["capture_time_source"] == "manual"
    assert response.get_json()["data"]["job"]["capture_timezone"] == "Asia/Shanghai"


def test_optional_location_has_a_database_migration():
    migration = Path("storage/migrations/008_optional_vision_anchor.sql")

    assert migration.is_file()
    assert "ALTER COLUMN anchor_gcj DROP NOT NULL" in migration.read_text(encoding="utf-8")


def test_capture_time_provenance_migration_keeps_historical_sources_unknown():
    migration = Path("storage/migrations/013_vision_capture_time_provenance.sql")
    sql = migration.read_text(encoding="utf-8")

    assert "capture_time_source TEXT NOT NULL DEFAULT 'unknown'" in sql
    assert "capture_timezone TEXT" in sql
    assert "('unknown', 'manual', 'dji_srt')" in sql


def test_repository_stores_a_missing_anchor_as_database_null(monkeypatch):
    from storage import database, vision_repository

    captured = {}

    class Cursor:
        def fetchone(self):
            return {"job_id": "00000000-0000-0000-0000-000000000002", "status": "queued"}

    class Connection:
        def execute(self, query, params):
            captured["query"] = query
            captured["params"] = params
            return Cursor()

    class ConnectionContext:
        def __enter__(self):
            return Connection()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(database, "connect", lambda _url: ConnectionContext())

    vision_repository.create_job(
        None,
        created_by="web",
        original_name="campus.png",
        media_kind="image",
        media_path="stored.png",
        sha256="a" * 64,
        anchor_gcj=None,
        captured_at="2026-10-09T06:00:00+00:00",
        capture_time_source="dji_srt",
        capture_timezone="Asia/Shanghai",
    )

    assert captured["params"][6] is None
    assert captured["params"][9] == "dji_srt"
    assert captured["params"][10] == "Asia/Shanghai"
    assert "capture_time_source" in captured["query"]
