import io
import json
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


def _enable_doodleverse_assistance(monkeypatch, tmp_path):
    import api.routes as routes

    model_path = tmp_path / "doodleverse-v7.onnx"
    model_path.write_bytes(b"pinned-onnx-test-fixture")
    monkeypatch.setattr(routes.config, "VISION_FLOOD_ENABLED", True)
    monkeypatch.setattr(routes.config, "VISION_FLOOD_RELEASE_MODE", "assisted_review")
    monkeypatch.setattr(routes.config, "VISION_FLOOD_MODEL_PATH", str(model_path))
    monkeypatch.setattr(routes.config, "VISION_FLOOD_REVISION", "doodleverse-floodnet-resunet-v7")
    monkeypatch.setattr(routes, "_vision_worker_capabilities", lambda *_args: {
        "flood_assistance_supported": True,
        "flood_assistance_automatic_supported": True,
        "flood_assistance_release_status": "approved",
        "flood_assistance_model_id": "doodleverse-floodnet-resunet-v7",
        "flood_assistance_model_version": "doodleverse-floodnet-resunet-v7",
        "flood_assistance_model_sha256": "a" * 64,
        "trial_flood_supported": True,
        "trial_flood_media": "image",
        "trial_flood_automatic_supported": True,
    })


def _flood_upload_data(*, purpose="operational", include_anchor=True):
    data = {
        "media": (io.BytesIO(b"\x89PNG\r\n\x1a\nvalid"), "campus.png", "image/png"),
        "purpose": purpose,
        "captured_at": "2026-11-10T02:00:00Z",
        "capture_time_source": "manual",
        "capture_timezone": "UTC",
        "observation_regions": json.dumps([
            {"id": "road-main", "kind": "road_surface", "polygon": [
                [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
            ]},
            {"id": "people", "kind": "pedestrian", "polygon": [
                [0.2, 0.2], [0.3, 0.2], [0.3, 0.3],
            ]},
        ]),
    }
    if include_anchor:
        data.update({"lng": "114.36", "lat": "30.53"})
    return data


def test_doodleverse_upload_tags_only_road_roi_and_keeps_operational_job(monkeypatch, tmp_path):
    _enable_doodleverse_assistance(monkeypatch, tmp_path)
    captured = {}
    client, csrf = _manager_client(monkeypatch, tmp_path, captured)

    response = client.post(
        "/api/manager/vision-jobs",
        data={**_flood_upload_data(), "flood_assistance": "true"},
        headers={"X-CSRF-Token": csrf},
    )

    assert response.status_code == 202, response.get_json()
    assert captured["purpose"] == "operational"
    regions = {item["id"]: item for item in captured["observation_regions"]}
    assert regions["road-main"]["flood_assistance_mode"] == "assisted_review"
    assert "flood_assistance_mode" not in regions["people"]
    assert captured["anchor_gcj"] is not None


def test_doodleverse_public_trial_uses_same_image_path_without_becoming_publishable(monkeypatch, tmp_path):
    _enable_doodleverse_assistance(monkeypatch, tmp_path)
    captured = {}
    client, csrf = _manager_client(monkeypatch, tmp_path, captured)
    data = _flood_upload_data(purpose="trial", include_anchor=False)
    data.pop("captured_at")
    data.pop("capture_time_source")
    data.pop("capture_timezone")

    response = client.post(
        "/api/manager/vision-jobs", data=data,
        headers={"X-CSRF-Token": csrf},
    )

    assert response.status_code == 202, response.get_json()
    assert captured["purpose"] == "trial"
    assert captured["anchor_gcj"] is None
    assert all("flood_assistance_mode" not in item for item in captured["observation_regions"])


def test_public_flood_image_accepts_upload_without_manual_road_polygon(monkeypatch, tmp_path):
    _enable_doodleverse_assistance(monkeypatch, tmp_path)
    captured = {}
    client, csrf = _manager_client(monkeypatch, tmp_path, captured)
    data = _flood_upload_data(purpose="trial", include_anchor=False)
    data["observation_regions"] = "[]"

    response = client.post("/api/manager/vision-jobs", data=data,
                           headers={"X-CSRF-Token": csrf})

    assert response.status_code == 202, response.get_json()
    assert captured["purpose"] == "trial"
    assert captured["observation_regions"] == []


def test_operational_flood_image_queues_automatic_analysis_without_manual_polygon(monkeypatch, tmp_path):
    _enable_doodleverse_assistance(monkeypatch, tmp_path)
    captured = {}
    client, csrf = _manager_client(monkeypatch, tmp_path, captured)
    data = _flood_upload_data()
    data.update({"observation_regions": "[]", "flood_assistance": "true"})

    response = client.post("/api/manager/vision-jobs", data=data,
                           headers={"X-CSRF-Token": csrf})

    assert response.status_code == 202, response.get_json()
    assert captured["purpose"] == "operational"
    marker = captured["observation_regions"]
    assert len(marker) == 1
    assert marker[0]["flood_assistance_mode"] == "assisted_review"
    assert marker[0]["kind"] == "automatic_flood_analysis"
    assert "polygon" not in marker[0]


def test_automatic_flood_rejects_upload_until_specialized_worker_supports_it(monkeypatch, tmp_path):
    import api.routes as routes

    _enable_doodleverse_assistance(monkeypatch, tmp_path)
    monkeypatch.setattr(routes, "_vision_worker_capabilities", lambda *_args: {
        "flood_assistance_supported": True, "flood_assistance_release_status": "approved",
        "flood_assistance_model_id": "doodleverse-floodnet-resunet-v7",
    })
    captured = {}
    client, csrf = _manager_client(monkeypatch, tmp_path, captured)
    data = _flood_upload_data()
    data.update({"observation_regions": "[]", "flood_assistance": "true"})

    response = client.post("/api/manager/vision-jobs", data=data,
                           headers={"X-CSRF-Token": csrf})

    assert response.status_code == 503
    assert response.get_json()["error"] == "flood_automatic_not_ready"
    assert not captured


def test_doodleverse_assisted_upload_rejects_video_and_missing_location(monkeypatch, tmp_path):
    _enable_doodleverse_assistance(monkeypatch, tmp_path)
    captured = {}
    client, csrf = _manager_client(monkeypatch, tmp_path, captured)
    video = client.post(
        "/api/manager/vision-jobs",
        data={
            "media": (io.BytesIO(b"video"), "campus.mp4", "video/mp4"),
            "flood_assistance": "true", "purpose": "operational",
        }, headers={"X-CSRF-Token": csrf},
    )
    assert video.status_code == 415
    assert video.get_json()["error"] == "flood_assistance_image_only"

    missing_location = client.post(
        "/api/manager/vision-jobs",
        data={**_flood_upload_data(include_anchor=False), "flood_assistance": "true"},
        headers={"X-CSRF-Token": csrf},
    )
    assert missing_location.status_code == 400
    assert missing_location.get_json()["error"] == "flood_assistance_anchor_required"


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
