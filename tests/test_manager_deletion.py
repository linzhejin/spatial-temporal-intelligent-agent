import uuid
from pathlib import Path

from app import create_app
from storage import database, vision_repository


def test_pending_delete_repository_uses_last_attempt_order(monkeypatch):
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    class Cursor:
        def fetchall(self):
            return [{"job_id": job_id, "media_path": job_id + ".png", "sha256": "digest"}
                    for job_id in ids]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params):
            normalized = " ".join(query.split())
            assert "WHERE deleted_at IS NOT NULL ORDER BY updated_at, job_id LIMIT %s" in normalized
            assert params == (200,)
            return Cursor()

    monkeypatch.setattr(database, "connect", lambda _url: Connection())

    pending = vision_repository.list_pending_deletions("postgresql://test", limit=200)

    assert [row["job_id"] for row in pending] == ids


def test_begin_delete_jobs_locks_duplicate_rows_in_stable_order(monkeypatch):
    digest = "same-content-sha256"
    locked_ids = ["job-a", "job-b", "job-c"]
    calls = []

    class Cursor:
        def __init__(self, rows=None, row=None):
            self.rows = rows or []
            self.row = row

        def fetchone(self):
            return self.row

        def fetchall(self):
            return self.rows

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params):
            normalized = " ".join(query.split())
            calls.append((normalized, params))
            if normalized.startswith("SELECT sha256, purpose FROM manager_vision_job"):
                return Cursor(row={"sha256": digest, "purpose": "trial"})
            if normalized.startswith("SELECT job_id FROM manager_vision_job"):
                assert "WHERE sha256=%s AND purpose=%s ORDER BY job_id FOR UPDATE" in normalized
                assert params == (digest, "trial")
                return Cursor(rows=[{"job_id": job_id} for job_id in locked_ids])
            if normalized.startswith("UPDATE manager_vision_job"):
                assert "WHERE sha256=%s AND purpose=%s AND deleted_at IS NULL" in normalized
                assert params == (digest, "trial")
                return Cursor()
            assert normalized.startswith(
                "SELECT job_id, media_path, sha256 FROM manager_vision_job WHERE sha256=%s AND purpose=%s"
            )
            assert params == (digest, "trial")
            return Cursor(rows=[{"job_id": job_id, "media_path": job_id + ".png", "sha256": digest}
                                for job_id in locked_ids])

    monkeypatch.setattr(database, "connect", lambda _url: Connection())

    deleted = vision_repository.begin_delete_jobs("postgresql://test", "job-b")

    assert [row["job_id"] for row in deleted] == locked_ids
    assert len(calls) == 4
    assert "ORDER BY job_id FOR UPDATE" in calls[1][0]
    assert calls[1][1] == (digest, "trial")


def install_fake_vision_repository(monkeypatch, rows):
    """Install a small durable-state model for manager deletion API tests."""
    retry_clock = {"value": 0}
    for row in rows.values():
        if "updated_at" not in row:
            row["updated_at"] = retry_clock["value"]
        retry_clock["value"] = max(retry_clock["value"], row["updated_at"] + 1)

    def begin_delete_jobs(_url, requested_id):
        target = rows.get(requested_id)
        if target is None:
            return []
        digest = target.get("sha256") or requested_id
        for row in rows.values():
            if (row.get("sha256") or row["job_id"]) == digest:
                row["deleted_at"] = row.get("deleted_at") or True
                row["worker_id"] = None
                row["lease_until"] = None
                row["updated_at"] = retry_clock["value"]
                retry_clock["value"] += 1
        return sorted(
            (dict(row) for row in rows.values()
             if row.get("deleted_at") and (row.get("sha256") or row["job_id"]) == digest),
            key=lambda row: (row["updated_at"], row["job_id"]),
        )

    def list_pending_deletions(_url, limit=200):
        pending = sorted((dict(row) for row in rows.values() if row.get("deleted_at")),
                         key=lambda row: (row["updated_at"], row["job_id"]))
        return pending[:limit]

    def mark_delete_retry_attempts(_url, job_ids):
        for job_id in job_ids:
            if job_id in rows and rows[job_id].get("deleted_at"):
                rows[job_id]["updated_at"] = retry_clock["value"]
                retry_clock["value"] += 1
        return len(job_ids)

    def finalize_delete_jobs(_url, job_ids):
        removed = 0
        for job_id in job_ids:
            if job_id in rows and rows[job_id].get("deleted_at"):
                del rows[job_id]
                removed += 1
        return removed

    def get_job(_url, job_id):
        row = rows.get(job_id)
        return dict(row) if row and not row.get("deleted_at") else None

    def list_jobs(_url, limit=50):
        return [dict(row) for row in rows.values() if not row.get("deleted_at")][:limit]

    monkeypatch.setattr(vision_repository, "begin_delete_jobs", begin_delete_jobs, raising=False)
    monkeypatch.setattr(vision_repository, "list_pending_deletions", list_pending_deletions, raising=False)
    monkeypatch.setattr(vision_repository, "mark_delete_retry_attempts", mark_delete_retry_attempts, raising=False)
    monkeypatch.setattr(vision_repository, "finalize_delete_jobs", finalize_delete_jobs, raising=False)
    monkeypatch.setattr(vision_repository, "get_job", get_job)
    monkeypatch.setattr(vision_repository, "list_jobs", list_jobs)


def manager_client(monkeypatch, upload_dir):
    import api.routes as routes

    monkeypatch.setattr(routes.config, "ROAD_CONDITION_ADMIN_TOKEN", "manager-delete-token")
    monkeypatch.setattr(routes.config, "VISION_UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(database, "initialize", lambda *_args: None)
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client(), {"X-Admin-Token": "manager-delete-token"}


def test_manager_deletes_vision_file_and_record_permanently(monkeypatch, tmp_path):
    job_id = str(uuid.uuid4())
    media_name = job_id + ".png"
    media_file = tmp_path / media_name
    media_file.write_bytes(b"private aerial image")
    rows = {job_id: {"job_id": job_id, "media_path": media_name, "sha256": "image-a",
                     "original_name": "campus.png", "deleted_at": None}}
    install_fake_vision_repository(monkeypatch, rows)
    client, headers = manager_client(monkeypatch, tmp_path)

    response = client.delete("/api/manager/vision-jobs/" + job_id, headers=headers)

    assert response.status_code == 200
    assert not media_file.exists()
    assert client.get("/api/manager/vision-jobs?limit=20", headers=headers).get_json()["data"]["jobs"] == []
    assert client.get("/api/manager/vision-jobs/" + job_id, headers=headers).status_code == 404
    assert client.get("/api/manager/vision-jobs/" + job_id + "/media", headers=headers).status_code == 404


def test_manager_delete_removes_all_visible_jobs_with_identical_media(monkeypatch, tmp_path):
    job_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    rows = {}
    files = []
    for job_id in job_ids:
        media_name = job_id + ".png"
        media_file = tmp_path / media_name
        media_file.write_bytes(b"same exact aerial image")
        files.append(media_file)
        rows[job_id] = {"job_id": job_id, "media_path": media_name,
                        "sha256": "same-content-sha256", "original_name": "flight.png",
                        "deleted_at": None}
    install_fake_vision_repository(monkeypatch, rows)
    client, headers = manager_client(monkeypatch, tmp_path)

    response = client.delete("/api/manager/vision-jobs/" + job_ids[0], headers=headers)

    assert response.status_code == 200
    assert response.get_json()["data"]["deleted_count"] == 2
    assert all(not path.exists() for path in files)
    assert client.get("/api/manager/vision-jobs?limit=20", headers=headers).get_json()["data"]["jobs"] == []


def test_manager_delete_rejects_path_escape_without_resurrecting_record(monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"keep")
    job_id = str(uuid.uuid4())
    rows = {job_id: {"job_id": job_id, "media_path": "../outside.png", "sha256": "escape",
                     "original_name": "outside.png", "deleted_at": None}}
    install_fake_vision_repository(monkeypatch, rows)
    client, headers = manager_client(monkeypatch, upload_dir)

    response = client.delete("/api/manager/vision-jobs/" + job_id, headers=headers)

    assert response.status_code == 202
    assert response.get_json()["data"]["cleanup_pending"] is True
    assert outside.read_bytes() == b"keep"
    assert rows[job_id]["deleted_at"]
    assert client.get("/api/manager/vision-jobs?limit=20", headers=headers).get_json()["data"]["jobs"] == []


def test_failed_media_unlink_stays_hidden_and_is_retried_after_reopen(monkeypatch, tmp_path):
    job_id = str(uuid.uuid4())
    media_name = job_id + ".png"
    media_file = tmp_path / media_name
    media_file.write_bytes(b"private aerial image")
    rows = {job_id: {"job_id": job_id, "media_path": media_name, "sha256": "image-retry",
                     "original_name": "campus.png", "deleted_at": None,
                     "status": "running", "worker_id": "vision-1", "lease_until": 123}}
    install_fake_vision_repository(monkeypatch, rows)
    client, headers = manager_client(monkeypatch, tmp_path)

    original_unlink = Path.unlink
    allow_unlink = {"value": False}

    def failing_unlink(path, *args, **kwargs):
        if path.resolve() == media_file.resolve() and not allow_unlink["value"]:
            raise OSError("simulated locked media file")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    response = client.delete("/api/manager/vision-jobs/" + job_id, headers=headers)

    assert response.status_code == 202
    assert response.get_json()["data"]["cleanup_pending"] is True
    assert rows[job_id]["deleted_at"]
    assert rows[job_id]["worker_id"] is None and rows[job_id]["lease_until"] is None
    assert media_file.exists()
    assert client.get("/api/manager/vision-jobs?limit=20", headers=headers).get_json()["data"]["jobs"] == []

    allow_unlink["value"] = True
    reopened_listing = client.get("/api/manager/vision-jobs?limit=20", headers=headers).get_json()["data"]["jobs"]
    assert reopened_listing == []
    assert not media_file.exists()
    assert job_id not in rows


def test_repeatedly_blocked_tombstones_do_not_starve_later_cleanup(monkeypatch, tmp_path):
    valid_id = str(uuid.uuid4())
    media_name = valid_id + ".png"
    media_file = tmp_path / media_name
    media_file.write_bytes(b"later cleanup")
    rows = {}
    for index in range(200):
        job_id = str(uuid.uuid4())
        rows[job_id] = {"job_id": job_id, "media_path": "../blocked-%03d.png" % index,
                        "sha256": "blocked-%03d" % index, "deleted_at": True,
                        "updated_at": index}
    rows[valid_id] = {"job_id": valid_id, "media_path": media_name,
                      "sha256": "later-valid", "deleted_at": True, "updated_at": 200}
    install_fake_vision_repository(monkeypatch, rows)
    client, headers = manager_client(monkeypatch, tmp_path)

    first_reopen = client.get("/api/manager/vision-jobs?limit=20", headers=headers)
    assert first_reopen.status_code == 200
    assert media_file.exists()

    second_reopen = client.get("/api/manager/vision-jobs?limit=20", headers=headers)
    assert second_reopen.status_code == 200
    assert not media_file.exists()
    assert valid_id not in rows
