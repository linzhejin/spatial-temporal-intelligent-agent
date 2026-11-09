import json
from datetime import datetime, timezone

import pytest
from psycopg.types.json import Jsonb

from api.routes import _parse_vision_camera_pose
from storage import database, vision_repository

DATABASE_URL = None  # assigned only after isolated_test_database validates the target


# ---- 纯函数单测（不依赖 DB） ----

def _legal_pose():
    return {
        "crs": "WGS84",
        "parsed": True,
        "sampled": False,
        "point_count": 1,
        "trajectory": [
            {
                "lng": 114.36, "lat": 30.53,
                "rel_alt_m": 12.5, "abs_alt_m": 45.0,
                "yaw_deg": 90.0, "gb_yaw_deg": 1.5, "gb_pitch_deg": -2.5,
                "gb_roll_deg": None,
                "focal_len_35mm": 24.0, "seconds": 0.0,
            },
        ],
    }


def test_parse_vision_camera_pose_roundtrip_dict():
    assert _parse_vision_camera_pose(_legal_pose()) == _legal_pose()


def test_parse_vision_camera_pose_accepts_json_string():
    assert _parse_vision_camera_pose(json.dumps(_legal_pose())) == _legal_pose()


def test_parse_vision_camera_pose_rejects_missing_or_invalid_payload():
    assert _parse_vision_camera_pose(None) is None
    assert _parse_vision_camera_pose("not json") is None
    assert _parse_vision_camera_pose(123) is None
    assert _parse_vision_camera_pose({"crs": "WGS84"}) is None  # no trajectory
    assert _parse_vision_camera_pose({"trajectory": []}) is None  # empty
    assert _parse_vision_camera_pose({"trajectory": [{"lng": 0, "lat": 0}] * 20001}) is None


def test_parse_vision_camera_pose_drops_out_of_range_points():
    payload = {
        "trajectory": [
            {"lng": 114.36, "lat": 30.53},
            {"lng": 114.36, "lat": 91.0},  # out of range, dropped
        ],
    }
    assert _parse_vision_camera_pose(payload) == {
        "trajectory": [{"lng": 114.36, "lat": 30.53,
                        "rel_alt_m": None, "abs_alt_m": None,
                        "yaw_deg": None, "gb_yaw_deg": None, "gb_pitch_deg": None,
                        "gb_roll_deg": None,
                        "focal_len_35mm": None, "seconds": None}],
    }


def test_parse_vision_camera_pose_nullifies_invalid_field_values():
    payload = {
        "trajectory": [
            {"lng": 114.36, "lat": 30.53, "rel_alt_m": "high", "seconds": "now",
             "focal_len_35mm": 0.0, "yaw_deg": 9999},
        ],
    }
    result = _parse_vision_camera_pose(payload)
    point = result["trajectory"][0]
    assert point["rel_alt_m"] is None
    assert point["seconds"] is None
    assert point["focal_len_35mm"] is None
    assert point["yaw_deg"] is None
    assert point["lng"] == 114.36 and point["lat"] == 30.53


def test_parse_vision_camera_pose_drops_all_invalid_points_to_none():
    assert _parse_vision_camera_pose({"trajectory": [{"lat": 30.53}]}) is None  # missing lng
    assert _parse_vision_camera_pose({"trajectory": ["not-a-point"]}) is None


def test_parse_vision_camera_pose_preserves_crs():
    payload = dict(_legal_pose(), crs="CGCS2000")
    assert _parse_vision_camera_pose(payload)["crs"] == "CGCS2000"


def test_parse_vision_camera_pose_keeps_only_whitelisted_point_keys():
    payload = {"trajectory": [{"lng": 114.36, "lat": 30.53, "extra": "dropped"}]}
    point = _parse_vision_camera_pose(payload)["trajectory"][0]
    assert set(point) == {"lng", "lat", "rel_alt_m", "abs_alt_m",
                          "yaw_deg", "gb_yaw_deg", "gb_pitch_deg", "gb_roll_deg",
                          "focal_len_35mm", "seconds"}


# ---- SQL 层单测（monkeypatch，无需 DB） ----

def _record_execution(monkeypatch, fetchone_value=None, fetchall_value=None):
    calls = {}

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, params):
            calls["statements"] = calls.get("statements", []) + [statement]
            calls["params"] = params
            return self

        def fetchone(self):
            return fetchone_value

        def fetchall(self):
            return fetchall_value or []

    monkeypatch.setattr(database, "connect", lambda url: FakeConnection())
    return calls


def test_create_job_insert_includes_camera_pose(monkeypatch):
    calls = _record_execution(
        monkeypatch,
        fetchone_value={"job_id": "abc", "status": "queued", "created_at": None,
                        "capture_time_source": "unknown", "capture_timezone": None},
    )
    pose = {"trajectory": [{"lng": 114.36, "lat": 30.53}]}
    vision_repository.create_job(
        None, created_by="web", original_name="x.mp4", media_kind="video",
        media_path="x.mp4", sha256="a" * 64,
        anchor_gcj={"lng": 114.36, "lat": 30.53}, camera_pose=pose,
    )
    statement = calls["statements"][0]
    assert "camera_pose" in statement
    assert "%s" in statement
    returning = statement.split("RETURNING", 1)[1]
    assert "camera_pose" in returning
    jsonb_values = [p.obj for p in calls["params"] if isinstance(p, Jsonb)]
    assert pose in jsonb_values


def test_get_job_select_includes_camera_pose(monkeypatch):
    calls = _record_execution(monkeypatch, fetchone_value={"job_id": "abc"})
    vision_repository.get_job(None, "abc")
    statement = calls["statements"][0]
    assert "camera_pose" in statement
    assert "observation_scene_id, camera_pose" in statement


def test_list_jobs_select_includes_camera_pose(monkeypatch):
    calls = _record_execution(monkeypatch, fetchone_value={"job_id": "abc"})
    vision_repository.list_jobs(None)
    statement = calls["statements"][0]
    assert "camera_pose" in statement
    assert "observation_scene_id, camera_pose" in statement


def test_claim_next_job_returning_includes_camera_pose(monkeypatch):
    calls = _record_execution(monkeypatch)
    assert vision_repository.claim_next_job(None, "worker-1") is None
    assert any("j.camera_pose" in statement for statement in calls["statements"])


# ---- 集成测试（需隔离库；未配置 TEST_DATABASE_URL 时由 fixture skip） ----

@pytest.mark.usefixtures("isolated_test_database")
def test_camera_pose_persisted_and_read_back():
    database.initialize(DATABASE_URL)
    pose = _legal_pose()
    item = vision_repository.create_job(
        DATABASE_URL, created_by="web", original_name="pose.mp4", media_kind="video",
        media_path="pose.mp4", sha256="b" * 64,
        anchor_gcj={"lng": 114.36, "lat": 30.53, "crs": "GCJ02"},
        captured_at=datetime.now(timezone.utc), camera_pose=pose,
    )
    job = vision_repository.get_job(DATABASE_URL, item["job_id"])
    assert job["camera_pose"] == pose


@pytest.mark.usefixtures("isolated_test_database")
def test_camera_pose_absent_for_legacy_jobs():
    database.initialize(DATABASE_URL)
    item = vision_repository.create_job(
        DATABASE_URL, created_by="web", original_name="legacy.mp4", media_kind="video",
        media_path="legacy.mp4", sha256="c" * 64,
        anchor_gcj={"lng": 114.36, "lat": 30.53, "crs": "GCJ02"},
        captured_at=datetime.now(timezone.utc),
    )
    job = vision_repository.get_job(DATABASE_URL, item["job_id"])
    assert job["camera_pose"] is None


@pytest.mark.usefixtures("isolated_test_database")
def test_camera_pose_null_fields_preserved():
    database.initialize(DATABASE_URL)
    pose = {"trajectory": [{"lng": 114.36, "lat": 30.53, "rel_alt_m": None,
                            "abs_alt_m": None, "yaw_deg": None, "gb_yaw_deg": None,
                            "gb_pitch_deg": None, "focal_len_35mm": None, "seconds": None}]}
    item = vision_repository.create_job(
        DATABASE_URL, created_by="web", original_name="nullfields.mp4", media_kind="video",
        media_path="nullfields.mp4", sha256="d" * 64,
        anchor_gcj={"lng": 114.36, "lat": 30.53, "crs": "GCJ02"},
        captured_at=datetime.now(timezone.utc), camera_pose=pose,
    )
    job = vision_repository.get_job(DATABASE_URL, item["job_id"])
    assert job["camera_pose"] == pose