"""标定输入预检与验收门禁的纯逻辑测试（不联网、不读生产配置）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "vision"))

import accept_calibration as accept  # noqa: E402
import prepare_calibration_session as prep  # noqa: E402


def _write_image(path: Path, size=(1000, 800)) -> Path:
    Image.new("RGB", size, (255, 255, 255)).save(path)
    return path


def _pose(**overrides) -> dict:
    point = {
        "lng": 114.3615,
        "lat": 30.5379,
        "rel_alt_m": 80.0,
        "gb_yaw_deg": 35.0,
        "gb_pitch_deg": -90.0,
        "gb_roll_deg": 0.0,
        "seconds": 0.0,
    }
    point.update(overrides)
    return {
        "source": "test_pose",
        "pose_quality": "full",
        "trajectory": [point],
    }


def _points() -> list[dict]:
    return [
        {"image_xy": [100, 100], "map_lng_gcj": 114.3610, "map_lat_gcj": 30.5383},
        {"image_xy": [900, 100], "map_lng_gcj": 114.3620, "map_lat_gcj": 30.5383},
        {"image_xy": [900, 700], "map_lng_gcj": 114.3620, "map_lat_gcj": 30.5375},
        {"image_xy": [100, 700], "map_lng_gcj": 114.3610, "map_lat_gcj": 30.5375},
    ]


def test_prepare_session_ready_for_spread_points(tmp_path):
    image = _write_image(tmp_path / "calib.jpg")
    session = prep.build_session(image, _points(), _pose())
    assert session["ready"] is True
    assert session["blockers"] == []
    assert session["point_count"] == 4
    assert len(session["image_sha256"]) == 64
    assert session["image_width"] == 1000
    assert session["pose"]["source"] == "test_pose"


def test_prepare_session_blocks_too_few_points(tmp_path):
    image = _write_image(tmp_path / "calib.jpg")
    session = prep.build_session(image, _points()[:3], _pose())
    assert session["ready"] is False
    assert any("控制点不足" in item for item in session["blockers"])


def test_prepare_session_blocks_out_of_bounds_and_missing_yaw(tmp_path):
    image = _write_image(tmp_path / "calib.jpg")
    points = _points()
    points[0]["image_xy"] = [1200, 100]
    pose = _pose(gb_yaw_deg=None)
    session = prep.build_session(image, points, pose)
    assert session["ready"] is False
    assert any("像素坐标越界" in item for item in session["blockers"])
    assert any("偏航角" in item for item in session["blockers"])


def test_prepare_session_blocks_unsupported_image_format(tmp_path):
    image = tmp_path / "calib.bmp"
    Image.new("RGB", (1000, 800), (255, 255, 255)).save(image)
    session = prep.build_session(image, _points(), _pose())
    assert session["ready"] is False
    assert any("不支持的影像格式" in item for item in session["blockers"])


def test_prepare_session_blocks_collinear_points(tmp_path):
    image = _write_image(tmp_path / "calib.jpg")
    points = [
        {"image_xy": [100, 400], "map_lng_gcj": 114.3610, "map_lat_gcj": 30.5379},
        {"image_xy": [400, 400], "map_lng_gcj": 114.3613, "map_lat_gcj": 30.5379},
        {"image_xy": [700, 400], "map_lng_gcj": 114.3616, "map_lat_gcj": 30.5379},
        {"image_xy": [900, 400], "map_lng_gcj": 114.3618, "map_lat_gcj": 30.5379},
    ]
    session = prep.build_session(image, points, _pose())
    assert session["ready"] is False
    assert any("共线" in item for item in session["blockers"])


def _case(name: str, **overrides) -> dict:
    row = {
        "name": name,
        "basis": "pose_footprint",
        "anchor_source": "pose",
        "center_error_m": 5.0,
        "top1_hit": True,
        "top3_hit": True,
        "suggested": [[1, 2, 0]],
    }
    row.update(overrides)
    return row


def _evaluation(rows: list[dict] | None = None, **summary_overrides) -> dict:
    cases = rows if rows is not None else [_case(f"case-{index}") for index in range(6)]
    summary = {
        "case_count": len(cases),
        "center_error_cases": len([row for row in cases if row.get("center_error_m") is not None]),
        "mean_center_error_m": 5.0,
        "max_center_error_m": 9.0,
        "top1_rate": 0.9,
        "top3_rate": 1.0,
    }
    summary.update(summary_overrides)
    return {"cases": cases, "summary": summary}


def _profile(**overrides) -> dict:
    report = {"calibrated": True, "rms_m": 4.2, "max_m": 7.8, "point_count": 6}
    report.update(overrides)
    return report


def test_acceptance_passes_for_calibrated_footprint_evidence():
    verdict = accept.apply_gates(_profile(), _evaluation())
    assert verdict["passed"] is True
    assert verdict["blockers"] == []


def test_acceptance_blocks_uncalibrated_profile():
    verdict = accept.apply_gates(_profile(calibrated=False), _evaluation())
    assert verdict["passed"] is False
    assert any("calibrated" in item for item in verdict["blockers"])


def test_acceptance_allows_nominal_preview_when_requested():
    verdict = accept.apply_gates(_profile(calibrated=False), _evaluation(), require_calibrated=False)
    assert verdict["passed"] is True


def test_acceptance_blocks_radius_fallback_and_low_top1():
    rows = [_case(f"case-{index}", basis="radius_only") for index in range(6)]
    rows[0]["top1_hit"] = False
    verdict = accept.apply_gates(_profile(), _evaluation(rows, top1_rate=0.5))
    assert verdict["passed"] is False
    assert any("pose_footprint" in item for item in verdict["blockers"])
    assert any("Top-1" in item for item in verdict["blockers"])


def test_acceptance_blocks_case_errors_and_missing_center_errors():
    rows = [_case(f"case-{index}") for index in range(6)]
    rows[2] = {"name": "broken", "image": "broken.jpg", "error": "没有可用姿态"}
    rows[3]["center_error_m"] = None
    verdict = accept.apply_gates(_profile(), _evaluation(rows, center_error_cases=5))
    assert verdict["passed"] is False
    assert any("执行失败" in item for item in verdict["blockers"])
    assert any("中心误差" in item for item in verdict["blockers"])


def test_templates_parse_and_keep_expected_sections():
    templates = ROOT / "scripts" / "vision" / "templates"
    points = json.loads((templates / "calibration_points.example.json").read_text(encoding="utf-8"))
    manifest = json.loads((templates / "acceptance_manifest.example.json").read_text(encoding="utf-8"))
    assert len(points["points"]) >= 4
    assert len(manifest["cases"]) >= 6
    assert all("image_xy" in item for item in points["points"])


def test_calibrate_accepts_session_pose(tmp_path):
    import calibrate_camera_profile as calibrate

    image = _write_image(tmp_path / "calib.jpg")
    pose = _pose()
    loaded = calibrate._load_pose(image, None, pose)
    assert loaded["trajectory"][0]["gb_yaw_deg"] == pytest.approx(35.0)
