"""DJI SRT → 前端解析 → 后端清洗 → 投影消费 的整条线路功能验证。

本测试不复制前端解析逻辑：它用 node 运行真实的 `static/js/manager.js`，
把 `tests/fixtures/dji_flight.srt` 解析成管理端实际上传的 `camera_pose`，
再交给后端 `_parse_vision_camera_pose`，最后用同一份轨迹点在**显式相机
标定参数**下走投影入口，验证「姿态解析链路」到投影消费之间没有字段断点。

刻意不覆盖的部分：生产 `suggested-edges` 仍固定为 `radius_only`，未标定
SRT 不参与投影（见 tests/test_vision_suggest.py 的
`test_uncalibrated_srt_pose_cannot_generate_a_footprint` 和 browser E2E）。
本文件只证明解析链路完整、投影入口可用，不改变生产行为。
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from api.routes import _parse_vision_camera_pose

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "dji_flight.srt"

_IMAGE_W, _IMAGE_H = 4000, 3000
# 显式测试内参：24mm 等效焦段按 36mm 画幅换算，代表「标定已完成后」的输入，
# 不从 SRT 的 focal_len 字段反推（该字段单位语义仍待现场核对）。
_INTRINSICS = {
    "fx_px": 24.0 * _IMAGE_W / 36.0,
    "fy_px": 24.0 * _IMAGE_W / 36.0,
    "cx_px": _IMAGE_W / 2.0,
    "cy_px": _IMAGE_H / 2.0,
}

_NODE_SCRIPT = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/js/manager.js', 'utf8');
const end = source.indexOf('function transformLatitude(x, y) {');
if (end < 0) throw new Error('manager.js pose boundary missing');
const context = vm.createContext({
  window: { sessionStorage: { getItem() { return null; }, setItem() {}, removeItem() {} } },
});
vm.runInContext(
  source.slice(0, end) +
  '\nglobalThis.poseApi = { parseDjiSubtitle: parseDjiSubtitle,' +
  ' convertPoseForUpload: convertPoseForUpload };\n})();',
  context,
);
const metadata = context.poseApi.parseDjiSubtitle(fs.readFileSync(process.argv[1], 'utf8'));
process.stdout.write(JSON.stringify({
  captureTime: metadata.captureTime,
  gpsPointCount: metadata.gpsPointCount,
  cameraPose: context.poseApi.convertPoseForUpload(metadata.cameraPose),
}));
"""


@pytest.fixture(scope="module")
def frontend_upload_payload():
    """用真实前端代码把 SRT fixture 解析成上传载荷。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to run the real manager.js parser")
    completed = subprocess.run(
        [node, "-e", _NODE_SCRIPT, str(FIXTURE)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_frontend_parses_the_whole_srt_file(frontend_upload_payload):
    assert frontend_upload_payload["captureTime"] == "2026-11-06T10:12:31"
    assert frontend_upload_payload["gpsPointCount"] == 3
    pose = frontend_upload_payload["cameraPose"]
    assert pose["crs"] == "WGS84"
    assert pose["parsed"] is True
    assert pose["point_count"] == 3
    assert [point["seconds"] for point in pose["trajectory"]] == [0, 1, 2]


def test_backend_accepts_what_the_frontend_actually_uploads(frontend_upload_payload):
    # 管理端以 JSON 字符串提交 camera_pose，这里走真实提交形态。
    cleaned = _parse_vision_camera_pose(json.dumps(frontend_upload_payload["cameraPose"]))
    assert cleaned is not None
    assert cleaned["crs"] == "WGS84"
    assert cleaned["point_count"] == 3
    first = cleaned["trajectory"][0]
    assert first == {
        "lng": 114.356321,
        "lat": 30.535117,
        "rel_alt_m": pytest.approx(82.104),
        "abs_alt_m": pytest.approx(109.538),
        "yaw_deg": None,
        "gb_yaw_deg": pytest.approx(-92.2),
        "gb_pitch_deg": pytest.approx(-79.3),
        "gb_roll_deg": pytest.approx(0.0),
        "focal_len_35mm": pytest.approx(240.0),
        "seconds": pytest.approx(0.0),
    }


def test_cleaned_pose_can_be_persisted_as_json_unchanged(frontend_upload_payload):
    # vision_repository.create_job 以 Jsonb 写入并原样读回；这里验证载荷可无损往返。
    cleaned = _parse_vision_camera_pose(frontend_upload_payload["cameraPose"])
    assert json.loads(json.dumps(cleaned)) == cleaned


def test_cleaned_pose_reaches_the_projection_entry_once_calibrated(frontend_upload_payload):
    pytest.importorskip("numpy")
    import numpy as np
    from spatial.camera_projection import camera_ground_homography, project_pixel
    from vision.registration import pose_to_homography_init

    cleaned = _parse_vision_camera_pose(frontend_upload_payload["cameraPose"])
    point = cleaned["trajectory"][0]
    calibrated = {
        "lng": point["lng"],
        "lat": point["lat"],
        # rel_alt_m 是相对起飞点高度；平地假设下作为离地高度使用，真实地形高差待现场标定。
        "height_agl_m": point["rel_alt_m"],
        "gb_yaw_deg": point["gb_yaw_deg"],
        "gb_pitch_deg": point["gb_pitch_deg"],
        "gb_roll_deg": point["gb_roll_deg"],
        "intrinsics": _INTRINSICS,
    }

    transform = camera_ground_homography(calibrated, _IMAGE_W, _IMAGE_H)
    origin = transform["origin"]
    assert origin["lng_wgs"] == pytest.approx(point["lng"], abs=1e-9)
    assert origin["lat_wgs"] == pytest.approx(point["lat"], abs=1e-9)
    assert origin["lng_gcj"] != point["lng"]  # 校园范围在 GCJ-02 偏移区内

    # 画幅中心射线落到地面：水平距离 = 高度 × tan(偏离天底角)，方向由云台偏航决定。
    east, north = project_pixel(transform["homography"], _IMAGE_W / 2, _IMAGE_H / 2)
    distance = math.hypot(east, north)
    tilt = math.radians(90.0 + point["gb_pitch_deg"])
    assert distance == pytest.approx(point["rel_alt_m"] * math.tan(tilt), rel=0.01)
    yaw = math.radians(point["gb_yaw_deg"])
    assert east / distance == pytest.approx(math.sin(yaw), abs=1e-3)
    assert north / distance == pytest.approx(math.cos(yaw), abs=1e-3)

    numpy_transform = pose_to_homography_init(calibrated, _IMAGE_W, _IMAGE_H)
    assert numpy_transform["homography"].shape == (3, 3)
    np.testing.assert_allclose(
        numpy_transform["homography"],
        np.asarray(transform["homography"], dtype=np.float64),
        atol=1e-9,
    )


def test_raw_srt_pose_is_not_a_calibration_input(frontend_upload_payload):
    from spatial.camera_projection import camera_ground_homography

    cleaned = _parse_vision_camera_pose(frontend_upload_payload["cameraPose"])
    point = cleaned["trajectory"][0]
    with pytest.raises(ValueError):
        # 直接拿 SRT 原始点：没有离地高度字段，投影入口必须拒绝。
        camera_ground_homography(point, _IMAGE_W, _IMAGE_H)
    with pytest.raises(ValueError):
        # 有高度但缺标定内参，同样拒绝。
        camera_ground_homography(
            {**point, "height_agl_m": point["rel_alt_m"]}, _IMAGE_W, _IMAGE_H)
