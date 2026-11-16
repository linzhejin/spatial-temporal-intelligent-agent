"""照片 EXIF/XMP 姿态读取与相机内参测试。"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from vision.camera_profile import intrinsics_for
from vision.pose_metadata import merge_camera_pose, parse_image_pose, with_height_agl

_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def _jpeg_with_xmp(tmp_path: Path, attributes: str) -> Path:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (120, 120, 120)).save(buffer, format="JPEG")
    data = buffer.getvalue()
    xmp = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
        f'{attributes}/></rdf:RDF></x:xmpmeta>'
    ).encode("utf-8")
    length = len(xmp) + len(_XMP_HEADER) + 2
    segment = b"\xff\xe1" + length.to_bytes(2, "big") + _XMP_HEADER + xmp
    path = tmp_path / "sample.jpg"
    path.write_bytes(data[:2] + segment + data[2:])
    return path


def test_full_xmp_pose(tmp_path):
    path = _jpeg_with_xmp(
        tmp_path,
        'drone-dji:GpsLatitude="30.5365" drone-dji:GpsLongitude="114.363" '
        'drone-dji:RelativeAltitude="+96.4" drone-dji:GimbalYawDegree="-12.5" '
        'drone-dji:GimbalPitchDegree="-42.0" drone-dji:GimbalRollDegree="1.5" '
        'drone-dji:FocalLengthIn35mmFormat="24"',
    )
    pose = parse_image_pose(path)
    assert pose is not None
    assert pose["source"] == "image_xmp"
    assert pose["pose_quality"] == "full"
    point = pose["trajectory"][0]
    assert point["lng"] == 114.363
    assert point["lat"] == 30.5365
    assert point["rel_alt_m"] == 96.4
    assert point["gb_yaw_deg"] == -12.5
    assert point["gb_pitch_deg"] == -42.0
    assert point["gb_roll_deg"] == 1.5
    assert point["focal_len_35mm"] == 24.0


def test_gps_only_defaults_to_nadir(tmp_path):
    path = _jpeg_with_xmp(
        tmp_path,
        'drone-dji:GpsLatitude="30.5" drone-dji:GpsLongitude="114.3"',
    )
    pose = parse_image_pose(path)
    assert pose is not None
    assert pose["pose_quality"] == "no_yaw"
    assert pose["yaw_unknown"] is True
    point = pose["trajectory"][0]
    assert point["gb_pitch_deg"] == -90.0
    assert point["gb_roll_deg"] == 0.0
    assert point["gb_yaw_deg"] is None


def test_no_gps_returns_none(tmp_path):
    path = _jpeg_with_xmp(tmp_path, 'drone-dji:RelativeAltitude="50"')
    assert parse_image_pose(path) is None


def test_exif_gps_only(tmp_path):
    path = tmp_path / "exif.jpg"
    exif = Image.Exif()
    exif[34853] = {
        1: "N", 2: (30.0, 32.0, 11.0),
        3: "E", 4: (114.0, 21.0, 47.0),
        5: 0, 6: 120.0,
    }
    Image.new("RGB", (16, 16)).save(path, exif=exif)
    pose = parse_image_pose(path)
    assert pose is not None
    assert pose["source"] == "image_exif"
    assert pose["pose_quality"] == "no_yaw"
    point = pose["trajectory"][0]
    assert round(point["lat"], 5) == round(30 + 32 / 60 + 11 / 3600, 5)
    assert round(point["lng"], 5) == round(114 + 21 / 60 + 47 / 3600, 5)
    assert point["abs_alt_m"] == 120.0
    assert point["gb_pitch_deg"] == -90.0


def test_srt_pose_wins_over_image_pose():
    client = {"trajectory": [{"lng": 114.3, "lat": 30.5, "seconds": 1.0}]}
    image = {"trajectory": [{"lng": 114.4, "lat": 30.6, "seconds": 0.0}]}
    assert merge_camera_pose(client, image) is client


def test_height_agl_prefers_absolute_when_ground_known():
    pose = {"trajectory": [
        {"lng": 114.3, "lat": 30.5, "rel_alt_m": 100.0, "abs_alt_m": 150.0},
    ]}
    result = with_height_agl(pose, ground_offset_m=5.0, ground_elevation_m=20.0)
    point = result["trajectory"][0]
    assert point["height_agl_m"] == 130.0
    assert result["height_source"] == "absolute_minus_ground"


def test_height_agl_falls_back_to_relative():
    pose = {"trajectory": [{"lng": 114.3, "lat": 30.5, "rel_alt_m": 100.0}]}
    result = with_height_agl(pose, ground_offset_m=5.0)
    assert result["trajectory"][0]["height_agl_m"] == 95.0
    assert result["height_source"] == "relative_to_takeoff"


def test_camera_profile_intrinsics_for_4k():
    intrinsics = intrinsics_for(3840, 2160, {"focal_35mm": 24.0})
    assert intrinsics["fx_px"] == 2560.0
    assert intrinsics["fy_px"] == 2560.0
    assert intrinsics["cx_px"] == 1920.0
    assert intrinsics["cy_px"] == 1080.0
