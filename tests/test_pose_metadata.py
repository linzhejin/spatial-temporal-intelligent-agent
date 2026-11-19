"""照片 EXIF/XMP 姿态读取与相机内参测试。"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from PIL.TiffImagePlugin import IFDRational
import networkx as nx

from spatial.vision_suggest import score_candidate_road_association
from vision import registration
from vision.camera_profile import intrinsics_for
from vision.pose_metadata import _ratio, merge_camera_pose, parse_image_pose, with_height_agl

_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def _inject_xmp(path: Path, attributes: str) -> None:
    data = path.read_bytes()
    xmp = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
        f'{attributes}/></rdf:RDF></x:xmpmeta>'
    ).encode("utf-8")
    length = len(xmp) + len(_XMP_HEADER) + 2
    segment = b"\xff\xe1" + length.to_bytes(2, "big") + _XMP_HEADER + xmp
    path.write_bytes(data[:2] + segment + data[2:])


def _jpeg_with_xmp(tmp_path: Path, attributes: str) -> Path:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (120, 120, 120)).save(buffer, format="JPEG")
    path = tmp_path / "sample.jpg"
    path.write_bytes(buffer.getvalue())
    _inject_xmp(path, attributes)
    return path


def _candidate_with_polygon():
    return {
        "kind": "possible_flooding",
        "evidence": {"summary": {"outline_polygons": [[
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
        ]] }},
    }


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


def test_gps_only_keeps_camera_orientation_unknown(tmp_path):
    path = _jpeg_with_xmp(
        tmp_path,
        'drone-dji:GpsLatitude="30.5" drone-dji:GpsLongitude="114.3"',
    )
    pose = parse_image_pose(path)
    assert pose is not None
    assert pose["pose_quality"] == "no_yaw"
    assert pose["yaw_unknown"] is True
    point = pose["trajectory"][0]
    assert point["gb_pitch_deg"] is None
    assert point["gb_roll_deg"] is None
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
    assert point["gb_pitch_deg"] is None
    assert point["gb_roll_deg"] is None


def test_exif_true_image_direction_sets_projection_yaw_and_provenance(tmp_path):
    path = tmp_path / "exif_direction.jpg"
    exif = Image.Exif()
    exif[34853] = {
        1: "N", 2: (30.0, 32.0, 11.0),
        3: "E", 4: (114.0, 21.0, 47.0),
        5: 0, 6: 120.0,
        16: "T", 17: IFDRational(12345, 100),
    }
    Image.new("RGB", (16, 16)).save(path, exif=exif)
    _inject_xmp(
        path,
        'drone-dji:GimbalPitchDegree="-90" drone-dji:GimbalRollDegree="0"',
    )

    pose = parse_image_pose(path)

    assert pose is not None
    assert pose["trajectory"][0]["gb_yaw_deg"] == 123.45
    assert pose["heading_source"] == "exif_gps_img_direction_true"
    assert pose["heading_quality"] == "exif_true_north"
    assert pose["pose_quality"] == "full"
    assert pose["yaw_unknown"] is False

    camera_pose = with_height_agl(merge_camera_pose(None, pose), ground_elevation_m=0.0)
    association = score_candidate_road_association(
        _candidate_with_polygon(), nx.MultiDiGraph(), camera_pose=camera_pose,
        media={"kind": "image", "width": 16, "height": 16},
        registration=registration, camera_profile={"focal_35mm": 24.0},
    )
    assert association["status"] == "no_match"
    assert association["reason_code"] == "no_roads_near_candidate"


def test_exif_magnetic_direction_is_not_promoted_or_used_to_select_edge(tmp_path):
    path = tmp_path / "magnetic_direction.jpg"
    exif = Image.Exif()
    exif[34853] = {
        1: "N", 2: (30.0, 32.0, 11.0),
        3: "E", 4: (114.0, 21.0, 47.0),
        5: 0, 6: 120.0,
        16: "M", 17: IFDRational(90, 1),
    }
    Image.new("RGB", (16, 16)).save(path, exif=exif)
    _inject_xmp(
        path,
        'drone-dji:GimbalPitchDegree="-90" drone-dji:GimbalRollDegree="0"',
    )

    pose = parse_image_pose(path)

    assert pose is not None
    assert pose["trajectory"][0]["gb_yaw_deg"] is None
    assert pose["heading_source"] == "exif_gps_img_direction_magnetic"
    assert pose["heading_quality"] == "magnetic_unconverted"
    assert pose["pose_quality"] == "no_yaw"
    assert pose["yaw_unknown"] is True

    camera_pose = with_height_agl(pose, ground_elevation_m=0.0)
    association = score_candidate_road_association(
        _candidate_with_polygon(), nx.MultiDiGraph(), camera_pose=camera_pose,
        media={"kind": "image", "width": 16, "height": 16},
        registration=registration, camera_profile={"focal_35mm": 24.0},
    )
    assert association["status"] == "insufficient_pose"
    assert association["reason_code"] == "gb_yaw_deg_missing_or_invalid"
    assert association["selected_edge"] is None


def test_exif_direction_without_reference_is_not_promoted_to_yaw(tmp_path):
    path = tmp_path / "unreferenced_direction.jpg"
    exif = Image.Exif()
    exif[34853] = {
        1: "N", 2: (30.0, 32.0, 11.0),
        3: "E", 4: (114.0, 21.0, 47.0),
        17: IFDRational(270, 1),
    }
    Image.new("RGB", (16, 16)).save(path, exif=exif)

    pose = parse_image_pose(path)

    assert pose is not None
    assert pose["trajectory"][0]["gb_yaw_deg"] is None
    assert pose["heading_quality"] == "reference_unknown"
    assert pose["yaw_unknown"] is True


def test_ratio_accepts_exif_rational_and_denominator_forms():
    assert _ratio(IFDRational(12345, 100)) == 123.45
    assert _ratio((12345, 100)) == 123.45
    assert _ratio("12345/100") == 123.45
    assert _ratio("90/0") is None


def test_xmp_gimbal_yaw_has_priority_over_exif_image_direction(tmp_path):
    path = tmp_path / "xmp_and_exif_direction.jpg"
    exif = Image.Exif()
    exif[34853] = {
        1: "N", 2: (30.0, 32.0, 11.0),
        3: "E", 4: (114.0, 21.0, 47.0),
        16: "T", 17: IFDRational(180, 1),
    }
    Image.new("RGB", (16, 16)).save(path, exif=exif)
    _inject_xmp(path, 'drone-dji:GimbalYawDegree="-12.5"')

    pose = parse_image_pose(path)

    assert pose is not None
    assert pose["trajectory"][0]["gb_yaw_deg"] == -12.5
    assert pose["heading_source"] == "image_xmp_gimbal_yaw"
    assert pose["heading_quality"] == "xmp_gimbal"


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
