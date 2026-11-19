"""从照片读取定位/姿态信息，作为 SRT 之外的另一条来源。

DJI 照片通常在 EXIF 里写 GPS，在 XMP 里写相对高度、云台角和等效焦距。
这里只做读取与归一化，不判断精度；是否可用于自动定位由标定状态决定。

输出结构与前端 SRT 解析一致：
{"trajectory": [{"lng", "lat", "rel_alt_m", "abs_alt_m",
                 "gb_yaw_deg", "gb_pitch_deg", "gb_roll_deg",
                 "focal_len_35mm", "seconds"}], "source", "pose_quality",
 "heading_source", "heading_quality"}

EXIF GPSImgDirection 仅在参考方向为真北时进入投影 yaw；磁北或缺少参考方向时
保留来源信息，但不把方向值作为真北角度使用。
缺少云台角度时保留为 None，不将 GPS 定位等同于正下视相机姿态。
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image

_XMP_PATTERNS = {
    "GpsLatitude": ("gps_latitude", ("GpsLatitude",)),
    "GpsLongitude": ("gps_longitude", ("GpsLongitude",)),
    "RelativeAltitude": ("relative_altitude", ("RelativeAltitude",)),
    "AbsoluteAltitude": ("absolute_altitude", ("AbsoluteAltitude",)),
    "GimbalYawDegree": ("gimbal_yaw", ("GimbalYawDegree",)),
    "GimbalPitchDegree": ("gimbal_pitch", ("GimbalPitchDegree",)),
    "GimbalRollDegree": ("gimbal_roll", ("GimbalRollDegree",)),
    "FocalLengthIn35mmFormat": ("focal_35mm", ("FocalLengthIn35mmFormat", "FocalLength")),
}


def _finite(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _ratio(value) -> float | None:
    if isinstance(value, str) and "/" in value:
        parts = value.strip().split("/")
        if len(parts) != 2:
            return None
        numerator, denominator = map(_finite, parts)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        numerator, denominator = map(_finite, value)
    elif hasattr(value, "numerator") and hasattr(value, "denominator"):
        numerator = _finite(value.numerator)
        denominator = _finite(value.denominator)
    else:
        return _finite(value)
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _degrees(value, ref) -> float | None:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    parts = [_ratio(item) for item in value[:3]]
    if not parts or parts[0] is None:
        return None
    degrees = parts[0]
    if len(parts) > 1 and parts[1] is not None:
        degrees += parts[1] / 60.0
    if len(parts) > 2 and parts[2] is not None:
        degrees += parts[2] / 3600.0
    if isinstance(ref, str) and ref.strip().upper() in {"S", "W"}:
        degrees = -degrees
    return degrees


def _read_exif_gps(path: Path) -> dict:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            gps = exif.get_ifd(34853) if exif else {}
    except Exception:
        return {}
    if not isinstance(gps, dict) or not gps:
        return {}
    lat = _degrees(gps.get(2), gps.get(1))
    lon = _degrees(gps.get(4), gps.get(3))
    altitude = _ratio(gps.get(6))
    altitude_ref = gps.get(5)
    if lat is None or lon is None:
        return {}
    if altitude is not None and altitude_ref == 1:
        altitude = -altitude
    result = {"lng": lon, "lat": lat}
    if altitude is not None:
        result["abs_alt_m"] = altitude
    direction = _ratio(gps.get(17))
    direction_ref_value = gps.get(16)
    if isinstance(direction_ref_value, bytes):
        direction_ref_value = direction_ref_value.decode("ascii", errors="ignore")
    direction_ref = (direction_ref_value.strip().upper()
                     if isinstance(direction_ref_value, str) else None)
    if gps.get(17) is not None:
        result["gps_img_direction_deg"] = direction
        result["gps_img_direction_ref"] = direction_ref
        if direction is None or not 0 <= direction <= 360:
            result["gps_img_direction_quality"] = "invalid_value"
        else:
            direction = 0.0 if direction == 360 else direction
            result["gps_img_direction_deg"] = direction
            if direction_ref == "T":
                result["true_heading_deg"] = direction
                result["gps_img_direction_quality"] = "exif_true_north"
            elif direction_ref == "M":
                result["gps_img_direction_quality"] = "magnetic_unconverted"
            else:
                result["gps_img_direction_quality"] = "reference_unknown"
    return result


def _read_xmp(path: Path) -> dict:
    try:
        raw = path.read_bytes()
    except OSError:
        return {}
    start = raw.find(b"<x:xmpmeta")
    end = raw.find(b"</x:xmpmeta>")
    if start < 0 or end <= start:
        return {}
    text = raw[start:end + 12].decode("utf-8", errors="ignore")
    values = {}
    for field, (key, names) in _XMP_PATTERNS.items():
        match = None
        for name in names:
            match = re.search(
                rf'(?:drone-dji:)?{name}\s*=\s*["\']([^"\']+)["\']', text)
            if match is None:
                match = re.search(
                    rf"<(?:drone-dji:)?{name}>([^<]+)</", text)
            if match is not None:
                break
        if match is not None:
            number = _finite(match.group(1).strip())
            if number is not None:
                values[key] = number
    return values


def parse_image_pose(path: str | Path) -> dict | None:
    """读取照片的 GPS/云台信息；无 GPS 返回 None。"""
    image_path = Path(path)
    if not image_path.is_file():
        return None
    xmp = _read_xmp(image_path)
    exif = _read_exif_gps(image_path)
    lng = xmp.get("gps_longitude", exif.get("lng"))
    lat = xmp.get("gps_latitude", exif.get("lat"))
    if lng is None or lat is None:
        return None
    if not (-180 <= lng <= 180 and -90 <= lat <= 90):
        return None

    rel_alt = xmp.get("relative_altitude")
    abs_alt = xmp.get("absolute_altitude", exif.get("abs_alt_m"))
    xmp_yaw = _finite(xmp.get("gimbal_yaw"))
    if xmp_yaw is not None and not -360 <= xmp_yaw <= 360:
        xmp_yaw = None
    exif_heading = exif.get("true_heading_deg")
    exif_direction = exif.get("gps_img_direction_deg")
    exif_direction_ref = exif.get("gps_img_direction_ref")
    if xmp_yaw is not None:
        yaw = xmp_yaw
        heading_source = "image_xmp_gimbal_yaw"
        heading_quality = "xmp_gimbal"
    elif exif_heading is not None:
        yaw = exif_heading
        heading_source = "exif_gps_img_direction_true"
        heading_quality = "exif_true_north"
    elif exif.get("gps_img_direction_quality") == "magnetic_unconverted":
        yaw = None
        heading_source = "exif_gps_img_direction_magnetic"
        heading_quality = "magnetic_unconverted"
    elif exif.get("gps_img_direction_quality") == "reference_unknown":
        yaw = None
        heading_source = "exif_gps_img_direction"
        heading_quality = "reference_unknown"
    elif exif.get("gps_img_direction_quality") == "invalid_value":
        yaw = None
        heading_source = "exif_gps_img_direction"
        heading_quality = "invalid_value"
    else:
        yaw = None
        heading_source = None
        heading_quality = "missing"
    pitch = xmp.get("gimbal_pitch")
    roll = xmp.get("gimbal_roll")
    focal_35mm = xmp.get("focal_35mm")
    angles_complete = pitch is not None and roll is not None and yaw is not None

    point = {
        "lng": float(lng),
        "lat": float(lat),
        "rel_alt_m": rel_alt,
        "abs_alt_m": abs_alt,
        "gb_yaw_deg": yaw,
        "gb_pitch_deg": pitch,
        "gb_roll_deg": roll,
        "focal_len_35mm": focal_35mm,
        "seconds": 0.0,
    }
    if xmp and exif:
        source = "image_xmp+exif"
    elif xmp:
        source = "image_xmp"
    else:
        source = "image_exif"
    quality = "full" if angles_complete else "heading_only" if yaw is not None else "no_yaw"
    return {
        "trajectory": [point],
        "source": source,
        "pose_quality": quality,
        "yaw_unknown": yaw is None,
        "heading_source": heading_source,
        "heading_quality": heading_quality,
        "heading_observation_deg": exif_direction,
        "heading_reference": exif_direction_ref,
    }


def merge_camera_pose(client_pose: dict | None, image_pose: dict | None) -> dict | None:
    """客户端 SRT 优先；没有 SRT 时使用照片自带信息。"""
    if isinstance(client_pose, dict):
        trajectory = client_pose.get("trajectory")
        if isinstance(trajectory, list) and trajectory:
            return client_pose
    return image_pose


def with_height_agl(camera_pose: dict | None, *, ground_offset_m: float = 0.0,
                    ground_elevation_m: float | None = None) -> dict | None:
    """给每个轨迹点补 height_agl_m；不修改入参。"""
    if not isinstance(camera_pose, dict):
        return camera_pose
    trajectory = camera_pose.get("trajectory")
    if not isinstance(trajectory, list):
        return camera_pose
    points = []
    height_source = None
    for point in trajectory:
        if not isinstance(point, dict):
            continue
        copied = dict(point)
        height = None
        if (ground_elevation_m is not None and copied.get("abs_alt_m") is not None):
            try:
                height = float(copied["abs_alt_m"]) - float(ground_elevation_m)
                height_source = "absolute_minus_ground"
            except (TypeError, ValueError):
                height = None
        if height is None and copied.get("rel_alt_m") is not None:
            try:
                height = float(copied["rel_alt_m"]) - float(ground_offset_m)
                height_source = "relative_to_takeoff"
            except (TypeError, ValueError):
                height = None
        if height is None and copied.get("height_agl_m") is not None:
            try:
                height = float(copied["height_agl_m"])
                height_source = "provided"
            except (TypeError, ValueError):
                height = None
        copied["height_agl_m"] = height
        points.append(copied)
    result = dict(camera_pose)
    result["trajectory"] = points
    result["height_source"] = height_source
    return result
