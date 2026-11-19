"""用真实影像与地面控制点标定相机内参比例和离地高度偏移。

输入：
  --image          真实影像路径
  --pose           可选，包含 trajectory 的 JSON；缺省读照片 EXIF/XMP
  --points         JSON；支持控制点列表，或 prepare_calibration_session.py 产出的会话对象
  --output         标定结果 JSON（拟合残差与阈值）
  --profile-output 可选，写出可直接用于 VISION_CAMERA_PROFILE_PATH 的相机配置 JSON

该方法只做小范围网格搜索，不引入 scipy；输出 calibrated 仅在残差达标时为 true。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spatial.camera_projection import camera_ground_homography, project_pixel  # noqa: E402
from spatial.coord_transform import wgs84_to_gcj02  # noqa: E402
from vision.camera_profile import intrinsics_for, resolve_profile  # noqa: E402
from vision.pose_metadata import parse_image_pose  # noqa: E402

_M_PER_DEG = 111320.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pose(path: Path, pose_json: Path | None, session_pose: dict | None = None) -> dict:
    if pose_json is not None:
        data = json.loads(pose_json.read_text(encoding="utf-8"))
    elif isinstance(session_pose, dict) and session_pose.get("trajectory"):
        data = session_pose
    else:
        data = parse_image_pose(path)
    if not isinstance(data, dict) or not isinstance(data.get("trajectory"), list):
        raise ValueError("没有可用的姿态轨迹；请提供 --pose 或含 GPS 的照片")
    if not data["trajectory"]:
        raise ValueError("姿态轨迹为空")
    return data


def _point_pose(base: dict, profile: dict, width: int, height: int,
                fx_scale: float, ground_offset_m: float) -> dict:
    pose = dict(base)
    base_height = base.get("rel_alt_m")
    if base_height is None:
        base_height = base.get("height_agl_m")
    if base_height is None:
        raise ValueError("姿态缺少 rel_alt_m/height_agl_m，无法计算离地高度")
    pose["height_agl_m"] = float(base_height) - float(ground_offset_m)
    nominal = intrinsics_for(width, height, profile)
    pose["intrinsics"] = {
        "fx_px": nominal["fx_px"] * fx_scale,
        "fy_px": nominal["fy_px"] * fx_scale,
        "cx_px": nominal["cx_px"],
        "cy_px": nominal["cy_px"],
    }
    return pose


def _project_to_gcj(pose: dict, width: int, height: int, image_xy) -> tuple[float, float]:
    transform = camera_ground_homography(pose, width, height)
    east, north = project_pixel(transform["homography"], float(image_xy[0]), float(image_xy[1]))
    lng = float(pose["lng"]) + east / (_M_PER_DEG * math.cos(math.radians(float(pose["lat"]))))
    lat = float(pose["lat"]) + north / _M_PER_DEG
    return wgs84_to_gcj02(lng, lat)


def _distance_m(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    cos_lat = math.cos(math.radians((lat1 + lat2) / 2.0))
    dx = (lng1 - lng2) * _M_PER_DEG * cos_lat
    dy = (lat1 - lat2) * _M_PER_DEG
    return math.hypot(dx, dy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--pose", type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-output", type=Path,
                        help="可选：写出可直接用于 VISION_CAMERA_PROFILE_PATH 的相机配置 JSON")
    parser.add_argument("--max-rms-m", type=float, default=10.0)
    parser.add_argument("--max-point-m", type=float, default=15.0)
    parser.add_argument("--min-points", type=int, default=3)
    args = parser.parse_args()

    raw_points = json.loads(args.points.read_text(encoding="utf-8"))
    session_pose = None
    if isinstance(raw_points, dict):
        session_pose = raw_points.get("pose")
        raw_points = raw_points.get("points")
    points = raw_points
    if not isinstance(points, list) or len(points) < args.min_points:
        raise ValueError(f"至少需要 {args.min_points} 个控制点")
    pose_data = _load_pose(args.image, args.pose, session_pose)
    base = pose_data["trajectory"][0]
    from PIL import Image
    with Image.open(args.image) as image:
        width, height = image.size
    profile = resolve_profile()

    best = None
    best_score = None
    for scale_index in range(-15, 16):
        fx_scale = 1.0 + scale_index * 0.02
        for offset_index in range(-20, 21):
            ground_offset = float(offset_index)
            try:
                pose = _point_pose(base, profile, width, height, fx_scale, ground_offset)
                residuals = []
                for point in points:
                    lng, lat = _project_to_gcj(pose, width, height, point["image_xy"])
                    residuals.append(_distance_m(
                        lng, lat, float(point["map_lng_gcj"]), float(point["map_lat_gcj"])))
            except (ValueError, KeyError, TypeError):
                continue
            rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
            # 焦距比例与离地高度在正下视时近似可互换；同等残差下优先接近标称值。
            score = rms + 0.01 * abs(fx_scale - 1.0) + 0.001 * abs(ground_offset)
            candidate = {
                "fx_scale": round(fx_scale, 3),
                "ground_offset_m": ground_offset,
                "rms_m": round(rms, 3),
                "max_m": round(max(residuals), 3),
                "residuals_m": [round(value, 3) for value in residuals],
            }
            if best_score is None or score < best_score:
                best_score = score
                best = candidate

    if best is None:
        raise ValueError("没有找到可用标定参数")
    calibrated = (best["rms_m"] <= args.max_rms_m
                  and best["max_m"] <= args.max_point_m
                  and len(points) >= args.min_points)
    report = {
        "image": str(args.image),
        "image_sha256": sha256(args.image),
        "image_width": width,
        "image_height": height,
        "point_count": len(points),
        "profile": profile.get("name"),
        "fx_scale": best["fx_scale"],
        "ground_offset_m": best["ground_offset_m"],
        "rms_m": best["rms_m"],
        "max_m": best["max_m"],
        "residuals_m": best["residuals_m"],
        "calibrated": calibrated,
        "thresholds": {"max_rms_m": args.max_rms_m, "max_point_m": args.max_point_m,
                       "min_points": args.min_points},
        "note": "只标定内参比例与离地高度偏移；正下视时二者近似可互换，同等残差优先接近标称值。"
                "建议加入不同深度或已知路面宽度的控制点；不改变姿态方向，也不证明绝对定位精度。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.profile_output is not None:
        camera_profile = {
            "name": f"{profile.get('name') or 'camera'}-calibrated",
            "sensor_width_mm": profile.get("sensor_width_mm"),
            "focal_35mm": round(float(profile.get("focal_35mm") or 24.0) * best["fx_scale"], 4),
            "image_width": profile.get("image_width"),
            "image_height": profile.get("image_height"),
            "calibrated": calibrated,
            "residual_m": best["rms_m"],
            "max_residual_m": best["max_m"],
            "ground_offset_m": best["ground_offset_m"],
            "source_report": str(args.output),
            "source_image_sha256": report["image_sha256"],
            "note": "由 calibrate_camera_profile.py 生成；calibrated=false 时只能用于预演。",
        }
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.profile_output.write_text(
            json.dumps(camera_profile, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(args.profile_output))
    print(json.dumps({key: report[key] for key in (
        "fx_scale", "ground_offset_m", "rms_m", "max_m", "calibrated")}, ensure_ascii=False))
    print(str(args.output))


if __name__ == "__main__":
    main()
