"""预检单张标定影像与地面控制点，输出可追溯的标定会话 JSON。

用法：
  python scripts/vision/prepare_calibration_session.py \
    --image D:/calib/example.jpg \
    --points D:/calib/example.points.json \
    --output D:/calib/example.session.json

points 文件支持两种格式：
  1. 列表：[{"image_xy": [x, y], "map_lng_gcj": ..., "map_lat_gcj": ...}, ...]
  2. 对象：{"points": [...]}，可直接使用 templates/calibration_points.example.json

输出中的 ready=true 才建议进入 calibrate_camera_profile.py；blockers 为阻断项，
warnings 需要人工判断但不阻断。脚本只读影像与点位，不修改生产配置。
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

from vision.pose_metadata import parse_image_pose  # noqa: E402

SCHEMA = "whu-vision-calibration-session-v1"


def _number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_points(path: Path) -> list:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("points")
    return raw if isinstance(raw, list) else []


def _pose_point_blockers(point: dict) -> list[str]:
    blockers = []
    if _number(point.get("lng")) is None or _number(point.get("lat")) is None:
        blockers.append("姿态缺少有效的经纬度")
    if _number(point.get("rel_alt_m")) is None and _number(point.get("height_agl_m")) is None:
        blockers.append("姿态缺少相对高度 rel_alt_m / height_agl_m")
    yaw = _number(point.get("gb_yaw_deg"))
    if yaw is None:
        blockers.append("姿态缺少云台偏航角 gb_yaw_deg")
    pitch = _number(point.get("gb_pitch_deg"))
    if pitch is None:
        blockers.append("姿态缺少云台俯仰角 gb_pitch_deg")
    elif not -90.0 <= pitch < 0.0:
        blockers.append("云台俯仰角必须为负值（-90 为正下视）")
    roll = _number(point.get("gb_roll_deg"))
    if roll is None:
        blockers.append("姿态缺少云台横滚角 gb_roll_deg")
    return blockers


def _point_blockers(points: list, width: float, height: float, min_points: int) -> list[str]:
    blockers = []
    if len(points) < min_points:
        blockers.append(f"控制点不足：{len(points)} < {min_points}")
    seen_image = set()
    seen_map = set()
    for index, point in enumerate(points, start=1):
        if not isinstance(point, dict):
            blockers.append(f"第 {index} 个控制点不是对象")
            continue
        xy = point.get("image_xy")
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            blockers.append(f"第 {index} 个控制点缺少 image_xy")
        else:
            x, y = _number(xy[0]), _number(xy[1])
            if x is None or y is None or not (0 <= x <= width and 0 <= y <= height):
                blockers.append(f"第 {index} 个控制点像素坐标越界")
            else:
                key = (round(x), round(y))
                if key in seen_image:
                    blockers.append(f"第 {index} 个控制点像素坐标重复")
                seen_image.add(key)
        lng = _number(point.get("map_lng_gcj"))
        lat = _number(point.get("map_lat_gcj"))
        if lng is None or lat is None or not (-180 <= lng <= 180 and -90 <= lat <= 90):
            blockers.append(f"第 {index} 个控制点地图坐标无效（需 GCJ-02）")
        else:
            key = (round(lng, 7), round(lat, 7))
            if key in seen_map:
                blockers.append(f"第 {index} 个控制点地图坐标重复")
            seen_map.add(key)
    return blockers


def _max_triangle_area(points: list) -> float:
    coords = []
    for point in points:
        xy = point.get("image_xy") if isinstance(point, dict) else None
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            x, y = _number(xy[0]), _number(xy[1])
            if x is not None and y is not None:
                coords.append((x, y))
    best = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            for k in range(j + 1, len(coords)):
                (x1, y1), (x2, y2), (x3, y3) = coords[i], coords[j], coords[k]
                area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0
                best = max(best, area)
    return best


def _coverage_warnings(points: list, width: float, height: float) -> list[str]:
    coords = []
    for point in points:
        xy = point.get("image_xy") if isinstance(point, dict) else None
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            x, y = _number(xy[0]), _number(xy[1])
            if x is not None and y is not None:
                coords.append((x, y))
    if not coords:
        return []
    xs = [item[0] for item in coords]
    ys = [item[1] for item in coords]
    if (max(xs) - min(xs)) < 0.35 * width or (max(ys) - min(ys)) < 0.35 * height:
        return ["控制点覆盖范围偏小，建议在画面四角或不同深度各取一点"]
    return []


def build_session(image_path: Path, points: list, pose: dict | None = None,
                  *, min_points: int = 4) -> dict:
    """校验影像、点位与姿态，返回会话报告（不写文件）。"""
    image_path = Path(image_path)
    blockers: list[str] = []
    warnings: list[str] = []
    if not image_path.is_file():
        raise FileNotFoundError(f"影像不存在：{image_path}")
    from PIL import Image
    with Image.open(image_path) as image:
        width, height = image.size
        image_format = (image.format or "").upper()
    if image_format not in {"JPEG", "PNG", "WEBP"}:
        blockers.append(f"不支持的影像格式：{image_format or '未知'}；请使用 JPEG/PNG/WebP")
    if not isinstance(pose, dict):
        pose = parse_image_pose(image_path)
    if not isinstance(pose, dict) or not isinstance(pose.get("trajectory"), list) or not pose["trajectory"]:
        blockers.append("没有可用姿态：请提供 --pose 或使用含 GPS/云台信息的照片")
        pose = {}
    trajectory = pose.get("trajectory") if isinstance(pose, dict) else None
    if isinstance(trajectory, list) and len(trajectory) > 1:
        warnings.append("姿态轨迹包含多个点，标定将使用第一个点")
    point = trajectory[0] if isinstance(trajectory, list) and trajectory else {}
    if point:
        blockers.extend(_pose_point_blockers(point))
        if pose.get("pose_quality") == "no_yaw":
            warnings.append("照片姿态缺少完整云台角，标定前请核对")
    blockers.extend(_point_blockers(points, width, height, min_points))
    if len(points) >= 3:
        area = _max_triangle_area(points)
        if area < 0.005 * width * height:
            blockers.append("控制点近似共线或过于集中，无法稳定标定")
    warnings.extend(_coverage_warnings(points, width, height))
    if len(points) < 6:
        warnings.append("控制点少于 6 个，标定残差可能不稳定")

    return {
        "schema": SCHEMA,
        "image": str(image_path),
        "image_sha256": _sha256(image_path),
        "image_width": width,
        "image_height": height,
        "pose": {
            "source": pose.get("source") if isinstance(pose, dict) else None,
            "pose_quality": pose.get("pose_quality") if isinstance(pose, dict) else None,
            "trajectory": trajectory or [],
        },
        "points": points,
        "point_count": len(points),
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--pose", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-points", type=int, default=4)
    args = parser.parse_args()

    points = _load_points(args.points)
    pose = None
    if args.pose is not None:
        pose = json.loads(args.pose.read_text(encoding="utf-8"))
    session = build_session(args.image, points, pose, min_points=args.min_points)
    session["next"] = (
        "python scripts/vision/calibrate_camera_profile.py "
        f"--image {args.image} --points {args.output} "
        "--output <calibration-report.json> --profile-output <camera-profile.json>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "ready": session["ready"],
        "point_count": session["point_count"],
        "blockers": session["blockers"],
        "warnings": session["warnings"],
    }, ensure_ascii=False))
    print(str(args.output))
    if not session["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
