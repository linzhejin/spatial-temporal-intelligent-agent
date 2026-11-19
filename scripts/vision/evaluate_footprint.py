"""评估姿态 footprint 与路段建议：中心误差、Top-1/Top-3 命中率。

清单格式：
{
  "cases": [
    {
      "name": "工学部东门",
      "image": "D:/.../image.jpg",
      "pose": "D:/.../pose.json",            # 可选；缺省读照片 EXIF/XMP
      "truth_center_lng_gcj": 114.36,
      "truth_center_lat_gcj": 30.53,
      "truth_edges": [[u, v, k], ...]         # 可选
    }
  ]
}

输出只用于离线评估，不修改场景或发布状态。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from spatial.network import get_network, load_or_download_network  # noqa: E402
from spatial.vision_suggest import score_suggested_edges  # noqa: E402
from vision.camera_profile import calibration_status, intrinsics_for, resolve_profile  # noqa: E402
from vision.pose_metadata import parse_image_pose, with_height_agl  # noqa: E402

_M_PER_DEG = 111320.0


def _distance_m(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    cos_lat = math.cos(math.radians((lat1 + lat2) / 2.0))
    dx = (lng1 - lng2) * _M_PER_DEG * cos_lat
    dy = (lat1 - lat2) * _M_PER_DEG
    return math.hypot(dx, dy)


def _load_pose(case: dict, image_path: Path, profile: dict) -> dict:
    pose_value = case.get("pose")
    if isinstance(pose_value, str):
        pose = json.loads(Path(pose_value).read_text(encoding="utf-8"))
    elif isinstance(pose_value, dict):
        pose = pose_value
    else:
        pose = parse_image_pose(image_path)
    if not isinstance(pose, dict) or not pose.get("trajectory"):
        raise ValueError("没有可用姿态")
    return with_height_agl(pose, ground_offset_m=float(profile.get("ground_offset_m") or 0.0))


def evaluate_manifest(manifest_path: Path, *, graph_path: Path | None = None,
                      max_mean_center_m: float = 10.0,
                      min_top1: float = 0.8) -> dict:
    """按清单评估全部案例并返回报告字典（不写文件、不改运行状态）。"""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest 必须包含非空 cases 列表")
    G = get_network() or load_or_download_network()
    if G is None:
        raise RuntimeError("校园路网不可用")
    if graph_path is not None:
        import networkx as nx
        G = nx.read_graphml(graph_path)
    profile = resolve_profile()
    calibration = calibration_status(profile)

    rows = []
    for case in cases:
        name = str(case.get("name") or "case")
        image_path = Path(case["image"])
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = (image.format or "").upper()
            if image_format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError(f"不支持的影像格式：{image_format or '未知'}；请使用 JPEG/PNG/WebP")
            pose = _load_pose(case, image_path, profile)
            pose["intrinsics"] = intrinsics_for(width, height, profile)
            media = {"kind": "image", "width": width, "height": height}
            result = score_suggested_edges(
                G, None, camera_pose=pose, media=media,
                camera_profile=profile, calibration=calibration,
            )
            edges = result.get("suggested_edges") or []
            truth_edges = {tuple(item) for item in (case.get("truth_edges") or [])}
            hits = [tuple([item["u"], item["v"], item["key"]]) for item in edges]
            top1 = bool(truth_edges and hits and hits[0] in truth_edges)
            top3 = bool(truth_edges and any(hit in truth_edges for hit in hits[:3]))
            center_error = None
            if case.get("truth_center_lng_gcj") is not None and case.get("truth_center_lat_gcj") is not None:
                footprint = result.get("footprint") or []
                if footprint:
                    lng = sum(point[0] for point in footprint) / len(footprint)
                    lat = sum(point[1] for point in footprint) / len(footprint)
                    center_error = round(_distance_m(
                        lng, lat,
                        float(case["truth_center_lng_gcj"]),
                        float(case["truth_center_lat_gcj"]),
                    ), 2)
            rows.append({
                "name": name, "image": str(image_path), "basis": result.get("basis"),
                "anchor_source": result.get("anchor_source"),
                "pose_source": result.get("pose_source"),
                "center_error_m": center_error,
                "top1_hit": top1 if truth_edges else None,
                "top3_hit": top3 if truth_edges else None,
                "suggested": hits[:5],
            })
        except Exception as error:
            rows.append({"name": name, "image": str(image_path), "error": str(error)})

    errors = [row["center_error_m"] for row in rows if row.get("center_error_m") is not None]
    top1_rows = [row for row in rows if row.get("top1_hit") is not None]
    top1_rate = (sum(1 for row in top1_rows if row["top1_hit"]) / len(top1_rows)) if top1_rows else None
    mean_center = (sum(errors) / len(errors)) if errors else None
    passed = bool(
        mean_center is not None and mean_center <= max_mean_center_m
        and (top1_rate is None or top1_rate >= min_top1)
    )
    return {
        "manifest": str(manifest_path),
        "profile": profile.get("name"),
        "calibration": calibration,
        "cases": rows,
        "summary": {
            "case_count": len(rows),
            "center_error_cases": len(errors),
            "mean_center_error_m": round(mean_center, 2) if mean_center is not None else None,
            "max_center_error_m": round(max(errors), 2) if errors else None,
            "top1_rate": round(top1_rate, 3) if top1_rate is not None else None,
            "top3_rate": round(
                sum(1 for row in top1_rows if row["top3_hit"]) / len(top1_rows), 3
            ) if top1_rows else None,
            "thresholds": {
                "max_mean_center_m": max_mean_center_m,
                "min_top1": min_top1,
            },
            "passed": passed,
        },
        "limits": [
            "离线评估；不修改场景、不发布事件",
            "中心误差以 footprint 顶点均值为代理，不等同于绝对定位精度",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-mean-center-m", type=float, default=10.0)
    parser.add_argument("--min-top1", type=float, default=0.8)
    args = parser.parse_args()

    report = evaluate_manifest(
        args.manifest, graph_path=args.graph,
        max_mean_center_m=args.max_mean_center_m, min_top1=args.min_top1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(str(args.output))


if __name__ == "__main__":
    main()
