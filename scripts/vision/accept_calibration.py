"""标定验收门禁：用独立验收影像评估 footprint 与路段建议，输出可复核的通过/阻断结论。

流程：
  1. prepare_calibration_session.py 产出标定会话；
  2. calibrate_camera_profile.py 产出 calibration-report.json 与 camera-profile.json；
  3. 本脚本用 camera-profile.json 运行 evaluate_footprint，按阈值判定是否可切换
     VISION_AUTO_LOCALIZATION=calibrated。

只做离线验收，不修改 .env、不重启服务、不发布事件。通过门禁也只代表当前验收集
上的几何一致性，不等同于校园现场绝对定位精度。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "vision"))

from evaluate_footprint import evaluate_manifest  # noqa: E402

SCHEMA = "whu-vision-calibration-acceptance-v1"


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def apply_gates(profile_report: dict, evaluation: dict, *,
                min_calibration_points: int = 4,
                max_calibration_rms_m: float = 10.0,
                max_calibration_point_m: float = 15.0,
                min_cases: int = 6,
                max_mean_center_m: float = 10.0,
                max_center_m: float = 20.0,
                min_top1: float = 0.8,
                min_top3: float = 0.9,
                require_calibrated: bool = True,
                required_basis: str = "pose_footprint",
                require_pose_anchor: bool = True) -> dict:
    """根据标定报告与验收评估报告给出门禁结论（纯函数）。"""
    blockers: list[str] = []
    gates: dict[str, bool | None] = {}

    profile_report = profile_report if isinstance(profile_report, dict) else {}
    calibrated = profile_report.get("calibrated") is True
    gates["profile_calibrated"] = calibrated
    if require_calibrated and not calibrated:
        blockers.append("标定报告 calibrated 不为 true")

    rms = _number(profile_report.get("rms_m"))
    gates["calibration_rms"] = rms is not None and rms <= max_calibration_rms_m
    if rms is None:
        blockers.append("标定报告缺少 rms_m")
    elif rms > max_calibration_rms_m:
        blockers.append(f"标定 RMS {rms} 米超过 {max_calibration_rms_m} 米")

    max_point = _number(profile_report.get("max_m"))
    gates["calibration_max_point"] = max_point is not None and max_point <= max_calibration_point_m
    if max_point is None:
        blockers.append("标定报告缺少 max_m")
    elif max_point > max_calibration_point_m:
        blockers.append(f"标定最大残差 {max_point} 米超过 {max_calibration_point_m} 米")

    point_count = _number(profile_report.get("point_count"))
    gates["calibration_point_count"] = point_count is not None and point_count >= min_calibration_points
    if point_count is None or point_count < min_calibration_points:
        blockers.append(f"标定点数不足：{point_count} < {min_calibration_points}")

    cases = evaluation.get("cases") if isinstance(evaluation, dict) else None
    cases = cases if isinstance(cases, list) else []
    summary = evaluation.get("summary") if isinstance(evaluation, dict) else {}
    summary = summary if isinstance(summary, dict) else {}

    case_count = len(cases)
    gates["acceptance_case_count"] = case_count >= min_cases
    if case_count < min_cases:
        blockers.append(f"验收案例不足：{case_count} < {min_cases}")

    failed_cases = [row.get("name") or row.get("image") for row in cases if row.get("error")]
    gates["no_case_errors"] = not failed_cases
    if failed_cases:
        blockers.append("验收案例执行失败：" + "、".join(str(item) for item in failed_cases[:5]))

    bases = [row.get("basis") for row in cases]
    gates["basis_is_pose_footprint"] = bool(bases) and all(item == required_basis for item in bases)
    if require_pose_anchor and bases and any(item != required_basis for item in bases):
        blockers.append(f"存在未使用 {required_basis} 的案例：" + "、".join(sorted({str(item) for item in bases})))

    anchors = [row.get("anchor_source") for row in cases]
    gates["anchor_is_pose"] = bool(anchors) and all(item == "pose" for item in anchors)
    if require_pose_anchor and anchors and any(item != "pose" for item in anchors):
        blockers.append("存在使用手工锚点的案例：" + "、".join(str(item) for item in anchors))

    center_cases = _number(summary.get("center_error_cases"))
    gates["all_cases_have_center_error"] = center_cases is not None and center_cases == case_count > 0
    if center_cases != case_count or not case_count:
        blockers.append("并非所有案例都生成了 footprint 中心误差")

    mean_center = _number(summary.get("mean_center_error_m"))
    gates["mean_center_error"] = mean_center is not None and mean_center <= max_mean_center_m
    if mean_center is None:
        blockers.append("验收报告缺少平均中心误差")
    elif mean_center > max_mean_center_m:
        blockers.append(f"平均中心误差 {mean_center} 米超过 {max_mean_center_m} 米")

    worst_center = _number(summary.get("max_center_error_m"))
    gates["max_center_error"] = worst_center is not None and worst_center <= max_center_m
    if worst_center is None:
        blockers.append("验收报告缺少最大中心误差")
    elif worst_center > max_center_m:
        blockers.append(f"最大中心误差 {worst_center} 米超过 {max_center_m} 米")

    top1 = _number(summary.get("top1_rate"))
    gates["top1_rate"] = top1 is not None and top1 >= min_top1
    if top1 is None:
        blockers.append("验收报告缺少 Top-1 命中率（需要 truth_edges）")
    elif top1 < min_top1:
        blockers.append(f"Top-1 命中率 {top1} 低于 {min_top1}")

    top3 = _number(summary.get("top3_rate"))
    gates["top3_rate"] = top3 is not None and top3 >= min_top3
    if top3 is None:
        blockers.append("验收报告缺少 Top-3 命中率（需要 truth_edges）")
    elif top3 < min_top3:
        blockers.append(f"Top-3 命中率 {top3} 低于 {min_top3}")

    return {"passed": not blockers, "blockers": blockers, "gates": gates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-report", required=True, type=Path)
    parser.add_argument("--camera-profile", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-calibration-points", type=int, default=4)
    parser.add_argument("--max-calibration-rms-m", type=float, default=10.0)
    parser.add_argument("--max-calibration-point-m", type=float, default=15.0)
    parser.add_argument("--min-cases", type=int, default=6)
    parser.add_argument("--max-mean-center-m", type=float, default=10.0)
    parser.add_argument("--max-center-m", type=float, default=20.0)
    parser.add_argument("--min-top1", type=float, default=0.8)
    parser.add_argument("--min-top3", type=float, default=0.9)
    parser.add_argument("--allow-nominal", action="store_true",
                        help="允许 calibrated=false 的标定报告通过（仅用于预演）")
    args = parser.parse_args()

    if not args.camera_profile.is_file():
        raise SystemExit(f"camera profile 不存在：{args.camera_profile}")
    os.environ["VISION_CAMERA_PROFILE_PATH"] = str(args.camera_profile.resolve())
    profile_report = json.loads(args.profile_report.read_text(encoding="utf-8"))
    evaluation = evaluate_manifest(
        args.manifest, graph_path=args.graph,
        max_mean_center_m=args.max_mean_center_m, min_top1=args.min_top1,
    )
    verdict = apply_gates(
        profile_report, evaluation,
        min_calibration_points=args.min_calibration_points,
        max_calibration_rms_m=args.max_calibration_rms_m,
        max_calibration_point_m=args.max_calibration_point_m,
        min_cases=args.min_cases,
        max_mean_center_m=args.max_mean_center_m,
        max_center_m=args.max_center_m,
        min_top1=args.min_top1,
        min_top3=args.min_top3,
        require_calibrated=not args.allow_nominal,
    )
    report = {
        "schema": SCHEMA,
        "passed": verdict["passed"],
        "blockers": verdict["blockers"],
        "gates": verdict["gates"],
        "camera_profile": str(args.camera_profile),
        "profile_report": profile_report,
        "evaluation": evaluation,
        "limits": [
            "离线验收；不修改 .env、不重启服务、不发布事件",
            "通过门禁只代表当前验收集上的几何一致性，不等同于现场绝对定位精度",
            "切换 VISION_AUTO_LOCALIZATION=calibrated 仍需人工确认现场标定与回退方案",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "blockers": report["blockers"],
        "gates": report["gates"],
    }, ensure_ascii=False))
    print(str(args.output))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
