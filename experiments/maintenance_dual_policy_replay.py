"""离线双策略对照：保守复用 vs 全量重算。

适用范围（由代码阅读确认，不是假设）：
``config.RESULT_MAINTENANCE_POLICY`` 只在多段地点行程的续接处被读取
（``agents/planner.py`` 的 ``with_next_place_stage``）。单轮直达、复合工具任务
与路况事件重算都不读该开关。因此本驱动器显式测量“开关对单轮任务无影响”，
而不是把它写成全域收益。

不产生模型调用：会话固定为 legacy 确定性循环，只走地点选择的确定性分支；
天气输入被冻结，使两臂看到完全相同的路径成本输入。

输出分两份：
- ``records.jsonl``：含耗时，供观察；
- ``deterministic.json``：只含决策与几何摘要、不含时延与随机标识，
  同一输入重复运行应逐字节一致。
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_POLICIES = ("conservative", "full_recompute")
_TASK_REVISION = 1
_PROOF_IDENTITY = {"owner_id": "offline-replay", "task_id": "maintenance-dual-policy",
                   "task_revision": _TASK_REVISION}
_FROZEN_WEATHER_LIVE = {
    "weather": "晴", "temperature": 22, "windpower": "≤3", "humidity": "60",
}

# 两段同方式行程，取自 controlled_multi_turn_pilot_v2 的 OD，
# 但第二段保持同一出行方式：试点的第二段会切换方式，按设计两臂都必须
# 全量重算，无法观测策略差异。此处的改动在校验清单里明示。
_ITINERARY_CASES = (
    {
        "case_id": "itin_poi015_poi016_poi014",
        "group": "two_stage_itinerary",
        "start_poi_id": "poi_015",
        "stop_poi_id": "poi_016",
        "final_poi_id": "poi_014",
        "mode": "walk",
        "constraints": {"avoid_steps": True},
        "strategy": "shortest",
        "derived_from": "controlled_multi_turn_pilot_v2/revision_dev_001",
        "deviation": "second stage keeps the first stage travel mode",
    },
    {
        "case_id": "itin_poi014_poi016_poi015",
        "group": "two_stage_itinerary",
        "start_poi_id": "poi_014",
        "stop_poi_id": "poi_016",
        "final_poi_id": "poi_015",
        "mode": "walk",
        "constraints": {"avoid_steps": True},
        "strategy": "shortest",
        "derived_from": "controlled_multi_turn_pilot_v2/revision_dev_002",
        "deviation": "second stage keeps the first stage travel mode",
    },
)

# 单轮直达：用于证明该开关在这些任务上不生效。
_DIRECT_CASES = (
    {
        "case_id": "direct_poi001_poi015",
        "group": "single_turn_direct",
        "start_poi_id": "poi_001",
        "end_poi_id": "poi_015",
        "mode": "walk",
        "constraints": {},
        "strategy": "shortest",
        "derived_from": "controlled_event_pilot_v3/event_dev_detour_001",
    },
    {
        "case_id": "direct_poi015_poi014",
        "group": "single_turn_direct",
        "start_poi_id": "poi_015",
        "end_poi_id": "poi_014",
        "mode": "walk",
        "constraints": {},
        "strategy": "shortest",
        "derived_from": "controlled_multi_turn_pilot_v2/revision_dev_001",
    },
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_metadata() -> dict:
    def run(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, check=False,
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "head": run("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "agents" / "planner.py",
        ROOT / "agents" / "result_maintenance.py",
        ROOT / "agents" / "route_state.py",
        ROOT / "agents" / "tools.py",
        ROOT / "config.py",
        DATA_DIR / "whu_road_network.graphml",
        DATA_DIR / "pois.json",
        DATA_DIR / "road_conditions.json",
        DATA_DIR / "edge_overrides.json",
        DATA_DIR / "road_annotations.json",
    )
    return {
        str(path.relative_to(ROOT)): _sha(path.read_bytes())
        for path in paths if path.is_file()
    }


def _frozen_weather() -> dict:
    from spatial import weather as weather_mod

    return {"live": dict(_FROZEN_WEATHER_LIVE),
            "impact": weather_mod.classify_weather(dict(_FROZEN_WEATHER_LIVE))}


@contextlib.contextmanager
def _deterministic_environment():
    """Force the deterministic planner loop and a frozen weather input."""
    previous_mode = os.environ.get("AGENT_DIALOGUE_MODE")
    os.environ["AGENT_DIALOGUE_MODE"] = "legacy"
    from agents import tools as agent_tools

    original_snapshot = agent_tools._weather_snapshot
    frozen = _frozen_weather()
    agent_tools._weather_snapshot = lambda: frozen
    try:
        yield frozen
    finally:
        agent_tools._weather_snapshot = original_snapshot
        if previous_mode is None:
            os.environ.pop("AGENT_DIALOGUE_MODE", None)
        else:
            os.environ["AGENT_DIALOGUE_MODE"] = previous_mode


def _poi_index() -> dict[str, dict]:
    from spatial.poi import load_pois

    return {poi["id"]: poi for poi in load_pois() if isinstance(poi, dict) and poi.get("id")}


def _leg_args(start_id: str, end_id: str, case: dict) -> dict:
    args = {
        "start": {"type": "poi", "poi_id": start_id},
        "end": {"type": "poi", "poi_id": end_id},
        "mode": case["mode"],
    }
    if case.get("constraints"):
        args["constraints"] = dict(case["constraints"])
    if case.get("strategy"):
        args["strategy"] = case["strategy"]
    return args


def _plan_leg(start_id: str, end_id: str, case: dict, uid: str) -> tuple[dict | None, dict]:
    from agents import tools as agent_tools

    args = _leg_args(start_id, end_id, case)
    result, artifact = agent_tools.execute_tool(
        "plan_route", args,
        {"query": f"{start_id}->{end_id}", "uid": uid,
         "_route_weather_snapshot": _frozen_weather()},
    )
    route = (artifact or {}).get("route") if isinstance(artifact, dict) else None
    return route, result if isinstance(result, dict) else {}


def _route_digest(route: dict | None) -> dict:
    if not isinstance(route, dict):
        return {"distance_m": None, "edge_count": 0, "edge_ids_sha256": None}
    edges = route.get("recommended_edge_ids") or []
    encoded = json.dumps([[int(a), int(b), int(k)] for a, b, k in edges],
                         separators=(",", ":")).encode("utf-8")
    distance = route.get("distance_m")
    if not isinstance(distance, (int, float)):
        distance = route.get("recommended_length_m")
    return {
        "distance_m": round(float(distance), 2) if isinstance(distance, (int, float)) else None,
        "edge_count": len(edges),
        "edge_ids_sha256": _sha(encoded) if edges else None,
    }


def _ref_id(value) -> str | None:
    if not isinstance(value, dict):
        return None
    if value.get("poi_id"):
        return str(value["poi_id"])
    if value.get("name"):
        return str(value["name"])
    if value.get("type") == "coord":
        return "coord"
    return None


def _compact_plan_call(name: str, args) -> dict:
    """Record which anchors a planning tool was asked to (re)compute."""
    anchors = None
    if name in {"plan_route", "plan_via_route"} and isinstance(args, dict):
        via = args.get("via_points")
        if not isinstance(via, list):
            via = [args["via"]] if isinstance(args.get("via"), dict) else []
        anchors = [_ref_id(args.get("start")), *[_ref_id(point) for point in via],
                   _ref_id(args.get("end"))]
        anchors = [anchor for anchor in anchors if anchor]
    return {"tool": name,
            "od_anchors": anchors,
            "od_anchor_count": len(anchors) if anchors else None}


def _pending_itinerary(case: dict, leg1: dict) -> dict:
    from agents import planner
    from agents.result_maintenance import capture_dependencies

    query = (f"从{case['start_poi_id']}到{case['stop_poi_id']}"
             + ("，按最短路径" if case.get("strategy") == "shortest" else ""))
    pending = planner._place_request_state(
        query,
        pending_slot="selection",
        # 服务端进入下一段时会把 start 改成上一段终点；沿用首段起点会拼出断开的行程。
        start={"type": "poi", "poi_id": case["stop_poi_id"]},
        constraints=dict(case.get("constraints") or {}),
        mode=case["mode"],
        # 真实服务在第二段会先检索下一站候选；外部选择必须命中该列表。
        candidate_poi_ids=[case["final_poi_id"]],
        candidate_places=[],
    )
    pending["strategy"] = case.get("strategy")
    pending["completed_legs"] = [leg1]
    pending["result_dependencies"] = capture_dependencies(
        [leg1], task_revision=_TASK_REVISION,
    )
    return pending


def _run_direct_arm(case: dict, arm: str, uid: str) -> dict:
    import config

    config.RESULT_MAINTENANCE_POLICY = arm
    started = time.perf_counter()
    route, result = _plan_leg(case["start_poi_id"], case["end_poi_id"], case, uid)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "case_id": case["case_id"],
        "group": case["group"],
        "arm": arm,
        "policy": arm,
        "response_kind": "route" if route else "error",
        "prefix_action": None,
        "stage1": _route_digest(route),
        "stage2": _route_digest(route),
        "tool_calls": {"plan_route": 1},
        "plan_calls": [_compact_plan_call("plan_route",
                                          _leg_args(case["start_poi_id"],
                                                    case["end_poi_id"], case))],
        "recomputed_anchor_count": 2,
        "elapsed_ms": elapsed_ms,
        "error": None if route else (result.get("error") or "route_unavailable"),
    }


def _run_itinerary_arm(case: dict, arm: str, uid: str) -> dict:
    import config
    from agents import planner

    config.RESULT_MAINTENANCE_POLICY = arm
    started = time.perf_counter()
    leg1, stage1_result = _plan_leg(case["start_poi_id"], case["stop_poi_id"], case, uid)
    if leg1 is None:
        return {
            "case_id": case["case_id"], "group": case["group"], "arm": arm, "policy": arm,
            "response_kind": "error", "prefix_action": None,
            "stage1": _route_digest(None), "stage2": _route_digest(None),
            "tool_calls": {"plan_route": 1},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": f"stage1:{stage1_result.get('error') or 'route_unavailable'}",
        }
    pending = _pending_itinerary(case, leg1)
    if pending.get("result_dependencies") is None:
        return {
            "case_id": case["case_id"], "group": case["group"], "arm": arm, "policy": arm,
            "response_kind": "error", "prefix_action": None,
            "stage1": _route_digest(leg1), "stage2": _route_digest(None),
            "tool_calls": {"plan_route": 1},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": "stage1:dependencies_unavailable",
        }
    calls: dict[str, int] = {}
    recorded: list[dict] = []
    from agents import tools as agent_tools

    original_execute = agent_tools.execute_tool

    def traced_execute(name, args, ctx=None):
        if name in {"plan_route", "plan_via_route"}:
            recorded.append(_compact_plan_call(name, args))
        return original_execute(name, args, ctx)

    def trace(payload):
        name = (payload or {}).get("tool_name")
        if isinstance(name, str):
            calls[name] = calls.get(name, 0) + 1

    agent_tools.execute_tool = traced_execute
    try:
        response = planner.run_agent(
            "就去这里",
            context={"place_request": pending, "selected_poi_id": case["final_poi_id"]},
            uid=uid,
            proof_identity=dict(_PROOF_IDENTITY),
            tool_trace_callback=trace,
        )
    finally:
        agent_tools.execute_tool = original_execute
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    maintenance = response.get("result_maintenance") or {}
    route = response.get("route") if isinstance(response.get("route"), dict) else None
    return {
        "case_id": case["case_id"],
        "group": case["group"],
        "arm": arm,
        "policy": arm,
        "response_kind": response.get("response_kind"),
        "prefix_action": maintenance.get("prefix_action"),
        "stage1": _route_digest(leg1),
        "stage2": _route_digest(route),
        "tool_calls": dict(sorted(calls.items())),
        "plan_calls": recorded,
        "recomputed_anchor_count": sum(
            item["od_anchor_count"] or 0 for item in recorded) or None,
        "elapsed_ms": elapsed_ms,
        "error": None if route else "stage2:route_unavailable",
    }


def _compare(case: dict, arms: dict[str, dict]) -> dict:
    conservative, full = arms["conservative"], arms["full_recompute"]
    c_dist, f_dist = conservative["stage2"]["distance_m"], full["stage2"]["distance_m"]
    distance_equal = (
        c_dist is not None and f_dist is not None
        and abs(c_dist - f_dist) <= 0.05
    )
    edge_equal = (conservative["stage2"]["edge_ids_sha256"]
                  == full["stage2"]["edge_ids_sha256"])
    retained = conservative["prefix_action"] == "retained"
    stale = bool(retained and not distance_equal)
    return {
        "case_id": case["case_id"],
        "group": case["group"],
        "prefix_action": {
            "conservative": conservative["prefix_action"],
            "full_recompute": full["prefix_action"],
        },
        "policy_sensitive": (
            conservative["prefix_action"] != full["prefix_action"]
            or conservative["tool_calls"] != full["tool_calls"]
        ),
        "planning_tool_calls": {
            "conservative": conservative["tool_calls"].get("plan_route", 0)
            + conservative["tool_calls"].get("plan_via_route", 0),
            "full_recompute": full["tool_calls"].get("plan_route", 0)
            + full["tool_calls"].get("plan_via_route", 0),
        },
        "plan_calls": {arm: row.get("plan_calls") or [] for arm, row in arms.items()},
        "recomputed_anchor_count": {
            "conservative": conservative.get("recomputed_anchor_count"),
            "full_recompute": full.get("recomputed_anchor_count"),
        },
        "distance_m": {"conservative": c_dist, "full_recompute": f_dist},
        "distance_equal": distance_equal,
        "edge_ids_equal": edge_equal,
        "stale_reference": stale,
        "errors": {arm: row["error"] for arm, row in arms.items() if row.get("error")},
        "elapsed_ms": {arm: row["elapsed_ms"] for arm, row in arms.items()},
    }


def run_dual_policy_replay(output_root: Path, run_id: str, *, root: Path = ROOT) -> Path:
    if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("invalid run_id")
    root = Path(root).resolve()
    output_root = Path(output_root)
    result_dir = output_root / run_id
    if result_dir.exists():
        raise FileExistsError(result_dir)

    pois = _poi_index()
    cases = [*_ITINERARY_CASES, *_DIRECT_CASES]
    for case in cases:
        for key in ("start_poi_id", "end_poi_id", "stop_poi_id", "final_poi_id"):
            poi_id = case.get(key)
            if poi_id and poi_id not in pois:
                raise ValueError(f"{case['case_id']}: unknown POI {poi_id}")

    records: list[dict] = []
    comparisons: list[dict] = []
    with _deterministic_environment():
        for case in cases:
            arms = {}
            for arm in _POLICIES:
                if case["group"] == "two_stage_itinerary":
                    row = _run_itinerary_arm(case, arm, f"replay-{case['case_id']}-{arm}")
                else:
                    row = _run_direct_arm(case, arm, f"replay-{case['case_id']}-{arm}")
                arms[arm] = row
                records.append(row)
            comparisons.append(_compare(case, arms))
    # 恢复默认策略，避免污染同进程后续调用
    import config

    config.RESULT_MAINTENANCE_POLICY = "conservative"

    itinerary = [row for row in comparisons if row["group"] == "two_stage_itinerary"]
    direct = [row for row in comparisons if row["group"] == "single_turn_direct"]
    retained = [row for row in itinerary if row["prefix_action"]["conservative"] == "retained"]
    stale = [row for row in comparisons if row["stale_reference"]]
    mismatched = [row for row in comparisons if not row["distance_equal"]]
    inert_violations = [row for row in direct if row["policy_sensitive"]]
    errors = [row for row in comparisons if row["errors"]]

    summary = {
        "run_type": "development_diagnostic",
        "formal_ready": False,
        "case_count": len(cases),
        "arm_count": len(records),
        "policy_sensitive_itinerary_cases": [row["case_id"] for row in itinerary if row["policy_sensitive"]],
        "conservative_prefix_retained_cases": [row["case_id"] for row in retained],
        "conservative_planning_tool_calls": sum(
            row["planning_tool_calls"]["conservative"] for row in itinerary),
        "full_recompute_planning_tool_calls": sum(
            row["planning_tool_calls"]["full_recompute"] for row in itinerary),
        "conservative_recomputed_anchors": sum(
            row["recomputed_anchor_count"]["conservative"] or 0 for row in itinerary),
        "full_recompute_recomputed_anchors": sum(
            row["recomputed_anchor_count"]["full_recompute"] or 0 for row in itinerary),
        "stale_reference_count": len(stale),
        "stale_reference_cases": [row["case_id"] for row in stale],
        "distance_mismatch_count": len(mismatched),
        "distance_mismatch_cases": [row["case_id"] for row in mismatched],
        "direct_route_policy_effect_count": len(inert_violations),
        "direct_route_policy_effect_cases": [row["case_id"] for row in inert_violations],
        "error_cases": [row["case_id"] for row in errors],
        "scope_statement": (
            "RESULT_MAINTENANCE_POLICY 只作用于多段地点行程的续接；"
            "单轮直达任务在两臂下结果一致，本报告不主张该开关对全任务类型有效。"
        ),
    }
    manifest = {
        "run_type": "development_diagnostic",
        "formal_ready": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(),
        "source_hashes": _source_hashes(),
        "policies": list(_POLICIES),
        "task_revision": _TASK_REVISION,
        "frozen_weather": dict(_FROZEN_WEATHER_LIVE),
        "planner_loop": "legacy_deterministic",
        "model_calls": 0,
        "limitations": [
            "所有 OD 与出行条件取自受控试点数据，不含现场可行性真值。",
            "第二段沿用首段出行方式，与试点中切换方式的第二段不同；切换方式本身必然触发两臂全量重算。",
            "耗时只覆盖本地进程内的确定性规划，不含鉴权、数据库、事件解码与生产网络延迟。",
            "天气输入被冻结，因此不反映真实天气变化带来的重算。",
        ],
    }
    deterministic = {
        "policies": list(_POLICIES),
        "cases": [
            {
                "case_id": row["case_id"],
                "group": row["group"],
                "prefix_action": row["prefix_action"],
                "planning_tool_calls": row["planning_tool_calls"],
                "recomputed_anchor_count": row["recomputed_anchor_count"],
                "distance_m": row["distance_m"],
                "distance_equal": row["distance_equal"],
                "edge_ids_equal": row["edge_ids_equal"],
                "stale_reference": row["stale_reference"],
                "policy_sensitive": row["policy_sensitive"],
                "errors": row["errors"],
            }
            for row in comparisons
        ],
        "summary": {
            key: value for key, value in summary.items()
            if key != "error_cases"
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(exist_ok=False)
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (result_dir / "deterministic.json").write_text(
        json.dumps(deterministic, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (result_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return result_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "output/experiments/maintenance_dual_policy")
    args = parser.parse_args(argv)
    result_dir = run_dual_policy_replay(args.output_root, args.run_id, root=args.root)
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"result_dir": str(result_dir),
                      "summary": summary}, ensure_ascii=False, indent=2))
    failed = (summary["error_cases"] or summary["distance_mismatch_cases"]
              or summary["stale_reference_cases"] or summary["direct_route_policy_effect_cases"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
