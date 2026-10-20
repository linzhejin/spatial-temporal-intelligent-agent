"""Report independent research gates and freeze the Agent experiment inputs.

``ready`` always means the Agent main experiment scope. Route field evidence,
preference attributes, and each UAV category have their own explicit status.
``--write-snapshot`` requires the Agent gate and a clean tracked worktree.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.validate.check_deployment_readiness import AUDIT, ROOT, assess, read_json  # noqa: E402


CHECKS = {
    "routing_policy": [sys.executable, "-m", "pytest", "tests/test_routing_policy.py", "-q"],
    "route_state": [
        sys.executable, "-m", "pytest", "tests/test_route_state.py",
        "tests/test_replan_api.py", "-q",
    ],
    "profile_telemetry": [
        sys.executable, "-m", "pytest", "tests/test_profile.py",
        "tests/test_telemetry.py", "-q",
    ],
    "routing": [sys.executable, "-m", "pytest", "tests/test_routing.py", "-q"],
    "task_planning": [
        sys.executable, "-m", "pytest", "tests/test_task_planner.py",
        "tests/test_task_patch.py", "-q",
    ],
    "poi_activity": [
        sys.executable, "-m", "pytest", "tests/test_poi_activity_recommendations.py",
        "tests/test_poi_activity_taxonomy.py", "-q",
    ],
    "multi_via": [sys.executable, "-m", "pytest", "tests/test_multi_via_agent.py", "-q"],
    "tool_registry": [
        sys.executable, "-m", "pytest", "tests/test_agent_tool_registry.py", "-q",
    ],
    "agent_capability_layer": [
        sys.executable, "-m", "pytest", "tests/test_tool_skills.py",
        "tests/test_reflection.py", "-q",
    ],
    "agent_trace_and_knowledge": [
        sys.executable, "-m", "pytest", "tests/test_provenance.py",
        "tests/test_experience.py", "-q",
    ],
    "frontend_state": ["npm.cmd" if sys.platform == "win32" else "npm", "run", "test:js"],
}

STRUCTURAL_CODES = {"structural_validation_missing", "structural_errors", "count_mismatch"}
ROUTE_EVIDENCE_CODES = {
    "reference_samples_missing", "unverified_reference_routes",
    "commercial_reference_missing", "reference_comparison_missing",
    "unresolved_route_disagreements", "unassessed_route_geometry_hazards",
    "unreviewed_route_building_crossings",
}
REQUIRED_INPUTS = (
    "poi", "network", "edge_overrides", "road_annotations", "attribute_master",
    "taxonomy", "tool_schemas", "model_config", "prompts", "task_cards", "event_state",
    "runtime_code", "requirements", "runtime_environment",
)

RUNTIME_SOURCE_PATHS = (
    "app.py", "agents", "api", "spatial", "storage", "jobs",
)
RUNTIME_PACKAGES = (
    "flask", "flask-cors", "openai", "httpx", "pydantic", "psycopg",
    "langgraph", "langgraph-checkpoint-postgres", "networkx", "shapely",
    "python-dotenv",
)


def _scope(blockers):
    return {"ready": not blockers, "blockers": blockers}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _vision_scopes(report):
    valid = (isinstance(report, dict)
             and report.get("task") == "campus_uav_field_readiness")
    checks = _mapping(report.get("checks")) if valid else {}
    campus = valid and report.get("campus_field_data_present") is True
    detection = _mapping(checks.get("object_detection"))
    event = _mapping(checks.get("event_evaluation"))
    targets = _mapping(event.get("event_metric_targets"))
    water = _mapping(checks.get("visible_water_segmentation"))
    models = _mapping(checks.get("models"))
    external = _mapping(checks.get("external_accident_evaluation"))

    def evidence(ok, reason):
        return _scope([] if campus and ok else
                      (["campus_field_evidence_missing"] if not campus else [reason]))

    result = {
        "detection": evidence(detection.get("accuracy_claim_ready") is True,
                              "detection_evidence_not_ready"),
    }
    for kind in ("congestion", "crowding"):
        target = _mapping(targets.get(kind))
        positive_flights = target.get("positive_test_flight_count")
        result[kind] = evidence(
            event.get("status") == "ready" and target.get("met") is True
            and target.get("flight_coverage_met") is True
            and type(positive_flights) is int and positive_flights > 0,
            f"{kind}_event_evidence_not_ready",
        )
    result["accident"] = evidence(
        event.get("campus_negative_safety_ready") is True
        and external.get("status") == "ready"
        and _mapping(models.get("accident")).get("status") == "present",
        "accident_evidence_not_ready",
    )
    result["flooding"] = evidence(
        water.get("status") == "ready"
        and _mapping(models.get("flood")).get("status") == "present",
        "flooding_evidence_not_ready",
    )
    return result


def assess_research_readiness(checks, deployment, attributes, *, vision=None, versions=None):
    """Keep evidence from one research line from certifying another line."""
    agent_blockers = {}
    failed = [name for name in CHECKS if not (checks or {}).get(name, {}).get("passed")]
    if failed:
        agent_blockers["code_checks"] = failed
    deployment_blockers = (deployment or {}).get("blockers") or []
    if not deployment:
        deployment_blockers = [{"code": "deployment_readiness_missing"}]
    elif deployment.get("ready") is False and not deployment_blockers:
        deployment_blockers = [{"code": "deployment_blockers_missing"}]
    structural = [row for row in deployment_blockers
                  if row.get("code") in STRUCTURAL_CODES
                  or row.get("code") in {
                      "deployment_readiness_missing", "deployment_blockers_missing",
                  }]
    if structural:
        agent_blockers["campus_structure"] = structural
    if versions is not None:
        missing = [name for name in REQUIRED_INPUTS
                   if not versions.get(name, {}).get("files")
                   or versions[name].get("missing")]
        if missing:
            agent_blockers["snapshot_inputs"] = missing

    route_blockers = [row for row in deployment_blockers
                      if row.get("code") in ROUTE_EVIDENCE_CODES | STRUCTURAL_CODES]
    if not deployment:
        route_blockers = deployment_blockers
    elif deployment.get("ready") is False and not route_blockers and (
            deployment_blockers[0].get("code") == "deployment_blockers_missing"):
        route_blockers = deployment_blockers
    preference_blockers = {}
    if (not attributes or not attributes.get("ready")
            or bool(attributes.get("degraded"))):
        preference_blockers["attribute_master_data"] = (attributes or {}).get(
            "blockers", ["attribute_master_data_report_missing"]
        )
    scopes = {
        "agent_main_experiment": _scope(agent_blockers),
        "sampled_route_evidence": _scope(route_blockers),
        "preference_attributes": _scope(preference_blockers),
        "vision": _vision_scopes(vision),
    }
    return {
        "scope": "agent_main_experiment",
        "ready": scopes["agent_main_experiment"]["ready"],
        "blockers": agent_blockers,
        "scopes": scopes,
        "interpretation": (
            "Top-level ready only certifies the Agent main experiment gate. "
            "Route, preference, and UAV evidence must be read from their own scopes."
        ),
    }


def _run_checks():
    results = {}
    for name, command in CHECKS.items():
        command = list(command)
        temp_root = None
        if "pytest" in command:
            temp_root = tempfile.mkdtemp(prefix=f"whu-research-{name}-")
            command.append(f"--basetemp={temp_root}")
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        results[name] = {
            "passed": completed.returncode == 0,
            "command": command,
            "exit_code": completed.returncode,
            "output_tail": output[-4000:],
        }
        if temp_root:
            shutil.rmtree(temp_root, ignore_errors=True)
    return results


def _deployment_result():
    return assess(
        read_json(ROOT / "data_validation_report.json"),
        read_json(AUDIT / "campus_spatial_quality.json"),
        read_json(AUDIT / "amap_comparison_metrics.json"),
        read_json(ROOT / "data" / "route_reference_samples.json"),
        read_json(ROOT / "data" / "campus_review_decisions.json") or {},
        read_json(AUDIT / "amap_comparison_candidate.json"),
    )


def _sha256_paths(paths, *, root=None, suffixes=None):
    """Hash names and bytes; report missing inputs instead of hashing an empty set."""
    root = ROOT if root is None else root
    digest = hashlib.sha256()
    existing = []
    missing = []
    for raw in paths:
        path = root / raw
        if path.is_dir():
            files = sorted(item for item in path.rglob("*")
                           if item.is_file() and (suffixes is None or item.suffix in suffixes))
        elif path.is_file():
            files = [path] if suffixes is None or path.suffix in suffixes else []
        else:
            files = []
        if not files:
            missing.append(str(raw).replace("\\", "/"))
        for item in files:
            rel = item.relative_to(root).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            existing.append(rel)
    return {"sha256": digest.hexdigest(), "files": existing, "missing": missing}


def _event_state_version():
    """Fingerprint the actual runtime file and events active at snapshot time."""
    from spatial import road_conditions

    path = Path(road_conditions._CONDITIONS_FILE)
    if not path.is_file():
        return {"sha256": None, "files": [], "missing": ["event_state"]}
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    try:
        active = road_conditions.list_conditions(strict=True)
    except road_conditions.RoadConditionsUnavailableError:
        return {"sha256": digest.hexdigest(), "files": ["event_state"],
                "missing": ["invalid_event_state"]}
    active_payload = json.dumps(active, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {"sha256": digest.hexdigest(), "files": ["event_state"],
            "missing": [], "active_sha256": hashlib.sha256(active_payload).hexdigest(),
            "active_count": len(active)}


def _runtime_model_config():
    """Record only non-credential model settings actually loaded by the app."""
    import config

    fields = (
        "LLM_MODEL", "VISION_MODEL_ID", "VISION_MODEL_REVISION",
        "VISION_ACCIDENT_REVISION", "VISION_FLOOD_REVISION",
        "VISION_CONFIDENCE_THRESHOLD", "VISION_TRACKING_MIN_CONFIDENCE",
        "VISION_ACCIDENT_THRESHOLD", "VISION_ACCIDENT_DENSE_TRIGGER_THRESHOLD",
        "VISION_FLOOD_MIN_AREA_RATIO",
    )
    values = {name: getattr(config, name) for name in fields}
    # The endpoint can contain credentials, so record only its fingerprint.
    values["OPENAI_BASE_URL_sha256"] = hashlib.sha256(
        config.OPENAI_BASE_URL.encode("utf-8")
    ).hexdigest()
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {"sha256": hashlib.sha256(payload).hexdigest(), "values": values}


def _runtime_environment_version():
    """Record the non-secret Python and installed library versions used by Agent."""
    packages = {}
    missing = []
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
            missing.append(f"package:{name}")
    values = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "timezone": datetime.now().astimezone().tzname(),
        "packages": packages,
    }
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": ["python_runtime", *[f"package:{name}" for name in RUNTIME_PACKAGES
                                       if packages[name] is not None]],
        "missing": missing,
        "values": values,
    }


def _git(*args):
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _versions():
    # Importing config loads .env, including the effective ROAD_CONDITIONS_FILE.
    runtime_model_config = _runtime_model_config()
    versions = {
        "git_commit": _git("rev-parse", "HEAD"),
        "model_config": _sha256_paths(["config.py"]),
        "prompts": _sha256_paths(["agents/prompts"]),
        "task_cards": _sha256_paths(["data/task_cards.json"]),
        "poi": _sha256_paths(["data/pois.json"]),
        "network": _sha256_paths(["data/whu_road_network.graphml"]),
        "road_annotations": _sha256_paths(["data/road_annotations.json"]),
        "edge_overrides": _sha256_paths(["data/edge_overrides.json"]),
        "attribute_master": _sha256_paths(["data/edge_attribute_master.json"]),
        "taxonomy": _sha256_paths(["spatial/poi_taxonomy.py"]),
        "tool_schemas": _sha256_paths(["agents/tools.py"]),
        "event_state": _event_state_version(),
        "runtime_code": _sha256_paths(RUNTIME_SOURCE_PATHS,
                                      suffixes={".py", ".sql"}),
        "runtime_model_config": runtime_model_config,
        "requirements": _sha256_paths(["requirements.txt"]),
        "runtime_environment": _runtime_environment_version(),
    }
    digest = hashlib.sha256()
    for name in REQUIRED_INPUTS:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((versions[name]["sha256"] or "missing").encode("ascii"))
        digest.update(b"\0")
    digest.update(versions["runtime_model_config"]["sha256"].encode("ascii"))
    digest.update((versions["event_state"].get("active_sha256") or "missing").encode("ascii"))
    versions["effective_inputs_sha256"] = digest.hexdigest()
    return versions


def _capture_diagnostic_inputs(path):
    """Persist an explicitly non-formal, non-overwriting input inventory."""
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    report = {
        "scope": "agent_main_experiment",
        "snapshot_kind": "diagnostic",
        "formal_eligible": False,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "tracked_worktree_clean": dirty == "",
        "versions": _versions(),
        "note": "Development input inventory only; not a frozen research snapshot.",
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as target:
            target.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite diagnostic snapshot: {output}") from error
    print(f"开发态输入清单已写入: {output}")
    print(f"有效输入指纹: {report['versions']['effective_inputs_sha256']}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-snapshot", action="store_true")
    parser.add_argument(
        "--capture-inputs", type=Path,
        help="write a non-formal development input inventory without running readiness tests",
    )
    args = parser.parse_args(argv)
    if args.capture_inputs:
        _capture_diagnostic_inputs(args.capture_inputs)
        return 0

    checks = _run_checks()
    deployment = _deployment_result()
    attribute_path = AUDIT / "attribute_master_data_readiness.json"
    attributes = read_json(attribute_path)
    versions = _versions()
    vision_path = Path(os.getenv("VISION_FIELD_READINESS_REPORT") or
                       AUDIT / "campus_uav_field_readiness.json")
    vision = read_json(vision_path)
    result = assess_research_readiness(
        checks, deployment, attributes, vision=vision, versions=versions,
    )
    if args.write_snapshot:
        dirty = _git("status", "--porcelain", "--untracked-files=no")
        if dirty is None:
            result["blockers"]["tracked_worktree"] = ["git_status_unavailable"]
        elif dirty:
            result["blockers"]["tracked_worktree"] = dirty.splitlines()
        if "tracked_worktree" in result["blockers"]:
            result["ready"] = False
            result["scopes"]["agent_main_experiment"]["ready"] = False
    report = {
        **result,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "campus_data": deployment,
        "attribute_master_data": attributes,
        "vision_field_readiness": vision,
        "versions": versions,
    }
    report_path = AUDIT / "agent_research_readiness.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.write_snapshot:
        if result["ready"]:
            output = ROOT / "experiments" / "research_snapshot.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"研究快照已写入: {output}")
        else:
            print("拒绝生成研究快照：仍有未通过门槛。")

    print(f"Agent 主实验就绪: {'通过' if result['ready'] else '未通过'}；报告: {report_path}")
    for name, detail in result["blockers"].items():
        print(f"- {name}: {detail}")
    for name in ("sampled_route_evidence", "preference_attributes"):
        status = result["scopes"][name]["ready"]
        print(f"- {name}: {'通过' if status else '未通过'}")
    for name, item in result["scopes"]["vision"].items():
        print(f"- vision.{name}: {'通过' if item['ready'] else '未通过'}")
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
