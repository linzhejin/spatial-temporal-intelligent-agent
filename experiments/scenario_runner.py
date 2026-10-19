"""Replay projected full-task observations against frozen scenario gold.

This offline CLI is deterministic and makes no service or database calls. An
external adapter may produce JSONL observations, but must project stable POI
IDs and keep independent review evidence separate from product self-reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from experiments.scenario_evaluator import evaluate_scenario, summarize, validate_dataset
from experiments.scenario_split_audit import audit_scenario_splits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments" / "protocol_v1.json"
DEFAULT_DATASET = ROOT / "experiments" / "datasets" / "scenarios_dev_v1.json"
DEFAULT_OUTPUT = ROOT / "output" / "experiments"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_OBSERVATION_FIELDS = {
    "scenario_id", "attempt", "snapshot_id", "input_messages", "turns",
    "latency_s", "tool_calls", "input_tokens", "output_tokens", "transport_error",
}
_TURN_FIELDS = {
    "status", "kind", "start_poi_id", "end_poi_id", "via_poi_ids",
    "travel_mode", "strategy", "hard_constraints", "candidate_poi_ids",
    "clarification_slot", "outputs", "route_edge_ids", "tool_trace",
    "independent_review", "unknowns",
}
_REVIEW_FIELDS = {"source", "reviewer", "evidence", "route", "hard_constraints", "outputs"}
_TOOL_TRACE_FIELDS = {"tool_name", "argument_keys", "outcome", "artifact_kind"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty() -> bool | None:
    result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return bool(result.stdout) if result.returncode == 0 else None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dependency_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("flask", "langgraph", "networkx", "requests", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_snapshot_manifest(report: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    """Check formal Agent gate and frozen input fingerprint against all cases."""
    from scripts.validate.check_agent_research_readiness import REQUIRED_INPUTS

    if report.get("scope") != "agent_main_experiment" or report.get("ready") is not True:
        raise ValueError("formal snapshot must be ready for agent_main_experiment")
    scopes = report.get("scopes") or {}
    if scopes and (scopes.get("agent_main_experiment") or {}).get("ready") is not True:
        raise ValueError("formal snapshot Agent scope is not ready")
    versions = report.get("versions") or {}
    fingerprint = versions.get("effective_inputs_sha256")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("formal snapshot has no effective input fingerprint")
    if any(case["snapshot_id"] != fingerprint for case in cases):
        raise ValueError("scenario snapshot_id differs from effective input fingerprint")
    for name in REQUIRED_INPUTS:
        item = versions.get(name) or {}
        digest = item.get("sha256")
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not isinstance(item.get("files"), list) or not item["files"]
                or item.get("missing") != []):
            raise ValueError(f"formal snapshot input {name} is incomplete")
    for name, key in (("event_state", "active_sha256"),
                      ("runtime_model_config", "sha256")):
        digest = (versions.get(name) or {}).get(key)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"formal snapshot input {name}.{key} is incomplete")
    return fingerprint


def _validate_observation_projection(row: dict[str, Any]) -> None:
    """Keep cookies, raw service payloads and credentials out of saved records."""
    if set(row) - _OBSERVATION_FIELDS:
        raise ValueError("replay has unprojected fields")
    if not isinstance(row.get("turns"), list):
        raise ValueError("replay turns must be a list")
    for turn in row["turns"]:
        if not isinstance(turn, dict) or set(turn) - _TURN_FIELDS:
            raise ValueError("replay has unprojected turn fields")
        tool_trace = turn.get("tool_trace")
        if tool_trace is not None:
            if not isinstance(tool_trace, list):
                raise ValueError("replay tool_trace must be a list or null")
            for call in tool_trace:
                if not isinstance(call, dict) or set(call) - _TOOL_TRACE_FIELDS:
                    raise ValueError("replay tool trace has unprojected fields")
                if (not isinstance(call.get("tool_name"), str) or not call["tool_name"]
                        or not isinstance(call.get("argument_keys"), list)
                        or any(not isinstance(key, str) for key in call["argument_keys"])
                        or call.get("outcome") not in {
                            "success", "tool_error", "invalid_args", "exception",
                            "skipped_after_route_verification_retry",
                            "skipped_after_route_verification_failure",
                        }
                        or call.get("artifact_kind") not in {
                            "route", "candidates", "clarify", "suggestions", "route_comparison", None,
                        }):
                    raise ValueError("replay tool trace has invalid values")
        review = turn.get("independent_review")
        if review is not None and (not isinstance(review, dict) or set(review) - _REVIEW_FIELDS):
            raise ValueError("replay has unprojected review fields")
    for field in ("latency_s", "tool_calls", "input_tokens", "output_tokens"):
        value = row.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
            raise ValueError("replay has invalid telemetry")
        if field != "latency_s" and value is not None and type(value) is not int:
            raise ValueError("replay count telemetry must be integer")
    if row.get("tool_calls") is not None and row["turns"] and all(
            isinstance(turn.get("tool_trace"), list) for turn in row["turns"]):
        traced_calls = sum(len(turn["tool_trace"]) for turn in row["turns"])
        if row["tool_calls"] != traced_calls:
            raise ValueError("replay tool_calls does not match projected tool_trace")
    if row.get("transport_error") not in {None, "timeout", "network_error", "adapter_error"}:
        raise ValueError("replay transport_error must be a category, not raw exception text")


def run_replay(dataset_path: Path, replay_path: Path, protocol_path: Path,
               output_root: Path, run_id: str, *, formal: bool = False,
               snapshot_manifest: Path | None = None,
               development_dataset_path: Path | None = None) -> Path:
    """Create a new, never-overwritten result directory for one replay run."""
    if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("invalid run_id")
    if formal and development_dataset_path is None:
        raise ValueError("formal runs require a development dataset for leakage audit")
    dataset_bytes = dataset_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    replay_bytes = replay_path.read_bytes()
    dataset = json.loads(dataset_bytes)
    protocol = json.loads(protocol_bytes)
    validate_dataset(dataset, formal=formal)
    if protocol.get("protocol_version") != dataset["protocol_version"]:
        raise ValueError("protocol and dataset versions differ")
    if formal and snapshot_manifest is None:
        raise ValueError("formal runs require a frozen snapshot manifest")
    if formal and _git_dirty() is not False:
        raise ValueError("formal runs require a clean Git worktree")
    snapshot_bytes = snapshot_manifest.read_bytes() if snapshot_manifest else None
    snapshot_fingerprint = None
    development_bytes = None
    split_audit = None
    if formal:
        snapshot_report = json.loads(snapshot_bytes)
        snapshot_fingerprint = validate_snapshot_manifest(snapshot_report, dataset["cases"])
        if snapshot_report["versions"].get("git_commit") != _git_head():
            raise ValueError("formal snapshot Git commit differs from current checkout")
        development_bytes = development_dataset_path.read_bytes()
        development_dataset = json.loads(development_bytes)
        split_audit = audit_scenario_splits(
            development_dataset, dataset,
            external_gates={
                "frozen_snapshot_manifest": snapshot_fingerprint is not None,
                "agent_readiness_gate": snapshot_report.get("ready") is True,
                "matching_git_commit": snapshot_report["versions"].get("git_commit") == _git_head(),
                "clean_tracked_worktree": _git_dirty() is False,
            },
        )
        if not split_audit["formal_ready"]:
            codes = ", ".join(item["code"] for item in split_audit["blockers"])
            raise ValueError(f"formal scenario split leakage/readiness audit failed: {codes}")
    observations: dict[tuple[str, int], dict[str, Any]] = {}
    for line_no, line in enumerate(replay_bytes.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("replay row must be an object")
        _validate_observation_projection(row)
        key = (row.get("scenario_id"), row.get("attempt"))
        if key in observations:
            raise ValueError(f"duplicate observation at line {line_no}")
        observations[key] = row
    expected_keys = {(case["id"], attempt)
                     for case in dataset["cases"]
                     for attempt in range(1, dataset.get("repeats", 1) + 1)}
    if not set(observations) <= expected_keys:
        raise ValueError("replay contains unknown scenario or attempt")
    records = []
    for case in dataset["cases"]:
        for attempt in range(1, dataset.get("repeats", 1) + 1):
            observation = observations.get((case["id"], attempt))
            records.append({"scenario_id": case["id"], "attempt": attempt,
                            "category": case["category"], "template_group": case["template_group"],
                            "od_group": case["od_group"], "snapshot_id": case["snapshot_id"],
                            "intent_labels": [turn.get("intent_label")
                                              for turn in case["expected_turns"]],
                            "observation": observation,
                            "score": evaluate_scenario(case, observation)})
    summary = summarize(records)
    output_root.mkdir(parents=True, exist_ok=True)
    result_dir = output_root / run_id
    result_dir.mkdir(exist_ok=False)
    manifest = {
        "protocol_version": dataset["protocol_version"],
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "formal_replay" if formal else "development_diagnostic",
        "dataset_split": dataset["split"],
        "dataset_version": dataset.get("version"),
        "gold_independence": "declared_manually_adjudicated" if formal else "unverified_draft_or_development",
        "dataset_sha256": _sha(dataset_bytes), "protocol_sha256": _sha(protocol_bytes),
        "replay_sha256": _sha(replay_bytes),
        "snapshot_manifest_sha256": _sha(snapshot_bytes) if snapshot_bytes is not None else None,
        "snapshot_effective_inputs_sha256": snapshot_fingerprint,
        "development_dataset_sha256": _sha(development_bytes) if development_bytes is not None else None,
        "scenario_split_audit": split_audit,
        "git_head": _git_head(), "git_tracked_worktree_dirty": _git_dirty(),
        "python_version": sys.version.split()[0], "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "dependency_versions": _dependency_versions(),
        "random_seed": (protocol.get("planned_design") or {}).get("random_seed"),
        "timezone": str(datetime.now().astimezone().tzinfo),
        "repeats": dataset.get("repeats", 1),
        "limitations": [
            "Replay scores only the supplied projected observations and independent review records.",
            "A review source label does not itself prove reviewer independence; audit evidence externally.",
            "Draft development gold and historical C1 cases are not an independent blind test.",
        ],
    }
    _write_json(result_dir / "manifest.json", manifest)
    (result_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    _write_json(result_dir / "summary.json", summary)
    return result_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--replay", type=Path, required=True,
                        help="JSONL with one projected observation per scenario and attempt")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--development-dataset", type=Path,
                        help="Required for formal runs; compared against the blind test split.")
    args = parser.parse_args(argv)
    result_dir = run_replay(args.dataset, args.replay, args.protocol,
                            args.output_root, args.run_id, formal=args.formal,
                            snapshot_manifest=args.snapshot_manifest,
                            development_dataset_path=args.development_dataset)
    print(json.dumps({"result_dir": str(result_dir), "run_type":
                      "formal_replay" if args.formal else "development_diagnostic"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
