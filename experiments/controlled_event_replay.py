"""Independent graph-scoped scoring for simulated-event development tasks.

The scorer verifies a successful road-condition query and the full route on
the frozen eligible graph. It does not establish field passability or event
truth, and an output flag alone cannot prove explanation quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import networkx as nx

from experiments.build_controlled_event_pilot import verify_event_pilot_dataset
from experiments.controlled_event_cases import replay_event_case
from experiments.controlled_event_manifest import verify_event_manifest
from experiments.controlled_gold_contract import validate_controlled_dataset
from experiments.controlled_route_replay import score_controlled_route_case
from experiments.independent_source_snapshot import active_closed_edges, mode_edge_evidence
from experiments.scenario_runner import _validate_observation_projection


ROOT = Path(__file__).resolve().parents[1]
PILOT = Path("experiments/datasets/controlled_event_pilot_v1")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_ROUTE_TOOLS = {"plan_route", "plan_via_route", "plan_itinerary",
                "plan_tour", "plan_multimodal_route"}


def _active_event_ids(snapshot: dict[str, Any], mode: str,
                      closed: set[tuple[Any, Any, Any]]) -> list[str]:
    epoch = snapshot["as_of_epoch"]
    return sorted(item["id"] for item in snapshot["conditions"]
                  if item["action"] == "closure"
                  and mode in item["affected_modes"]
                  and item["start_time"] <= epoch
                  and (item["end_time"] == 0 or epoch < item["end_time"])
                  and (item.get("revoked_at") is None or epoch < item["revoked_at"])
                  and (item["edge"]["u"], item["edge"]["v"],
                       item["edge"]["key"]) in closed)


def score_controlled_event_case(
    case: dict[str, Any], observed: dict[str, Any] | None,
    graph: Any, anchors: dict[str, dict[str, Any]],
    edge_evidence: dict[str, set[tuple[Any, Any, Any]]],
    event_snapshot: dict[str, Any], *, template: str,
) -> dict[str, Any]:
    """Score one answer using exact active closures and source-replayed gold."""
    if case.get("category") not in {"compound_tools", "event_infeasible"}:
        raise ValueError("unsupported controlled event category")
    replay_event_case(case, graph, anchors, edge_evidence, event_snapshot,
                      template=template)
    mode = case["gold_derivation"]["parameters"]["mode"]
    closed = active_closed_edges(graph, event_snapshot, mode)
    projected = deepcopy(case)
    projected["category"] = "mode_access"
    score = score_controlled_route_case(
        projected, observed, graph, anchors,
        permitted_edge_ids=edge_evidence["permitted"],
        unknown_edge_ids=edge_evidence["unknown"],
        closed_edge_ids=closed,
    )
    if isinstance(observed, dict):
        turns = observed.get("turns")
        actual = turns[0] if isinstance(turns, list) and len(turns) == 1 else None
        trace = actual.get("tool_trace") if isinstance(actual, dict) else None
        trace = trace if isinstance(trace, list) else []
        query_indexes = [index for index, call in enumerate(trace)
                         if isinstance(call, dict)
                         and call.get("tool_name") == "list_road_conditions"
                         and call.get("outcome") == "success"]
        route_indexes = [index for index, call in enumerate(trace)
                         if isinstance(call, dict)
                         and call.get("tool_name") in _ROUTE_TOOLS]
        successful_route = any(trace[index].get("outcome") == "success"
                               for index in route_indexes)
        if not query_indexes:
            score["failures"].append("road_query_missing")
        if not successful_route:
            score["failures"].append("route_plan_trace_missing")
        if route_indexes and query_indexes and min(query_indexes) > min(route_indexes):
            score["failures"].append("road_query_after_route")
        event_ids = actual.get("road_event_ids") if isinstance(actual, dict) else None
        if not isinstance(event_ids, list) or not event_ids:
            score["failures"].append("road_event_ids_missing")
        elif (any(not isinstance(value, str) or not value for value in event_ids)
              or len(set(event_ids)) != len(event_ids)
              or sorted(event_ids) != _active_event_ids(event_snapshot, mode, closed)):
            score["failures"].append("road_event_mismatch")
        score["task_success"] = score["task_success"] and not score["failures"]
    return score


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _validate_event_projection(row: dict[str, Any]) -> None:
    """Allow only event IDs in addition to the frozen generic projection."""
    clean = deepcopy(row)
    turns = clean.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict):
                ids = turn.pop("road_event_ids", None)
                if ids is not None and (not isinstance(ids, list)
                                        or any(not isinstance(value, str) or not value
                                               for value in ids)
                                        or len(ids) != len(set(ids))):
                    raise ValueError("invalid projected road_event_ids")
    _validate_observation_projection(clean)


def run_controlled_event_replay(
    replay_path: Path, output_root: Path, run_id: str, *,
    root: Path = ROOT, dataset_path: Path | None = None,
    manifest_path: Path | None = None, spec_path: Path | None = None,
    event_path: Path | None = None,
) -> Path:
    """Validate frozen event inputs then score projected development attempts."""
    if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("invalid run_id")
    root = Path(root).resolve()
    if root != ROOT.resolve():
        raise ValueError("event replay root must match its imported code checkout")
    output_root = Path(output_root)
    result_dir = output_root / run_id
    if result_dir.exists():
        raise FileExistsError(result_dir)
    dataset_path = Path(dataset_path or root / PILOT / "scenarios_event_pilot_dev_v2.json")
    manifest_path = Path(manifest_path or root / PILOT / "source_manifest_event_pilot_dev_v1.json")
    spec_path = Path(spec_path or root / PILOT / "event_pilot_spec.json")
    event_path = Path(event_path or root / PILOT / "event_snapshots_dev_v1.json")
    dataset_bytes, manifest_bytes, spec_bytes, event_bytes, replay_bytes = (
        path.read_bytes() for path in
        (dataset_path, manifest_path, spec_path, event_path, replay_path)
    )
    dataset = json.loads(dataset_bytes)
    manifest = json.loads(manifest_bytes)
    spec = json.loads(spec_bytes)
    event_source = json.loads(event_bytes)
    if dataset.get("split") != "development":
        raise ValueError("event replay accepts development pilot only")
    validate_controlled_dataset(dataset)
    if _sha(manifest_bytes) != dataset["source_manifest_sha256"]:
        raise ValueError("event manifest bytes differ from dataset")
    if verify_event_manifest(manifest, root) != dataset["snapshot_id"]:
        raise ValueError("event manifest differs from dataset snapshot")
    inputs = manifest["inputs"]
    if (spec_path.resolve() != (root / inputs["templates"]["path"]).resolve()
            or event_path.resolve() != (root / inputs["event_snapshot"]["path"]).resolve()):
        raise ValueError("event spec or snapshot differs from frozen source paths")
    graph = nx.read_graphml(root / inputs["graph"]["path"],
                            node_type=int, force_multigraph=True)
    anchor_data = json.loads((root / inputs["anchors"]["path"]).read_text(encoding="utf-8"))
    anchors = {item["poi_id"]: item for item in anchor_data["anchors"]}
    if len(anchors) != len(anchor_data["anchors"]):
        raise ValueError("duplicate frozen anchor id")
    boundary_wkt = (root / inputs["boundary"]["path"]).read_text(encoding="utf-8")
    modes = {case["gold_derivation"]["parameters"]["mode"] for case in dataset["cases"]}
    evidence = {
        mode: mode_edge_evidence(graph, mode, require_boundary=True,
                                 boundary_wkt=boundary_wkt)
        for mode in modes
    }
    verify_event_pilot_dataset(dataset, graph, anchors, evidence, spec, event_source,
                               manifest, _sha(manifest_bytes))
    snapshots = {item["id"]: item for item in event_source["snapshots"]}
    attempts: dict[tuple[str, int], dict[str, Any]] = {}
    expected_keys = {(case["id"], attempt)
                     for case in dataset["cases"]
                     for attempt in range(1, dataset["repeats"] + 1)}
    for line_no, line in enumerate(replay_bytes.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("replay row must be an object")
        _validate_event_projection(row)
        if (not isinstance(row.get("scenario_id"), str)
                or type(row.get("attempt")) is not int):
            raise ValueError(f"invalid scenario or attempt at line {line_no}")
        key = (row["scenario_id"], row["attempt"])
        if key not in expected_keys or key in attempts:
            raise ValueError(f"unknown or duplicate scenario attempt at line {line_no}")
        attempts[key] = row
    records = []
    for case in dataset["cases"]:
        params = case["gold_derivation"]["parameters"]
        snapshot = snapshots[params["event_snapshot_id"]]
        template = spec["templates"][case["gold_derivation"]["template_id"]]["text"]
        for attempt in range(1, dataset["repeats"] + 1):
            score = score_controlled_event_case(
                case, attempts.get((case["id"], attempt)), graph, anchors,
                evidence[params["mode"]], snapshot, template=template,
            )
            records.append({"scenario_id": case["id"], "attempt": attempt,
                            "category": case["category"], "score": score})
    failures = Counter(code for record in records
                       for code in record["score"]["failures"])
    summary = {
        "run_type": "development_diagnostic", "formal_ready": False,
        "case_count": len(dataset["cases"]), "attempt_count": len(records),
        "observed_attempt_count": len(attempts),
        "task_success_count": sum(record["score"]["task_success"] for record in records),
        "failure_counts": dict(sorted(failures.items())),
        "by_category": {
            category: {
                "attempts": sum(row["category"] == category for row in records),
                "successes": sum(row["category"] == category and
                                 row["score"]["task_success"] for row in records),
            } for category in ("compound_tools", "event_infeasible")
        },
    }
    manifest_out = {
        "run_type": "development_diagnostic", "formal_ready": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_profile": "event_pilot", "source_snapshot_id": dataset["snapshot_id"],
        "dataset_sha256": _sha(dataset_bytes),
        "source_manifest_sha256": _sha(manifest_bytes),
        "spec_sha256": _sha(spec_bytes), "event_snapshot_sha256": _sha(event_bytes),
        "replay_sha256": _sha(replay_bytes),
        "scorer_sha256": _sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "limitations": [
            "Synthetic closures are not real campus road conditions.",
            "Graph correctness does not establish field passability.",
            "Explanation presence is projected, not a semantic quality judgment.",
            "These two development tasks are not a blind or formal experiment.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(exist_ok=False)
    _write_json(result_dir / "manifest.json", manifest_out)
    _write_json(result_dir / "summary.json", summary)
    (result_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return result_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "output/experiments/controlled_event_pilot")
    args = parser.parse_args(argv)
    result_dir = run_controlled_event_replay(
        args.replay, args.output_root, args.run_id, root=args.root,
    )
    print(json.dumps({"result_dir": str(result_dir),
                      "run_type": "development_diagnostic", "formal_ready": False},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
