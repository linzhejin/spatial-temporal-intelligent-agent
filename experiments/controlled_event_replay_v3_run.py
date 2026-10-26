"""Score the frozen v3 synthetic-event development packet with v2 rules.

Projected observations are diagnostics, not campus road truth or blind results.
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
from experiments.controlled_event_manifest import verify_event_manifest
from experiments.controlled_event_replay_v2 import (
    _ROUTE_TOOLS, score_controlled_event_case,
)
from experiments.controlled_gold_contract import validate_controlled_dataset
from experiments.independent_source_snapshot import mode_edge_evidence
from experiments.scenario_runner import _validate_observation_projection


PILOT = Path("experiments/datasets/controlled_event_pilot_v3")
BASE_PILOT = Path("experiments/datasets/controlled_event_pilot_v2")
ROOT = Path(__file__).resolve().parents[1]
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _runtime_source_tree_sha(root: Path) -> str:
    """Pin local Python code beyond the enumerated packet inputs for this run."""
    files = [root / name for name in ("app.py", "config.py") if (root / name).is_file()]
    for folder in ("agents", "api", "experiments", "spatial", "storage"):
        files.extend((root / folder).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(_sha(path.read_bytes().replace(b"\r\n", b"\n")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _validate_event_projection(row: dict[str, Any]) -> None:
    """Permit only event IDs beyond the generic privacy-safe projection."""
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


def _validate_v3_projection(row: dict[str, Any]) -> None:
    """Allow only the exact no-path code on a failed route tool call."""
    clean = deepcopy(row)
    turns = clean.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            trace = turn.get("tool_trace")
            if not isinstance(trace, list):
                continue
            for call in trace:
                if not isinstance(call, dict) or "error_code" not in call:
                    continue
                code = call.pop("error_code")
                digest = call.pop("request_digest", None)
                if (call.get("tool_name") not in _ROUTE_TOOLS
                        or call.get("outcome") != "tool_error"
                        or code != "graph_no_path"
                        or not isinstance(digest, str)
                        or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                    raise ValueError("invalid projected error_code")
    _validate_event_projection(clean)


def run_controlled_event_replay_v3(
    replay_path: Path, output_root: Path, run_id: str, *,
    observation_origin: str, root: Path = ROOT,
) -> Path:
    """Validate exact source hashes and write one never-overwritten run."""
    if observation_origin not in {"synthetic_projection", "unverified_projection"}:
        raise ValueError("observation_origin must label an unverified projection")
    if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("invalid run_id")
    root = Path(root).resolve()
    if root != ROOT.resolve():
        raise ValueError("event replay root must match its imported code checkout")
    result_dir = Path(output_root) / run_id
    if result_dir.exists():
        raise FileExistsError(result_dir)

    pilot = root / PILOT
    dataset_path = pilot / "scenarios_event_pilot_dev_v3.json"
    manifest_path = pilot / "source_manifest_event_pilot_dev_v1.json"
    base_manifest_path = root / BASE_PILOT / "source_manifest_event_pilot_dev_v1.json"
    spec_path = pilot / "event_pilot_spec.json"
    event_path = pilot / "event_snapshots_dev_v1.json"
    replay_path = Path(replay_path)
    dataset_bytes, manifest_bytes, base_manifest_bytes, spec_bytes, event_bytes, replay_bytes = (
        path.read_bytes() for path in
        (dataset_path, manifest_path, base_manifest_path, spec_path,
         event_path, replay_path)
    )
    dataset = json.loads(dataset_bytes)
    manifest = json.loads(manifest_bytes)
    base_manifest = json.loads(base_manifest_bytes)
    spec = json.loads(spec_bytes)
    event_source = json.loads(event_bytes)
    if dataset.get("split") != "development":
        raise ValueError("event replay accepts development pilot only")
    validate_controlled_dataset(dataset)
    if _sha(manifest_bytes) != dataset["source_manifest_sha256"]:
        raise ValueError("event manifest bytes differ from dataset")
    if verify_event_manifest(manifest, root) != dataset["snapshot_id"]:
        raise ValueError("event manifest differs from dataset snapshot")
    base_snapshot_id = verify_event_manifest(base_manifest, root)
    base_inputs = base_manifest["inputs"]
    inputs = manifest["inputs"]
    overrides = {"templates", "event_snapshot", "dataset_builder",
                 "event_scorer", "collector"}
    if any(inputs[name] != base_inputs[name] for name in inputs
           if name not in overrides):
        raise ValueError("event source differs from verified base inventory")
    if (base_inputs["dataset_builder"]["path"] !=
            "experiments/build_controlled_event_pilot.py"
            or base_inputs["collector"]["path"] !=
            "experiments/scenario_live_adapter.py"):
        raise ValueError("base event dependencies are not pinned")
    if (spec_path.resolve() != (root / inputs["templates"]["path"]).resolve()
            or event_path.resolve() != (root / inputs["event_snapshot"]["path"]).resolve()):
        raise ValueError("event spec or snapshot differs from frozen source paths")
    if (inputs["event_scorer"]["path"] !=
            "experiments/controlled_event_replay_v2.py"):
        raise ValueError("unexpected event scorer")
    graph = nx.read_graphml(root / inputs["graph"]["path"],
                            node_type=int, force_multigraph=True)
    anchor_data = json.loads((root / inputs["anchors"]["path"])
                             .read_text(encoding="utf-8"))
    anchors = {item["poi_id"]: item for item in anchor_data["anchors"]}
    if len(anchors) != len(anchor_data["anchors"]):
        raise ValueError("duplicate frozen anchor id")
    boundary_wkt = (root / inputs["boundary"]["path"]).read_text(encoding="utf-8")
    modes = {case["gold_derivation"]["parameters"]["mode"]
             for case in dataset["cases"]}
    evidence = {
        mode: mode_edge_evidence(graph, mode, require_boundary=True,
                                 boundary_wkt=boundary_wkt)
        for mode in modes
    }
    verify_event_pilot_dataset(dataset, graph, anchors, evidence, spec,
                               event_source, manifest, _sha(manifest_bytes))
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
        _validate_v3_projection(row)
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
        "campus_truth": False, "agent_result_verified": False,
        "observation_origin": observation_origin,
        "case_count": len(dataset["cases"]), "attempt_count": len(records),
        "submitted_projection_count": len(attempts),
        "projection_score_pass_count": sum(record["score"]["task_success"]
                                           for record in records),
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
        "campus_truth": False, "agent_result_verified": False,
        "observation_origin": observation_origin,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_profile": "event_pilot_v3", "source_snapshot_id": dataset["snapshot_id"],
        "base_source_snapshot_id": base_snapshot_id,
        "base_source_manifest_sha256": _sha(base_manifest_bytes),
        "dataset_sha256": _sha(dataset_bytes),
        "source_manifest_sha256": _sha(manifest_bytes),
        "spec_sha256": _sha(spec_bytes), "event_snapshot_sha256": _sha(event_bytes),
        "replay_sha256": _sha(replay_bytes),
        "scorer_sha256": _sha((root / inputs["event_scorer"]["path"])
                              .read_bytes().replace(b"\r\n", b"\n")),
        "runner_sha256": _sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "runtime_source_tree_sha256": _runtime_source_tree_sha(root),
        "event_guard_sha256": _sha((root / "experiments/event_replay_guard.py")
                                   .read_bytes().replace(b"\r\n", b"\n")),
        "collector_sha256": _sha((root / inputs["collector"]["path"])
                                 .read_bytes().replace(b"\r\n", b"\n")),
        "limitations": [
            "Synthetic closures are not real campus road conditions.",
            "Graph correctness does not establish field passability.",
            "Explanation presence is projected, not a semantic quality judgment.",
            "These two development tasks are not a blind or formal experiment.",
            "Supplied projections have no verified Agent collector provenance.",
        ],
    }
    result_dir.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--observation-origin", required=True,
                        choices=["synthetic_projection", "unverified_projection"])
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "output/experiments/controlled_event_pilot_v3")
    args = parser.parse_args(argv)
    result = run_controlled_event_replay_v3(args.replay, args.output_root,
                                            args.run_id,
                                            observation_origin=args.observation_origin)
    print(json.dumps({"result_dir": str(result),
                      "run_type": "development_diagnostic", "formal_ready": False},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
