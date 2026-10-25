"""Independent graph-scoped scoring for source-derived route observations.

This scores one controlled route task from projected facts and frozen edge
evidence. It never trusts a product verifier flag or a claimed review. Passing
here establishes only correctness within the frozen graph, not field access.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import networkx as nx

from experiments.build_controlled_route_pilot import verify_route_pilot_dataset
from experiments.controlled_gold_contract import validate_controlled_dataset
from experiments.controlled_source_manifest import verify_source_manifest
from experiments.independent_route_oracle import inspect_route_edges
from experiments.independent_source_snapshot import mode_edge_evidence
from experiments.scenario_runner import _validate_observation_projection


_ROUTE_CATEGORIES = {"mode_access", "multi_via_itinerary"}
ROOT = Path(__file__).resolve().parents[1]
PILOT = Path("experiments/datasets/controlled_route_pilot_v1")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def score_controlled_route_case(
    case: dict[str, Any], observed: dict[str, Any] | None,
    graph: Any, anchors: dict[str, dict[str, Any]], *,
    permitted_edge_ids: set[tuple[Any, Any, Any]],
    unknown_edge_ids: set[tuple[Any, Any, Any]],
    closed_edge_ids: set[tuple[Any, Any, Any]],
) -> dict[str, Any]:
    """Compare a projected attempt with one independently derived route case."""
    if case.get("category") not in _ROUTE_CATEGORIES:
        raise ValueError("controlled route scorer only supports route categories")
    expected_turns = case.get("expected_turns")
    derivation = case.get("gold_derivation")
    if (not isinstance(expected_turns, list) or len(expected_turns) != 1
            or not isinstance(derivation, dict)
            or not isinstance(derivation.get("parameters"), dict)
            or not isinstance(derivation.get("oracle_result"), dict)):
        raise ValueError("controlled route case lacks derivation")
    expected = expected_turns[0]
    params = derivation["parameters"]
    gold = derivation["oracle_result"]
    if gold.get("verdict") not in {"feasible", "infeasible"}:
        raise ValueError("controlled route gold is not decidable")

    failures: list[str] = []
    unknowns: list[str] = []
    route_issues: list[str] = []
    route_verdict = "not_applicable"
    length_m = None

    def fail(code: str) -> None:
        if code not in failures:
            failures.append(code)

    if not isinstance(observed, dict):
        fail("missing_observation")
        return {"task_success": False, "failures": failures, "unknowns": unknowns,
                "route_verdict": route_verdict, "route_issues": route_issues,
                "length_m": length_m}
    if observed.get("scenario_id") != case.get("id"):
        fail("scenario_mismatch")
    if observed.get("snapshot_id") != case.get("snapshot_id"):
        fail("snapshot_mismatch")
    if observed.get("input_messages") != case.get("messages"):
        fail("input_mismatch")
    if observed.get("transport_error"):
        fail("transport_error")
    turns = observed.get("turns")
    if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], dict):
        fail("turn_count_mismatch")
        return {"task_success": False, "failures": failures, "unknowns": unknowns,
                "route_verdict": route_verdict, "route_issues": route_issues,
                "length_m": length_m}
    actual = turns[0]
    for key, code in (
        ("status", "wrong_status"), ("kind", "wrong_kind"),
        ("start_poi_id", "wrong_poi"), ("end_poi_id", "wrong_poi"),
        ("via_poi_ids", "wrong_poi_order"),
        ("travel_mode", "wrong_travel_mode"), ("strategy", "wrong_strategy"),
        ("candidate_poi_ids", "unexpected_candidates"),
        ("clarification_slot", "unexpected_clarification"),
    ):
        if actual.get(key) != expected[key]:
            fail(code)
    actual_constraints = actual.get("hard_constraints")
    if not isinstance(actual_constraints, dict):
        fail("dropped_constraint")
    else:
        for key, value in expected["hard_constraints"].items():
            if actual_constraints.get(key) != value:
                fail("dropped_constraint")
        if any(key not in expected["hard_constraints"] and value
               for key, value in actual_constraints.items()):
            fail("unexpected_constraint")
    outputs = actual.get("outputs")
    if (not isinstance(outputs, list) or any(not isinstance(value, str) for value in outputs)
            or not set(expected["required_outputs"]) <= set(outputs)):
        fail("missing_output")
    reported_unknowns = actual.get("unknowns")
    if (not isinstance(reported_unknowns, list)
            or any(value not in expected["allowed_unknowns"] for value in reported_unknowns)):
        fail("unexpected_unknown")

    edges = actual.get("route_edge_ids")
    if gold["verdict"] == "infeasible":
        if edges != []:
            fail("unexpected_route_edges")
    elif not isinstance(edges, list) or not edges:
        fail("route_edges_missing")
    else:
        start = anchors.get(params["start_poi_id"])
        end = anchors.get(params["end_poi_id"])
        via = [anchors.get(poi_id) for poi_id in params["via_poi_ids"]]
        if not isinstance(start, dict) or not isinstance(end, dict) or any(
                not isinstance(anchor, dict) for anchor in via):
            raise ValueError("frozen route anchors missing")
        check = inspect_route_edges(
            graph, edges, start_node=start["node_id"], end_node=end["node_id"],
            via_nodes=[anchor["node_id"] for anchor in via], mode=params["mode"],
            permitted_edge_ids=permitted_edge_ids,
            uncertain_edge_ids=unknown_edge_ids,
            closed_edge_ids=closed_edge_ids,
            avoid_steps=params["avoid_steps"],
            max_distance_m=params.get("max_distance_m"),
        )
        route_verdict = check["verdict"]
        route_issues = check["issues"]
        length_m = check["length_m"]
        if route_verdict == "fail":
            fail("route_invalid")
        elif route_verdict == "unknown":
            unknowns.extend(check["unknowns"])
        elif (expected["strategy"] == "shortest"
              and length_m > gold["length_m"] + 0.001):
            fail("not_shortest")
    return {"task_success": not failures and not unknowns,
            "failures": failures, "unknowns": unknowns,
            "route_verdict": route_verdict, "route_issues": route_issues,
            "length_m": length_m}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def run_controlled_route_replay(
    replay_path: Path, output_root: Path, run_id: str, *,
    root: Path = ROOT,
    dataset_path: Path | None = None,
    manifest_path: Path | None = None,
    spec_path: Path | None = None,
) -> Path:
    """Score a frozen development packet; missing attempts count as failures."""
    if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("invalid run_id")
    root = Path(root).resolve()
    output_root = Path(output_root)
    result_dir = output_root / run_id
    if result_dir.exists():
        raise FileExistsError(result_dir)
    dataset_path = Path(dataset_path or root / PILOT / "scenarios_route_pilot_dev_v2.json")
    manifest_path = Path(manifest_path or root / PILOT / "source_manifest_route_pilot_dev_v1.json")
    spec_path = Path(spec_path or root / PILOT / "route_pilot_spec.json")
    dataset_bytes, manifest_bytes, spec_bytes, replay_bytes = (
        path.read_bytes() for path in (dataset_path, manifest_path, spec_path, replay_path)
    )
    dataset = json.loads(dataset_bytes)
    manifest = json.loads(manifest_bytes)
    spec = json.loads(spec_bytes)
    if dataset.get("split") != "development" or manifest.get("profile") != "route_pilot":
        raise ValueError("route replay accepts development pilot only")
    validate_controlled_dataset(dataset)
    if _sha(manifest_bytes) != dataset["source_manifest_sha256"]:
        raise ValueError("source manifest bytes differ from dataset")
    if verify_source_manifest(manifest, root) != dataset["snapshot_id"]:
        raise ValueError("source manifest differs from dataset snapshot")
    recorded_spec = (root / manifest["inputs"]["templates"]["path"]).resolve()
    if spec_path.resolve() != recorded_spec:
        raise ValueError("route specification differs from frozen template source")

    inputs = manifest["inputs"]
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
    verify_route_pilot_dataset(dataset, graph, anchors, spec, manifest,
                               _sha(manifest_bytes), evidence)

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
        _validate_observation_projection(row)
        if (not isinstance(row.get("scenario_id"), str)
                or type(row.get("attempt")) is not int):
            raise ValueError(f"invalid scenario or attempt at line {line_no}")
        key = (row.get("scenario_id"), row.get("attempt"))
        if key not in expected_keys:
            raise ValueError(f"unknown scenario or attempt at line {line_no}")
        if key in attempts:
            raise ValueError(f"duplicate scenario attempt at line {line_no}")
        attempts[key] = row

    records = []
    for case in dataset["cases"]:
        mode = case["gold_derivation"]["parameters"]["mode"]
        for attempt in range(1, dataset["repeats"] + 1):
            observation = attempts.get((case["id"], attempt))
            score = score_controlled_route_case(
                case, observation, graph, anchors,
                permitted_edge_ids=evidence[mode]["permitted"],
                unknown_edge_ids=evidence[mode]["unknown"],
                closed_edge_ids=set(),
            )
            records.append({"scenario_id": case["id"], "attempt": attempt,
                            "category": case["category"], "score": score})
    failure_counts = Counter(code for record in records
                             for code in record["score"]["failures"])
    summary = {
        "run_type": "development_diagnostic",
        "case_count": len(dataset["cases"]),
        "attempt_count": len(records),
        "observed_attempt_count": len(attempts),
        "task_success_count": sum(record["score"]["task_success"] for record in records),
        "failure_counts": dict(sorted(failure_counts.items())),
        "by_category": {
            category: {
                "attempts": sum(record["category"] == category for record in records),
                "successes": sum(record["category"] == category
                                 and record["score"]["task_success"] for record in records),
            } for category in sorted(_ROUTE_CATEGORIES)
        },
        "formal_ready": False,
    }
    manifest_out = {
        "run_type": "development_diagnostic", "formal_ready": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_profile": "route_pilot",
        "source_snapshot_id": dataset["snapshot_id"],
        "dataset_sha256": _sha(dataset_bytes),
        "source_manifest_sha256": _sha(manifest_bytes),
        "spec_sha256": _sha(spec_bytes), "replay_sha256": _sha(replay_bytes),
        "scorer_sha256": _sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "scorer_content_encoding": "utf8_lf_canonical_v1",
        "limitations": [
            "Only frozen graph validity is checked; field passability is unverified.",
            "These six development cases are not a blind or formal experiment.",
            "Explanation presence is projected, not semantic explanation quality.",
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
                        default=ROOT / "output/experiments/controlled_route_pilot")
    args = parser.parse_args(argv)
    result_dir = run_controlled_route_replay(
        args.replay, args.output_root, args.run_id, root=args.root,
    )
    print(json.dumps({"result_dir": str(result_dir),
                      "run_type": "development_diagnostic", "formal_ready": False},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
