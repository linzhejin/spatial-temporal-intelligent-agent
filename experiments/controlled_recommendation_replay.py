"""Independently score frozen, development-only place recommendation attempts.

The gold pool is rebuilt from reviewed POI activities and frozen graph edges.
This does not certify venue opening, last-metre access, or field passability.
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

from experiments.build_controlled_recommendation_pilot import verify_recommendation_pilot_dataset
from experiments.controlled_gold_contract import validate_controlled_dataset
from experiments.controlled_recommendation_gold import rank_reachable_activity_pois
from experiments.controlled_recommendation_manifest import verify_recommendation_manifest
from experiments.independent_source_snapshot import mode_edge_evidence
from experiments.scenario_runner import _validate_observation_projection


ROOT = Path(__file__).resolve().parents[1]
PILOT = Path("experiments/datasets/controlled_recommendation_pilot_v2")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def score_controlled_recommendation_case(
    case: dict[str, Any], observed: dict[str, Any] | None,
    graph: Any, anchors: dict[str, dict[str, Any]], pois: list[dict[str, Any]], *,
    permitted_edge_ids: set[tuple[Any, Any, Any]],
    unknown_edge_ids: set[tuple[Any, Any, Any]],
    closed_edge_ids: set[tuple[Any, Any, Any]],
) -> dict[str, Any]:
    """Compare a projected attempt against the complete eligible POI pool."""
    if case.get("category") != "place_recommendation":
        raise ValueError("controlled recommendation scorer only supports place recommendations")
    turns = case.get("expected_turns")
    derivation = case.get("gold_derivation")
    if (not isinstance(turns, list) or len(turns) != 1
            or not isinstance(derivation, dict)
            or not isinstance(derivation.get("parameters"), dict)
            or not isinstance(derivation.get("oracle_result"), dict)):
        raise ValueError("controlled recommendation case lacks derivation")
    expected = turns[0]
    params = derivation["parameters"]
    gold = rank_reachable_activity_pois(
        graph, pois, anchors, start_poi_id=params["start_poi_id"],
        activity=params["activity"], mode=params["mode"], limit=params["limit"],
        permitted_edge_ids=permitted_edge_ids,
        unknown_edge_ids=unknown_edge_ids, closed_edge_ids=closed_edge_ids,
    )
    if gold != derivation["oracle_result"] or gold["candidate_poi_ids"] != expected["candidate_poi_ids"]:
        raise ValueError("recommendation gold differs from independent source calculation")

    failures: list[str] = []
    unknowns: list[str] = []

    def fail(code: str) -> None:
        if code not in failures:
            failures.append(code)

    def result() -> dict[str, Any]:
        return {"task_success": not failures and not unknowns,
                "failures": failures, "unknowns": unknowns,
                "eligible_count": len(gold["all_reachable_poi_ids"]),
                "expected_candidate_poi_ids": gold["candidate_poi_ids"]}

    if not isinstance(observed, dict):
        fail("missing_observation")
        return result()
    if observed.get("scenario_id") != case.get("id"):
        fail("scenario_mismatch")
    if observed.get("snapshot_id") != case.get("snapshot_id"):
        fail("snapshot_mismatch")
    if observed.get("input_messages") != case.get("messages"):
        fail("input_mismatch")
    if observed.get("transport_error"):
        fail("transport_error")
    actual_turns = observed.get("turns")
    if (not isinstance(actual_turns, list) or len(actual_turns) != 1
            or not isinstance(actual_turns[0], dict)):
        fail("turn_count_mismatch")
        return result()
    actual = actual_turns[0]
    for key, code in (
        ("status", "wrong_status"), ("kind", "wrong_kind"),
        ("start_poi_id", "wrong_start"), ("end_poi_id", "unexpected_destination"),
        ("via_poi_ids", "unexpected_via"), ("travel_mode", "wrong_travel_mode"),
        ("clarification_slot", "unexpected_clarification"),
    ):
        if actual.get(key) != expected[key]:
            fail(code)
    if actual.get("route_edge_ids") != []:
        fail("unexpected_route")
    if not isinstance(actual.get("outputs"), list) or "candidates" not in actual["outputs"]:
        fail("missing_candidates_output")

    # The product ranks graph distance plus endpoint snap distances. When that
    # may reorder the frozen graph-only gold, leave the ordering ungraded.
    full_pool = gold["all_reachable_poi_ids"]
    start_snap = anchors[params["start_poi_id"]].get("snap_distance_m")
    snaps = [anchors[poi_id].get("snap_distance_m") for poi_id in full_pool]
    if (not isinstance(start_snap, (int, float))
            or any(not isinstance(value, (int, float)) for value in snaps)):
        unknowns.append("snap_distance_unavailable")
    else:
        total_order = sorted(
            full_pool,
            key=lambda poi_id: (gold["graph_distance_m"][poi_id] + start_snap
                                + anchors[poi_id]["snap_distance_m"], poi_id),
        )
        if total_order[:params["limit"]] != gold["candidate_poi_ids"]:
            unknowns.append("distance_metric_order_ambiguous")

    actual_ids = actual.get("candidate_poi_ids")
    if not isinstance(actual_ids, list) or any(not isinstance(poi_id, str) for poi_id in actual_ids):
        fail("candidate_identity_missing")
    elif not unknowns:
        if len(actual_ids) != params["limit"] or set(actual_ids) != set(gold["candidate_poi_ids"]):
            fail("wrong_candidate_set")
        elif actual_ids != gold["candidate_poi_ids"]:
            fail("wrong_candidate_order")
    return result()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def run_controlled_recommendation_replay(
    replay_path: Path, output_root: Path, run_id: str, *,
    root: Path = ROOT, dataset_path: Path | None = None,
    manifest_path: Path | None = None, spec_path: Path | None = None,
) -> Path:
    """Validate frozen inputs, then count missing attempts as failures."""
    if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("invalid run_id")
    root = Path(root).resolve()
    result_dir = Path(output_root) / run_id
    if result_dir.exists():
        raise FileExistsError(result_dir)
    dataset_path = Path(dataset_path or root / PILOT / "scenarios_recommendation_dev_v2.json")
    manifest_path = Path(manifest_path or root / PILOT / "source_manifest_recommendation_dev_v1.json")
    spec_path = Path(spec_path or root / PILOT / "recommendation_pilot_spec.json")
    dataset_bytes, manifest_bytes, spec_bytes, replay_bytes = (
        path.read_bytes() for path in (dataset_path, manifest_path, spec_path, replay_path)
    )
    dataset, manifest, spec = (json.loads(data) for data in
                               (dataset_bytes, manifest_bytes, spec_bytes))
    if dataset.get("split") != "development" or manifest.get("profile") != "recommendation_pilot":
        raise ValueError("recommendation replay accepts development pilot only")
    validate_controlled_dataset(dataset, formal=False)
    if _sha(manifest_bytes) != dataset["source_manifest_sha256"]:
        raise ValueError("source manifest bytes differ from dataset")
    if verify_recommendation_manifest(manifest, root) != dataset["snapshot_id"]:
        raise ValueError("source manifest differs from dataset snapshot")
    inputs = manifest["inputs"]
    recorded_spec = (root / inputs["templates"]["path"]).resolve()
    if spec_path.resolve() != recorded_spec:
        raise ValueError("recommendation specification differs from frozen source")
    graph = nx.read_graphml(root / inputs["graph"]["path"],
                            node_type=int, force_multigraph=True)
    poi_data = json.loads((root / inputs["poi"]["path"]).read_text(encoding="utf-8"))
    if not isinstance(poi_data, dict) or not isinstance(poi_data.get("pois"), list):
        raise ValueError("frozen POI catalog lacks pois")
    pois = poi_data["pois"]
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
    if modes != {"walk"}:
        raise ValueError("recommendation pilot supports walk only")
    verify_recommendation_pilot_dataset(
        dataset, graph, anchors, pois, spec, manifest,
        permitted_edge_ids=evidence["walk"]["permitted"],
        unknown_edge_ids=evidence["walk"]["unknown"],
    )

    expected_keys = {(case["id"], attempt)
                     for case in dataset["cases"]
                     for attempt in range(1, dataset["repeats"] + 1)}
    attempts: dict[tuple[str, int], dict[str, Any]] = {}
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
        key = (row["scenario_id"], row["attempt"])
        if key not in expected_keys:
            raise ValueError(f"unknown scenario or attempt at line {line_no}")
        if key in attempts:
            raise ValueError(f"duplicate scenario attempt at line {line_no}")
        attempts[key] = row

    records = []
    for case in dataset["cases"]:
        for attempt in range(1, dataset["repeats"] + 1):
            score = score_controlled_recommendation_case(
                case, attempts.get((case["id"], attempt)), graph, anchors, pois,
                permitted_edge_ids=evidence["walk"]["permitted"],
                unknown_edge_ids=evidence["walk"]["unknown"],
                closed_edge_ids=set(),
            )
            records.append({"scenario_id": case["id"], "attempt": attempt,
                            "category": case["category"], "score": score})
    failure_counts = Counter(code for record in records
                             for code in record["score"]["failures"])
    unknown_counts = Counter(code for record in records
                             for code in record["score"]["unknowns"])
    summary = {
        "run_type": "development_diagnostic", "formal_ready": False,
        "case_count": len(dataset["cases"]), "attempt_count": len(records),
        "observed_attempt_count": len(attempts),
        "task_success_count": sum(record["score"]["task_success"] for record in records),
        "failure_counts": dict(sorted(failure_counts.items())),
        "unknown_counts": dict(sorted(unknown_counts.items())),
    }
    manifest_out = {
        "run_type": "development_diagnostic", "formal_ready": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_profile": "recommendation_pilot",
        "source_snapshot_id": dataset["snapshot_id"],
        "dataset_sha256": _sha(dataset_bytes),
        "source_manifest_sha256": _sha(manifest_bytes),
        "spec_sha256": _sha(spec_bytes), "replay_sha256": _sha(replay_bytes),
        "scorer_sha256": _sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "scorer_content_encoding": "utf8_lf_canonical_v1",
        "limitations": [
            "Only frozen reviewed POIs and graph distance are checked.",
            "Last-metre access and venue opening are unverified.",
            "These development cases are not a blind or formal experiment.",
        ],
    }
    output_root = Path(output_root)
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
                        default=ROOT / "output/experiments/controlled_recommendation_pilot")
    args = parser.parse_args(argv)
    result_dir = run_controlled_recommendation_replay(
        args.replay, args.output_root, args.run_id, root=args.root,
    )
    print(json.dumps({"result_dir": str(result_dir),
                      "run_type": "development_diagnostic", "formal_ready": False},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
