"""Score two-turn route revisions against a frozen, independent graph oracle.

Only development diagnostics are supported. A passing score is graph-scoped,
not evidence that the route's final metre or field access was verified.
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

from experiments.build_controlled_multi_turn_pilot import verify_multi_turn_pilot_dataset
from experiments.controlled_gold_contract import validate_controlled_dataset
from experiments.controlled_multi_turn_cases import score_controlled_multi_turn_case
from experiments.controlled_source_manifest import verify_source_manifest
from experiments.independent_source_snapshot import mode_edge_evidence
from experiments.scenario_runner import _validate_observation_projection


ROOT = Path(__file__).resolve().parents[1]
PILOT = Path("experiments/datasets/controlled_multi_turn_pilot_v1")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def run_controlled_multi_turn_replay(
    replay_path: Path, output_root: Path, run_id: str, *,
    root: Path = ROOT, dataset_path: Path | None = None,
    manifest_path: Path | None = None, spec_path: Path | None = None,
) -> Path:
    """Verify sources and score all attempts, counting absent rows as failures."""
    if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("invalid run_id")
    root = Path(root).resolve()
    output_root = Path(output_root)
    result_dir = output_root / run_id
    if result_dir.exists():
        raise FileExistsError(result_dir)
    dataset_path = Path(dataset_path or root / PILOT / "scenarios_multi_turn_dev_v2.json")
    manifest_path = Path(manifest_path or root / PILOT / "source_manifest_multi_turn_dev_v1.json")
    spec_path = Path(spec_path or root / PILOT / "multi_turn_pilot_spec.json")
    dataset_bytes, manifest_bytes, spec_bytes, replay_bytes = (
        path.read_bytes() for path in (dataset_path, manifest_path, spec_path, replay_path)
    )
    dataset, manifest, spec = (json.loads(data) for data in
                               (dataset_bytes, manifest_bytes, spec_bytes))
    if dataset.get("split") != "development" or manifest.get("profile") != "route_pilot":
        raise ValueError("multi-turn replay accepts development pilot only")
    validate_controlled_dataset(dataset, formal=False)
    if any(case.get("category") != "multi_turn_revision" for case in dataset["cases"]):
        raise ValueError("multi-turn replay accepts revision cases only")
    if _sha(manifest_bytes) != dataset["source_manifest_sha256"]:
        raise ValueError("source manifest bytes differ from dataset")
    if verify_source_manifest(manifest, root) != dataset["snapshot_id"]:
        raise ValueError("source manifest differs from dataset snapshot")
    inputs = manifest["inputs"]
    recorded_spec = (root / inputs["templates"]["path"]).resolve()
    if spec_path.resolve() != recorded_spec:
        raise ValueError("multi-turn specification differs from frozen source")
    graph = nx.read_graphml(root / inputs["graph"]["path"],
                            node_type=int, force_multigraph=True)
    anchor_data = json.loads((root / inputs["anchors"]["path"]).read_text(encoding="utf-8"))
    anchors = {item["poi_id"]: item for item in anchor_data["anchors"]}
    if len(anchors) != len(anchor_data["anchors"]):
        raise ValueError("duplicate frozen anchor id")
    boundary_wkt = (root / inputs["boundary"]["path"]).read_text(encoding="utf-8")
    modes = {case["gold_derivation"]["parameters"][name]
             for case in dataset["cases"]
             for name in ("initial_mode", "revised_mode")}
    evidence = {mode: mode_edge_evidence(
        graph, mode, require_boundary=True, boundary_wkt=boundary_wkt,
    ) for mode in modes}
    verify_multi_turn_pilot_dataset(
        dataset, graph, anchors, spec, manifest, _sha(manifest_bytes), evidence,
    )

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
        key = (row["scenario_id"], row["attempt"])
        if key not in expected_keys:
            raise ValueError(f"unknown scenario or attempt at line {line_no}")
        if key in attempts:
            raise ValueError(f"duplicate scenario attempt at line {line_no}")
        attempts[key] = row

    records = []
    for case in dataset["cases"]:
        for attempt in range(1, dataset["repeats"] + 1):
            observation = attempts.get((case["id"], attempt))
            score = score_controlled_multi_turn_case(
                case, observation, graph, anchors, evidence,
            )
            records.append({"scenario_id": case["id"], "attempt": attempt,
                            "category": case["category"], "score": score})
    failure_counts = Counter(code for record in records
                             for code in record["score"]["failures"])
    summary = {
        "run_type": "development_diagnostic", "case_count": len(dataset["cases"]),
        "attempt_count": len(records), "observed_attempt_count": len(attempts),
        "task_success_count": sum(record["score"]["task_success"] for record in records),
        "failure_counts": dict(sorted(failure_counts.items())),
        "formal_ready": False,
    }
    manifest_out = {
        "run_type": "development_diagnostic", "formal_ready": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_profile": "route_pilot", "source_snapshot_id": dataset["snapshot_id"],
        "dataset_sha256": _sha(dataset_bytes),
        "source_manifest_sha256": _sha(manifest_bytes),
        "spec_sha256": _sha(spec_bytes), "replay_sha256": _sha(replay_bytes),
        "scorer_sha256": _sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "scorer_content_encoding": "utf8_lf_canonical_v1",
        "limitations": [
            "Only frozen graph validity is checked; field passability is unverified.",
            "These two development cases are neither blind nor formal results.",
            "No model attempt was run by this scorer; a replay must be supplied.",
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
                        default=ROOT / "output/experiments/controlled_multi_turn_pilot")
    args = parser.parse_args(argv)
    result_dir = run_controlled_multi_turn_replay(
        args.replay, args.output_root, args.run_id, root=args.root,
    )
    print(json.dumps({"result_dir": str(result_dir),
                      "run_type": "development_diagnostic", "formal_ready": False},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
