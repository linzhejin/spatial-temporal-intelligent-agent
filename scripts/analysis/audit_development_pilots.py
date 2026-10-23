"""Project historical development pilots without treating them as formal scores.

No model is called. Missing prompt binding, origin or clarification slot remains
visible; the case's expected answer is never used to fill an observed field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agents import tools as campus_tools
from experiments.baselines.ordinary_projection import project_ordinary_turn
from experiments.baselines.ordinary_tool_agent import SYSTEM_PROMPT
from experiments.scenario_live_adapter import build_poi_lookup


ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = (
    "data/pois.json", "data/whu_road_network.graphml",
    "data/edge_overrides.json", "data/campus_review_decisions.json",
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_pilot(report: dict, *, report_sha256: str, case: dict,
                dataset_sha256: str, lookup: dict,
                component_sha256: dict[str, str], tool_schema_sha256: str,
                prompt_sha256: str) -> dict:
    if report.get("case_id") != case.get("id"):
        raise ValueError("pilot case ID differs from development case")
    if not str(report.get("status", "")).startswith("development_pilot_only"):
        raise ValueError("only development pilots may be projected")
    turns = report.get("turns")
    if not isinstance(turns, list) or len(turns) != len(case["messages"]):
        raise ValueError("pilot turn count differs from development case")
    if sum(len(turn.get("tool_trace") or []) for turn in turns) != report.get("tool_calls"):
        raise ValueError("pilot trace differs from reported tool count")
    projections = [project_ordinary_turn(turn, lookup) for turn in turns]
    input_bound = (report.get("development_dataset_sha256") == dataset_sha256
                   and report.get("case_messages_sha256") == _digest(case["messages"]))
    data_bound = report.get("data_component_sha256") == component_sha256
    gaps = []
    if not input_bound:
        gaps.append("input_hash_absent_or_mismatch")
    if not data_bound:
        gaps.append("data_component_hash_mismatch")
    if report.get("tool_schema_sha256") != tool_schema_sha256:
        gaps.append("tool_schema_hash_absent_or_mismatch")
    if report.get("baseline_system_prompt_sha256") != prompt_sha256:
        gaps.append("baseline_prompt_hash_absent_or_mismatch")
    if not report.get("resolved_model_id"):
        gaps.append("provider_response_model_unobserved")
    for index, projection in enumerate(projections):
        if projection["kind"] == "candidates" and projection["start_poi_id"] is None:
            gaps.append(f"turn_{index + 1}:candidate_origin_unobserved")
        if projection["kind"] == "clarify" and projection["clarification_slot"] is None:
            gaps.append(f"turn_{index + 1}:clarification_slot_unobserved")
    return {
        "case_id": case["id"], "condition": report.get("condition", "ordinary_tool_agent"),
        "report_sha256": report_sha256, "input_bound": input_bound,
        "data_bound": data_bound, "projection_gaps": gaps,
        "measured_usage": report.get("measured_usage"),
        "tool_calls": report["tool_calls"],
        "turns": [{
            "status": item["status"], "kind": item["kind"],
            "start_poi_id": item["start_poi_id"],
            "end_poi_id": item["end_poi_id"],
            "candidate_poi_ids": item["candidate_poi_ids"],
            "clarification_slot": item["clarification_slot"],
            "route_edge_count": len(item["route_edge_ids"]),
        } for item in projections],
        "assessment": "projection_only_not_scored",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "experiments/datasets/scenarios_dev_v2.json")
    parser.add_argument("--pois", type=Path, default=ROOT / "data/pois.json")
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if dataset.get("split") != "development":
        raise ValueError("only development cases may be audited here")
    by_id = {case["id"]: case for case in dataset["cases"]}
    pois = json.loads(args.pois.read_text(encoding="utf-8"))["pois"]
    lookup = build_poi_lookup(pois)
    component_sha256 = {name: _file_hash(ROOT / name) for name in COMPONENTS}
    rows = []
    for path in args.report:
        report = json.loads(path.read_text(encoding="utf-8"))
        case = by_id.get(report.get("case_id"))
        if case is None:
            raise ValueError("pilot case is absent from development dataset")
        rows.append(audit_pilot(
            report, report_sha256=_file_hash(path), case=case,
            dataset_sha256=_file_hash(args.dataset), lookup=lookup,
            component_sha256=component_sha256,
            tool_schema_sha256=_digest(campus_tools.TOOL_SCHEMAS),
            prompt_sha256=_digest(SYSTEM_PROMPT),
        ))
    output = {"status": "development_projection_only_not_scored",
              "dataset_sha256": _file_hash(args.dataset), "pilots": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"pilots": len(rows), "gaps": [row["projection_gaps"] for row in rows],
                      "status": output["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
