"""Index development references by turn without treating them as formal gold.

Only visible scenario-v2 authoring drafts are accepted. The result is a triage
ledger for missing semantic, access, and field evidence, not an Agent score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.scenario_v2 import validate_dataset


ROOT = Path(__file__).resolve().parents[2]


def _index(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[Any, dict[str, Any]]:
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in indexed:
            raise ValueError(f"duplicate reference: {key}")
        indexed[key] = row
    return indexed


def _matching_keys(actual: set, expected: set, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} references are missing or stale: "
                         f"missing={len(expected - actual)}, extra={len(actual - expected)}")


def audit_turns(dataset: dict[str, Any], t1: dict[str, Any], t2: dict[str, Any],
                t2_uncertainty: dict[str, Any], multi_routes: dict[str, Any],
                t5: dict[str, Any]) -> dict[str, Any]:
    """Map each authored turn to its narrowest available development reference."""
    cases = dataset["cases"]
    t1_rows = _index(t1["cases"], ("case_id",))
    t2_rows = _index(t2["cases"], ("case_id",))
    uncertain_rows = _index(t2_uncertainty["cases"], ("case_id",))
    multi_rows = _index(multi_routes["routes"], ("case_id", "turn_index"))
    t5_rows = _index(t5["cases"], ("case_id",))
    _matching_keys(set(t1_rows), {(c["id"],) for c in cases if c["category"] == "T1"}, "T1")
    _matching_keys(set(t2_rows), {(c["id"],) for c in cases if c["category"] == "T2"}, "T2")
    _matching_keys(set(uncertain_rows), set(t2_rows), "T2 uncertainty")
    _matching_keys(
        set(multi_rows),
        {(c["id"], i) for c in cases if c["category"] in {"T3", "T4", "T5"}
         for i, turn in enumerate(c["expected_turns"]) if turn["kind"] == "route"},
        "T3–T5 route",
    )
    _matching_keys(
        set(t5_rows),
        {(c["id"],) for c in cases if c["category"] == "T5"
         and any(turn["kind"] == "candidates" for turn in c["expected_turns"])},
        "T5 selection",
    )

    turns = []
    for case in cases:
        case_id = case["id"]
        category = case["category"]
        for index, turn in enumerate(case["expected_turns"]):
            kind = turn["kind"]
            scope, verdict = "none", "not_independently_referenced"
            remaining = ["concrete_task_semantics_not_adjudicated"]
            if kind == "route":
                if category == "T1":
                    ref = t1_rows[(case_id,)]["proposed_route_reference"]
                else:
                    source = multi_rows[(case_id, index)]
                    ordered = [turn["start_poi_id"], *turn["via_poi_ids"],
                               turn["end_poi_id"]]
                    if source["ordered_poi_ids"] != ordered:
                        raise ValueError(f"{case_id} turn {index}: ordered POIs differ")
                    ref = source["reference"]
                scope = "graph_route"
                verdict = "graph_feasible" if ref["graph_feasible"] else "graph_unreachable"
                remaining.append("field_passability_unverified")
                if ref["access_uncertain"]:
                    remaining.append("endpoint_access_unverified")
            elif kind == "candidates" and category == "T2":
                t2_rows[(case_id,)]  # exact coverage was checked above
                scope = "graph_candidate_rank"
                verdict = uncertain_rows[(case_id,)]["reference_top_four_verdict"]
                if verdict not in {"pass", "unknown"}:
                    raise ValueError(f"{case_id}: unsupported candidate verdict {verdict}")
                remaining.append("poi_access_unverified" if verdict == "unknown"
                                 else "field_access_not_proven")
            elif kind == "candidates" and category == "T5":
                selected = t5_rows[(case_id,)]
                candidate_index = sum(t["kind"] == "candidates"
                                      for t in case["expected_turns"][:index])
                if candidate_index > 1:
                    raise ValueError(f"{case_id}: unsupported third candidate turn")
                in_top = (selected["selected_in_first_four"] if candidate_index == 0
                          else selected["meal_selected_in_first_four"])
                if type(in_top) is not bool:
                    raise ValueError(f"{case_id} turn {index}: selection verdict missing")
                scope = "candidate_selection_membership"
                verdict = "selected_in_first_four" if in_top else "selected_outside_first_four"
                remaining.extend(["full_candidate_order_not_referenced",
                                  "poi_access_unverified"])
            elif kind == "information":
                remaining.extend(["information_source_not_referenced",
                                  "client_route_preservation_not_referenced"])
            elif kind == "clarify":
                remaining.append("clarification_rule_needs_case_application")
            turns.append({
                "case_id": case_id,
                "category": category,
                "turn_index": index,
                "kind": kind,
                "reference_scope": scope,
                "reference_verdict": verdict,
                "formal_adjudicated": case["adjudication"]["status"] == "adjudicated",
                "remaining_evidence": remaining,
            })

    return {
        "status": "development_evidence_coverage_not_gold",
        "case_count": len(cases),
        "turn_count": len(turns),
        "formal_adjudicated_cases": sum(
            c["adjudication"]["status"] == "adjudicated" for c in cases),
        "by_reference_scope": dict(sorted(Counter(t["reference_scope"] for t in turns).items())),
        "by_category": dict(sorted(Counter(t["category"] for t in turns).items())),
        "t2_rank_verdicts": dict(sorted(Counter(
            t["reference_verdict"] for t in turns if t["category"] == "T2").items())),
        "turns": turns,
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=ROOT / "experiments/datasets/scenarios_test_v2_diverse_draft.json")
    for name in ("t1", "t2", "t2-uncertainty", "multi-routes", "t5"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence ledger: {args.output}")
    paths = {
        "draft": args.draft, "t1": args.t1, "t2": args.t2,
        "t2_uncertainty": args.t2_uncertainty,
        "multi_routes": args.multi_routes, "t5": args.t5,
    }
    input_sha256 = {name: _digest(path) for name, path in paths.items()}
    reports = {name: _load(path) for name, path in paths.items()}
    draft = reports["draft"]
    if (draft.get("version") != "scenarios-test-v2-diverse-authoring-draft"
            or not draft.get("authoring_status")):
        raise ValueError("only visible diverse authoring drafts are accepted")
    validate_dataset(draft)
    for name in ("t1", "t2", "multi_routes", "t5"):
        source = reports[name].get("source_sha256", {})
        if source.get("draft") != input_sha256["draft"]:
            raise ValueError(f"{name}: reference belongs to a different draft")
        for component in ("poi", "graph", "overrides", "reviews"):
            keys = ("poi", "pois") if component == "poi" else (component,)
            digests = {source[key] for key in keys if key in source}
            if digests != {draft.get(f"source_{component}_sha256")}:
                raise ValueError(f"{name}: {component} input differs from the draft")
    uncertainty_source = reports["t2_uncertainty"]
    if (uncertainty_source.get("reference_sha256") != input_sha256["t2"]
            or uncertainty_source.get("poi_sha256") != draft.get("source_poi_sha256")
            or uncertainty_source.get("graph_sha256") != draft.get("source_graph_sha256")):
        raise ValueError("T2 uncertainty audit belongs to different reference inputs")
    result = audit_turns(draft, reports["t1"], reports["t2"],
                         reports["t2_uncertainty"], reports["multi_routes"],
                         reports["t5"])
    if any(_digest(path) != input_sha256[name] for name, path in paths.items()):
        raise ValueError("an input changed during audit")
    result["input_sha256"] = input_sha256
    result["limitations"] = [
        "All cases are visible authoring drafts, not held-out test cases.",
        "Graph references share the authored POI and network inputs with the product.",
        "A graph route, rank, or selection-membership reference is not field access or semantic gold.",
        "No Agent response is scored by this ledger.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "case_count", "turn_count", "formal_adjudicated_cases",
        "by_reference_scope", "t2_rank_verdicts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
