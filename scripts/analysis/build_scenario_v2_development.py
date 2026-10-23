"""Prepare a source-linked v2 development bank; it remains unadjudicated."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "experiments/datasets/scenarios_dev_v1.json"
DEFAULT_OUTPUT = ROOT / "experiments/datasets/scenarios_dev_v2.json"


_CLARIFICATIONS = (
    ("去打球", "sport", None, None, {}),
    ("从玉兰2门出发去打球，别走台阶", "sport", "poi_449", "walk", {"avoid_steps": True}),
    ("我在信息学部，想找地方打球", "sport", None, None, {}),
    ("骑车去打球", "sport", None, "bike", {}),
    ("去运动", "activity", None, None, {}),
    ("我想从玉兰2门出发运动", "activity", "poi_449", None, {}),
    ("我从桂园操场出发，帮我规划一条路线", "destination", None, None, {}),
    ("我想步行过去，别走台阶", "destination", None, "walk", {"avoid_steps": True}),
)


def _clarification_case(index: int, specification: tuple) -> dict:
    prompt, slot, start, mode, constraints = specification
    return {
        "id": f"v2_dev_clarify_{index:02d}", "category": "T3",
        "template_group": f"v2_clarification_{index:02d}",
        "od_group": f"development_clarification_{index:02d}",
        "snapshot_id": "PENDING_FROZEN_SNAPSHOT",
        "source_group": "v2_hand_authored_clarification_development",
        "adjudication": {"status": "draft", "method": None, "reviewer": None,
                         "evidence": []},
        "messages": [{"role": "user", "content": prompt}],
        "expected_turns": [{
            "status": "needs_input", "kind": "clarify", "feasibility": "not_applicable",
            "start_poi_id": start, "end_poi_id": None, "via_poi_ids": [],
            "travel_mode": mode, "strategy": None,
            "hard_constraints": constraints, "required_outputs": [],
            "candidate_poi_ids": [], "clarification_slot": slot,
            "allowed_unknowns": [], "intent_label": None, "expected_tools": None,
        }],
    }


def build_development_bank(source_path: Path) -> dict:
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    selected = []
    for category, target, limit in (
        ("mode_access", "T1", 10),
        ("multi_via_itinerary", "T1", 2),
        ("place_recommendation", "T2", 10),
        ("multi_turn_revision", "T4", 10),
        ("compound_tools", "T5", 10),
        ("event_infeasible", "T5", 10),
    ):
        matches = [case for case in source["cases"] if case["category"] == category]
        if len(matches) < limit:
            raise ValueError(f"source bank lacks {category} cases")
        for item in matches[:limit]:
            case = deepcopy(item)
            case["source_case_id"] = case["id"]
            case["source_group"] = "scenarios_dev_v1_authoring_bank"
            case["category"] = "T3" if any(turn["kind"] == "clarify"
                                           for turn in case["expected_turns"]) else target
            selected.append(case)
    selected.extend(_clarification_case(index, spec)
                    for index, spec in enumerate(_CLARIFICATIONS, 1))
    if len(selected) != 60:
        raise AssertionError("development bank must contain 60 cases")
    return {
        "protocol_version": "scenario-v2", "version": "scenarios-dev-v2-draft60",
        "split": "development", "repeats": 1,
        "authoring_status": "DRAFT ONLY: expected turns are hypotheses, not formal gold.",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "cases": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    from experiments.scenario_v2 import validate_dataset

    bank = build_development_bank(args.source)
    validate_dataset(bank)
    args.output.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(bank["cases"]),
                      "status": "draft"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
