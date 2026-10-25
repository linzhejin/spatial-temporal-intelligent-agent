"""A generated gold label needs inspectable inputs, not only a plausible answer."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.controlled_gold_contract import validate_controlled_dataset
from experiments.controlled_source_manifest import REQUIRED_INPUTS, make_source_manifest
from experiments.scenario_evaluator import CATEGORIES


_HASH = "a" * 64
_OTHER_HASH = "b" * 64


def _digest_json(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def _dataset():
    parameters = {"start_poi_id": "poi_a", "end_poi_id": "poi_b",
                  "via_poi_ids": [], "travel_mode": "walk", "avoid_steps": True}
    return {
        "protocol_version": "scenario-v2",
        "gold_layer": "controlled_graph",
        "split": "development",
        "repeats": 1,
        "source_manifest_sha256": _HASH,
        "snapshot_id": _OTHER_HASH,
        "cases": [{
            "id": "controlled_route_1",
            "category": "mode_access",
            "template_group": "controlled_route_group",
            "od_group": "north_south",
            "snapshot_id": _OTHER_HASH,
            "messages": [{"role": "user", "content": "从甲到乙步行，不走台阶"}],
            "expected_turns": [{
                "status": "completed", "kind": "route", "feasibility": "feasible",
                "start_poi_id": "poi_a", "end_poi_id": "poi_b", "via_poi_ids": [],
                "travel_mode": "walk", "strategy": "shortest",
                "hard_constraints": {"avoid_steps": True},
                "required_outputs": ["route"], "candidate_poi_ids": [],
                "clarification_slot": None, "allowed_unknowns": [],
                "intent_label": "route_planning", "expected_tools": None,
            }],
            "gold_derivation": {
                "kind": "controlled_template", "generator_id": "controlled-v1",
                "generator_sha256": _HASH, "template_id": "route_no_steps",
                "template_sha256": _OTHER_HASH,
                "parameters": parameters, "parameters_sha256": _digest_json(parameters),
                "evidence_basis": ["template_parameter", "catalog_attribute",
                                   "independent_graph_oracle"],
                "unresolved_ambiguity": False,
            },
            "adjudication": {
                "status": "generated_verified", "method": "deterministic_source_rules",
                "evidence": [{"kind": "independent_graph_oracle", "sha256": _HASH}],
            },
        }],
    }


def test_controlled_development_contract_accepts_source_derived_gold():
    validate_controlled_dataset(_dataset())


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.pop("source_manifest_sha256"), "source_manifest"),
    (lambda d: d["cases"][0].update(snapshot_id=_HASH), "snapshot_id"),
    (lambda d: d["cases"][0]["gold_derivation"].update(unresolved_ambiguity=True),
     "ambiguity"),
    (lambda d: d["cases"][0]["gold_derivation"].update(evidence_basis=["template_parameter"]),
     "independent_graph_oracle"),
    (lambda d: d["cases"][0]["gold_derivation"].update(parameters_sha256=_HASH),
     "parameters_sha256"),
    (lambda d: d["cases"][0]["adjudication"].update(method="manual_independent"),
     "deterministic_source_rules"),
    (lambda d: d["cases"][0]["adjudication"].update(evidence=[]), "evidence"),
    (lambda d: d["cases"][0]["expected_turns"][0].update(feasibility="unknown"),
     "feasibility"),
])
def test_controlled_contract_rejects_missing_or_ambiguous_gold(mutation, match):
    dataset = _dataset()
    mutation(dataset)
    with pytest.raises(ValueError, match=match):
        validate_controlled_dataset(dataset)


def test_formal_gate_keeps_300_case_quota_and_frozen_manifest_requirement():
    dataset = _dataset()
    dataset["split"] = "test"
    dataset["repeats"] = 3
    with pytest.raises(ValueError, match="300"):
        validate_controlled_dataset(dataset, formal=True)

    dataset = _dataset()
    dataset["cases"][0]["adjudication"]["status"] = "draft"
    with pytest.raises(ValueError, match="generated_verified"):
        validate_controlled_dataset(dataset)


def test_formal_contract_verifies_manifest_bytes_and_current_sources(tmp_path):
    dataset = _dataset()
    dataset["split"] = "test"
    dataset["repeats"] = 3
    dataset["cases"] = []
    for category in sorted(CATEGORIES):
        for number in range(50):
            case = deepcopy(_dataset()["cases"][0])
            case["id"] = f"{category}_{number}"
            case["category"] = category
            if category == "event_infeasible":
                case["gold_derivation"]["evidence_basis"].append("frozen_event_snapshot")
            dataset["cases"].append(case)

    paths = {}
    for name in REQUIRED_INPUTS:
        target = tmp_path / f"{name}.txt"
        target.write_text(name, encoding="utf-8")
        paths[name] = target.name
    manifest = make_source_manifest(tmp_path, paths, {
        "graph_crs": "EPSG:4326", "boundary_crs": "EPSG:4326",
        "boundary_provenance": "fixed fixture", "event_scope": "controlled_synthetic",
    })
    source_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    dataset["source_manifest_sha256"] = sha256(source_bytes).hexdigest()
    dataset["snapshot_id"] = manifest["effective_inputs_sha256"]
    for case in dataset["cases"]:
        case["snapshot_id"] = dataset["snapshot_id"]
        case["gold_derivation"]["generator_sha256"] = manifest["inputs"]["generator"]["sha256"]
        case["gold_derivation"]["template_sha256"] = manifest["inputs"]["templates"]["sha256"]

    with pytest.raises(ValueError, match="source_manifest_bytes"):
        validate_controlled_dataset(dataset, formal=True)
    validate_controlled_dataset(dataset, formal=True,
                                source_manifest_bytes=source_bytes, source_root=tmp_path)
    dataset["cases"][0]["gold_derivation"]["generator_sha256"] = _HASH
    with pytest.raises(ValueError, match="generator_sha256"):
        validate_controlled_dataset(dataset, formal=True,
                                    source_manifest_bytes=source_bytes, source_root=tmp_path)
    dataset["cases"][0]["gold_derivation"]["generator_sha256"] = manifest["inputs"]["generator"]["sha256"]
    (tmp_path / paths["graph"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="graph.*sha256"):
        validate_controlled_dataset(dataset, formal=True,
                                    source_manifest_bytes=source_bytes, source_root=tmp_path)


def test_controlled_gold_must_not_contain_manual_review_claims():
    dataset = deepcopy(_dataset())
    dataset["cases"][0]["adjudication"]["reviewer"] = "human-1"
    with pytest.raises(ValueError, match="reviewer"):
        validate_controlled_dataset(dataset)


def test_dynamic_event_and_information_outputs_need_their_own_evidence():
    event_case = _dataset()
    event_case["cases"][0]["category"] = "event_infeasible"
    with pytest.raises(ValueError, match="frozen_event_snapshot"):
        validate_controlled_dataset(event_case)

    info_case = _dataset()
    info_case["cases"][0]["category"] = "compound_tools"
    info_case["cases"][0]["expected_turns"][0]["required_outputs"] = ["route", "weather"]
    with pytest.raises(ValueError, match="independent_information_source"):
        validate_controlled_dataset(info_case)


def test_protocol_v2_matches_enforced_categories_and_source_inputs():
    path = Path(__file__).resolve().parents[1] / "experiments" / "protocol_v2.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["protocol_version"] == "scenario-v2"
    assert protocol["controlled_layer"]["categories"] == {
        category: 50 for category in CATEGORIES
    }
    assert set(protocol["controlled_layer"]["required_source_inputs"]) == REQUIRED_INPUTS
