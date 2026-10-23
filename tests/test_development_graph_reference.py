from copy import deepcopy

import pytest

from experiments.development_graph_reference import build_graph_development_set
from experiments.scenario_split_audit import audit_scenario_splits
from scripts.analysis.build_scenario_v2_authoring import build_authoring_bank


def _inputs():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    full = build_authoring_bank(root / "data/pois.json")
    ids = {"v2_T1_explicit_1", "v2_T1_scenic_1",
           "v2_T2_meal_1", "v2_T2_canteen_1"}
    draft = {**full, "cases": [case for case in full["cases"] if case["id"] in ids]}
    source = {name: (str(index) * 64) for index, name in enumerate(
        ("draft", "pois", "graph", "overrides", "reviews"), 1)}
    route_ref = {"start_node": "s", "end_node": "e", "via_nodes": [],
                 "graph_feasible": True, "access_uncertain": False}
    route = {"source_sha256": source, "cases": [
        {"case_id": case_id, "needs_review": False,
         "proposed_route_reference": route_ref}
        for case_id in ("v2_T1_explicit_1", "v2_T1_scenic_1")
    ]}
    candidate_ref = {"distances_m": {"a": 1, "b": 2, "c": 3, "d": 4},
                     "uncertain_access_ids": [], "max_display": 4,
                     "rank_tolerance_m": 30}
    candidates = {"source_sha256": source, "cases": [
        {"case_id": case_id, "review_needed": case_id.endswith("canteen_1"),
         "top_four": [{"poi_id": key} for key in ("a", "b", "c", "d")],
         "proposed_candidate_reference": candidate_ref}
        for case_id in ("v2_T2_meal_1", "v2_T2_canteen_1")
    ]}
    return draft, route, candidates, source


def test_graph_development_set_excludes_preference_and_uncertain_cases():
    result = build_graph_development_set(*_inputs())
    assert {case["source_case_id"] for case in result["cases"]} == {
        "v2_T1_explicit_1", "v2_T2_meal_1"
    }
    assert len(result["excluded_cases"]) == 2
    assert all(case["snapshot_id"] == result["snapshot_id"]
               for case in result["cases"])
    assert all(case["adjudication"]["method"] == "graph_computed"
               for case in result["cases"])
    assert result == build_graph_development_set(*_inputs())


def test_graph_development_set_rejects_stale_reference_report():
    draft, route, candidates, source = _inputs()
    stale = deepcopy(candidates)
    stale["source_sha256"] = {**source, "graph": "f" * 64}
    with pytest.raises(ValueError, match="stale source digests"):
        build_graph_development_set(draft, route, stale, source)


def test_visible_source_cannot_be_relabelled_as_an_independent_holdout():
    draft, route, candidates, source = _inputs()
    development = build_graph_development_set(draft, route, candidates, source)
    report = audit_scenario_splits(development, draft)
    assert report["leakage_free"] is False
    assert "source_group_overlap" in {row["code"] for row in report["blockers"]}
