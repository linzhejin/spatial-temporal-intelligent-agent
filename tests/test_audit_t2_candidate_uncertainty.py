"""The T2 authoring audit must preserve unknown access without human inflation."""

import pytest

from scripts.analysis.audit_t2_candidate_uncertainty import audit_reference


def _case(index: int, *, uncertain: bool) -> dict:
    distances = {"near": 100.0, "next": 120.0, "third": 140.0, "fourth": 160.0,
                 "fifth": 165.0}
    return {
        "case_id": f"T2_{index}", "activity_family": "meal",
        "top_four": [{"poi_id": key} for key in list(distances)[:4]],
        "fourth_fifth_gap_m": 5.0, "review_needed": True,
        "proposed_candidate_reference": {
            "distances_m": distances, "rank_tolerance_m": 30.0, "max_display": 4,
            "uncertain_access_ids": ["fifth"] if uncertain else [],
        },
    }


def test_close_fifth_is_graph_decidable_until_its_access_is_uncertain():
    report = {"status": "development_reference_only_not_adjudicated",
              "cases": [_case(i, uncertain=i <= 9) for i in range(1, 25)]}
    result = audit_reference(report, {"fifth": "候选地点"})
    assert result["reference_top_four_verdict_counts"] == {"pass": 15, "unknown": 9}
    assert result["affected_unique_pois"] == [{"poi_id": "fifth", "name": "候选地点"}]
    assert result["status"] == "development_reference_uncertainty_only_not_agent_score"


def test_duplicate_authoring_case_is_rejected():
    report = {"status": "development_reference_only_not_adjudicated",
              "cases": [_case(1, uncertain=False) for _ in range(24)]}
    with pytest.raises(ValueError, match="distinct"):
        audit_reference(report, {})
