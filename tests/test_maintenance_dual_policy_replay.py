"""Decision bookkeeping of the offline dual-policy replay (no routing here)."""
from experiments.maintenance_dual_policy_replay import _compact_plan_call, _compare


def _arm(**overrides):
    row = {
        "case_id": "case",
        "group": "two_stage_itinerary",
        "arm": "conservative",
        "policy": "conservative",
        "response_kind": "route",
        "prefix_action": "retained",
        "stage1": {"distance_m": 100.0, "edge_count": 2, "edge_ids_sha256": "a"},
        "stage2": {"distance_m": 150.0, "edge_count": 3, "edge_ids_sha256": "b"},
        "tool_calls": {"plan_route": 1},
        "plan_calls": [],
        "recomputed_anchor_count": 2,
        "elapsed_ms": 1.0,
        "error": None,
    }
    row.update(overrides)
    return row


def test_plan_call_records_every_anchor_it_was_asked_to_recompute():
    call = _compact_plan_call("plan_via_route", {
        "start": {"poi_id": "poi_a"},
        "via_points": [{"poi_id": "poi_b"}],
        "end": {"poi_id": "poi_c"},
    })
    assert call["od_anchors"] == ["poi_a", "poi_b", "poi_c"]
    assert call["od_anchor_count"] == 3
    assert _compact_plan_call("get_weather", {})["od_anchor_count"] is None


def test_retained_prefix_with_a_different_route_is_reported_as_stale():
    case = {"case_id": "case", "group": "two_stage_itinerary"}
    comparison = _compare(case, {
        "conservative": _arm(),
        "full_recompute": _arm(
            arm="full_recompute", policy="full_recompute",
            prefix_action="recomputed", tool_calls={"plan_via_route": 1},
            recomputed_anchor_count=3,
            stage2={"distance_m": 151.0, "edge_count": 3, "edge_ids_sha256": "c"},
        ),
    })
    assert comparison["stale_reference"] is True
    assert comparison["distance_equal"] is False
    assert comparison["policy_sensitive"] is True
    assert comparison["recomputed_anchor_count"] == {"conservative": 2, "full_recompute": 3}


def test_retained_prefix_with_the_same_route_is_not_stale():
    case = {"case_id": "case", "group": "two_stage_itinerary"}
    comparison = _compare(case, {
        "conservative": _arm(),
        "full_recompute": _arm(
            arm="full_recompute", policy="full_recompute",
            prefix_action="recomputed", tool_calls={"plan_via_route": 1},
            recomputed_anchor_count=3,
        ),
    })
    assert comparison["stale_reference"] is False
    assert comparison["distance_equal"] is True
    assert comparison["edge_ids_equal"] is True
    assert comparison["policy_sensitive"] is True
