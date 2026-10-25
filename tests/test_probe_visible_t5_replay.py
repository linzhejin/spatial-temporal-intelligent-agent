from scripts.analysis.probe_visible_t5_replay import compare_fields, selected_followup_poi


def test_selected_followup_poi_finds_meal_before_original_destination():
    current = {"kind": "candidates", "start_poi_id": "origin"}
    following = {"kind": "route", "via_poi_ids": ["meal"], "end_poi_id": "destination"}
    assert selected_followup_poi(current, following) == "meal"


def test_selected_followup_poi_tracks_both_selections_in_three_stage_task():
    first = {"kind": "candidates", "start_poi_id": "origin"}
    second = {"kind": "candidates", "start_poi_id": "court"}
    final = {"kind": "route", "via_poi_ids": ["court"], "end_poi_id": "meal"}
    assert selected_followup_poi(first, second) == "court"
    assert selected_followup_poi(second, final) == "meal"


def test_candidate_projection_checks_observable_fields_and_normalizes_false_constraints():
    expected = {"status": "needs_input", "kind": "candidates",
                "start_poi_id": "origin", "end_poi_id": "destination",
                "via_poi_ids": [], "travel_mode": "walk", "strategy": "shortest",
                "hard_constraints": {}}
    actual = {**expected, "end_poi_id": None, "strategy": None,
              "hard_constraints": {"avoid_steps": False, "avoid_slope": False,
                                   "excluded_edge_ids": [], "max_distance_m": None}}
    checks = compare_fields(expected, actual)
    assert "end_poi_id" not in checks
    assert "strategy" not in checks
    assert all(checks.values())


def test_route_projection_checks_final_destination_and_strategy():
    expected = {"status": "completed", "kind": "route", "start_poi_id": "origin",
                "end_poi_id": "destination", "via_poi_ids": ["meal"],
                "travel_mode": "walk", "strategy": "shortest",
                "hard_constraints": {"avoid_steps": True}}
    actual = {**expected, "end_poi_id": "meal"}
    checks = compare_fields(expected, actual)
    assert checks["end_poi_id"] is False
    assert checks["strategy"] is True
