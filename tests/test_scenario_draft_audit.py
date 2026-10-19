import networkx as nx

from experiments.scenario_draft_audit import audit_scenario_drafts


def _dataset():
    return {
        "protocol_version": "scenario-v1",
        "split": "development",
        "cases": [{
            "id": "route-1",
            "category": "mode_access",
            "template_group": "simple",
            "od_group": "od-1",
            "snapshot_id": "PENDING_FROZEN_SNAPSHOT",
            "adjudication": {"status": "draft", "method": None,
                             "reviewer": None, "evidence": []},
            "messages": [{"role": "user", "content": "从玉兰2门到桂操，步行"}],
            "expected_turns": [{
                "status": "completed", "kind": "route", "feasibility": "unknown",
                "start_poi_id": "start", "end_poi_id": "end", "via_poi_ids": [],
                "travel_mode": "walk", "strategy": "shortest", "hard_constraints": {},
                "required_outputs": ["route"], "candidate_poi_ids": [],
                "clarification_slot": None, "allowed_unknowns": [],
                "intent_label": None, "expected_tools": None,
            }],
        }],
    }


def _poi(poi_id, name, alias, lng, lat):
    return {"id": poi_id, "name": name, "aliases": [alias],
            "coordinates": {"lng": lng, "lat": lat}}


def _graph(connected=True):
    graph = nx.MultiDiGraph()
    graph.add_node("1", x=114.0, y=30.0)
    graph.add_node("2", x=114.001, y=30.0)
    if connected:
        graph.add_edge("1", "2", key=0, length=100.0)
    return graph


def test_draft_audit_reports_explicit_alias_and_base_graph_evidence_without_claiming_truth():
    report = audit_scenario_drafts(
        _dataset(),
        [_poi("start", "玉兰二门", "玉兰2门", 114.0, 30.0),
         _poi("end", "桂园操场", "桂操", 114.001, 30.0)],
        {"walk": _graph()}, poi_coordinate_system="WGS-84",
    )

    turn = report["cases"][0]["turns"][0]
    assert turn["entity_check"]["status"] == "explicit_catalog_names_match"
    assert turn["graph_check"]["status"] == "connected_in_base_mode_graph"
    assert turn["graph_check"]["scope"] == "base_mode_graph_only"
    assert turn["research_gold_status"] == "still_unadjudicated"
    assert {gap["code"] for gap in turn["review_gaps"]} >= {
        "intent_label_missing", "tool_use_expectation_missing"
    }


def test_draft_audit_surfaces_name_conflict_and_disconnected_base_graph():
    report = audit_scenario_drafts(
        _dataset(),
        [_poi("start", "玉兰二门", "玉兰2门", 114.0, 30.0),
         _poi("end", "桂园操场", "桂园操场", 114.001, 30.0),
         _poi("wrong", "桂操附近设施", "桂操", 114.001, 30.0),
         _poi("other", "梅园操场", "梅操", 114.001, 30.0)],
        {"walk": _graph(connected=False)}, poi_coordinate_system="WGS-84",
    )
    turn = report["cases"][0]["turns"][0]
    assert turn["entity_check"]["status"] == "possible_role_mismatch_needs_review"
    assert turn["graph_check"]["status"] == "not_connected_in_base_mode_graph"


def test_draft_audit_keeps_ambiguous_aliases_visible_for_manual_review():
    report = audit_scenario_drafts(
        {**_dataset(), "cases": [{**_dataset()["cases"][0],
                                    "messages": [{"role": "user", "content": "从玉兰2门到操场，步行"}]}]},
        [_poi("start", "玉兰二门", "玉兰2门", 114.0, 30.0),
         _poi("end", "桂园操场", "操场", 114.001, 30.0),
         _poi("other", "梅园操场", "操场", 114.001, 30.0)],
        {"walk": _graph()}, poi_coordinate_system="WGS-84",
    )
    check = report["cases"][0]["turns"][0]["entity_check"]
    assert check["status"] == "manual_review_required"
    assert "操场" in check["ambiguous_aliases"]


def test_draft_audit_does_not_use_product_candidate_outputs_as_gold():
    report = audit_scenario_drafts(
        _dataset(),
        [_poi("start", "玉兰二门", "玉兰2门", 114.0, 30.0),
         _poi("end", "桂园操场", "桂操", 114.001, 30.0)],
        {"walk": _graph()}, poi_coordinate_system="WGS-84",
    )
    assert "agent_observation" not in report
    assert report["cases"][0]["turns"][0]["candidate_gold"] == "unadjudicated"


def test_draft_audit_lists_review_gaps_without_promoting_expectations_to_gold():
    report = audit_scenario_drafts(
        _dataset(),
        [_poi("start", "玉兰二门", "玉兰2门", 114.0, 30.0),
         _poi("end", "桂园操场", "桂操", 114.001, 30.0)],
        {"walk": _graph()}, poi_coordinate_system="WGS-84",
    )

    case = report["cases"][0]
    assert {gap["code"] for gap in case["review_gaps"]} == {
        "gold_not_independently_adjudicated", "snapshot_not_frozen",
    }
    turn = case["turns"][0]
    assert turn["draft_expectation"]["start_poi_id"] == "start"
    assert turn["draft_expectation"]["end_poi_id"] == "end"
    assert turn["context_messages"] == ["从玉兰2门到桂操，步行"]
    assert {gap["code"] for gap in turn["review_gaps"]} == {
        "intent_label_missing", "tool_use_expectation_missing",
        "route_feasibility_undetermined", "route_feasibility_evidence_missing",
    }
    assert report["summary"]["review_gap_counts"]["tool_use_expectation_missing"] == 1
    assert turn["research_gold_status"] == "still_unadjudicated"


def test_draft_audit_flags_candidate_gold_and_event_snapshot_gaps_separately():
    dataset = _dataset()
    case = dataset["cases"][0]
    case["category"] = "event_infeasible"
    case["messages"] = [{"role": "user", "content": "如果路况里有道路封闭，请按快照规划。"}]
    case["expected_turns"][0].update({
        "kind": "candidates", "feasibility": "unknown", "travel_mode": None,
    })
    report = audit_scenario_drafts(
        dataset, [], {"walk": _graph()}, poi_coordinate_system="WGS-84",
    )

    case_gaps = {gap["code"] for gap in report["cases"][0]["review_gaps"]}
    turn_gaps = {gap["code"] for gap in report["cases"][0]["turns"][0]["review_gaps"]}
    assert "event_snapshot_not_defined" in case_gaps
    assert "candidate_set_unadjudicated" in turn_gaps


def test_event_audit_does_not_treat_non_event_cases_as_missing_snapshots():
    dataset = _dataset()
    case = dataset["cases"][0]
    case["category"] = "event_infeasible"
    case["messages"] = [{"role": "user", "content": "从科技门到图书馆，总距离上限50米。"}]
    report = audit_scenario_drafts(
        dataset, [], {"walk": _graph()}, poi_coordinate_system="WGS-84",
    )

    codes = {gap["code"] for gap in report["cases"][0]["review_gaps"]}
    assert "event_snapshot_not_defined" not in codes
    assert "event_category_without_event_cue" in codes


def test_generic_accessibility_snapshot_is_not_misread_as_road_event_snapshot():
    dataset = _dataset()
    case = dataset["cases"][0]
    case["category"] = "event_infeasible"
    case["messages"] = [{"role": "user", "content": "快照没有无障碍证据，请说明未知。"}]
    report = audit_scenario_drafts(
        dataset, [], {"walk": _graph()}, poi_coordinate_system="WGS-84",
    )

    codes = {gap["code"] for gap in report["cases"][0]["review_gaps"]}
    assert "event_snapshot_not_defined" not in codes
    assert "event_category_without_event_cue" in codes


def test_pending_event_snapshot_placeholder_is_reported_as_missing():
    dataset = _dataset()
    case = dataset["cases"][0]
    case["messages"] = [{"role": "user", "content": "请先查路况事件快照，再规划路线。"}]
    case["event_snapshot_id"] = "PENDING_EVENT_FIXTURE_AND_FROZEN_SNAPSHOT"
    report = audit_scenario_drafts(
        dataset, [], {"walk": _graph()}, poi_coordinate_system="WGS-84",
    )

    codes = {gap["code"] for gap in report["cases"][0]["review_gaps"]}
    assert "event_snapshot_not_defined" in codes


def test_synthetic_road_event_fixtures_are_isolated_and_exercise_runtime_semantics(
    tmp_path, monkeypatch,
):
    import json
    from pathlib import Path

    from spatial import road_conditions as rc

    fixture_path = Path(__file__).resolve().parents[1] / "experiments" / "datasets" / "road_events_synthetic_v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["scope"] == "synthetic_unit_only"
    assert fixture["campus_truth"] is False
    assert "data/road_conditions.json" not in fixture.get("writes_to", [])

    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.0, y=30.0)
    graph.add_node(2, x=114.001, y=30.0)
    graph.add_edge(1, 2, key=0, length=100.0)
    graph.add_edge(2, 1, key=0, length=100.0)
    fixture_by_id = {snapshot["id"]: snapshot for snapshot in fixture["snapshots"]}
    closure = fixture_by_id["synthetic_mode_specific_closure"]["conditions"]
    _penalties, walk_closed, _ = rc.resolve_condition_effects(graph, closure, "walk", strict=True)
    _penalties, bike_closed, _ = rc.resolve_condition_effects(graph, closure, "bike", strict=True)
    assert not walk_closed
    assert bike_closed

    slowdown = fixture_by_id["synthetic_walk_slowdown"]["conditions"]
    penalties, closed, _ = rc.resolve_condition_effects(graph, slowdown, "walk", strict=True)
    assert penalties
    assert not closed

    notice = fixture_by_id["synthetic_notice_only"]["conditions"]
    penalties, closed, applied = rc.resolve_condition_effects(graph, notice, "walk", strict=True)
    assert penalties == {}
    assert not closed
    assert applied == 1

    expired = fixture_by_id["synthetic_expired_closure"]
    revoked = fixture_by_id["synthetic_revoked_closure"]
    assert expired["as_of_epoch"] >= expired["conditions"][0]["end_time"]
    assert revoked["conditions"][0]["revoked_at"] <= revoked["as_of_epoch"]

    event_store = tmp_path / "synthetic-road-conditions.json"
    all_conditions = [condition for snapshot in fixture["snapshots"]
                      for condition in snapshot["conditions"]]
    event_store.write_text(json.dumps(all_conditions), encoding="utf-8")
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(event_store))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    monkeypatch.setattr(rc, "_cache_error", None)
    monkeypatch.setattr(rc.time, "time", lambda: fixture["evaluation_epoch"])
    active = rc.list_conditions(strict=True)
    assert {condition["id"] for condition in active} == {
        "synthetic-bike-closure", "synthetic-walk-slowdown", "synthetic-notice",
    }


def test_prior_turn_location_is_not_misreported_as_a_current_turn_conflict():
    dataset = _dataset()
    dataset["cases"][0]["messages"].append(
        {"role": "user", "content": "刚才那条路线改为平坦优先"})
    dataset["cases"][0]["expected_turns"].append(
        dict(dataset["cases"][0]["expected_turns"][0]))
    report = audit_scenario_drafts(
        dataset,
        [_poi("start", "玉兰二门", "玉兰2门", 114.0, 30.0),
         _poi("end", "桂园操场", "桂操", 114.001, 30.0)],
        {"walk": _graph()}, poi_coordinate_system="WGS-84",
    )
    second = report["cases"][0]["turns"][1]["entity_check"]
    assert second["status"] == "explicit_catalog_names_match"
    assert second["explicit_unique_poi_ids"] == []
    assert set(second["prior_context_poi_ids"]) == {"start", "end"}


def test_draft_audit_catches_clear_start_end_order_mismatch():
    dataset = _dataset()
    dataset["cases"][0]["messages"] = [
        {"role": "user", "content": "\u4ece\u6842\u64cd\u5230\u7389\u51702\u95e8\uff0c\u6b65\u884c"}]
    report = audit_scenario_drafts(
        dataset,
        [_poi("start", "\u7389\u5170\u4e8c\u95e8", "\u7389\u51702\u95e8", 114.0, 30.0),
         _poi("end", "\u6842\u56ed\u64cd\u573a", "\u6842\u64cd", 114.001, 30.0)],
        {"walk": _graph()}, poi_coordinate_system="WGS-84",
    )
    check = report["cases"][0]["turns"][0]["entity_check"]["route_order_check"]
    assert check["status"] == "literal_reference_order_differs_semantics_unresolved"


def test_constraint_audit_detects_distance_and_duration_values():
    dataset = _dataset()
    dataset["cases"][0]["messages"] = [{
        "role": "user",
        "content": "\u8ddd\u79bb\u4e0a\u9650\u6536\u7d27\u4e3a1000\u7c73\uff0c\u603b\u65f6\u95f4\u4e0d\u8d85\u8fc720\u5206\u949f",
    }]
    dataset["cases"][0]["expected_turns"][0]["hard_constraints"] = {
        "max_distance_m": 1000, "max_duration_s": 1200,
    }
    report = audit_scenario_drafts(dataset, [], {"walk": _graph()}, poi_coordinate_system="WGS-84")
    check = report["cases"][0]["turns"][0]["constraint_check"]
    assert check["status"] == "textually_supported"
    assert check["detected_constraints"] == {"max_distance_m": 1000, "max_duration_s": 1200}


def test_constraint_audit_flags_text_and_draft_value_disagreement():
    dataset = _dataset()
    dataset["cases"][0]["messages"] = [{
        "role": "user", "content": "\u8ddd\u79bb\u4e0a\u96501000\u7c73",
    }]
    dataset["cases"][0]["expected_turns"][0]["hard_constraints"] = {"max_distance_m": 1500}
    report = audit_scenario_drafts(dataset, [], {"walk": _graph()}, poi_coordinate_system="WGS-84")
    assert report["cases"][0]["turns"][0]["constraint_check"]["status"] == "possible_mismatch"


def test_constraint_audit_preserves_and_cancels_step_restrictions_across_turns():
    dataset = _dataset()
    case = dataset["cases"][0]
    case["messages"] = [
        {"role": "user", "content": "\u4e0d\u8d70\u5df2\u6807\u8bb0\u53f0\u9636"},
        {"role": "user", "content": "\u8bb0\u4f4f\u4e0d\u8d70\u53f0\u9636\u7684\u6761\u4ef6"},
        {"role": "user", "content": "\u5141\u8bb8\u8d70\u53f0\u9636\u4e86"},
    ]
    case["expected_turns"] = [dict(case["expected_turns"][0]) for _ in range(3)]
    case["expected_turns"][0]["hard_constraints"] = {"avoid_steps": True}
    case["expected_turns"][1]["hard_constraints"] = {"avoid_steps": True}
    case["expected_turns"][2]["hard_constraints"] = {}
    report = audit_scenario_drafts(dataset, [], {"walk": _graph()}, poi_coordinate_system="WGS-84")
    checks = [turn["constraint_check"] for turn in report["cases"][0]["turns"]]
    assert [check["detected_constraints"].get("avoid_steps") for check in checks] == [True, True, False]
    assert all(check["status"] == "textually_supported" for check in checks)


def test_constraint_audit_does_not_confuse_accessibility_evidence_with_avoiding_steps():
    dataset = _dataset()
    dataset["cases"][0]["messages"] = [{
        "role": "user",
        "content": "\u4e0d\u8981\u628a\u672a\u6807\u53f0\u9636\u5f53\u6210\u5df2\u6838\u5b9e\u7684\u65e0\u969c\u788d\u8bc1\u636e",
    }]
    report = audit_scenario_drafts(dataset, [], {"walk": _graph()}, poi_coordinate_system="WGS-84")
    detected = report["cases"][0]["turns"][0]["constraint_check"]["detected_constraints"]
    assert "avoid_steps" not in detected
    assert detected["require_verified_accessibility"] is True


def test_constraint_audit_does_not_turn_expired_closure_into_a_current_avoidance_rule():
    dataset = _dataset()
    dataset["cases"][0]["messages"] = [{
        "role": "user",
        "content": "\u82e5\u4e4b\u524d\u5c01\u8def\u5df2\u5230\u671f\uff0c\u8bf7\u6838\u5b9e\u6700\u65b0\u4e8b\u4ef6\u5feb\u7167",
    }]
    report = audit_scenario_drafts(dataset, [], {"walk": _graph()}, poi_coordinate_system="WGS-84")
    assert "avoid_closed_edges" not in report["cases"][0]["turns"][0]["constraint_check"]["detected_constraints"]


def test_constraint_audit_detects_closed_edge_avoidance_after_multiple_prohibitions():
    dataset = _dataset()
    case = dataset["cases"][0]
    case["messages"] = [{
        "role": "user",
        "content": "如果通路封闭，不允许省略途经点或穿越封闭路段。",
    }]
    case["expected_turns"][0]["hard_constraints"] = {"avoid_closed_edges": True}
    report = audit_scenario_drafts(dataset, [], {"walk": _graph()}, poi_coordinate_system="WGS-84")
    check = report["cases"][0]["turns"][0]["constraint_check"]
    assert check["status"] == "textually_supported"
    assert check["detected_constraints"]["avoid_closed_edges"] is True
