from experiments.scenario_split_audit import audit_scenario_splits


def _case(case_id, *, template="template-a", od="od-a", prompt="从甲到乙"):
    return {
        "id": case_id,
        "category": "place_recommendation",
        "template_group": template,
        "od_group": od,
        "messages": [{"role": "user", "content": prompt}],
        "adjudication": {"status": "draft"},
    }


def _datasets(dev_case, test_case):
    return (
        {"split": "development", "repeats": 1, "cases": [dev_case]},
        {"split": "test", "repeats": 3, "cases": [test_case]},
    )


def test_split_audit_rejects_shared_template_od_and_exact_prompt():
    dev, test = _datasets(_case("dev-1"), _case("test-1"))

    report = audit_scenario_splits(dev, test)

    assert report["leakage_free"] is False
    assert {
        "template_group_overlap", "od_group_overlap", "prompt_overlap",
    } <= {item["code"] for item in report["blockers"]}


def test_split_audit_does_not_call_empty_test_split_ready():
    dev = {"split": "development", "repeats": 1, "cases": [_case("dev-1")]}
    test = {"split": "test", "repeats": 3, "cases": []}

    report = audit_scenario_splits(dev, test)

    assert report["leakage_evaluable"] is False
    assert report["leakage_free"] is False
    assert report["formal_ready"] is False
    assert "test_split_empty" in {item["code"] for item in report["blockers"]}


def test_split_audit_detects_independent_groups_without_claiming_formal_readiness():
    dev, test = _datasets(
        _case("dev-1", template="template-a", od="od-a", prompt="甲到乙"),
        _case("test-1", template="template-b", od="od-b", prompt="丙到丁"),
    )

    report = audit_scenario_splits(dev, test)

    assert report["leakage_free"] is True
    assert report["formal_ready"] is False
    assert report["overlap_counts"] == {
        "scenario_ids": 0,
        "template_groups": 0,
        "od_groups": 0,
        "prompt_fingerprints": 0,
        "source_groups": 0,
    }

def test_split_audit_detects_shared_source_group_and_hides_prompt_hashes():
    secret_prompt = "".join(map(chr, (0x53bb, 0x98df, 0x5802)))
    dev_case = _case("dev-1", prompt=secret_prompt)
    test_case = _case("test-1", template="new-template", od="new-od", prompt=f" {secret_prompt} ")
    dev_case["flight_id"] = "flight-7"
    test_case["flight_id"] = "flight-7"
    dev, test = _datasets(dev_case, test_case)

    report = audit_scenario_splits(dev, test)

    codes = {item["code"] for item in report["blockers"]}
    assert {"prompt_overlap", "source_group_overlap"} <= codes
    assert secret_prompt not in str(report)


def test_formal_ready_requires_complete_formal_contract_and_verified_gates():
    from experiments.scenario_evaluator import PROTOCOL_VERSION

    snapshot = "a" * 64
    cases = []
    for index in range(300):
        category = sorted({
            "place_recommendation", "mode_access", "multi_via_itinerary",
            "compound_tools", "multi_turn_revision", "event_infeasible",
        })[index % 6]
        cases.append({
            "id": f"test-{index}",
            "category": category,
            "template_group": f"template-{index}",
            "od_group": f"od-{index}",
            "snapshot_id": snapshot,
            "messages": [{"role": "user", "content": f"unique prompt {index}"}],
            "expected_turns": [{
                "status": "completed", "kind": "information", "feasibility": "not_applicable",
                "start_poi_id": None, "end_poi_id": None, "via_poi_ids": [],
                "hard_constraints": {}, "required_outputs": [], "candidate_poi_ids": [],
                "clarification_slot": None, "allowed_unknowns": [], "travel_mode": None,
                "strategy": None, "intent_label": "information_lookup",
                "expected_tools": [["get_weather"]],
            }],
            "adjudication": {"status": "adjudicated", "method": "manual_independent",
                             "reviewer": "reviewer-a", "evidence": ["case-note-1"]},
        })
    development = {"split": "development", "cases": [_case("dev", template="dev-t", od="dev-o", prompt="dev prompt")]}
    test = {"protocol_version": PROTOCOL_VERSION, "split": "test", "repeats": 3, "cases": cases}
    gates = {name: True for name in (
        "frozen_snapshot_manifest", "agent_readiness_gate", "matching_git_commit", "clean_tracked_worktree",
    )}

    report = audit_scenario_splits(development, test, external_gates=gates)

    assert report["leakage_free"] is True
    assert report["formal_ready"] is True
    assert report["blockers"] == []

def test_split_audit_reports_malformed_fields_without_crashing():
    dev_case = _case("dev-1")
    malformed = {"id": "test-1", "category": [], "template_group": "new-template",
                 "od_group": "new-od", "messages": None, "adjudication": {"status": []}}
    dev, test = _datasets(dev_case, malformed)

    report = audit_scenario_splits(dev, test)

    assert report["formal_ready"] is False
    assert report["leakage_free"] is False
    assert "test_category_invalid" in {item["code"] for item in report["blockers"]}
