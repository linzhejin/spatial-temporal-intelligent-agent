"""Contracts for the multi-turn development harness; no model calls."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.complex_multiturn_dev.budget import BudgetLedger, BudgetUnavailable
from experiments.complex_multiturn_dev.contracts import (
    assess_turn, read_dataset, validate_dataset,
)
from experiments.complex_multiturn_dev.observation import project_observation
from experiments.complex_multiturn_dev.event_runtime import load_event_bundle
from experiments.complex_multiturn_dev.graph_checks import inspect_delivery_route


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "experiments/datasets/complex_multiturn_dev_v1/cases.json"


def test_dataset_has_distinct_core_and_variant_cases_with_real_multi_turns():
    data = read_dataset(DATASET)
    validate_dataset(data)
    assert len(data["cases"]) == 32
    for group in ("search", "itinerary", "clarification", "events"):
        cases = [c for c in data["cases"] if c["group"] == group]
        assert sum(c["phase"] == "core" for c in cases) == 6
        assert sum(c["phase"] == "variant" for c in cases) == 2
        assert all(3 <= len(c["turns"]) <= 5 for c in cases)
        assert all(all(t["checks"] for t in c["turns"]) for c in cases)


def test_additional_time_radius_cases_are_separate_from_the_fixed_32():
    from experiments.complex_multiturn_dev.contracts import validate_supplement
    data = read_dataset(DATASET.parent / "supplement_time_radius_v2.json")
    validate_supplement(data)
    assert len(data["cases"]) == 2
    assert all(case["id"].startswith("supplement_") for case in data["cases"])
    with pytest.raises(ValueError):
        validate_dataset(data)
    broken = deepcopy(data)
    broken["cases"][0]["id"] = "search_01"
    with pytest.raises(ValueError):
        validate_supplement(broken)


def test_time_only_supplement_has_a_bound_independent_graph_witness():
    from experiments.complex_multiturn_dev.contracts import validate_supplement
    dataset = read_dataset(DATASET.parent / "supplement_time_only.json")
    validate_supplement(dataset)
    proof = read_dataset(DATASET.parent / "supplement_time_only_preflight.json")
    assert proof["graph_sha256"] == hashlib.sha256((ROOT / "data/whu_road_network.graphml").read_bytes()).hexdigest()
    assert proof["poi_sha256"] == hashlib.sha256((ROOT / "data/pois.json").read_bytes()).hexdigest()
    assert proof["checker_sha256"] == hashlib.sha256((ROOT / "experiments/complex_multiturn_dev/candidate_checks.py").read_bytes()).hexdigest()
    assert proof["within_1500m"]["verdict"] == "pass"
    assert proof["outside_1000m"]["verdict"] == "fail"
    assert proof["within_20min"]["verdict"] == "pass"


def test_missing_observation_is_unknown_and_never_a_pass():
    expected = {"checks": [{"field": "origin", "op": "eq", "value": "poi_418"}]}
    assert assess_turn(expected, {"status": "completed"})["verdict"] == "unknown"


def test_explicit_null_can_prove_absence_but_missing_evidence_cannot():
    expected = {"checks": [{"field": "route_max_distance_m", "op": "absent"}]}
    assert assess_turn(expected, {})["verdict"] == "unknown"
    assert assess_turn(expected, {"route_max_distance_m": None})["verdict"] == "pass"
    assert assess_turn(expected, {"route_max_distance_m": 500})["verdict"] == "fail"


def test_explicit_null_conflicts_with_a_required_value():
    expected = {"checks": [{"field": "max_duration_s", "op": "eq", "value": 1200}]}
    assert assess_turn(expected, {})["verdict"] == "unknown"
    assert assess_turn(expected, {"max_duration_s": None})["verdict"] == "fail"
    assert assess_turn(expected, {"max_duration_s": 1200})["verdict"] == "pass"


def test_observed_changed_origin_is_a_failure():
    expected = {"checks": [{"field": "origin", "op": "eq", "value": "poi_418"}]}
    result = assess_turn(expected, {"status": "completed", "origin": "poi_004"})
    assert result["verdict"] == "fail"
    assert result["failures"] == ["origin:eq"]


def test_order_requires_all_stops_in_order_and_missing_evidence_is_unknown():
    expected = {"checks": [{"field": "stops", "op": "ordered_contains", "value": ["a", "b"]}]}
    assert assess_turn(expected, {"status": "completed", "stops": ["a", "b", "c"]})["verdict"] == "pass"
    assert assess_turn(expected, {"status": "completed", "stops": ["b", "a"]})["verdict"] == "fail"
    assert assess_turn(expected, {"status": "completed"})["verdict"] == "unknown"


def test_transport_failure_is_not_a_success_even_if_some_fields_match():
    expected = {"checks": [{"field": "origin", "op": "eq", "value": "poi_418"}]}
    result = assess_turn(expected, {"status": "failed", "origin": "poi_418"})
    assert result["verdict"] == "fail"


def test_audit_labels_deadline_or_model_timeout_separately_from_general_failure():
    from experiments.complex_multiturn_dev.graph_checks import timeout_evidence
    assert timeout_evidence({"failure_code": "deadline_exceeded"}) == "timeout:deadline_exceeded"
    assert timeout_evidence({"failure_code": "model_timeout"}) == "timeout:model_timeout"
    assert timeout_evidence({"failure_code": "place_search_limit"}) is None


def test_report_audit_cache_changes_when_new_evidence_arrives(tmp_path):
    from experiments.complex_multiturn_dev.report import audit_fingerprint
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "turns.jsonl").write_text("{}\n", encoding="utf-8")
    before = audit_fingerprint(run, dataset)
    (run / "search_01_candidate_scope_audit.jsonl").write_text("{}\n", encoding="utf-8")
    assert audit_fingerprint(run, dataset) != before


def test_audit_does_not_accept_route_for_an_ambiguous_destination(tmp_path):
    from experiments.complex_multiturn_dev.audit import audit
    dataset = DATASET.parent / "cases_event_endpoint_revised.json"
    manifest = {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "case_ids": ["clarification_03"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    item = {"case": "clarification_03", "turn": 1, "run_id": "run",
            "observed": {"status": "completed", "origin": "poi_140", "mode": "walk"},
            "result": {"response_kind": "route", "recommended_edge_ids": ["edge"]}}
    (tmp_path / "turns.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    audit(tmp_path, dataset)
    row = json.loads((tmp_path / "independent_route_audit.jsonl").read_text(encoding="utf-8"))
    assert "ambiguous_destination_chosen_without_clarification" in row["failures"]


def test_audit_checks_activity_state_when_a_search_returns_empty(tmp_path):
    from experiments.complex_multiturn_dev.audit import audit
    dataset = DATASET.parent / "cases_event_endpoint_revised.json"
    manifest = {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "case_ids": ["search_02"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    item = {"case": "search_02", "turn": 2, "run_id": "run",
            "observed": {"status": "completed", "origin": "poi_424", "mode": "walk",
                         "radius_m": 1000, "avoid_steps": True, "activities": ["basketball"]},
            "result": {"response_kind": "chat", "assistant_text": "暂未找到羽毛球场"},
            "tool_observations": [{"tool": "search_poi_candidates", "args": {
                "start": {"poi_id": "poi_424"}, "mode": "walk", "max_distance_m": 1000,
                "activities": ["badminton"], "constraints": {"avoid_steps": True}},
                "result": {"count": 0, "online_status": "ok"}}]}
    (tmp_path / "turns.jsonl").write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    audit(tmp_path, dataset)
    row = json.loads((tmp_path / "independent_route_audit.jsonl").read_text(encoding="utf-8"))
    assert "activity_state_conflict" in row["failures"]
    assert "forbidden_activity_retained" in row["failures"]


def test_audit_recovers_an_explicit_null_from_a_saved_bound_task_result(tmp_path):
    from experiments.complex_multiturn_dev.audit import audit
    dataset = DATASET.parent / "supplement_time_radius_v2.json"
    case_id = "supplement_time_radius_same_origin"
    manifest = {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "case_ids": [case_id]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    item = {"case": case_id, "turn": 1, "run_id": "run", "task_id": "bound-task",
            "observed": {"status": "completed", "origin": "poi_424", "mode": "walk", "radius_m": 500},
            "result": {"response_kind": "chat", "place_request": {"constraints": {
                "max_duration_s": None, "max_distance_m": None}}}}
    (tmp_path / "turns.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    audit(tmp_path, dataset)
    row = json.loads((tmp_path / "independent_route_audit.jsonl").read_text(encoding="utf-8"))
    assert "max_duration_s:eq" in row["failures"]


def test_audit_treats_a_failed_candidate_time_witness_as_failure(tmp_path):
    from experiments.complex_multiturn_dev.audit import audit
    dataset = DATASET.parent / "supplement_time_radius_v2.json"
    case_id = "supplement_time_radius_separate_center"
    manifest = {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "case_ids": [case_id]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    item = {"case": case_id, "turn": 1, "run_id": "run", "task_id": "bound-task",
            "observed": {"status": "completed", "origin": "poi_140", "search_anchor": "poi_257",
                         "mode": "walk", "radius_m": 1000, "max_duration_s": 1200, "route_max_distance_m": None},
            "result": {"response_kind": "candidates", "candidates": [{"poi_id": "poi_257"}]}}
    (tmp_path / "turns.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    checker = ROOT / "experiments/complex_multiturn_dev/candidate_checks.py"
    evidence = {"case": case_id, "turn": 1, "candidate": "poi_257", "verdict": "fail",
                "reason": "no_path_in_frozen_eligible_graph", "checker_sha256": hashlib.sha256(checker.read_bytes()).hexdigest()}
    (tmp_path / (case_id + "_candidate_time_audit.jsonl")).write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    audit(tmp_path, dataset)
    row = json.loads((tmp_path / "independent_route_audit.jsonl").read_text(encoding="utf-8"))
    assert any(item.startswith("candidate_time:poi_257:") for item in row["failures"])


def test_projection_distinguishes_origin_search_anchor_and_ordered_route():
    lookup = {"names": {"出发点": "poi_418", "搜索中心": "poi_004",
                        "第一站": "poi_423", "终点": "poi_137"},
              "ids": {"poi_418", "poi_004", "poi_423", "poi_137"}}
    run = {"status": "completed", "result": {
        "place_request": {"start": {"name": "出发点"}, "search_anchor": {"name": "搜索中心"},
                          "mode": "walk", "constraints": {"avoid_steps": True}},
        "route_state": {"start": {"name": "出发点"}, "end": {"name": "终点"},
                        "via": {"name": "第一站"}, "travel_mode": "walk"}}}
    obs = project_observation(run, None, lookup)
    assert obs["origin"] == "poi_418"
    assert obs["search_anchor"] == "poi_004"
    assert obs["stops"] == ["poi_423", "poi_137"]
    assert obs["avoid_steps"] is True


def test_projection_does_not_invent_event_or_verified_route():
    obs = project_observation({"status": "completed", "result": {}}, None,
                              {"names": {}, "ids": set()})
    assert "event_state" not in obs
    assert "route_integrity" not in obs


def test_projection_reads_actual_clarification_slot_and_saved_time_constraint():
    run = {"status": "needs_input", "result": {"response_kind": "clarify",
           "clarify": {"slot": "start", "question": "你从哪里出发？"},
           "place_request": {"constraints": {"max_duration_s": 1200}}}}
    obs = project_observation(run, None, {"names": {}, "ids": set()})
    assert obs["clarification_slot"] == "start"
    assert obs["max_duration_s"] == 1200


def test_projection_records_explicit_absence_of_a_route_distance_limit():
    run = {"status": "completed", "result": {"place_request": {"constraints": {"max_distance_m": None}}}}
    obs = project_observation(run, None, {"names": {}, "ids": set()})
    assert "route_max_distance_m" in obs and obs["route_max_distance_m"] is None


def test_projection_never_mutates_a_multi_stop_delivery():
    lookup = {"names": {}, "ids": {"poi_001", "poi_004", "poi_137"}}
    route = {"start": {"poi_id": "poi_001"}, "end": {"poi_id": "poi_137"},
             "via": {"type": "multi", "points": [{"poi_id": "poi_004"}]}}
    run = {"status": "completed", "result": {"route_state": route}}
    original = deepcopy(run)
    assert project_observation(run, None, lookup)["stops"] == ["poi_004", "poi_137"]
    assert run == original


def test_projection_uses_the_saved_route_end_when_followup_is_plain_chat():
    lookup = {"names": {}, "ids": {"poi_001", "poi_004", "poi_137"}}
    route = {"start": {"poi_id": "poi_001"}, "end": {"poi_id": "poi_137"},
             "via": {"poi_id": "poi_004"}}
    run = {"status": "completed", "result": {"response_kind": "chat", "place_request": {
        "start": {"poi_id": "poi_001"}, "semantic_request": {"tour_stops": ["poi_004"],
                                                               "route_to_modify": route}}}}
    observed = project_observation(run, None, lookup)
    assert observed["stops"] == ["poi_004", "poi_137"]
    assert observed["destination"] == "poi_137"


def test_projection_prefers_accepted_task_over_unbound_chat_defaults():
    lookup = {"names": {}, "ids": {"poi_001"}}
    run = {"status": "completed", "task_id": None,
           "result": {"place_request": {"mode": "walk", "search_radius_m": 1000}}}
    task = {"state": {"place_request": {"start": {"poi_id": "poi_001"},
                                        "mode": "bike", "search_radius_m": 2000}}}
    observed = project_observation(run, task, lookup)
    assert observed["mode"] == "bike"
    assert observed["radius_m"] == 2000
    assert observed["origin"] == "poi_001"


def test_budget_reserves_before_request_and_accounts_peak_prices(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.jsonl", limit_cny=Decimal("20"))
    reservation = ledger.reserve(request_bytes=10000, max_output_tokens=2600)
    assert ledger.spent_or_reserved() > 0
    ledger.settle(reservation, prompt_tokens=1000, completion_tokens=500)
    assert ledger.spent_or_reserved() == Decimal("0.006")


def test_approved_increased_budget_keeps_previous_ledger_and_hard_cap(tmp_path):
    path = tmp_path / "budget.jsonl"
    original = BudgetLedger(path, limit_cny=Decimal("20"))
    ident = original.reserve_fixed(amount_cny=Decimal("0.005"), purpose="search")
    original.settle_fixed(ident)
    extended = BudgetLedger(path, limit_cny=Decimal("40"))
    assert extended.spent_or_reserved() == Decimal("0.005")
    with pytest.raises(ValueError, match="40"):
        BudgetLedger(path, limit_cny=Decimal("40.01"))


def test_budget_counts_reported_cache_hits_at_conservative_peak_rate_without_editing_ledger(tmp_path):
    path = tmp_path / "budget.jsonl"
    ledger = BudgetLedger(path, limit_cny=Decimal("40"))
    ident = ledger.reserve(request_bytes=10000, max_output_tokens=2600)
    ledger.settle(ident, prompt_tokens=1000, completion_tokens=500,
                  cache_hit_tokens=800, cache_miss_tokens=200)
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["cache_hit_tokens"] == 800 and record["cache_miss_tokens"] == 200
    assert record["usage_recorded_at_utc"]
    original = path.read_bytes()
    assert ledger.spent_or_reserved() == Decimal("0.004432")
    assert path.read_bytes() == original


def test_provider_evidence_keeps_only_poi_fields_not_keys_or_request_urls():
    from experiments.complex_multiturn_dev.run_dev import _safe_provider_evidence
    payload = {"status": "1", "count": "1", "key": "PRIVATE", "pois": [{
        "id": "A123", "name": "咖啡店", "type": "餐饮服务;咖啡厅", "typecode": "050500",
        "location": "114.36,30.53", "key": "PRIVATE", "extra": "PRIVATE"}]}
    captured = _safe_provider_evidence(payload)
    assert captured["pois"][0]["id"] == "A123"
    assert captured["pois"][0]["typecode"] == "050500"
    assert "PRIVATE" not in json.dumps(captured, ensure_ascii=False)


def test_runner_does_not_select_a_candidate_hidden_by_a_completed_route():
    from experiments.complex_multiturn_dev.run_dev import _selection_payload
    previous = {"task_id": "task", "current_task_revision": 2,
                "result": {"response_kind": "route", "needs_input": False}}
    with pytest.raises(ValueError, match="visible"):
        _selection_payload(previous, {"candidate_ids": ["poi_001"]})
    previous["result"] = {"response_kind": "candidates"}
    payload, selected = _selection_payload(previous, {"candidate_ids": ["poi_001"]})
    assert selected == payload["selected_poi_id"] == "poi_001"


def test_inconsistent_cache_details_do_not_lower_the_budget_or_lose_usage(tmp_path):
    path = tmp_path / "budget.jsonl"
    ledger = BudgetLedger(path, limit_cny=Decimal("40"))
    ident = ledger.reserve(request_bytes=10000, max_output_tokens=2600)
    ledger.settle(ident, prompt_tokens=1000, completion_tokens=500,
                  cache_hit_tokens=800, cache_miss_tokens=300)
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["cache_usage_status"] == "unavailable_or_inconsistent"
    assert ledger.spent_or_reserved() == Decimal("0.006")


def test_corrupted_recorded_cache_counts_fall_back_to_the_original_upper_bound(tmp_path):
    path = tmp_path / "budget.jsonl"
    ledger = BudgetLedger(path, limit_cny=Decimal("40"))
    ident = ledger.reserve(request_bytes=10000, max_output_tokens=2600)
    ledger.settle(ident, prompt_tokens=1000, completion_tokens=500,
                  cache_hit_tokens=800, cache_miss_tokens=200)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["cache_hit_tokens"] = 900
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert ledger.spent_or_reserved() == Decimal("0.006")


def test_unconfirmed_usage_keeps_reservation_and_stops_further_calls(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.jsonl", limit_cny=Decimal("20"))
    ledger.reserve(request_bytes=1000, max_output_tokens=100)
    reopened = BudgetLedger(tmp_path / "budget.jsonl", limit_cny=Decimal("20"))
    with pytest.raises(BudgetUnavailable, match="unsettled"):
        reopened.reserve(request_bytes=10, max_output_tokens=10)


def test_uncertain_usage_requires_explicit_approval_to_book_reserved_upper_bound(tmp_path):
    path = tmp_path / "budget.jsonl"
    ledger = BudgetLedger(path, limit_cny=Decimal("20"))
    ident = ledger.reserve(request_bytes=1000, max_output_tokens=100)
    upper = ledger.spent_or_reserved()
    with pytest.raises(BudgetUnavailable, match="approval"):
        ledger.settle_uncertain_at_reserved_cap(ident, reason="HTTP 400; usage unavailable", approved=False)
    with pytest.raises(BudgetUnavailable, match="unsettled"):
        ledger.reserve(request_bytes=10, max_output_tokens=10)
    ledger.settle_uncertain_at_reserved_cap(ident, reason="HTTP 400; usage unavailable", approved=True)
    assert ledger.spent_or_reserved() == upper
    entry = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["accounting"] == "approved_uncertain_usage_reserved_upper_bound"
    assert entry["usage"] == "unknown" and Decimal(entry["amount_cny"]) == upper
    ledger.reserve(request_bytes=10, max_output_tokens=10)


def test_budget_does_not_start_a_request_that_might_exceed_the_cap(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.jsonl", limit_cny=Decimal("0.001"))
    with pytest.raises(BudgetUnavailable, match="limit"):
        ledger.reserve(request_bytes=10000, max_output_tokens=2600)
    assert ledger.spent_or_reserved() == 0


def test_budget_stop_is_reported_to_the_runner_before_any_provider_call(tmp_path):
    from experiments.complex_multiturn_dev.run_dev import _guard_client
    ledger = BudgetLedger(tmp_path / "budget.jsonl", limit_cny=Decimal("0.001"))
    provider = Mock()
    client = SimpleNamespace(base_url="https://api.deepseek.com/v1",
                             chat=SimpleNamespace(completions=SimpleNamespace(create=provider)))
    state = {"stop_reason": None}
    guarded = _guard_client(lambda: client, ledger, tmp_path / "journal.jsonl", state)()
    with pytest.raises(BudgetUnavailable):
        guarded.chat.completions.create(model="deepseek-flash", messages=[], max_tokens=2600)
    assert state["stop_reason"]
    provider.assert_not_called()


def test_browser_bridge_does_not_process_an_idempotent_retry_twice(monkeypatch):
    from experiments.complex_multiturn_dev.browser import BrowserCase
    from jobs import worker
    process = Mock(return_value=None)
    monkeypatch.setattr(worker, "process_next_run", process)
    response = SimpleNamespace(status_code=202, headers={"content-type": "application/json"},
                               get_json=lambda **_kw: {"data": {"run_id": "same-run"}},
                               get_data=lambda: b"{}")
    client = SimpleNamespace(open=Mock(return_value=response))
    case = BrowserCase.__new__(BrowserCase)
    case.client, case.host = client, SimpleNamespace(database_url="isolated", workflow=None)
    case.jobs = {"same-run": object()}
    case.conversation = "conversation"
    request = SimpleNamespace(url="http://localhost/api/conversations/conversation/messages", method="POST",
                              post_data="{}", post_data_json={}, headers={"content-type": "application/json"})
    route = SimpleNamespace(request=request, fulfill=Mock(), abort=Mock(), continue_=Mock())

    case._bridge(route)

    route.fulfill.assert_called_once()
    process.assert_not_called()


def test_browser_bridge_returns_accepted_response_while_worker_runs(monkeypatch):
    import threading
    from experiments.complex_multiturn_dev.browser import BrowserCase
    from jobs import worker
    started, release = threading.Event(), threading.Event()
    def slow_worker(*_args, **_kwargs):
        started.set()
        release.wait(2)
        return {"run_id": "new-run"}
    monkeypatch.setattr(worker, "process_next_run", slow_worker)
    accepted = SimpleNamespace(status_code=202, headers={"content-type": "application/json"},
                               get_json=lambda **_kw: {"data": {"run_id": "new-run"}},
                               get_data=lambda: b"{}")
    client = SimpleNamespace(open=Mock(return_value=accepted))
    case = BrowserCase.__new__(BrowserCase)
    case.client, case.host = client, SimpleNamespace(database_url="isolated", workflow=None)
    case.jobs, case.conversation = {}, "conversation"
    request = SimpleNamespace(url="http://localhost/api/conversations/conversation/messages", method="POST",
                              post_data="{}", post_data_json={}, headers={"content-type": "application/json"})
    route = SimpleNamespace(request=request, fulfill=Mock(), abort=Mock(), continue_=Mock())
    try:
        case._bridge(route)
        assert started.wait(1)
        route.fulfill.assert_called_once()
        assert case.jobs["new-run"]["thread"].is_alive()
    finally:
        release.set()
        case.jobs["new-run"]["thread"].join(timeout=2)
    assert case.jobs["new-run"]["outcome"]["run_id"] == "new-run"


def test_search_and_fetch_are_reserved_and_charged_conservatively(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.jsonl", limit_cny=Decimal("20"))
    ident = ledger.reserve_fixed(amount_cny=Decimal("0.005"), purpose="search")
    ledger.settle_fixed(ident)
    assert ledger.spent_or_reserved() == Decimal("0.005")


def test_map_search_is_also_reserved_at_the_documented_upper_unit_price(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.jsonl", limit_cny=Decimal("20"))
    ident = ledger.reserve_fixed(amount_cny=Decimal("0.003"), purpose="amap_search")
    ledger.settle_fixed(ident)
    assert ledger.spent_or_reserved() == Decimal("0.003")


def test_projection_keeps_a_valid_provider_candidate_identity():
    lookup = {"names": {}, "ids": set()}
    run = {"status": "needs_input", "result": {"candidates": [{"poi_id": "amap:B123456"}]}}
    assert project_observation(run, None, lookup)["candidate_ids"] == ["amap:B123456"]


def test_event_bundle_rejects_changed_dataset_or_sources(tmp_path):
    graph = tmp_path / "graph"; graph.write_bytes(b"graph")
    pois = tmp_path / "pois"; pois.write_bytes(b"pois")
    dataset = tmp_path / "dataset"; dataset.write_bytes(b"dataset")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    data = {"scope": "synthetic_development_only", "campus_truth": False,
            "graph_sha256": sha(graph), "poi_sha256": sha(pois), "dataset_sha256": sha(dataset),
            "fixtures": [{"case_id": "events_01", "stages": {}}]}
    fixture = tmp_path / "events.json"; fixture.write_text(json.dumps(data), encoding="utf-8")
    assert "events_01" in load_event_bundle(fixture, dataset, graph, pois)
    dataset.write_bytes(b"different")
    with pytest.raises(ValueError, match="dataset"):
        load_event_bundle(fixture, dataset, graph, pois)


def test_route_check_uses_independent_edges_and_active_closures():
    import networkx as nx

    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, length=5.0, highway="footway")
    graph.add_edge(2, 3, key=0, length=6.0, highway="footway")
    anchors = {"poi_001": {"node_id": 1}, "poi_004": {"node_id": 3}}
    lookup = {"names": {}, "ids": set(anchors)}
    edges = {(1, 2, 0), (2, 3, 0)}
    evidence = {"walk": {"permitted": edges, "unknown": set()}}
    result = {"response_kind": "route", "recommended_edge_ids": [[1, 2, 0], [2, 3, 0]],
              "route_state": {"start": {"poi_id": "poi_001"}, "end": {"poi_id": "poi_004"},
                              "travel_mode": "walk"}}
    assert inspect_delivery_route(graph, anchors, evidence, lookup, result, set())["verdict"] == "pass"
    failed = inspect_delivery_route(graph, anchors, evidence, lookup, result, {(2, 3, 0)})
    assert failed["verdict"] == "fail"
    assert "closed_edge_used" in failed["issues"]


def test_route_check_does_not_certify_unanchored_or_missing_edges():
    import networkx as nx

    graph = nx.MultiDiGraph()
    result = {"response_kind": "route", "route_state": {"travel_mode": "walk"}}
    checked = inspect_delivery_route(graph, {}, {}, {"names": {}, "ids": set()}, result, set())
    assert checked["verdict"] == "unknown"


def test_route_edges_are_still_checked_when_a_provider_endpoint_is_unanchored():
    import networkx as nx

    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, length=5.0, highway="footway")
    graph.add_edge(2, 3, key=0, length=6.0, highway="footway")
    anchors = {"poi_001": {"node_id": 1}}
    lookup = {"names": {}, "ids": set(anchors)}
    edges = {(1, 2, 0), (2, 3, 0)}
    evidence = {"walk": {"permitted": edges, "unknown": set()}}
    result = {"response_kind": "route", "recommended_edge_ids": [[1, 2, 0], [2, 3, 0]],
              "route_state": {"start": {"poi_id": "poi_001"}, "end": {"poi_id": "amap:ABC123"},
                              "travel_mode": "walk"}}
    checked = inspect_delivery_route(graph, anchors, evidence, lookup, result, set())
    assert checked["verdict"] == "unknown"
    assert "endpoint_anchor_missing" in checked["unknowns"]
    assert checked["length_m"] == 11.0
    closed = inspect_delivery_route(graph, anchors, evidence, lookup, result, {(2, 3, 0)})
    assert closed["verdict"] == "fail"
    assert "closed_edge_used" in closed["issues"]


def test_route_delivery_must_match_expected_origin_and_mode_not_just_task_state():
    from experiments.complex_multiturn_dev.graph_checks import inspect_expected_route_fields
    lookup = {"names": {}, "ids": {"poi_001", "poi_004"}}
    turn = {"checks": [{"field": "origin", "op": "eq", "value": "poi_001"},
                       {"field": "mode", "op": "eq", "value": "walk"}]}
    route = {"route_state": {"start": {"poi_id": "poi_004"}, "travel_mode": "bike"}}
    checked = inspect_expected_route_fields(turn, route, lookup)
    assert set(checked["failures"]) == {"route_origin_mismatch", "route_mode_mismatch"}
    route["route_state"] = {"start": {"poi_id": "poi_001"}, "travel_mode": "walk"}
    assert inspect_expected_route_fields(turn, route, lookup) == {"failures": [], "unknowns": []}
    route["route_state"].pop("start")
    assert "route_origin_missing" in inspect_expected_route_fields(turn, route, lookup)["unknowns"]


def test_radius_audit_rejects_unrequested_expansion_and_accepts_requested_changes():
    from experiments.complex_multiturn_dev.graph_checks import inspect_radius_transition
    assert inspect_radius_transition(1000, 2000, "请找附近球场") == "search_radius_change_unrequested"
    assert inspect_radius_transition(1000, 2000, "不要扩大范围") == "search_radius_change_unrequested"
    assert inspect_radius_transition(1000, 2000, "范围放宽一些") is None
    assert inspect_radius_transition(1000, 1500, "范围改成一千五百米") is None


def test_independent_reference_route_still_checks_the_stated_endpoints():
    from experiments.complex_multiturn_dev.graph_checks import inspect_reference_endpoints
    lookup = {"names": {}, "ids": {"poi_001", "poi_004", "poi_137"}}
    rubric = {"reference_origin": "poi_001", "reference_destination": "poi_137"}
    result = {"route_state": {"start": {"poi_id": "poi_004"}, "end": {"poi_id": "poi_137"}}}
    checked = inspect_reference_endpoints(rubric, result, lookup)
    assert checked["failures"] == ["reference_origin_mismatch"]
    assert checked["unknowns"] == []


def test_candidate_scope_oracle_rejects_out_of_range_and_allows_in_range():
    import networkx as nx
    from experiments.complex_multiturn_dev.candidate_checks import inspect_candidate_scope
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, length=600.0, highway="footway")
    graph.add_edge(2, 3, key=0, length=600.0, highway="footway")
    anchors = {"a": {"node_id": 1, "snap_distance_m": 0},
               "b": {"node_id": 2, "snap_distance_m": 0},
               "c": {"node_id": 3, "snap_distance_m": 0}}
    evidence = {"walk": {"permitted": {(1, 2, 0), (2, 3, 0)}, "unknown": set()}}
    assert inspect_candidate_scope(graph, anchors, evidence, "a", "b", "walk", 1000)["verdict"] == "pass"
    assert inspect_candidate_scope(graph, anchors, evidence, "a", "c", "walk", 1000)["verdict"] == "fail"
    assert inspect_candidate_scope(graph, anchors, evidence, "a", "missing", "walk", 1000)["verdict"] == "unknown"


def test_time_audit_does_not_treat_default_radius_as_an_explicit_time_only_limit():
    from experiments.complex_multiturn_dev.candidate_checks import time_distance_limit
    observed = {"origin": "a", "search_anchor": "a", "radius_m": 1000, "max_duration_s": 1200}
    assert time_distance_limit(observed) is None
    assert time_distance_limit({**observed, "search_radius_explicit": True}) == 1000
    assert time_distance_limit({**observed, "route_max_distance_m": 700}) == 700
    assert time_distance_limit({**observed, "search_anchor": "b", "search_radius_explicit": True}) is None


def test_scope_audit_does_not_apply_the_default_radius_to_time_only_search():
    from experiments.complex_multiturn_dev.candidate_audit import scope_radius_for_audit
    observed = {"origin": "a", "search_anchor": "a", "radius_m": 1000, "max_duration_s": 1200}
    assert scope_radius_for_audit(observed) is None
    assert scope_radius_for_audit({**observed, "search_radius_explicit": True}) == 1000
    assert scope_radius_for_audit({**observed, "search_anchor": "b"}) == 1000
    assert scope_radius_for_audit({**observed, "max_duration_s": None}) == 1000


def test_time_only_tool_audit_detects_an_unrequested_distance_cap():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_time_only_scope
    lookup = {"names": {}, "ids": {"a"}}
    observed = {"origin": "a", "mode": "walk", "max_duration_s": 1200, "route_max_distance_m": None}
    call = {"tool": "search_poi_candidates", "args": {"start": {"poi_id": "a"}, "mode": "walk",
             "max_distance_m": 1500, "constraints": {"max_duration_s": 1200}},
            "result": {"error": None}, "trusted_search_scope": {}}
    assert inspect_time_only_scope(observed, [call], lookup) is True
    call["args"]["max_distance_m"] = 1000
    assert inspect_time_only_scope(observed, [call], lookup) is False
    reach = {"tool": "find_reachable_places", "args": {"start": {"poi_id": "a"}, "mode": "walk",
              "max_time_min": 20, "constraints": {"max_duration_s": 1200}},
             "result": {"error": None}, "trusted_search_scope": {"_search_scope_distance_m": 1000}}
    assert inspect_time_only_scope(observed, [reach], lookup) is False
    reach.pop("trusted_search_scope")
    assert inspect_time_only_scope(observed, [reach], lookup) is None


def test_candidate_time_oracle_requires_one_witness_path_within_both_limits():
    import networkx as nx
    from experiments.complex_multiturn_dev.candidate_checks import inspect_candidate_time
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, length=100.0, highway="footway")
    graph.add_edge(2, 3, key=0, length=100.0, highway="footway")
    anchors = {"start": {"node_id": 1, "snap_distance_m": 10},
               "goal": {"node_id": 3, "snap_distance_m": 10}}
    evidence = {"walk": {"permitted": {(1, 2, 0), (2, 3, 0)}, "unknown": set()}}
    answer = inspect_candidate_time(graph, anchors, evidence, "start", "goal", "walk", 180, radius=250)
    assert answer["verdict"] == "pass"
    assert answer["route_edge_ids"] == [[1, 2, 0], [2, 3, 0]]
    assert inspect_candidate_time(graph, anchors, evidence, "start", "goal", "walk", 60, radius=250)["verdict"] != "pass"
    assert inspect_candidate_time(graph, anchors, evidence, "start", "goal", "walk", 180, radius=150)["verdict"] == "fail"
    assert inspect_candidate_time(graph, anchors, evidence, "start", "missing", "walk", 180)["verdict"] == "unknown"


def test_clarification_audit_requires_a_visible_question_about_the_missing_start():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_start_clarification
    expected = {"checks": [{"field": "clarification_slot", "op": "eq", "value": "start"}]}
    assert inspect_start_clarification(expected, {"response_kind": "clarify", "assistant_text": "你从哪里出发？",
        "clarify": {"slot": "start", "question": "你从哪里出发？"}})["verdict"] == "pass"
    assert inspect_start_clarification(expected, {"response_kind": "route"})["verdict"] == "fail"
    assert inspect_start_clarification(expected, {"response_kind": "chat", "message": "好的。"})["verdict"] == "unknown"


def test_ambiguous_destination_audit_requires_a_visible_choice_or_question():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_destination_clarification
    result = {"response_kind": "clarify", "assistant_text": "校内有几个体育馆，你要去哪一个？",
              "clarify": {"slot": "destination", "options": ["场馆甲", "场馆乙"]}}
    assert inspect_destination_clarification(result)["verdict"] == "pass"
    assert inspect_destination_clarification({"response_kind": "route"})["verdict"] == "fail"


def test_no_path_audit_requires_matching_endpoints_event_and_tool_evidence():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_expected_no_path
    lookup = {"names": {}, "ids": {"poi_001", "poi_423"}}
    fixture = {"kind": "blocked", "start_poi_id": "poi_001", "end_poi_id": "poi_423", "mode": "walk",
               "active_reference": {"verdict": "infeasible"}}
    item = {"event_proof": {"stage": "active", "active_ids": ["event-1"]},
            "consumed_event_ids": [["event-1"]], "tool_observations": [{"tool": "plan_route",
                "args": {"start": {"poi_id": "poi_001"}, "end": {"poi_id": "poi_423"}, "mode": "walk"},
                "result": {"reason_code": "graph_no_path"}}]}
    result = {"failure_code": "route_not_found", "assistant_text": "道路封闭，暂时无法通行。"}
    assert inspect_expected_no_path(fixture, item, result, lookup)["verdict"] == "pass"
    item["tool_observations"][0]["args"]["end"] = {"poi_id": "poi_001"}
    assert inspect_expected_no_path(fixture, item, result, lookup)["verdict"] == "unknown"


def test_activity_audit_detects_stale_requirements_and_repeated_candidates():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_candidate_semantics
    rule = {"activity": "badminton", "forbidden_activity": "basketball", "must_not_repeat": True}
    local = {"poi_a": {"activities": ["badminton"]}, "poi_b": {"activities": ["basketball"]}}
    check = inspect_candidate_semantics(rule, [{"poi_id": "poi_a"}], ["basketball"], local, [], {"poi_a"})
    assert "activity_state_conflict" in check["failures"]
    assert "forbidden_activity_retained" in check["failures"]
    assert "candidate_repeated" in check["failures"]
    check = inspect_candidate_semantics(rule, [{"poi_id": "poi_a"}], ["badminton"], local, [], set())
    assert check == {"failures": [], "unknowns": []}
    check = inspect_candidate_semantics(rule, [{"poi_id": "poi_b"}], ["badminton"], local, [], set())
    assert "candidate_activity_unverified:poi_b" in check["unknowns"]


def test_candidate_audit_detects_same_named_colocated_venue_with_new_id():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_candidate_semantics
    first = {"poi_id": "poi_a", "name": "示例球场", "lng": 114.36, "lat": 30.53}
    duplicate = {**first, "poi_id": "amap:B"}
    local = {"poi_a": {"activities": ["basketball"]}}
    result = inspect_candidate_semantics({"activity": "basketball", "must_not_repeat": True},
        [duplicate], ["basketball"], local, [], {"poi_a"}, previously_shown_rows=[first])
    assert "candidate_repeated_identity" in result["failures"]
    distinct = {**duplicate, "lng": 114.37}
    result = inspect_candidate_semantics({"activity": "basketball", "must_not_repeat": True},
        [distinct], ["basketball"], local, [], {"poi_a"}, previously_shown_rows=[first])
    assert "candidate_repeated_identity" not in result["failures"]


def test_audit_detects_an_unconfirmed_previous_destination_as_current_location():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_unconfirmed_position
    lookup = {"names": {"甲地": "poi_a", "乙地": "poi_b"}, "ids": {"poi_a", "poi_b"}}
    previous = {"start": {"name": "甲地"}, "end": {"name": "乙地"}}
    assert inspect_unconfirmed_position("继续步行", "接着从乙地往前走，下一个去哪儿？", previous, lookup) == [
        "previous_destination_as_unconfirmed_location"]
    assert inspect_unconfirmed_position("我已经到乙地了", "接着从乙地往前走，下一个去哪儿？", previous, lookup) == []
    assert inspect_unconfirmed_position("继续步行", "你要从乙地出发吗？", previous, lookup) == []


def test_audit_detects_contradictory_saved_stop_order_in_arrow_and_split_forms():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_saved_stop_orders
    lookup = {"names": {"起点": "poi_start", "甲": "poi_a", "乙": "poi_b"},
              "ids": {"poi_start", "poi_a", "poi_b"}}
    expected = ["poi_b", "poi_a"]
    assert inspect_saved_stop_orders(["顺序：起点→甲→乙", "顺序：起点→乙→甲"], expected, lookup) == [
        "stale_stop_order_retained"]
    assert inspect_saved_stop_orders(["先到甲", "再到乙", "先到乙", "再到甲"], expected, lookup) == [
        "stale_stop_order_retained"]
    assert inspect_saved_stop_orders(["避开台阶", "顺序：起点→乙→甲"], expected, lookup) == []


def test_activity_audit_uses_direct_provider_type_evidence():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_candidate_semantics
    candidates = [{"poi_id": "amap:A123", "name": "某咖啡", "lat": 30.53, "lng": 114.36}]
    raw = [{"id": "A123", "name": "某咖啡", "type": "餐饮服务;咖啡厅", "typecode": "050500",
            "location": "114.36,30.53"}]
    assert inspect_candidate_semantics({"activity": "coffee"}, candidates, ["coffee"], {}, raw, set()) == {
        "failures": [], "unknowns": []}
    raw[0]["location"] = "114.37,30.53"
    assert "candidate_activity_unverified:amap:A123" in inspect_candidate_semantics(
        {"activity": "coffee"}, candidates, ["coffee"], {}, raw, set())["unknowns"]


def test_audit_detects_denial_of_an_identified_provider_type_record():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_provider_type_denial
    candidate = {"poi_id": "amap:EXAMPLE", "name": "示例运动场馆", "lng": 114.36, "lat": 30.53}
    raw = [{"id": "EXAMPLE", "name": "示例运动场馆", "type": "体育休闲服务;运动场馆;羽毛球场",
            "typecode": "080118", "location": "114.36,30.53"}]
    assert inspect_provider_type_denial("示例运动场馆记录里没有羽毛球场信息。", [candidate], raw) == [
        "provider_type_denied:amap:EXAMPLE:badminton"]
    assert inspect_provider_type_denial("示例运动场馆在地图中标为羽毛球场，是否开放尚未核实。", [candidate], raw) == []
    mismatched = [{**raw[0], "location": "114.38,30.53"}]
    assert inspect_provider_type_denial("示例运动场馆记录里没有羽毛球场信息。", [candidate], mismatched) == []


def test_activity_audit_rejects_a_provider_sports_place_displayed_as_dining():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_candidate_semantics
    candidates = [{"poi_id": "amap:A123", "name": "校园篮球场", "lat": 30.53, "lng": 114.36,
                   "type": "dining"}]
    raw = [{"id": "A123", "name": "校园篮球场", "type": "体育休闲服务;运动场馆;篮球场",
            "typecode": "080104", "location": "114.36,30.53"}]
    result = inspect_candidate_semantics({"activity": "basketball"}, candidates, ["basketball"], {}, raw, set())
    assert "candidate_type_conflict:amap:A123" in result["failures"]
    candidates[0]["type"] = "sports"
    assert inspect_candidate_semantics({"activity": "basketball"}, candidates, ["basketball"], {}, raw, set()) == {
        "failures": [], "unknowns": []}


def test_search_rubric_distinguishes_missing_and_unrequested_route_distance():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_route_distance_requirement
    rule = {"route_distance_absent": True}
    assert inspect_route_distance_requirement(rule, {"route_max_distance_m": None}) == {"failures": [], "unknowns": []}
    assert inspect_route_distance_requirement(rule, {"route_max_distance_m": 2000})["failures"] == ["unrequested_route_distance"]
    assert inspect_route_distance_requirement(rule, {})["unknowns"] == ["route_distance_unverified"]
    assert inspect_route_distance_requirement({}, {"route_max_distance_m": 2000}) == {"failures": [], "unknowns": []}


def test_sport_clarification_audit_requires_a_question_before_candidates():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_sport_clarification
    question = {"response_kind": "clarify", "assistant_text": "想打哪种球？",
                "clarify": {"question": "想打哪种球？", "slot": "sport"}, "place_request": {"activities": []}}
    assert inspect_sport_clarification(question)["verdict"] == "pass"
    question["place_request"]["activities"] = ["basketball", "badminton"]
    assert inspect_sport_clarification(question)["verdict"] == "fail"
    assert inspect_sport_clarification({"response_kind": "candidates", "candidates": [{"poi_id": "poi_001"}]})["verdict"] == "fail"
    assert inspect_sport_clarification({"response_kind": "chat", "assistant_text": "好的"})["verdict"] == "unknown"


def test_activity_audit_accepts_explicit_saved_requirements_without_structured_activity():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_candidate_semantics
    local = {"poi_a": {"activities": ["coffee"]}}
    result = inspect_candidate_semantics({"activity": "coffee"}, [{"poi_id": "poi_a"}], [], local, [], set(),
                                         requirements=["咖啡店", "避开台阶"])
    assert result == {"failures": [], "unknowns": []}
    negated = inspect_candidate_semantics({"activity": "coffee"}, [{"poi_id": "poi_a"}], [], local, [], set(),
                                          requirements=["不要咖啡店"])
    assert "activity_state_unverified" in negated["unknowns"]


def test_search_center_audit_detects_a_stale_old_location_requirement():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_stale_center_requirement
    lookup = {"names": {"旧地点": "a", "新地点": "b"}, "ids": {"a", "b"}}
    assert inspect_stale_center_requirement("a", "b", ["旧地点附近", "避开台阶"], lookup) == [
        "previous_search_center_retained_in_requirements"]
    assert inspect_stale_center_requirement("a", "b", ["新地点附近", "避开台阶"], lookup) == []


def test_provider_anchor_requires_matching_raw_identity_and_coordinates():
    import networkx as nx
    from experiments.complex_multiturn_dev.candidate_checks import provider_anchor
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.36, y=30.53)
    candidate = {"poi_id": "amap:A123", "name": "某咖啡", "lng": 114.36, "lat": 30.53}
    source = {"id": "A123", "name": "某咖啡", "type": "餐饮服务;咖啡厅", "location": "114.36,30.53"}
    anchor = provider_anchor(graph, {1}, candidate, [source], to_wgs84=lambda x, y: (x, y))
    assert anchor["node_id"] == 1 and anchor["poi_id"] == "amap:A123"
    source["location"] = "114.37,30.53"
    assert provider_anchor(graph, {1}, candidate, [source], to_wgs84=lambda x, y: (x, y)) is None


def test_empty_search_requires_matching_scope_category_and_honest_message():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_empty_search
    lookup = {"names": {}, "ids": {"poi_001"}}
    obs = {"origin": "poi_001", "mode": "walk", "radius_m": 100, "activities": ["badminton"]}
    call = {"tool": "search_poi_candidates", "args": {"start": {"poi_id": "poi_001"},
            "mode": "walk", "max_distance_m": 100, "activities": ["badminton"]},
            "result": {"count": 0, "online_status": "ok", "error": None}}
    result = {"response_kind": "chat", "assistant_text": "当前范围没查到羽毛球场。", "candidates": []}
    assert inspect_empty_search({"activity": "badminton"}, obs, result, [call], lookup)["verdict"] == "pass"
    result["assistant_text"] = "本次检索没有返回可核验的羽毛球场候选记录。"
    assert inspect_empty_search({"activity": "badminton"}, obs, result, [call], lookup)["verdict"] == "pass"
    result["assistant_text"] = "没查到羽毛球场，说明附近路网数据不足。"
    assert inspect_empty_search({"activity": "badminton"}, obs, result, [call], lookup)["verdict"] == "fail"
    result["assistant_text"] = "本次检索没有返回可核验的羽毛球场候选记录。"
    failed_call = deepcopy(call)
    failed_call["result"]["online_status"] = "provider_error"
    assert inspect_empty_search({"activity": "badminton"}, obs, result, [failed_call], lookup)["verdict"] == "unknown"
    incomplete = deepcopy(call)
    incomplete["args"]["subcategories"] = ["sports"]
    assert inspect_empty_search({"activity": "badminton"}, obs, result, [incomplete], lookup)["verdict"] == "unknown"
    call["args"]["max_distance_m"] = 500
    assert inspect_empty_search({"activity": "badminton"}, obs, result, [call], lookup)["verdict"] == "unknown"


def test_dataset_rejects_duplicate_ids_or_exposed_formal_claim():
    data = read_dataset(DATASET)
    broken = deepcopy(data)
    broken["cases"][1]["id"] = broken["cases"][0]["id"]
    with pytest.raises(ValueError):
        validate_dataset(broken)
    broken = deepcopy(data)
    broken["formal_test"] = True
    with pytest.raises(ValueError):
        validate_dataset(broken)


def test_activity_audit_distinguishes_withdrawal_from_a_retained_activity():
    from experiments.complex_multiturn_dev.semantic_checks import inspect_candidate_semantics
    rule = {"activity": "meal", "forbidden_activity": "coffee"}
    candidate = [{"poi_id": "poi_a"}]
    local = {"poi_a": {"activities": ["canteen_meal"]}}
    withdrawn = inspect_candidate_semantics(
        rule, candidate, ["meal"], local, [], set(),
        requirements=["附近的吃饭地点，不再找咖啡店"],
    )
    assert "forbidden_activity_retained" not in withdrawn["failures"]
    retained = inspect_candidate_semantics(
        rule, candidate, ["meal"], local, [], set(),
        requirements=["附近的吃饭地点，仍要找咖啡店"],
    )
    assert "forbidden_activity_retained" in retained["failures"]



@pytest.mark.parametrize("autocrlf", ["false", "true"])
def test_frozen_sources_survive_git_newline_settings(tmp_path, autocrlf):
    import subprocess
    from experiments.complex_multiturn_dev.event_runtime import load_event_bundle

    paths = [
        "data/whu_road_network.graphml",
        "data/pois.json",
        "experiments/datasets/complex_multiturn_dev_v1/cases.json",
        "experiments/datasets/complex_multiturn_dev_v1/cases_event_endpoint_revised.json",
        "experiments/datasets/complex_multiturn_dev_v1/anchor_snapshot_60m.json",
        "experiments/complex_multiturn_dev/candidate_checks.py",
        "experiments/datasets/complex_multiturn_dev_v1/supplement_time_only.json",
        "experiments/datasets/complex_multiturn_dev_v1/supplement_time_radius.json",
        "experiments/datasets/complex_multiturn_dev_v1/supplement_time_radius_v2.json",
    ]
    prefix = tmp_path.as_posix() + "/"
    subprocess.run(["git", "-c", "core.autocrlf=" + autocrlf, "checkout-index", "--prefix=" + prefix, "--", *paths],
                   cwd=ROOT, check=True, capture_output=True)
    copied = lambda path: tmp_path / path
    assert copied(paths[2]).read_bytes() == (ROOT / paths[2]).read_bytes()
    assert load_event_bundle(DATASET.parent / "event_fixtures_v2.json", copied(paths[3]), copied(paths[0]), copied(paths[1]))
    proof = json.loads((DATASET.parent / "supplement_time_only_preflight.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(copied(paths[4]).read_bytes()).hexdigest() == proof["anchor_sha256"]
    assert hashlib.sha256(copied(paths[5]).read_bytes()).hexdigest() == proof["checker_sha256"]
    for path in paths[6:]:
        assert copied(path).read_bytes() == (ROOT / path).read_bytes()
