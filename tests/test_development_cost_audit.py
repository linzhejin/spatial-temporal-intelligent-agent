import hashlib
import json

import pytest

from scripts.analysis.estimate_development_cost import _price, _usage, estimate


def test_cache_accounting_is_observed_only_when_reported():
    ordinary = {"resolved_model_id": "deepseek-flash", "measured_usage": {
        "requests": 2, "input_tokens": 100, "output_tokens": 10,
        "cache_hit_input_tokens": 80, "cache_miss_input_tokens": 20,
    }}
    known = _usage(ordinary, "ordinary_tool_agent")
    rates = {"input_cache_hit": 1, "input_cache_miss": 2, "output": 3}
    assert known["cache_accounting_observed"] is True
    assert _price(known, rates)["cache_observed_usd"] == 0.00015
    harness = {"resolved_model_id": "deepseek-flash", "measured_usage": {
        "requests": 2, "input_tokens": 100, "output_tokens": 10,
    }}
    assert _price(_usage(harness, "deepseek_harness"), rates)["cache_observed_usd"] is None
    ordinary["measured_usage"]["cache_miss_input_tokens"] = 19
    with pytest.raises(ValueError, match="cache accounting"):
        _usage(ordinary, "ordinary_tool_agent")


def test_product_deterministic_pilot_records_zero_model_cost():
    report = {"run_evidence": [
        {"status": "needs_input", "model_requests": []},
        {"status": "completed", "model_requests": []},
    ]}
    usage = _usage(report, "luojia")
    assert usage["response_model_id"] is None
    assert usage["requests"] == 0
    assert usage["input_tokens"] == usage["output_tokens"] == 0
    estimate = _price(usage, {"input_cache_hit": 1, "input_cache_miss": 2,
                              "output": 3})
    assert estimate["cache_all_miss_usd"] == 0
    assert estimate["cache_observed_usd"] is None


def test_zero_request_pilot_with_http_audit_has_observed_zero_cost():
    report = {"run_evidence": [{"model_requests": []}],
              "provider_http_attempt_count": 0}
    usage = _usage(report, "luojia")
    assert usage["provider_http_attempt_count"] == 0
    assert _price(usage, {"input_cache_hit": 1, "input_cache_miss": 2,
                          "output": 3})["cache_observed_usd"] == 0


def test_product_live_pilot_sums_observed_model_requests_across_turns():
    report = {"run_evidence": [
        {"model_requests": [{"resolved_model_id": "deepseek-flash",
                             "input_tokens": 100, "output_tokens": 10,
                             "cache_hit_input_tokens": 80,
                             "cache_miss_input_tokens": 20}]},
        {"model_requests": [{"resolved_model_id": "deepseek-flash",
                             "input_tokens": 200, "output_tokens": 20,
                             "cache_hit_input_tokens": 150,
                             "cache_miss_input_tokens": 50}]},
    ]}
    usage = _usage(report, "luojia")
    assert usage["response_model_id"] == "deepseek-flash"
    assert usage["requests"] == 2
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 30
    assert usage["cache_hit_input_tokens"] == 230
    assert usage["cache_miss_input_tokens"] == 70


def test_cost_audit_requires_hashed_reports_and_matching_weather(tmp_path):
    root = tmp_path
    output = root / "output" / "experiments"
    output.mkdir(parents=True)
    shared = {"case_id": "visible-T5", "status": "development_pilot_only",
              "development_dataset_sha256": "a", "case_messages_sha256": "b",
              "tool_schema_sha256": "c", "data_component_sha256": {"pois": "d"},
              "weather_fixture": {"sha256": "weather"}}
    items = []
    for condition in ("luojia", "ordinary_tool_agent", "deepseek_harness"):
        report = dict(shared)
        if condition == "luojia":
            report["model_trace"] = [{"resolved_model_id": "deepseek-flash",
                                      "input_tokens": 100, "output_tokens": 10,
                                      "cache_hit_input_tokens": 80,
                                      "cache_miss_input_tokens": 20}]
        else:
            report["resolved_model_id"] = "deepseek-flash"
            report["measured_usage"] = {"requests": 1, "input_tokens": 100,
                                        "output_tokens": 10}
        relative = f"output/experiments/{condition}.json"
        raw = json.dumps(report).encode()
        (output / f"{condition}.json").write_bytes(raw)
        items.append({"cohort": "matched_T5_weather", "group": "T5",
                      "condition": condition, "path": relative,
                      "sha256": hashlib.sha256(raw).hexdigest()})
    pricing = {"model": "deepseek-flash", "unit": "USD_per_1M_tokens",
               "off_peak": {"input_cache_hit": 1, "input_cache_miss": 2, "output": 3},
               "peak": {"input_cache_hit": 2, "input_cache_miss": 4, "output": 6}}
    catalog = {"reports": items}
    result = estimate(root, catalog, pricing)
    assert result["formal_budget_ready"] is False
    assert result["observed_distinct_cases_by_group_condition"]["T5"]["luojia"] == 1
    assert result["observed_distinct_cases_by_group_condition"]["T1"]["luojia"] == 0
    catalog["reports"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        estimate(root, catalog, pricing)
    catalog["reports"][0]["sha256"] = hashlib.sha256(
        (output / "luojia.json").read_bytes()).hexdigest()
    path = output / "luojia.json"
    report = json.loads(path.read_text())
    report["weather_fixture"]["sha256"] = "different"
    raw = json.dumps(report).encode()
    path.write_bytes(raw)
    catalog["reports"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="different weather fixtures"):
        estimate(root, catalog, pricing)


def test_cost_audit_keeps_zero_request_product_row_in_matched_group(tmp_path):
    output = tmp_path / "output" / "experiments"
    output.mkdir(parents=True)
    items = []
    for cohort, group in (("matched_T5_weather", "T5"), ("matched_20261025", "T1")):
        for condition in ("luojia", "ordinary_tool_agent", "deepseek_harness"):
            report = {"case_id": f"visible-{group}", "status": "development_pilot_only",
                      "development_dataset_sha256": "a", "case_messages_sha256": group,
                      "tool_schema_sha256": "c", "data_component_sha256": {"pois": "d"},
                      "weather_fixture": {"sha256": "weather"}}
            if condition == "luojia" and group == "T1":
                report["run_evidence"] = [{"model_requests": []}]
            elif condition == "luojia":
                report["model_trace"] = [{"resolved_model_id": "deepseek-flash",
                                          "input_tokens": 100, "output_tokens": 10}]
            else:
                report["resolved_model_id"] = "deepseek-flash"
                report["measured_usage"] = {"requests": 1, "input_tokens": 100,
                                            "output_tokens": 10}
            path = output / f"{cohort}_{condition}.json"
            raw = json.dumps(report).encode()
            path.write_bytes(raw)
            items.append({"cohort": cohort, "group": group, "condition": condition,
                          "path": path.relative_to(tmp_path).as_posix(),
                          "sha256": hashlib.sha256(raw).hexdigest()})
    pricing = {"model": "deepseek-flash", "unit": "USD_per_1M_tokens",
               "off_peak": {"input_cache_hit": 1, "input_cache_miss": 2, "output": 3},
               "peak": {"input_cache_hit": 2, "input_cache_miss": 4, "output": 6}}
    result = estimate(tmp_path, {"reports": items}, pricing)
    row = next(row for row in result["rows"]
               if row["cohort"] == "matched_20261025" and row["condition"] == "luojia")
    assert row["usage"]["requests"] == 0
    assert row["peak"]["cache_all_miss_usd"] == 0
    assert result["observed_distinct_cases_by_group_condition"]["T1"]["luojia"] == 1
    assert any("1/5 task groups" in blocker for blocker in result["formal_blockers"])
    assert any("zero-request" in blocker for blocker in result["formal_blockers"])


def test_matched_development_group_rejects_different_case_messages(tmp_path):
    output = tmp_path / "output" / "experiments"
    output.mkdir(parents=True)
    items = []
    for cohort, group in (("matched_T5_weather", "T5"), ("matched_20261025", "T1")):
        for condition in ("luojia", "ordinary_tool_agent", "deepseek_harness"):
            report = {"case_id": f"visible-{group}", "status": "development_pilot_only",
                      "development_dataset_sha256": "a",
                      "case_messages_sha256": "wrong" if group == "T1" and condition == "deepseek_harness" else group,
                      "tool_schema_sha256": "c", "data_component_sha256": {"pois": "d"},
                      "weather_fixture": {"sha256": "weather"}}
            if condition == "luojia":
                report["model_trace"] = [{"resolved_model_id": "deepseek-flash",
                                          "input_tokens": 100, "output_tokens": 10}]
            else:
                report["resolved_model_id"] = "deepseek-flash"
                report["measured_usage"] = {"requests": 1, "input_tokens": 100,
                                            "output_tokens": 10}
            path = output / f"{cohort}_{condition}.json"
            raw = json.dumps(report).encode()
            path.write_bytes(raw)
            items.append({"cohort": cohort, "group": group, "condition": condition,
                          "path": path.relative_to(tmp_path).as_posix(),
                          "sha256": hashlib.sha256(raw).hexdigest()})
    pricing = {"model": "deepseek-flash", "unit": "USD_per_1M_tokens",
               "off_peak": {"input_cache_hit": 1, "input_cache_miss": 2, "output": 3},
               "peak": {"input_cache_hit": 2, "input_cache_miss": 4, "output": 6}}
    with pytest.raises(ValueError, match="matched development pilots differ on case_messages_sha256"):
        estimate(tmp_path, {"reports": items}, pricing)
