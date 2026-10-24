"""Audit hashed development pilot usage and estimate a planning-only model cost.

The result deliberately cannot authorize a formal batch. Reports lacking cache
accounting receive a possible-cost interval, not a fabricated exact bill.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = {"luojia", "ordinary_tool_agent", "deepseek_harness"}
GROUPS = {"T1", "T2", "T3", "T4", "T5"}


def _read_report(root: Path, item: dict) -> dict:
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root.resolve() / "output" / "experiments"):
        raise ValueError("cost report must be under ignored output/experiments")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"cost report is too large: {item['path']}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError(f"cost report hash mismatch: {item['path']}")
    return json.loads(raw)


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid usage {label}")
    return value


def _usage(report: dict, condition: str) -> dict:
    if condition == "luojia":
        traces = report.get("model_trace")
        if not isinstance(traces, list) or not traces:
            raise ValueError("product pilot lacks model trace")
        ids = {entry.get("resolved_model_id") for entry in traces}
        if len(ids) != 1:
            raise ValueError("product response models differ")
        entries = traces
        model_id = ids.pop()
    else:
        measured = report.get("measured_usage")
        if not isinstance(measured, dict):
            raise ValueError("baseline pilot lacks measured usage")
        entries = [measured]
        model_id = report.get("resolved_model_id")
    requests = sum(_nonnegative_int(row.get("requests", 1), "requests") for row in entries)
    inputs = sum(_nonnegative_int(row.get("input_tokens"), "input_tokens") for row in entries)
    outputs = sum(_nonnegative_int(row.get("output_tokens"), "output_tokens") for row in entries)
    if not model_id or requests < 1:
        raise ValueError("pilot has no observed response model or requests")
    cache_known = all("cache_hit_input_tokens" in row and "cache_miss_input_tokens" in row
                      for row in entries)
    hits = misses = None
    if cache_known:
        hits = sum(_nonnegative_int(row["cache_hit_input_tokens"], "cache hits")
                   for row in entries)
        misses = sum(_nonnegative_int(row["cache_miss_input_tokens"], "cache misses")
                     for row in entries)
        if hits + misses != inputs:
            raise ValueError("cache accounting differs from total input tokens")
    return {"response_model_id": model_id, "requests": requests,
            "input_tokens": inputs, "output_tokens": outputs,
            "cache_hit_input_tokens": hits, "cache_miss_input_tokens": misses,
            "cache_accounting_observed": cache_known}


def _price(usage: dict, rates: dict) -> dict:
    inputs, outputs = usage["input_tokens"], usage["output_tokens"]
    minimum = (inputs * rates["input_cache_hit"] + outputs * rates["output"]) / 1_000_000
    maximum = (inputs * rates["input_cache_miss"] + outputs * rates["output"]) / 1_000_000
    exact = None
    if usage["cache_accounting_observed"]:
        exact = (usage["cache_hit_input_tokens"] * rates["input_cache_hit"]
                 + usage["cache_miss_input_tokens"] * rates["input_cache_miss"]
                 + outputs * rates["output"]) / 1_000_000
    return {"cache_all_hit_usd": round(minimum, 6),
            "cache_all_miss_usd": round(maximum, 6),
            "cache_observed_usd": round(exact, 6) if exact is not None else None}


def estimate(root: Path, catalog: dict, pricing: dict) -> dict:
    if pricing.get("unit") != "USD_per_1M_tokens":
        raise ValueError("unexpected pricing unit")
    rows = []
    seen = set()
    for item in catalog["reports"]:
        key = (item["cohort"], item["group"], item["condition"])
        if key in seen or item["group"] not in GROUPS or item["condition"] not in CONDITIONS:
            raise ValueError("duplicate or invalid pilot stratum")
        seen.add(key)
        report = _read_report(root, item)
        if report.get("case_id") is None or not str(report.get("status", "")).startswith("development_pilot_only"):
            raise ValueError("only labelled development pilot reports are accepted")
        usage = _usage(report, item["condition"])
        if usage["response_model_id"] != pricing["model"]:
            raise ValueError("response model differs from pricing model")
        rows.append({**{name: item[name] for name in ("cohort", "group", "condition", "path", "sha256")},
                     "case_id": report["case_id"], "usage": usage,
                     "off_peak": _price(usage, pricing["off_peak"]),
                     "peak": _price(usage, pricing["peak"])})
    matched = [row for row in rows if row["cohort"] == "matched_T5_weather"]
    if {row["condition"] for row in matched} != CONDITIONS or len(matched) != 3:
        raise ValueError("T5 weather cohort must contain exactly the three conditions")
    if len({row["case_id"] for row in matched}) != 1:
        raise ValueError("T5 weather cohort case IDs differ")
    matched_reports = [_read_report(root, item) for item in catalog["reports"]
                       if item["cohort"] == "matched_T5_weather"]
    for field in ("development_dataset_sha256", "case_messages_sha256",
                  "tool_schema_sha256", "data_component_sha256"):
        values = [report.get(field) for report in matched_reports]
        if any(value is None for value in values) or len({json.dumps(value, sort_keys=True)
                                                           for value in values}) != 1:
            raise ValueError(f"matched T5 pilots differ on {field}")
    fixtures = [report.get("weather_fixture") for report in matched_reports]
    if any(not isinstance(item, dict) or not item.get("sha256") for item in fixtures):
        raise ValueError("matched T5 pilots lack weather fixture provenance")
    if len({item["sha256"] for item in fixtures}) != 1:
        raise ValueError("matched T5 pilots used different weather fixtures")
    # A single visible case scaled to its 24 x 3 planned runs is a planning
    # illustration only. Other groups and repeated token distributions are missing.
    t5_runs_per_condition = 24 * 3
    illustration = {
        "status": "single_visible_case_scaling_not_a_budget_bound",
        "runs_per_condition": t5_runs_per_condition,
        "peak_all_miss_usd": round(sum(row["peak"]["cache_all_miss_usd"]
                                       for row in matched) * t5_runs_per_condition, 4),
        "off_peak_cache_all_hit_usd": round(sum(row["off_peak"]["cache_all_hit_usd"]
                                             for row in matched) * t5_runs_per_condition, 4),
    }
    coverage = {group: {condition: len({row["case_id"] for row in rows
                                        if row["group"] == group and row["condition"] == condition})
                        for condition in sorted(CONDITIONS)} for group in sorted(GROUPS)}
    return {"status": "development_cost_audit_only", "formal_budget_ready": False,
            "pricing": pricing, "rows": rows, "matched_T5_illustration": illustration,
            "observed_distinct_cases_by_group_condition": coverage,
            "formal_blockers": [
                "Only one matched T5 case; no three-condition T1-T4 coverage or per-group repeat distributions.",
                "Some Harness reports lack cache hit/miss accounting; observed bill is unknown.",
                "Current official prices, Search/Fetch charges and a user-approved batch ceiling must be rechecked before formal paid calls.",
            ]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path,
                        default=ROOT / "experiments/fixtures/development_cost_inputs_20261024.json")
    parser.add_argument("--pricing", type=Path,
                        default=ROOT / "experiments/fixtures/deepseek_flash_pricing_20261024.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = estimate(ROOT, json.loads(args.catalog.read_text(encoding="utf-8")),
                      json.loads(args.pricing.read_text(encoding="utf-8")))
    report["catalog_sha256"] = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    report["pricing_sha256"] = hashlib.sha256(args.pricing.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"rows": len(report["rows"]),
                      "formal_budget_ready": report["formal_budget_ready"],
                      "matched_T5_peak_all_miss_usd": report["matched_T5_illustration"]["peak_all_miss_usd"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
