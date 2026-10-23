from collections import Counter
from pathlib import Path

from experiments.scenario_v2 import validate_dataset
from scripts.analysis.build_scenario_v2_development import build_development_bank


ROOT = Path(__file__).resolve().parents[1]


def test_v2_development_bank_keeps_60_cases_and_real_clarifications():
    dataset = build_development_bank(
        ROOT / "experiments/datasets/scenarios_dev_v1.json")
    validate_dataset(dataset)
    assert len(dataset["cases"]) == 60
    counts = Counter(case["category"] for case in dataset["cases"])
    assert counts["T3"] == 10
    assert all(case["adjudication"]["status"] == "draft"
               for case in dataset["cases"])
    copied = [case for case in dataset["cases"] if case.get("source_case_id")]
    assert len(copied) == 52
    assert {case["source_group"] for case in copied} == {
        "scenarios_dev_v1_authoring_bank"}
    assert len(dataset["source_sha256"]) == 64
    clarifications = [case for case in dataset["cases"]
                      if case["category"] == "T3"]
    assert all(case["expected_turns"][-1]["kind"] == "clarify"
               for case in clarifications)
