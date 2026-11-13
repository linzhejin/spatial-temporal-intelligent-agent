"""Freeze additional user-visible activity and clarification requirements."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/datasets/complex_multiturn_dev_v1"


def main():
    dataset = DATA / "cases_event_endpoint_revised.json"
    cases = {row["id"]: row for row in json.loads(dataset.read_text(encoding="utf-8"))["cases"]}
    activities = {
        "search_01": ["coffee", "meal", "meal"],
        "search_02": ["basketball", "badminton", "badminton"],
        "search_03": ["badminton", "badminton", "badminton"],
        "search_04": ["basketball", "basketball", "basketball"],
        "search_05": ["coffee", "coffee", "coffee"],
        "search_06": ["basketball", "basketball", None],
        "search_07": ["basketball", "badminton", "badminton"],
        "search_08": ["coffee", "coffee", None, None],
        "clarification_02": [None, "badminton", "badminton"],
    }
    rules = {}
    for ident, expected in activities.items():
        if len(expected) != len(cases[ident]["turns"]):
            raise ValueError("rubric length differs from frozen case")
        rules[ident] = [{"activity": activity, "must_not_repeat": ident == "search_04" and index > 0}
                        for index, activity in enumerate(expected)]
        if ident in {"search_01", "search_02", "search_07"}:
            forbidden = "coffee" if ident == "search_01" else "basketball"
            for row in rules[ident][1:]:
                row["forbidden_activity"] = forbidden
    rules["clarification_03"] = [{"clarification": "ambiguous_destination"}, {}, {}]
    result = {"scope": "development_rubric_strengthening", "formal_test": False,
              "note": "按题目原文明示的活动及换批要求补足检查；只增强，不替换或放宽原标准。",
              "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "cases": rules}
    path = DATA / "semantic_rubric_v2.json"
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(path)
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    print(str(path))
    strengthened = deepcopy(result)
    strengthened["note"] = "按题目原文明示的活动、换批和路线距离条件补足检查；v2 保留，v3 只增强不放宽原标准。"
    for ident, rows in strengthened["cases"].items():
        if ident.startswith("search_"):
            for row in rows:
                row["route_distance_absent"] = True
    path = DATA / "semantic_rubric_v3.json"
    text = json.dumps(strengthened, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(path)
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    print(str(path))
    clarified = deepcopy(strengthened)
    clarified["note"] = "在 v3 基础上补查模糊球类需求的必要澄清；保留旧版本，只增强不放宽原标准。"
    clarified["cases"]["clarification_02"][0]["clarification"] = "ambiguous_sport"
    path = DATA / "semantic_rubric_v4.json"
    text = json.dumps(clarified, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(path)
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    print(str(path))


if __name__ == "__main__":
    main()
