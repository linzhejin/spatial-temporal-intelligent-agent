"""Extra checks explicitly implied by the independent-reference-route prompts.

Added during development after detecting that the original machine checks did
not specify the reference-route origin. This only strengthens the rubric and
does not turn an earlier failure into a success.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/datasets/complex_multiturn_dev_v1"


def main():
    dataset = DATA / "cases_event_endpoint_revised.json"
    cases = {row["id"]: row for row in json.loads(dataset.read_text(encoding="utf-8"))["cases"]}
    extra = {
        "clarification_04": {"reference_origin": "poi_004", "reference_destination": "poi_137",
                             "quoted_basis": "万林艺术博物馆到凌波门"},
        "clarification_08": {"reference_origin": "poi_140", "reference_destination": "poi_004",
                             "quoted_basis": "珞珈门到万林艺术博物馆"},
    }
    for ident, row in extra.items():
        if row["quoted_basis"] not in cases[ident]["turns"][0]["user"]:
            raise ValueError("reference endpoint evidence does not match the user prompt")
    result = {"scope": "development_rubric_strengthening", "formal_test": False,
              "note": "补充核对题目明示的参考路线两端；先前运行及原标准完整保留。",
              "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "cases": extra}
    path = DATA / "reference_route_rubric.json"
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(path)
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    print(str(path))


if __name__ == "__main__":
    main()
