"""Export a small human review pack from unadjudicated scenario-v2 drafts.

The suggested answers are authoring hypotheses, never independent gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "experiments/datasets/scenarios_test_v2_draft.json"
DEFAULT_OUTPUT = ROOT / "docs/research/10_任务歧义审查卡_待确认.md"

REVIEW_FAMILIES = (
    ("T3", "ball", "追问球类；保留出发点。可选篮球、羽毛球、乒乓球、足球、其他球类。"),
    ("T3", "exercise", "追问运动类型；保留出发点。可选跑步、打球、健身、其他。"),
    ("T3", "destination_weather", "目的地缺失；可先回答天气，再问去哪里，不能沿用旧路线终点。"),
    ("T5", "sport_then_food", "复合任务不能直接给一条固定路线；先确定篮球场，再按其位置检索餐饮。"),
    ("T5", "food_before_route", "先给附近正餐候选，确定场所后再规划经该场所到终点。"),
)
EXTRA_CASE = ("T4", "weather_after", 1,
              "只补充天气；保留原路线、起终点和方式，不重新解释为新目的地。")


def build_cards(dataset: dict) -> list[dict]:
    if dataset.get("protocol_version") != "scenario-v2" or dataset.get("split") != "test":
        raise ValueError("expected scenario-v2 test authoring bank")
    cases = {case["id"]: case for case in dataset.get("cases", [])}
    selected = []
    for group, family, suggestion in REVIEW_FAMILIES:
        for variant in (1, 2, 3):
            case_id = f"v2_{group}_{family}_{variant}"
            case = cases[case_id]
            if case["adjudication"]["status"] != "draft":
                raise ValueError(f"{case_id}: review pack only accepts drafts")
            selected.append({"id": case_id, "group": group,
                             "prompt": " / ".join(msg["content"] for msg in case["messages"]),
                             "suggestion": suggestion,
                             "current_gold_kind": case["expected_turns"][-1]["kind"]})
    group, family, variant, suggestion = EXTRA_CASE
    case_id = f"v2_{group}_{family}_{variant}"
    case = cases[case_id]
    selected.append({"id": case_id, "group": group,
                     "prompt": " / ".join(msg["content"] for msg in case["messages"]),
                     "suggestion": suggestion,
                     "current_gold_kind": case["expected_turns"][-1]["kind"]})
    return selected


def render_cards(cards: list[dict], dataset_sha256: str) -> str:
    lines = [
        "# 任务歧义审查卡（待确认）", "",
        f"- 来源：`scenarios_test_v2_draft.json`，SHA-256 `{dataset_sha256}`。",
        f"- 共 {len(cards)} 张。以下答案是程序侧**建议**，不是独立金标准。",
        "- 请只在建议与实际用户意图不一致处改写。未答复前所有卡片状态为 `pending`。",
        "- 审查完成后另行生成独立裁决记录；本文件本身不会使测试集可正式计分。", "",
    ]
    for number, card in enumerate(cards, 1):
        lines.extend([
            f"## {number}. {card['id']}", "",
            f"**用户说法：** {card['prompt']}", "",
            f"**建议处理：** {card['suggestion']}", "",
            f"**现有草稿结果类型：** `{card['current_gold_kind']}`（仍需检查，尤其复合任务）。", "",
            "**你的判断：** □ 同意建议　□ 修改：________", "",
        ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.dataset.read_bytes()
    cards = build_cards(json.loads(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_cards(cards, hashlib.sha256(source).hexdigest()), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cards": len(cards), "status": "pending"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
