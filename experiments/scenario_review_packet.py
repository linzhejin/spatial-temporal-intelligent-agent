"""Select a small development-only semantic review packet without filling gold."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any


def build_review_packet(dataset: dict[str, Any], *, per_category: int = 3,
                        seed: int = 20261026) -> dict[str, Any]:
    """Sample draft cases by category; proposals remain explicitly unreviewed."""
    if dataset.get("split") != "development":
        raise ValueError("review packet must use development drafts")
    if type(per_category) is not int or per_category < 1:
        raise ValueError("per_category must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    cases = dataset.get("cases")
    if not isinstance(cases, list):
        raise ValueError("dataset cases must be a list")
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        case_id = case.get("id")
        category = case.get("category")
        if (not isinstance(case_id, str) or not case_id
                or case_id in seen_ids or not isinstance(category, str) or not category):
            raise ValueError("case IDs and categories must be unique/nonempty")
        seen_ids.add(case_id)
        if (case.get("adjudication") or {}).get("status") == "draft":
            grouped.setdefault(category, []).append(case)
    if not grouped:
        raise ValueError("no draft cases available")
    selected = []
    for category in sorted(grouped):
        group = sorted(grouped[category], key=lambda case: case["id"])
        if len(group) < per_category:
            raise ValueError(f"not enough draft cases in {category}")
        rng = random.Random(f"{seed}:{category}")
        selected.extend(sorted(rng.sample(group, per_category), key=lambda case: case["id"]))
    cards = []
    for case in selected:
        messages = case.get("messages")
        turns = case.get("expected_turns")
        if not isinstance(messages, list) or not isinstance(turns, list) or len(messages) != len(turns):
            raise ValueError(f"{case['id']}: message/turn mismatch")
        cards.append({
            "case_id": case["id"],
            "category": case["category"],
            "messages": [message.get("content") for message in messages],
            "proposed_interpretation": [{
                key: turn.get(key) for key in (
                    "status", "kind", "feasibility", "start_poi_id", "end_poi_id",
                    "via_poi_ids", "travel_mode", "strategy", "hard_constraints",
                    "required_outputs", "candidate_poi_ids", "clarification_slot",
                )
            } for turn in turns],
            "review_status": "pending",
            "human_decision": None,
            "proposal_is_gold": False,
        })
    return {
        "format": "development-semantic-review-packet-v1",
        "source_split": "development",
        "source_version": dataset.get("version"),
        "seed": seed,
        "per_category": per_category,
        "card_count": len(cards),
        "formal_gold": False,
        "cards": cards,
    }


def render_review_packet(packet: dict[str, Any], poi_names: dict[str, str]) -> str:
    """Render a compact human semantics checklist, not a route-field verdict."""
    def place(poi_id: Any) -> str:
        if not isinstance(poi_id, str) or not poi_id:
            return "未指定"
        name = poi_names.get(poi_id)
        return f"{name}（{poi_id}）" if name else f"未在目录中找到（{poi_id}）"

    lines = [
        "# 场景语义审查卡（开发集）", "",
        f"共 {packet['card_count']} 张；来源版本：`{packet.get('source_version') or '未标记'}`。",
        "**以下是草稿建议，不是金标准。这里只请核对用户意思、地点和条件；路线实际可走性另行核验。**",
        "", "每张卡可答“正确”或指出要改的地方；暂不能判断就写“未知”。", "",
    ]
    for number, card in enumerate(packet["cards"], 1):
        lines.extend([f"## {number}. {card['case_id']}（{card['category']}）", ""])
        for turn_no, (message, proposed) in enumerate(
                zip(card["messages"], card["proposed_interpretation"], strict=True), 1):
            vias = proposed.get("via_poi_ids") or []
            lines.extend([
                f"**第 {turn_no} 轮原话：**{message}", "",
                f"- 建议处理：{proposed.get('status') or '未标记'} / {proposed.get('kind') or '未标记'}",
                f"- 起点：{place(proposed.get('start_poi_id'))}；终点：{place(proposed.get('end_poi_id'))}",
                f"- 按顺序途经：{' → '.join(place(item) for item in vias) if vias else '无'}",
                f"- 出行方式：{proposed.get('travel_mode') or '未指定'}；策略：{proposed.get('strategy') or '未指定'}",
                f"- 条件：`{json.dumps(proposed.get('hard_constraints') or {}, ensure_ascii=False, sort_keys=True)}`",
                f"- 待补证据：路线可行性={proposed.get('feasibility') or '未标记'}；审查结论=待人工确认",
                "",
            ])
        lines.extend(["**你的判断：**待人工确认", "", "---", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("experiments/datasets/scenarios_dev_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--pois", type=Path, default=Path("data/pois.json"))
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    source_bytes = args.dataset.read_bytes()
    packet = build_review_packet(json.loads(source_bytes),
                                 per_category=args.per_category, seed=args.seed)
    packet["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(packet, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if args.markdown_output:
        poi_data = json.loads(args.pois.read_text(encoding="utf-8"))
        poi_names = {item["id"]: item["name"] for item in poi_data["pois"]}
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        with args.markdown_output.open("x", encoding="utf-8") as stream:
            stream.write(render_review_packet(packet, poi_names))
    print(json.dumps({"output": str(args.output), "cards": packet["card_count"],
                      "formal_gold": False,
                      "markdown_output": str(args.markdown_output) if args.markdown_output else None},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
