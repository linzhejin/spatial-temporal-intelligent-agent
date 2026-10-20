"""离线蒸馏经验知识层：data/agent_experience.jsonl → data/experience_cards.json。

运行时只跑 spec 脚本本身；Agent 主链路只读蒸馏产物，不在线改写提示词。

用法：
    python scripts/analysis/build_experience_cards.py
    python scripts/analysis/build_experience_cards.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents import experience  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="蒸馏 Agent 经验卡")
    parser.add_argument("--input", type=Path, default=experience.EXPERIENCE_PATH,
                        help="样本 JSONL 路径（默认 data/agent_experience.jsonl）")
    parser.add_argument("--output", type=Path, default=experience.CARDS_PATH,
                        help="经验卡输出路径（默认 data/experience_cards.json）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印统计，不写文件")
    args = parser.parse_args(argv)

    if args.dry_run:
        payload = _preview(args.input)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    payload = experience.distill(args.input, args.output)
    cards = payload.get("cards", [])
    confident = sum(1 for c in cards
                    if experience._is_confident(c))
    print(f"样本文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"聚合模式 {len(cards)} 条，其中可召回（高置信）{confident} 条")
    return 0


def _preview(input_path: Path) -> dict:
    """不落盘地预览蒸馏结果。"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        payload = experience.distill(input_path, Path(tmp) / "preview.json")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
