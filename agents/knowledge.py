"""珞珈智行 — 知识注入（三层知识库）。

OpenGA 把知识分为工具库 / 知识库 / 技能库三层。本模块把它落到 Agent 上下文：

- 能力知识层：工具能力的结构与边界，事实源在 agents/tool_skills.py
  （由 planner 以固定前缀注入，保证提示词前段稳定；本模块只提供访问器）；
- 领域知识层：开发者沉淀的校园场景任务卡（data/task_cards.json）；
- 经验知识层：从已成功/已核验的编排中离线蒸馏出的经验卡（agents/experience.py）。

运行期只读：任务卡与经验卡都只读不改，不在线改写提示词。
用户个体的"动态记忆"（EMA 画像）在 agents/profile.py。
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from agents import experience, tool_skills

logger = logging.getLogger(__name__)

_CARDS_PATH = Path(__file__).parent.parent / "data" / "task_cards.json"
_cards_cache = None

MAX_CARDS_PER_QUERY = 2  # 注入上限，避免稀释主提示词

# 月份 → 季节，用于 season_hint 加权
_SEASON_BY_MONTH = {
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
    12: "winter", 1: "winter", 2: "winter",
}


def load_cards() -> list:
    """加载任务卡（进程内缓存）。"""
    global _cards_cache
    if _cards_cache is None:
        try:
            data = json.loads(_CARDS_PATH.read_text(encoding="utf-8"))
            _cards_cache = data.get("cards", [])
        except Exception as e:
            logger.warning("任务卡加载失败，按空知识库运行: %s", e)
            _cards_cache = []
    return _cards_cache


def retrieve_domain_knowledge(query: str, limit: int = MAX_CARDS_PER_QUERY,
                              now: datetime = None) -> list:
    """领域知识层：按触发词命中数召回任务卡；季节匹配的卡 +1 分。

    纯关键词子串匹配——卡片数量少（<50），不需要向量检索。
    """
    if not query:
        return []
    season = _SEASON_BY_MONTH[(now or datetime.now()).month]
    scored = []
    for card in load_cards():
        hits = sum(1 for t in card.get("triggers", []) if t and t in query)
        if hits == 0:
            continue
        score = hits + (1 if card.get("season_hint") == season else 0)
        scored.append((score, card))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:limit]]


# 兼容既有调用点与测试：旧名指向领域知识层
retrieve_cards = retrieve_domain_knowledge


def retrieve_capability_knowledge() -> str:
    """能力知识层：工具能力说明表（事实源在 agents/tool_skills.py）。"""
    return tool_skills.build_skill_manifest()


def retrieve_experience_knowledge(query: str, limit: int = None) -> list:
    """经验知识层：召回高置信的成功编排经验卡。"""
    if limit is None:
        return experience.retrieve_experience(query)
    return experience.retrieve_experience(query, limit=limit)


def _domain_message(query: str, now: datetime = None) -> str | None:
    cards = retrieve_domain_knowledge(query, now=now)
    if not cards:
        return None
    lines = ["校园场景经验（供决策参考，优先级低于用户显式要求）："]
    for c in cards:
        line = f"【{c['name']}】{c['advice']}"
        if c.get("default_weights"):
            w = c["default_weights"]
            line += (f" 参考权重 distance={w['distance']}, "
                     f"slope={w['slope']}, scenery={w['scenery']}。")
        lines.append(line)
    return "\n".join(lines)


def _experience_message(query: str) -> str | None:
    cards = retrieve_experience_knowledge(query)
    if not cards:
        return None
    lines = ["历史成功编排（仅统计已核验通过或明确成功的执行，供编排参考）："]
    for c in cards:
        sequence = " → ".join(c.get("tool_sequence") or [])
        lines.append(
            f"【{c.get('scenario_key')}】成功 {c.get('success_count')} 次、"
            f"失败 {c.get('failure_count')} 次：{sequence}"
        )
    return "\n".join(lines)


def build_knowledge_message(query: str, now: datetime = None) -> str | None:
    """生成注入 LLM 的知识消息文本；无命中返回 None。

    只拼接「经验 → 领域」两层：能力知识层由 planner 以固定前缀单独注入，
    避免同一张能力说明表被重复塞进上下文（也保住提示词前段可缓存）。

    经验层在前：它是从真实成功执行中沉淀的编排模式，比人工写的场景经验
    更贴近当前实际调用方式；两者都明确低于用户显式要求的优先级。
    """
    layers = [layer for layer in (
        _experience_message(query),
        _domain_message(query, now=now),
    ) if layer]
    if not layers:
        return None
    return "\n".join(layers)
