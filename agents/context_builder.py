"""Explicit context-selection rules for multi-turn route conversations."""
from __future__ import annotations

import re


_ROUTE_FOLLOWUP = re.compile(
    r"换一条|再规划|重新规划|再短一点|更短一点|改走|从那里|从这里|从那儿|"
    r"到那里|到这里|终点改为|起点改为|这条路线|刚才(?:的)?路线|沿途|"
    r"(?:切换到?|改成|改为|换成|改走).{0,5}(?:步行|走路|骑行|骑车|开车|驾车)|增加途经"
)
_FRESH_ROUTE = re.compile(r"从.{1,35}(?:到|去).{1,35}(?:走|路线|怎么|最短)?")


def named_destination_followup(query: str) -> list[dict]:
    """Find a named-place alternative without treating an activity as a place.

    Exact local names and aliases are used here: a fuzzy match must not silently
    turn a question about something else into a route destination.
    """
    text = (query or "").strip().rstrip("？?！!。 ")
    prefix = r"(?:那|那么|要不|要不然|不然|或者|还是)?"
    ending = r"(?:不行吗|可以吗|行不行|能行吗|行吗|怎么样|呢|吧)?"
    patterns = (
        prefix + r"(?:把|将)?(?:终点|目的地)(?:改为|改成|改到|换成|换为|换到|设为|定为)(.+?)" + ending,
        prefix + r"(?:换成|换为|换到|换去|改成|改为|改到|改去)(?:去|到)?(.+?)" + ending,
        prefix + r"(?:去|到)?(.+?)(?:不行吗|可以吗|行不行|能行吗|行吗|怎么样|呢)",
    )
    name = next((match.group(1).strip() for pattern in patterns
                 if (match := re.fullmatch(pattern, text))), "")
    if not 2 <= len(name) <= 50 or re.search(r"[，,；;。？！?!]", name):
        return []
    from spatial.poi import find_poi_candidates

    return [poi for poi, score in find_poi_candidates(name, limit=20, min_score=1.0)
            if score >= 1.0]


def is_implicit_destination_followup(query: str) -> bool:
    """An elliptical alternative needs the immediately preceding route."""
    return bool(named_destination_followup(query)) and not re.search(
        r"路线|终点|目的地", query or "",
    )


def is_route_followup(query: str) -> bool:
    """Return true only when the user explicitly refers to a prior route."""
    text = (query or "").strip()
    if not text or not (_ROUTE_FOLLOWUP.search(text) or named_destination_followup(text)):
        return False
    # An explicit pair of new endpoints starts a new route unless the user also
    # explicitly asks to modify the previous one.
    if _FRESH_ROUTE.search(text) and not re.search(r"刚才|这条路线|原路线", text):
        return False
    return True


def build_agent_context(query: str, history: list[dict] | None,
                        previous_route_state: dict | None = None,
                        *, same_task: bool = False) -> dict:
    """Keep a short transcript, but carry route slots only on explicit follow-up."""
    transcript = [
        {"role": item.get("role"), "content": str(item.get("content", ""))[:500]}
        for item in (history or [])[-8:]
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
        and item.get("content")
    ]
    context = {"history": transcript}
    if previous_route_state and (same_task or is_route_followup(query)):
        context["previous_route_state"] = previous_route_state
    if transcript:
        context["context_policy"] = (
            "本轮是同一未完成任务的澄清续答：保留原任务的目标、途经点、出行方式与约束；"
            "本轮用户明确修正的内容优先。不要把回答当成新的独立任务。"
            if same_task else
            "仅在本轮明确引用上一条路线时继承其起终点和约束；新地点、新问题或出行方式切换不得"
            "从旧任务推断端点。用户本轮明确提供的信息优先。"
        )
    return context
