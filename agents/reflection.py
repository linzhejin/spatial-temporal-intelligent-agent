"""珞珈智行 — 路线核验失败的反思与定向重规划（GAReflector 式）。

现有链路在核验失败后只回注一句泛化固定指令（「依据违规项修正参数」），
模型往往不知道到底该改什么，容易用同一组参数再撞一次墙。
本模块把 spatial/result_verifier 的结构化报告（checks / violations / unknowns）
转成可执行的诊断，再由 planner 注入定向修正指令。

边界：
- 只诊断，不重算、不改约束、不放宽硬约束；
- 只消费已核验输出，不信任规划器自报；
- 重试预算与轮次/时间护栏仍由 planner 负责，本模块只给「该不该再试」的判断。

诊断键：
- constraint_vs_connectivity：硬约束剔除与连通性冲突，或用到被排除/封闭边
- budget_exceeded：距离或时间超预算
- stale_event_snapshot：路网/事件版本与规划快照不一致
- endpoint_unresolved：起终点吸附不到路网
- invalid_request_shape：结果结构或形状非法
- evidence_insufficient：全部为 unknown，没有任何可修正的确定性违规
- unclassified：有失败但不在上述分组
"""

from __future__ import annotations

# 诊断键 → 定向修正指令（必须具体到「改什么参数」，而不是笼统「再试一次」）
_ADVICE = {
    "constraint_vs_connectivity": (
        "硬约束与路网连通性冲突：检查是否存在过度收紧的约束"
        "（如同时要求避开陡坡与避开台阶），或误用了被排除/已封闭的边。"
        "只能按用户的原话调整可调整项；用户没松口的约束不得自行放宽。"
    ),
    "budget_exceeded": (
        "距离或时间超出用户给出的预算：先缩短路线或减少途经点/游览点，"
        "不要改写用户给定的预算数字。"
    ),
    "stale_event_snapshot": (
        "路网或路况版本与核验快照不一致：重新读取当前路况/数据版本后再规划，"
        "不要沿用上一轮的边序列或旧快照。"
    ),
    "endpoint_unresolved": (
        "起点或终点无法吸附到可用路网：改用最近的可达校门/道路点，"
        "或向用户说明该点与道路的接驳未核实，不要换成更远的节点硬算。"
    ),
    "invalid_request_shape": (
        "结果结构非法：检查工具参数的类型与必填项（如 via_points 是否有序、"
        "legs 段数是否在 2~4 之间），按 schema 修正后重试。"
    ),
    "evidence_insufficient": (
        "本次失败来自证据不足（未知项），不是确定性违规："
        "不要重复调用同一组参数，应如实说明哪些条件无法核实。"
    ),
    "unclassified": (
        "按核验返回的违规项逐条修正参数；若无法定位原因，改用更简单的一次调用"
        "（如先 plan_route 再补充其他需求）。"
    ),
}

# 分组优先级：越靠前越可能是真正的根因
_GROUP_PRIORITY = (
    ("stale_event_snapshot", ("road_conditions",)),
    ("invalid_request_shape", ("route_structure",)),
    ("endpoint_unresolved", ("endpoints",)),
    ("constraint_vs_connectivity", ("mode_and_constraints", "exact_edges")),
    ("budget_exceeded", ("duration_and_budget",)),
)

_GROUP_EXPLANATION = {
    "route_structure": "结果结构",
    "road_conditions": "路网/事件版本",
    "endpoints": "起终点吸附",
    "exact_edges": "边序列",
    "mode_and_constraints": "出行方式与约束",
    "length_and_geometry": "长度与画线",
    "duration_and_budget": "时长与预算",
    "other": "其他",
}


def _checks_of(verification: dict) -> dict:
    checks = verification.get("checks")
    return checks if isinstance(checks, dict) else {}


def diagnose(verification: dict) -> dict:
    """把 result_verifier 的结构化报告转成定向诊断。

    入参是 tools 返回的错误字典里的 route_verification（或直接是核验报告）。
    """
    verification = verification if isinstance(verification, dict) else {}
    checks = _checks_of(verification)

    failed = []
    unknown = []
    for name, entry in checks.items():
        entry = entry if isinstance(entry, dict) else {}
        status = entry.get("status")
        issues = [str(item) for item in (entry.get("issues") or [])]
        if status == "fail":
            failed.append({"name": name, "issues": issues})
        elif status == "unknown":
            unknown.append({"name": name, "issues": issues})

    failed_names = {item["name"] for item in failed}
    primary = None
    for key, groups in _GROUP_PRIORITY:
        if failed_names & set(groups):
            primary = key
            break

    if primary is None:
        if failed:
            primary = "unclassified"
        elif unknown:
            primary = "evidence_insufficient"
        else:
            # 没有结构化的 checks（旧格式报告）：退回状态判断
            status = verification.get("status")
            issues = [str(item) for item in (verification.get("issues") or [])]
            if status == "unknown":
                primary = "evidence_insufficient"
            elif status in ("fail", "unknown") or issues:
                primary = "unclassified"
            else:
                primary = None

    should_give_up = primary == "evidence_insufficient"

    return {
        "primary_conflict": primary,
        "failed_checks": failed,
        "unknown_checks": [item["name"] for item in unknown],
        "failed_groups": [item["name"] for item in failed],
        "advice": _ADVICE.get(primary, "") if primary else "",
        "should_give_up": should_give_up,
        "status": verification.get("status"),
    }


def summarize(diagnosis: dict) -> str:
    """一句话诊断摘要，用于终态答复（让用户看到是约束冲突而非系统故障）。

    未分类失败不产出摘要——终态答复本身会说明未通过核验，重复一遍反而含糊。
    """
    if not isinstance(diagnosis, dict):
        return ""
    key = diagnosis.get("primary_conflict")
    if not key or key == "unclassified":
        return ""
    labels = {
        "constraint_vs_connectivity": "当前通行约束与路网连通性冲突",
        "budget_exceeded": "距离或时间超出你给的预算",
        "stale_event_snapshot": "路况已变化，需要按最新路况重算",
        "endpoint_unresolved": "起点或终点无法接入可通行路网",
        "invalid_request_shape": "请求格式不完整",
        "evidence_insufficient": "关键通行条件缺少可核实的证据",
        "unclassified": "路线未通过核验",
    }
    groups = [g for g in (diagnosis.get("failed_groups") or [])
              if g in _GROUP_EXPLANATION]
    text = labels.get(key, "路线未通过核验")
    if groups:
        text += "（未通过项：" + "、".join(_GROUP_EXPLANATION[g] for g in groups) + "）"
    return text


def build_reflection_message(diagnosis: dict, attempt: int = 1) -> str:
    """生成注入 system 的定向重规划指令。

    attempt=1：按诊断精确修正；
    attempt>=2：换策略（改端点/换更简单的工具），仍不得放宽用户约束。
    """
    advice = (diagnosis or {}).get("advice") or _ADVICE["unclassified"]
    if attempt <= 1:
        head = ("路径工具的结果未通过独立核验。这是一次定向修正机会："
                f"{advice}")
    else:
        head = ("路径工具的结果仍未通过独立核验，这是最后一次修正机会。"
                "请换一种做法：优先改用更简单的一次调用（如 plan_route），"
                "或调整起终点到可达道路；仍不得省略或放宽用户明确提出的约束。"
                f"{advice}")
    tail = ("不要用同一组参数重复调用同一工具；若确实无法满足，"
            "不得声称路线已生成，也不要省略硬约束，应如实说明原因。")
    return head + tail
