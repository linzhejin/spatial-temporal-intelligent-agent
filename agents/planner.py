"""珞珈智行 — Agent 规划器（Plan-Act-Observe 循环）。

v2 全 Agent 架构的唯一入口：所有用户输入（含明确 A→B、闲聊、天气、路况）
都进入本循环，由 LLM 决策调用工具或直接回答。

护栏：
- 轮次上限 MAX_TURNS、总时长 TIME_BUDGET_S（gunicorn timeout 60s 内留余量）
- 工具参数/执行失败 → 错误信息回注，LLM 自我修正
- LLM API 故障 → 抛 PlannerError，由 routes.py 落回旧管道（停电保险）
"""

import json
import logging
import math
import re
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx

import config
from agents import tools as agent_tools
from agents import experience, knowledge, profile, provenance, reflection, tool_skills
from agents.preferences import detect_strategy_hint, route_preference_requested
from agents.timings import normalize_timings
from spatial.route_constraints import positive_budget

logger = logging.getLogger(__name__)

MAX_TURNS = 6
TIME_BUDGET_S = 40.0
LLM_MAX_TOKENS = 800

# 核验失败后的定向修正预算（反思-重规划）。次数上限之外，仍受 MAX_TURNS 与
# TIME_BUDGET_S 双重约束：剩余轮次不足或剩余时间不够一次 LLM 往返时不再重试。
MAX_REFLECTION_ATTEMPTS = 2
_REFLECTION_MIN_BUDGET_S = 8.0

_PROMPT_PATH = Path(__file__).parent / "prompts" / "agent_system.txt"
_system_prompt_cache = None  # NOTE: 修改 agent_system.txt 后需清此缓存或重启进程


class PlannerError(Exception):
    """LLM 层面的故障（网络/鉴权/限流），触发路由层兜底。"""


def _load_system_prompt() -> str:
    global _system_prompt_cache
    if _system_prompt_cache is None:
        _system_prompt_cache = _PROMPT_PATH.read_text(encoding="utf-8")
    return _system_prompt_cache


def _make_client():
    from openai import OpenAI
    return OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        # Some supported runtimes cannot decode Brotli responses from this API.
        default_headers={"Accept-Encoding": "gzip, deflate"},
        # 循环内每次调用给 25s read；总时长由 TIME_BUDGET_S 兜底
        timeout=httpx.Timeout(connect=5.0, read=25.0, write=10.0, pool=5.0),
        # SDK 默认会对超时自动重试；两次 25 秒重试可耗尽任务截止时间。
        max_retries=0,
    )


# DeepSeek 偶发把内部 DSML 工具调用标记当纯文本吐出，必须剥掉，绝不能进前端
_DSML_RE = re.compile(r"<\s*[｜|]{2,}\s*DSML[\s\S]*", re.IGNORECASE)


def _sanitize_message(text: str) -> str:
    if not text:
        return ""
    return _DSML_RE.split(text)[0].strip()


_MODE_VERB = {"walk": "步行", "bike": "骑行", "drive": "开车"}


def _fmt_dist(m):
    if m is None:
        return None
    return f"{m / 1000:.1f} 公里" if m >= 1000 else f"{round(m)} 米"


def _build_route_message(route: dict) -> str:
    """路线工具成功后直接用结构化数据拼一句话，不再调 LLM（快且不会泄漏 DSML）。"""
    mode = route.get("mode", "walk")
    verb = _MODE_VERB.get(mode, "步行")
    s_name = route.get("start_name") or "起点"
    e_name = route.get("end_name") or "终点"
    length = route.get("recommended_length_m") or route.get("distance_m")
    dur = route.get("duration_min")
    shortest = route.get("shortest_length_m")
    shortest_dur = route.get("shortest_duration_min")

    intermediate_names = [
        str(leg.get("end_name")).strip()
        for leg in (route.get("legs") or [])[:-1]
        if isinstance(leg, dict) and leg.get("end_name")
    ]
    via_text = ""
    if intermediate_names:
        via_text = "经" + "、".join(intermediate_names[:3])
        if len(intermediate_names) > 3:
            via_text += f"等 {len(intermediate_names)} 处"
    head = f"已为你规划好从{s_name}{via_text}到{e_name}的{verb}路线"
    ld = _fmt_dist(length)
    if ld:
        head += f"，约 {ld}"
        if dur is not None:
            head += f"、{dur:g} 分钟"
    # 最短路线明显更短（>7%）才提示，地图上灰虚线可直接对比
    tail = ""
    if (route.get("shortest_baseline_feasible") is not False
            and shortest and length and shortest < length * 0.93):
        sd = _fmt_dist(shortest)
        if sd:
            tail = f"最短路线约 {sd}"
            if shortest_dur is not None:
                tail += f"、{shortest_dur:g} 分钟"
            tail += "，地图灰虚线可对比"
    return head + "。" + (tail + "。" if tail else "")


def _road_condition_feedback(route: dict, request: str) -> str:
    if not re.search(r"路况|管制|封路|封闭|禁行|拥堵|积水|绕行|绕路|(?:能否|是否|能不能|可以).{0,4}(?:通行|走通)", request or ""):
        return ""
    applied = route.get("road_conditions_applied")
    if type(applied) is not int:
        return " 本次未取得可核实的路况应用信息。"
    if not applied:
        return " 本次算路未应用生效路况记录；未覆盖路段仍需现场确认。"
    mode_label = {"walk": "步行", "bike": "骑行", "drive": "驾车"}.get(
        route.get("mode"), "本次出行方式")
    mode_scope = "本次各段出行方式" if route.get("road_condition_scope") == "multi_mode" else mode_label
    message = f" 本次算路纳入 {applied} 条适用于{mode_scope}的生效路况记录。"
    overview = route.get("road_condition_overview")
    if not isinstance(overview, list) or not overview:
        return message
    labels = {"closure": "封闭", "slowdown": "缓行", "notice": "提醒"}
    id_counts = {}
    for event in overview:
        if isinstance(event, dict) and isinstance(event.get("id"), (str, int)) and not isinstance(event.get("id"), bool):
            identifier = event["id"]
            id_counts[identifier] = id_counts.get(identifier, 0) + 1
    along_ids = {event["id"] for event in route.get("route_events", [])
                 if isinstance(event, dict) and isinstance(event.get("id"), (str, int))
                 and not isinstance(event.get("id"), bool) and isinstance(event.get("action"), str)
                 and event.get("action") in {"notice", "slowdown"}
                 and id_counts.get(event["id"]) == 1} if isinstance(route.get("route_events"), list) else set()
    def is_along(event):
        identifier = event.get("id")
        return (isinstance(identifier, (str, int)) and not isinstance(identifier, bool)
                and identifier in along_ids and event.get("action") in ("notice", "slowdown"))
    ordered = sorted((event for event in overview if isinstance(event, dict)),
                     key=lambda event: 0 if is_along(event) else 1)
    items = []
    for event in ordered[:3]:
        name = event.get("name") if isinstance(event.get("name"), str) else "未命名路况"
        road = event.get("road_name") if isinstance(event.get("road_name"), str) else ""
        raw_action = event.get("action")
        action = labels.get(raw_action, "路况") if isinstance(raw_action, str) else "路况"
        modes = event.get("travel_modes")
        mode_text = "、".join({"walk": "步行", "bike": "骑行", "drive": "驾车"}[mode]
                             for mode in modes if isinstance(mode, str)
                             and mode in {"walk", "bike", "drive"}) if isinstance(modes, list) else ""
        along = "沿途，" if is_along(event) else ""
        items.append(f"{name}（{along}{road or '位置未标注'}，{action}{'，' + mode_text if mode_text else ''}）")
    if items:
        omitted = max(0, applied - len(items))
        remainder = f"另有 {omitted} 条生效记录。" if omitted else ""
        caveat = ("这些记录与无法规划之间的关系尚未确认。" if route.get("no_route")
                  else "这些记录不一定影响本路线。")
        message += " 当前记录：" + "；".join(items) + "。" + remainder + caveat
    return message


def _build_route_comparison_message(comparison: dict) -> str:
    """Summarize a comparison without asking the model to restate its measurements."""
    start = comparison.get("start_name") or "起点"
    end = comparison.get("end_name") or "终点"
    verb = _MODE_VERB.get(comparison.get("mode", "walk"), "步行")
    alternatives = comparison.get("alternatives") or []
    unavailable = comparison.get("unavailable") or []
    if not alternatives:
        return (f"目前无法生成从{start}到{end}的{verb}方案。"
                "请检查起终点和通行限制后再试。")
    message = (f"已比较从{start}到{end}的{verb}方案："
               f"{len(alternatives)} 种可用，{len(unavailable)} 种未能生成。"
               "距离和估算时间见下方；比较不会改变当前地图路线。")
    return message


def _route_verification_failure_response(result, *, message, turns=0):
    verification = (result or {}).get("route_verification") if isinstance(result, dict) else None
    issues = set(verification.get("issues") or []) if isinstance(verification, dict) else set()
    if (isinstance(verification, dict) and verification.get("status") == "unknown"
            and issues == {"slope_evidence_unknown"}):
        message = "部分路段缺少可核实的坡度信息，无法确认这条路线满足坡度要求；本次未展示新路线。"
        detail = ""
    else:
        detail = str((result or {}).get("message") or "") if isinstance(result, dict) else ""
    if detail:
        message += f"原因：{detail}"
    return {
        "response_kind": "chat", "message": message,
        "route": None, "route_kind": None, "candidates": None,
        "clarify": None, "suggestions": None, "turns": turns,
        "route_verification": verification,
    }


_NAMED_VIA_RE = re.compile(
    r"^(?:请|带我|帮我)?\s*(?:从(?P<start>.+?))?"
    r"(?:途经|经过|经由|经)(?P<via>.+?)(?:去|到)"
    r"(?P<end>[^，,。]+)"
)


def _named_via_request(text: str) -> dict | None:
    """Recognize an explicit, single named stop before the destination."""
    match = _NAMED_VIA_RE.match((text or "").strip())
    if not match:
        return None
    via = match.group("via").strip()
    end = re.split(r"\s*(?:不走|避开|顺便|同时|查天气)", match.group("end"), maxsplit=1)[0].strip()
    if not via or not end or re.search(r"、|和|及|再经|途经|经过", via):
        return None
    if any(agent_tools.is_open_ended_activity_stop(point) for point in (via, end)):
        return None
    return {"start": (match.group("start") or "").strip(), "via": via, "end": end}


def _explicit_start_name(query: str) -> str | None:
    """Return a clearly named origin so a map-selected start does not override it."""
    text = (query or "").strip()
    field = re.search(r"(?:起点|出发地)(?:设为|是|为|在|[:：])\s*([^，,。；;]+)", text)
    if field:
        name = field.group(1).strip()
        return name if name and not agent_tools.is_deictic_origin(name) else None

    # Common conversational form: “我在教五这边，帮我导航到图书馆”.
    located = re.search(r"(?:^|[，,。；;\s])我(?:现在)?在\s*([^，,。；;]{1,40})", text)
    if located:
        name = re.split(r"(?:出发|前往|去|到|帮我|请你|规划|导航|带我)",
                        located.group(1), maxsplit=1)[0].strip()
        name = re.sub(r"(?:这边|附近|旁边|周围|一带)$", "", name).strip()
        if name and name not in {"我这", "我这里", "我这边", "这里", "当前位置", "当前"} \
                and not agent_tools.is_deictic_origin(name):
            return name

    from_match = re.search(
        r"(?:从|由)\s*([^，,。；;]{1,30}?)\s*(?:出发|到|去|前往)", text
    )
    if from_match:
        name = from_match.group(1).strip()
        if name and name not in {"我这", "我这里", "我的位置", "当前位置", "地图起点"} \
                and not agent_tools.is_deictic_origin(name):
            return name
        if name == "地图起点":
            return name
        return None

    direct_match = re.match(r"^([^，,。；;]{2,20}?)\s*(?:到|至|去|前往)", text)
    if not direct_match:
        return None
    name = direct_match.group(1).strip()
    # Bare imperative/request lead-ins are not place names. Without this guard,
    # phrases such as “带我去最近的食堂” look like an origin named “带我”.
    if re.match(
        r"^(?:我想|想要|我要|我希望|帮我|帮忙|请帮我|请你|请问|请|麻烦|带我|带我们|带您|送我|给我|推荐我|现在)",
        name,
    ):
        return None
    return name if name and not agent_tools.is_deictic_origin(name) else None


def _manual_map_start_ref(coord_start: dict) -> dict | None:
    if not isinstance(coord_start, dict) or coord_start.get("name") != "地图起点":
        return None
    try:
        lng, lat = float(coord_start.get("lng")), float(coord_start.get("lat"))
    except (TypeError, ValueError):
        return None
    if not (-180 <= lng <= 180 and -90 <= lat <= 90):
        return None
    return {"name": "地图起点", "type": "coord", "lng": lng, "lat": lat}


def _asks_for_origin(question: str) -> bool:
    """Identify a clarification that asks only for a route origin."""
    text = (question or "").strip()
    if not text or re.search(r"终点|目的地|要去哪里|去哪(?:里)?|想去的地点", text):
        return False
    return bool(re.search(
        r"起点|出发地|从哪(?:里)?(?:出发|开始)|你(?:现在)?在哪(?:里)?|当前位置|从哪里出发",
        text,
    ))


def _continuation_start(query: str) -> str:
    text = (query or "").strip().strip("，,。")
    match = re.fullmatch(r"(?:从|我在|起点(?:是|在)?)(.+?)(?:出发)?", text)
    if match:
        return match.group(1).strip()
    if text and len(text) <= 30 and not re.search(r"[？?]|路线|天气|介绍|查询|去|到", text):
        return text
    return ""


def _explicit_route_edit(query: str, prior_route: dict) -> dict | None:
    """Apply a bounded spoken edit to the saved route, retaining untouched slots."""
    from agents.route_state import apply_change, validate_route_state

    text = (query or "").strip().rstrip("。")
    prefix = r"(?:把|将)?(?:刚才(?:的)?|这条|原)路线"
    strategy = re.fullmatch(
        prefix + r"(?:改为|改成|走)(平坦优先|风景优先|最短路径)"
        r"(?:[，,；;]\s*((?:别走|不走|避开)(?:已标记的)?(?:台阶|楼梯)(?:仍然|继续)保留))?",
        text,
    )
    if strategy:
        name = {"平坦优先": "flat", "风景优先": "scenery",
                "最短路径": "shortest"}[strategy.group(1)]
        changed = apply_change(prior_route, {"strategy": name})
        if strategy.group(2):
            changed["hard_constraints"]["avoid_steps"] = True
        changed["strategy"]["source"] = "explicit_nl"
        return changed

    endpoint = re.fullmatch(prefix + r"(起点|终点)改为([^，,。；;]+)", text)
    if endpoint:
        place = endpoint.group(2).strip()
        if not place or place in {"我这里", "我的位置", "当前位置"}:
            return None
        changed = deepcopy(validate_route_state(prior_route))
        changed["start" if endpoint.group(1) == "起点" else "end"] = {
            "name": place, "type": "poi"
        }
        changed["route_id"] = f"route-{uuid4().hex}"
        return validate_route_state(changed)

    constraint = re.fullmatch(
        prefix + r"(不走台阶|避开台阶|避开陡坡|不走楼梯|可以走台阶了)", text
    )
    if constraint:
        changed = deepcopy(validate_route_state(prior_route))
        command = constraint.group(1)
        if command == "可以走台阶了":
            changed["hard_constraints"].pop("avoid_steps", None)
        elif command == "避开陡坡":
            changed["hard_constraints"]["slope"] = "avoid"
        else:
            changed["hard_constraints"]["avoid_steps"] = True
        changed["route_id"] = f"route-{uuid4().hex}"
        return validate_route_state(changed)
    return None



def _build_messages(query: str, context: dict = None, history: list = None,
                    coord_start: dict = None, coord_end: dict = None,
                    uid: str = None, travel_mode: str = None,
                    coord_waypoints: list = None) -> list:
    messages = [{"role": "system", "content": _load_system_prompt()}]

    # 能力说明表（GeoSkill 式技能规范）：固定在提示词前段，利于前缀缓存
    messages.append({"role": "system", "content": tool_skills.build_skill_manifest()})

    # 任务卡知识注入（静态校园经验）
    knowledge_msg = knowledge.build_knowledge_message(query)
    if knowledge_msg:
        messages.append({"role": "system", "content": knowledge_msg})

    # 用户画像注入（动态个体偏好）
    active_task_request = (context or {}).get("active_task_request")
    preference_query = active_task_request if isinstance(active_task_request, str) and active_task_request.strip() else query
    if active_task_request and re.search(r"风景优先|景观优先|平坦优先|少爬坡|少走坡|赏樱|赏花", query):
        preference_query = query
    allow_preferences = route_preference_requested(preference_query, context, history)
    if active_task_request and re.search(r"最短|赶时间|赶课|快一点|快点|尽快|直接到|不绕路|别绕路", query):
        allow_preferences = False
    profile_msg = profile.build_profile_message(uid) if allow_preferences else None
    if profile_msg:
        messages.append({"role": "system", "content": profile_msg})
    if not allow_preferences and re.search(r"从.{1,20}(?:到|去)|路线|怎么走|通勤|上课|宿舍", query):
        messages.append({"role": "system", "content":
                         "本次为普通通勤或距离优先请求。所有规划工具省略 weights，"
                         "由策略层执行纯最短路径 distance/slope/scenery=1.00/0.00/0.00。"
                         "目的地为景点、天气、任务卡和历史画像均不能改变这个策略。"})

    # 出行方式注入：前端切换了步行/骑行/驾车，LLM 应在 plan_route 等工具调用中设 mode
    if travel_mode and travel_mode in ("walk", "bike", "drive"):
        messages.append({
            "role": "system",
            "content": f"用户当前选择的出行方式为 {travel_mode}（步行/骑行/驾车）。"
                        f"调用 plan_route / plan_via_route 时 mode 参数请设为 '{travel_mode}'。"
                        f"除非用户在消息中明确说「骑车/开车/步行」覆盖，否则用此值。",
        })

    # GPS 定位或地图手选点注入：地图手选点是明确的路由端点，不只是当前位置提示。
    gps_lines = []
    for label, coord in (("起点（用户当前位置）", coord_start), ("终点", coord_end)):
        if isinstance(coord, dict) and coord.get("lng") is not None and coord.get("lat") is not None:
            if label.startswith("起点") and coord.get("name") == "地图起点":
                label = "起点（用户在地图上手动选定的起点）"
            elif label == "终点" and coord.get("name") == "地图终点":
                label = "终点（用户在地图上手动选定的终点）"
            gps_lines.append(
                f"- {label}: WGS-84 坐标 lng={coord['lng']}, lat={coord['lat']}"
                f"（必须将此点作为对应坐标端点传入）"
            )
    if gps_lines:
        messages.append({"role": "system", "content": "用户本次请求携带了可用于路线规划的精确坐标：\n"
            + "\n".join(gps_lines)})
    if _manual_map_start_ref(coord_start):
        messages.append({"role": "system", "content":
            "用户已在地图上手动选定路线起点。若本轮文本没有明确写出另一个具体起点，"
            "无论需求写成‘去食堂’、‘去最近的食堂’等简略说法，都必须使用地图起点坐标规划，"
            "不能再询问用户当前在哪里。用户若在本轮明确指定了另一个起点，则以本轮文字为准。"})

    # 地图途经点注入：用户在地图上选的途经点坐标
    if isinstance(coord_waypoints, list) and coord_waypoints:
        wp_lines = []
        for i, wp in enumerate(coord_waypoints):
            if isinstance(wp, dict) and wp.get("lng") is not None and wp.get("lat") is not None:
                wp_lines.append(
                    f"- 途经点{i+1}: WGS-84 坐标 lng={wp['lng']}, lat={wp['lat']}"
                    f"（调用 plan_via_route 时按原顺序放入 via_points，使用"
                    f"{{name:'地图途经点{i+1}', type:'coord', lng:{wp['lng']}, lat:{wp['lat']}}}）"
                )
        if wp_lines:
            messages.append({
                "role": "system",
                "content": "用户在地图上按顺序标记了途经点，规划必须依次经过所有点；"
                           "调用 plan_via_route 时必须一次传入完整、有序的 via_points 数组：\n"
                           + "\n".join(wp_lines),
            })

    # 多轮历史（前端在 P4 接入；兼容旧 context.history 形态）
    history = history or (context or {}).get("history") or []
    for turn in history[-8:]:
        if not isinstance(turn, dict):
            continue
        role, content = turn.get("role"), turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)[:500]})
        elif turn.get("query"):  # 旧形态兼容
            messages.append({"role": "user", "content": str(turn["query"])})

    # 旧 context 的规划槽位（上一轮路线状态），供承接"换一条/从那里出发"
    if context:
        active_task_request = context.get("active_task_request")
        if isinstance(active_task_request, str) and active_task_request.strip():
            messages.append({
                "role": "system",
                "content": "当前正在继续同一个未完成任务。下一条用户消息是原始请求的引用，"
                           "末尾用户消息是对上次追问的回答或修正；保留原请求中的其他目标、"
                           "途经点、出行方式和硬约束，明确冲突时以本轮用户输入为准。"
                           "只有完成整项请求或继续追问尚缺的关键信息后才能结束。",
            })
            messages.append({"role": "user", "content": active_task_request.strip()[:500]})
        policy = context.get("context_policy")
        if policy:
            messages.append({"role": "system", "content": policy})
        slot = {k: context[k] for k in ("start", "end") if context.get(k)}
        if slot:
            messages.append({
                "role": "system",
                "content": "上一轮规划状态（供承接，不要复述）: " + json.dumps(slot, ensure_ascii=False),
            })

    messages.append({"role": "user", "content": query})
    return messages


def _join_leg_paths(legs: list[dict], field: str) -> list:
    joined = []
    for leg in legs:
        part = leg.get(field) or []
        if joined and part and joined[-1] == part[0]:
            part = part[1:]
        joined.extend(part)
    return joined


def _vision_condition_summary(result: dict, query: str = "") -> str | None:
    """Describe vision-linked road records from evidence, never model inference."""
    conditions = result.get("conditions") if isinstance(result, dict) else None
    if (conditions == [] and type(result.get("count")) is int and result["count"] == 0
            and re.search(r"路况|管制|封路|封闭|拥堵|积水|事故|施工", query)):
        return "当前系统没有已生效的路况记录；未覆盖路段的现场通行情况仍需确认。"
    if not isinstance(conditions, list) or not any(
        isinstance(item, dict) and item.get("source_kind") != "manager_record" for item in conditions
    ):
        return None
    def requested_event(event):
        if not isinstance(event, dict):
            return False
        if any(isinstance(event.get(key), str) and len(event[key].strip()) >= 2
               and event[key].strip() in query for key in ("road_name", "name", "type_label")):
            return True
        label = str(event.get("type_label") or "")
        return any(term in query and term in label for term in ("积水", "事故", "拥堵", "施工"))

    conditions = sorted(conditions, key=lambda event: not requested_event(event))
    lines = []
    mode_names = {"walk": "步行", "bike": "骑行", "drive": "驾车"}
    for event in conditions[:5]:
        if not isinstance(event, dict):
            lines.append("有一条路况记录格式不完整，详情未知。")
            continue
        name = str(event.get("name") or "未命名事件")[:100]
        location = str(event.get("location") or event.get("road_name") or "位置未记录")[:100]
        status = event.get("status")
        source = event.get("source_kind")
        if status == "confirmed_active":
            action = str(event.get("action_label") or "影响方式未记录")[:30]
            affected = event.get("affected_modes")
            modes = "、".join(mode_names[x] for x in affected if x in mode_names) if isinstance(affected, list) else ""
            event_type = str(event.get("type_label") or "事件类型未记录")[:30]
            detail = f"{name}（{location}，{event_type}）：{action}"
            detail += f"，影响{modes}。" if modes else "，受影响的出行方式未记录。"
            end_time = event.get("end_time")
            if isinstance(end_time, (int, float)) and not isinstance(end_time, bool):
                try:
                    rendered_end = datetime.fromtimestamp(end_time, timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
                except (ValueError, OverflowError, OSError):
                    rendered_end = "未能识别"
                detail += f"记录结束时间：{rendered_end}。"
            elif isinstance(end_time, str) and end_time.strip():
                try:
                    parsed_end = datetime.fromisoformat(end_time.strip().replace("Z", "+00:00"))
                    if parsed_end.tzinfo is None:
                        rendered_end = parsed_end.strftime("%Y-%m-%d %H:%M") + "（记录未注明时区）"
                    else:
                        rendered_end = parsed_end.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    rendered_end = "未能识别"
                detail += f"记录结束时间：{rendered_end}。"
            elif re.search(r"什么时候|何时|几点|多久|截止|结束时间|解封|解除", query):
                detail += "结束时间未记录。"
        else:
            detail = f"{name}（{location}）：尚不能确认该记录已发布并生效。"
        if source == "vision_job":
            evidence = event.get("vision_evidence")
            evidence = evidence if isinstance(evidence, dict) else {}
            candidate = evidence.get("candidate_index")
            has_review = (type(candidate) is int and candidate >= 0
                          and bool(evidence.get("reviewed_at")) and bool(evidence.get("reviewed_by")))
            if status == "confirmed_active" and has_review:
                detail += "影像候选经管理端复核后发布，是否现场核实未知。"
            elif status == "pending_review":
                detail += "影像候选尚未复核，不能作为已发布路况。"
            else:
                detail += "该记录与影像有关，候选复核或发布情况未记录，无法确认是否经过复核。"
        elif source == "manager_record":
            detail += "来源：管理记录。"
        elif source == "research_simulated":
            detail += "来源：实验模拟事件，不是现场事实。"
        else:
            detail += "来源未记录或无法确认。"
        lines.append(detail)
    if len(conditions) > 5:
        lines.append(f"另有 {len(conditions) - 5} 条记录未展开。")
    return "当前路况记录：\n" + "\n".join(lines)


def _mixed_evidence_summary(summary: str, *, route: dict | None, candidates: list | None,
                            place_results: list, resolved_results: list, last_tool_results: dict,
                            weather_result: dict | None, weather_failed: bool,
                            tools_used: set, query: str) -> str:
    """Deliver available tool evidence without retaining conflicting free-text road claims."""
    parts = []
    if route:
        parts.append(_build_route_message(route))
    if candidates:
        names = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            campus = item.get("campus")
            label = name.strip()[:100]
            if isinstance(campus, str) and campus.strip():
                label += f"（{campus.strip()[:40]}）"
            names.append(label)
        if names:
            parts.append("候选地点：" + "、".join(names) + "。")
        else:
            parts.append("已找到候选地点，但暂时无法整理名称。")
    described_poi_ids = set()
    for result in place_results:
        if not isinstance(result, dict) or result.get("error"):
            parts.append(str(result.get("message") or "地点详情暂时获取不到。")[:300]
                         if isinstance(result, dict) else "地点详情暂时获取不到。")
            continue
        if result.get("found") is False:
            parts.append(str(result.get("message") or "当前地点资料没有匹配结果。")[:300])
            continue
        place = result.get("place") if isinstance(result.get("place"), dict) else result
        name = str(place.get("name") or "地点")[:100]
        poi_id = place.get("poi_id") or place.get("id")
        if poi_id:
            described_poi_ids.add(str(poi_id))
        details = []
        for key, label in (("campus", "所在区域"), ("location", "位置"), ("description", "资料")):
            value = place.get(key)
            if isinstance(value, str) and value.strip():
                details.append(f"{label}：{value.strip()[:200]}")
        opening = place.get("opening_hours")
        opening_value = opening.get("value") if isinstance(opening, dict) else None
        if (isinstance(opening, dict) and opening.get("status") == "known" and opening_value
                and str(opening_value).strip().lower() not in {"unknown", "未知", "未登记"}):
            details.append(f"开放时间：{str(opening_value)[:100]}")
        elif isinstance(opening, dict):
            details.append("开放时间未登记")
        if re.search(r"无障碍|轮椅|坡道", query):
            access = place.get("accessibility")
            if isinstance(access, str) and access.strip().lower() not in {"", "unknown", "未知", "未登记"}:
                details.append(f"无障碍信息：{access.strip()[:100]}")
            else:
                details.append("无障碍情况未核实")
        parts.append(f"{name}：" + "；".join(details) + "。" if details else f"{name}：暂无可核对的详情。")
    for result in resolved_results:
        if isinstance(result, dict) and result.get("found") and isinstance(result.get("poi"), dict):
            poi = result["poi"]
            name = str(poi.get("name") or "名称未记录")[:100]
            poi_id = poi.get("poi_id") or poi.get("id")
            if not poi_id or str(poi_id) not in described_poi_ids:
                campus = str(poi.get("campus") or "")[:100]
                parts.append(f"已找到地点：{name}" + (f"（{campus}）" if campus else "") + "。")
                if poi_id:
                    described_poi_ids.add(str(poi_id))
        elif isinstance(result, dict):
            parts.append(str(result.get("message") or "当前地点资料没有匹配结果。")[:300])
    labels = {"plan_route": "路线规划", "plan_via_route": "途经路线规划", "plan_tour": "游览路线规划",
              "plan_itinerary": "行程规划", "plan_multimodal_route": "分段路线规划",
              "search_poi_candidates": "地点查找", "find_reachable_places": "地点查找"}
    for (tool_name, _identity), result in last_tool_results.items():
        if tool_name in labels and isinstance(result, dict) and result.get("error"):
            parts.append(f"{labels[tool_name]}未完成：{str(result.get('message') or '服务暂不可用')[:200]}。")
    if "get_weather" in tools_used:
        if weather_result:
            description = weather_result["weather"]
            temperature = weather_result.get("temperature")
            if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
                description += f"、{temperature:g}°C"
            parts.append(f"天气参考：{description}。")
            advice = weather_result.get("advice")
            if isinstance(advice, str) and advice.strip():
                parts.append(advice.strip())
        elif weather_failed:
            parts.append("天气暂时获取不到。")
    supported = {"list_road_conditions", "get_weather", "get_place_details", "resolve_poi", "suggest_followup",
                 "plan_route", "plan_via_route", "plan_tour", "plan_itinerary", "plan_multimodal_route",
                 "search_poi_candidates", "find_reachable_places"}
    if tools_used - supported or (not parts and tools_used - {"list_road_conditions", "suggest_followup"}):
        parts.append("其他查询结果暂无法可靠整理，请单独询问。")
    parts.append(summary)
    return "\n".join(parts)


def _place_route_join_issue(legs: list[dict]) -> str | None:
    states = [leg.get("route_state") for leg in legs]
    if not all(isinstance(state, dict)
               and all(isinstance(state.get(key), str) and state.get(key)
                       for key in ("data_version", "road_condition_version")) for state in states):
        return "各段缺少可核实的路网或路况版本"
    if len({state["data_version"] for state in states}) > 1 or len({
        state["road_condition_version"] for state in states
    }) > 1:
        return "两段路线使用的路网或路况版本不同"
    for first, second in zip(legs, legs[1:]):
        if not all(isinstance(leg.get("recommended_edge_ids"), list)
                   and leg["recommended_edge_ids"] for leg in (first, second)):
            return "路段缺少可核对的实际边序列"
        last_edge = first["recommended_edge_ids"][-1]
        first_edge = second["recommended_edge_ids"][0]
        if (not isinstance(last_edge, (list, tuple)) or len(last_edge) != 3
                or not isinstance(first_edge, (list, tuple)) or len(first_edge) != 3):
            return "路段缺少可核对的实际边序列"
        try:
            if any(not (type(value) is int or (isinstance(value, str) and re.fullmatch(r"-?\d+", value)))
                   for value in (*last_edge, *first_edge)):
                raise ValueError("invalid edge component")
            last_end, next_start = int(last_edge[1]), int(first_edge[0])
        except (TypeError, ValueError, OverflowError):
            return "路段缺少可核对的实际边序列"
        if last_end != next_start:
            return "相邻路段的路网节点没有接通"
        first_path, second_path = first.get("recommended"), second.get("recommended")
        if not first_path or not second_path:
            return "路段缺少可核对的路线几何"
        try:
            def lon_lat(point):
                if isinstance(point, dict):
                    return float(point.get("lng", point.get("lon"))), float(point["lat"])
                return float(point[0]), float(point[1])

            lon1, lat1 = lon_lat(first_path[-1])
            lon2, lat2 = lon_lat(second_path[0])
            if not all(math.isfinite(value) for value in (lon1, lat1, lon2, lat2)):
                return "路段衔接坐标无效"
            gap_m = math.hypot((lon2 - lon1) * 96000, (lat2 - lat1) * 111000)
        except (IndexError, KeyError, TypeError, ValueError):
            return "路段衔接坐标无效"
        if gap_m > 3:
            return "相邻路段尚未在同一位置接通"
    return None


def _place_tool_constraints(requested: dict) -> dict:
    if not isinstance(requested, dict):
        raise ValueError("任务约束无法确认")
    allowed = {"avoid_slope", "avoid_steps", "require_verified_accessibility", "max_distance_m",
               "max_duration_s", "excluded_edge_ids"}
    if any(key not in allowed and value for key, value in requested.items()):
        raise ValueError("存在无法表达的任务约束")
    result = {key: deepcopy(value) for key, value in requested.items()
              if key in allowed and key != "avoid_slope" and value}
    if requested.get("avoid_slope"):
        result["slope"] = "avoid"
    return result


def _place_strategy_signature(strategy: dict):
    if not isinstance(strategy, dict) or not strategy.get("name"):
        raise ValueError("路线策略无法确认")
    name = strategy["name"]
    weights, cap = strategy.get("weights"), strategy.get("detour_cap")
    if name == "shortest":
        weights = weights if weights is not None else {"distance": 1.0, "slope": 0.0, "scenery": 0.0}
        cap = cap if cap is not None else 1.0
    return name, deepcopy(weights), cap


def _place_route_constraint_issue(legs: list[dict], requested: dict) -> str | None:
    try:
        expected = _place_tool_constraints(requested)
        states = [leg["route_state"] for leg in legs]
        if any({key: value for key, value in state["hard_constraints"].items() if value} != expected
               for state in states):
            return "各段没有保留全部任务约束"
        signature = _place_strategy_signature(states[0].get("strategy"))
        if any(_place_strategy_signature(state.get("strategy")) != signature for state in states[1:]):
            return "各段路线策略不同，无法确认全程策略"
        distance_limit = positive_budget(expected, "max_distance_m")
        time_limit = positive_budget(expected, "max_duration_s")
        for limit, field, factor in ((distance_limit, "recommended_length_m", 1),
                                     (time_limit, "duration_min", 60)):
            if limit is not None:
                values = [leg.get(field) for leg in legs]
                if any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value < 0 for value in values):
                    return "整程距离或时间约束无法核验"
                if sum(values) * factor > limit + (0.15 if factor == 1 else 3):
                    return "整程超过你设定的距离或时间上限"
        return None
    except (KeyError, TypeError, ValueError, IndexError):
        return "整程约束或策略无法核验"


def _place_route_snapshot_issue(legs: list[dict]) -> str | None:
    from agents.route_state import current_road_condition_version

    try:
        version = current_road_condition_version(agent_tools.list_conditions(strict=True))
    except Exception:
        return "当前路况暂时无法确认"
    if any(leg["route_state"].get("road_condition_version") != version for leg in legs):
        return "当前路况已更新"
    return None


def _replan_changed_place_route(legs: list[dict], context: dict) -> tuple:
    """Replan equivalent ordered legs using one fresh snapshot, or fail closed."""
    from agents.route_state import current_road_condition_version, validate_route_state

    try:
        states = [validate_route_state(leg.get("route_state")) for leg in legs]
        if len(states) < 2 or any(state["route_kind"] not in {"direct", "via"} for state in states):
            raise ValueError("暂不支持自动重算这种行程")
        first = states[0]
        if any(state["travel_mode"] != first["travel_mode"]
               or state["hard_constraints"] != first["hard_constraints"]
               or _place_strategy_signature(state["strategy"]) != _place_strategy_signature(first["strategy"])
               or state.get("data_version") != first.get("data_version") for state in states):
            raise ValueError("各段方式、约束、策略或路网版本不同，无法安全合并重算")
        if "task_constraints" in context:
            requested = context["task_constraints"]
            expected_constraints = _place_tool_constraints(requested)
            actual_constraints = {key: value for key, value in first["hard_constraints"].items() if value}
            if expected_constraints != actual_constraints:
                raise ValueError("路线没有保留全部任务约束")
        pois = agent_tools.load_pois()

        def endpoint(value):
            if not isinstance(value, dict):
                raise ValueError("地点信息不完整")
            if value.get("type") == "coord":
                point = value.get("coordinates") or value
                lng, lat = float(point["lng"]), float(point["lat"])
                if not (math.isfinite(lng) and math.isfinite(lat) and -180 <= lng <= 180 and -90 <= lat <= 90):
                    raise ValueError("地点坐标无效")
                return {"type": "coord", "name": value.get("name") or "地图选点",
                        "coordinates": {"lng": lng, "lat": lat}}, ("coord", lng, lat)
            selected = value.get("poi_id")
            if selected:
                found = [poi for poi in pois if poi.get("id") == selected]
            else:
                found = [poi for poi in pois if poi.get("name") == value.get("name")]
            if len(found) != 1 or not found[0].get("id"):
                raise ValueError("已选地点无法唯一确认，请重新选择")
            poi = found[0]
            return {"type": "poi", "name": poi["name"], "poi_id": poi["id"]}, ("poi", poi["id"])

        ordered, identities = [], []
        for index, state in enumerate(states):
            start, start_id = endpoint(state["start"])
            if index and start_id != identities[-1]:
                raise ValueError("各段地点衔接无法确认")
            if not index:
                ordered.append(start)
                identities.append(start_id)
            if state["route_kind"] == "via":
                via = state.get("via") or {}
                points = via.get("points") if via.get("type") == "multi" else [via]
                if not isinstance(points, list) or not points:
                    raise ValueError("途经点信息不完整")
                for point in points:
                    item, identity = endpoint(point)
                    ordered.append(item)
                    identities.append(identity)
            item, identity = endpoint(state["end"])
            ordered.append(item)
            identities.append(identity)
        if any(a == b for a, b in zip(identities, identities[1:])):
            raise ValueError("相邻地点重复，暂不能安全自动重算")
        if len(ordered) - 2 > agent_tools._MAX_VIA_POINTS:
            raise ValueError("途经点数量超过自动重算上限")

        fresh, error = agent_tools._prepare_route_context({
            key: context[key] for key in ("query", "uid") if key in context
        })
        if error:
            return None, error.get("message") or "暂时无法获取当前路况"
        snapshot = fresh["_route_conditions_snapshot"]
        version = current_road_condition_version(snapshot)
        data_version = agent_tools._data_version_for_graph(fresh["_route_graph_snapshot"])
        if data_version == "unavailable":
            raise ValueError("当前路网版本无法确认")
        strategy = first["strategy"]
        args = {"start": ordered[0], "end": ordered[-1], "via_points": ordered[1:-1],
                "mode": first["travel_mode"], "constraints": deepcopy(first["hard_constraints"]),
                "strategy": strategy["name"], "strategy_source": strategy.get("source", "explicit_nl")}
        if strategy.get("weights") is not None:
            args["weights"] = deepcopy(strategy["weights"])
        result, artifact = agent_tools.execute_tool("plan_via_route", args, fresh)
        route = artifact.get("route") if isinstance(artifact, dict) else None
        if not isinstance(route, dict) or (route.get("route_verification") or {}).get("status") != "pass":
            return None, (result or {}).get("message") or "当前条件下无法核验完整路线"
        new_state = validate_route_state(route.get("route_state"))
        actual_via = new_state.get("via") or {}
        actual_points = actual_via.get("points") if actual_via.get("type") == "multi" else [actual_via]
        actual_refs = [new_state["start"], *actual_points, new_state["end"]]
        actual_ids = [endpoint(item)[1] for item in actual_refs]
        if (actual_ids != identities or new_state.get("travel_mode") != first["travel_mode"]
                or new_state.get("hard_constraints") != first["hard_constraints"]
                or _place_strategy_signature(new_state["strategy"]) != _place_strategy_signature(strategy)
                or new_state.get("data_version") != data_version
                or new_state.get("road_condition_version") != version):
            raise ValueError("重算结果与原行程或快照不一致")
        if (current_road_condition_version(agent_tools.list_conditions(strict=True)) != version
                or agent_tools._data_version_for_graph(fresh["_route_graph_snapshot"]) != data_version):
            raise ValueError("路况或路网再次变化，请重试")
        return route, None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "无法确认各段地点、约束或最新路况，请保留原需求后重试。"
    except Exception:
        logger.exception("按最新快照重算完整行程失败")
        return None, "当前无法可靠重算完整行程，请稍后重试。"


def _combined_place_route(legs: list[dict]) -> dict:
    """Retain each chosen leg's actual geometry and edges in order."""
    from agents.route_state import build_route_state

    combined = deepcopy(legs[-1])
    combined["start_name"] = legs[0].get("start_name", "")
    combined["recommended"] = _join_leg_paths(legs, "recommended")
    if all(leg.get("shortest") for leg in legs):
        combined["shortest"] = _join_leg_paths(legs, "shortest")
    else:
        combined.pop("shortest", None)
    for field in ("recommended_edge_ids", "shortest_edge_ids"):
        if all(isinstance(leg.get(field), list) for leg in legs):
            combined[field] = [edge for leg in legs for edge in leg[field]]
        else:
            combined.pop(field, None)
    for field in ("recommended_length_m", "shortest_length_m", "distance_m",
                  "duration_min", "shortest_duration_min"):
        if all(isinstance(leg.get(field), (int, float)) for leg in legs):
            combined[field] = round(sum(leg[field] for leg in legs), 2)
        elif field == "distance_m" and all(
            isinstance(leg.get("recommended_length_m"), (int, float)) for leg in legs
        ):
            combined[field] = round(sum(leg["recommended_length_m"] for leg in legs), 2)
        else:
            combined.pop(field, None)
    combined["legs"] = deepcopy(legs)
    combined["steps"] = agent_tools._merge_leg_steps(legs)
    combined["pois"] = [poi for leg in legs for poi in leg.get("pois") or []]
    combined["via"] = {"type": "multi", "points": [
        {"name": leg.get("end_name", "")} for leg in legs[:-1]
    ]}
    combined["degraded"] = any(bool(leg.get("degraded")) for leg in legs)
    leg_proofs = [leg.get("route_verification") for leg in legs]
    first_state = legs[0].get("route_state") or {}
    combined_constraints = deepcopy(first_state.get("hard_constraints") or {})
    if combined_constraints.get("slope") == "avoid":
        combined_constraints.pop("slope")
        combined_constraints["avoid_slope"] = True
    if (all(isinstance(proof, dict) and proof.get("status") == "pass" for proof in leg_proofs)
            and _place_route_join_issue(legs) is None
            and _place_route_constraint_issue(legs, combined_constraints) is None):
        combined["route_verification"] = {
            "status": "pass", "scope": "verified_legs_and_checked_join",
            "field_truth_verified": False,
            "leg_count": len(legs),
        }
    else:
        combined.pop("route_verification", None)
    combined["length_capped"] = any(bool(leg.get("length_capped")) for leg in legs)
    combined["hard_budget_fallback"] = any(bool(leg.get("hard_budget_fallback")) for leg in legs)
    combined["shortest_baseline_feasible"] = all(
        leg.get("shortest_baseline_feasible") is True for leg in legs
    )
    statuses = [leg.get("filter_status") for leg in legs if leg.get("filter_status")]
    if statuses:
        combined["filter_status"] = ";".join(dict.fromkeys(statuses))
    if all(isinstance(leg.get("endpoint_access"), dict) for leg in legs):
        combined["endpoint_access"] = {
            "start": legs[0]["endpoint_access"].get("start"),
            "via": [leg["endpoint_access"].get("end") for leg in legs[:-1]],
            "end": legs[-1]["endpoint_access"].get("end"),
            "max_snap_m": legs[-1]["endpoint_access"].get("max_snap_m"),
        }
    states = [leg.get("route_state") for leg in legs]
    event_versions = [(state.get("data_version"), state.get("road_condition_version"))
                      for state in states if isinstance(state, dict)]
    if (len(event_versions) == len(legs) and len(set(event_versions)) == 1
            and all(isinstance(value, str) and value for value in event_versions[0])
            and all(isinstance(leg.get("route_events"), list) for leg in legs)):
        notices, seen = [], {}
        for leg_index, leg in enumerate(legs, 1):
            for event in leg["route_events"]:
                if not isinstance(event, dict):
                    continue
                identifier = event.get("id")
                if isinstance(identifier, (str, int)) and not isinstance(identifier, bool):
                    if identifier in seen:
                        existing = seen[identifier]
                        mode = leg.get("mode")
                        if mode in {"walk", "bike", "drive"} and mode not in existing["travel_modes"]:
                            existing["travel_modes"].append(mode)
                        if leg_index not in existing["leg_indices"]:
                            existing["leg_indices"].append(leg_index)
                        continue
                entry = deepcopy(event)
                entry["travel_modes"] = [leg["mode"]] if leg.get("mode") in {"walk", "bike", "drive"} else []
                entry["leg_indices"] = [leg_index]
                if isinstance(identifier, (str, int)) and not isinstance(identifier, bool):
                    seen[identifier] = entry
                notices.append(entry)
        combined["route_events"] = notices
        combined["route_events_status"] = "complete"
        if all(isinstance(leg.get("road_condition_overview"), list) for leg in legs):
            overview = {}
            for leg in legs:
                for event in leg["road_condition_overview"]:
                    if not isinstance(event, dict):
                        continue
                    key = event.get("id")
                    if not isinstance(key, (str, int)) or isinstance(key, bool):
                        key = (event.get("name"), event.get("road_name"), event.get("action"))
                    if key not in overview:
                        overview[key] = {**deepcopy(event), "travel_modes": []}
                    mode = leg.get("mode")
                    if mode in {"walk", "bike", "drive"} and mode not in overview[key]["travel_modes"]:
                        overview[key]["travel_modes"].append(mode)
            combined["road_condition_overview"] = list(overview.values())
            combined["road_conditions_applied"] = len(overview)
            if len({leg.get("mode") for leg in legs}) > 1:
                combined["road_condition_scope"] = "multi_mode"
        else:
            combined.pop("road_condition_overview", None)
            combined["road_conditions_applied"] = None
    else:
        combined.pop("route_events", None)
        combined["route_events_status"] = "unknown"
        combined.pop("road_condition_overview", None)
        combined["road_conditions_applied"] = None
    if all(isinstance(state, dict) for state in states):
        data_versions = {state.get("data_version") for state in states}
        road_versions = {state.get("road_condition_version") for state in states}
        same_modes = len({state.get("travel_mode") for state in states}) == 1
        same_constraints = all(
            state.get("hard_constraints") == states[0].get("hard_constraints")
            for state in states
        )
        if (len(data_versions) == len(road_versions) == 1 and same_modes and same_constraints
                and all(state.get("route_kind") in {"direct", "via"} for state in states)):
            constraints = states[0].get("hard_constraints") or {}
            try:
                max_distance = positive_budget(constraints, "max_distance_m")
                max_duration = positive_budget(constraints, "max_duration_s")
                if max_distance is not None:
                    shortest_distance = combined.get("shortest_length_m")
                    combined["shortest_baseline_feasible"] &= (
                        isinstance(shortest_distance, (int, float))
                        and shortest_distance <= max_distance + 0.15
                    )
                if max_duration is not None:
                    shortest_duration = combined.get("shortest_duration_min")
                    combined["shortest_baseline_feasible"] &= (
                        isinstance(shortest_duration, (int, float))
                        and shortest_duration * 60 <= max_duration + 3
                    )
            except ValueError:
                combined["shortest_baseline_feasible"] = False
            via_points = []
            for index, state in enumerate(states):
                if state.get("route_kind") == "via":
                    via = state.get("via") or {}
                    via_points.extend(via.get("points", []) if via.get("type") == "multi" else [via])
                if index < len(states) - 1:
                    via_points.append(state["end"])
            if not via_points or len(via_points) > agent_tools._MAX_VIA_POINTS:
                combined.pop("route_state", None)
                combined["route_state_unavailable"] = "完整途经点超过可重算范围"
                return combined
            first, last = states[0], states[-1]
            combined["route_state"] = build_route_state(
                route_kind="via",
                original_query=" → ".join(
                    [combined["start_name"]] + [leg.get("end_name", "") for leg in legs]
                ),
                start=first["start"], end=last["end"],
                via=(via_points[0] if len(via_points) == 1 else
                     {"type": "multi", "points": via_points}),
                legs=[],
                travel_mode=last["travel_mode"],
                hard_constraints=last["hard_constraints"],
                strategy=last["strategy"],
                data_version=last["data_version"],
                road_condition_version=last["road_condition_version"],
            )
        else:
            combined.pop("route_state", None)
            combined["route_state_unavailable"] = "各段方式或约束不同，不能统一重算"
    else:
        combined.pop("route_state", None)
    return combined


def _explicit_place_strategy(query: str) -> str | None:
    """Keep only a supported, explicitly requested strategy in task memory."""
    if not re.search(
        r"最短|赶时间|赶课|不绕路|平坦|平路|少爬坡|少走坡|省力|"
        r"风景优先|景观优先|看风景|风景好|游览|旅游|散步|赏樱|赏花|逛",
        query or "",
    ):
        return None
    strategy = detect_strategy_hint(query)
    return strategy if strategy in {"shortest", "recommended", "scenery", "flat"} else None


def _requested_recommendation_limit(query: str, *, catalog: bool) -> int:
    """Honor a small explicit shortlist size without truncating the search pool."""
    match = re.search(
        r"(?:推荐|介绍|列出|给我|找|选)[^，,。；;！？?!]{0,12}?"
        r"([1-5一二三四五两])\s*(?:个|家|处|座)",
        query or "",
    )
    if match:
        return {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}.get(
            match.group(1), int(match.group(1)) if match.group(1).isdigit() else 4,
        )
    return 20 if catalog else 4


def _place_request_state(query: str, *, activities=None, subcategories=None,
                         pending_slot=None, start=None, constraints=None,
                         previous=None, candidate_poi_ids=None, selected_poi_id=None,
                         follow_up_requests=None,
                         mode="walk") -> dict:
    previous = previous if isinstance(previous, dict) else {}
    strategy = previous.get("strategy") or _explicit_place_strategy(query)
    return {
        "request_id": previous.get("request_id") or uuid4().hex,
        "original_query": previous.get("original_query") or query,
        "activities": list(activities if activities is not None else previous.get("activities") or []),
        "subcategories": list(subcategories if subcategories is not None else previous.get("subcategories") or []),
        "follow_up_requests": list(follow_up_requests if follow_up_requests is not None
                                   else previous.get("follow_up_requests") or []),
        "completed_legs": list(previous.get("completed_legs") or []),
        "pending_slot": pending_slot,
        "start": start if start is not None else previous.get("start"),
        "selected_poi_id": (selected_poi_id if selected_poi_id is not None
                             else previous.get("selected_poi_id")),
        "candidate_poi_ids": list(candidate_poi_ids if candidate_poi_ids is not None
                                   else previous.get("candidate_poi_ids") or []),
        "mode": previous.get("mode") or mode,
        "strategy": strategy,
        "constraints": constraints if constraints is not None else previous.get("constraints") or {},
    }


def _selected_candidate_from_text(query: str, candidate_poi_ids: list[str]) -> str | None:
    """Accept a clear spoken choice only from the current candidate list."""
    candidate_ids = set(candidate_poi_ids)
    if not candidate_ids:
        return None
    choices = []
    for poi in agent_tools.load_pois():
        if poi.get("id") not in candidate_ids:
            continue
        names = [poi.get("name"), *(poi.get("aliases") or [])]
        choices.extend((str(name).strip(), poi["id"]) for name in names
                       if name and len(str(name).strip()) >= 3)
    for clause in re.split(r"[，,。；;]|然后|接着|随后|最后", query or ""):
        clause = clause.strip()
        if not clause:
            continue
        # A bare name or an affirmative selection is a choice. A rejected
        # candidate mentioned in the same sentence is not.
        matches = set()
        for name, poi_id in choices:
            if name not in clause:
                continue
            before = clause[:clause.find(name)]
            if re.search(r"(?:不(?:要|想)?去|别去|不是|排除)\s*$", before):
                continue
            affirmative = re.search(
                r"(?:就去|还是去|改为|换成|选(?:择)?|要去|去|定)\s*$", before,
            )
            if clause == name or (affirmative and not clause.endswith(("吗", "？", "?"))):
                matches.add(poi_id)
        if len(matches) == 1:
            return matches.pop()
        if len(matches) > 1:
            return None
    return None


def _apply_place_request_correction(saved: dict, query: str) -> dict:
    """Keep unanswered slots while applying only explicit current-turn changes."""
    from agents.parser import detect_travel_mode

    updated = deepcopy(saved)
    mode, explicit_mode = detect_travel_mode(query)
    if explicit_mode:
        updated["mode"] = mode
    strategy = _explicit_place_strategy(query)
    if strategy:
        updated["strategy"] = strategy
    constraints = dict(updated.get("constraints") or {})
    if re.search(r"(?:不走|避开|不要走|绕开)(?:楼梯|台阶)", query):
        constraints["avoid_steps"] = True
    elif re.search(r"(?:可以|允许)(?:走|经过)?(?:楼梯|台阶)|不(?:用|需要)避开(?:楼梯|台阶)", query):
        constraints["avoid_steps"] = False
    if re.search(r"避坡|少爬坡|不想爬坡|避开陡坡", query):
        constraints["avoid_slope"] = True
    elif re.search(r"不(?:用|需要)(?:再)?避(?:开)?(?:陡坡|坡)|可以(?:爬|走)(?:坡|陡坡)", query):
        constraints["avoid_slope"] = False
    updated["constraints"] = constraints
    return updated


def _response_verification(response: dict) -> dict | None:
    """取响应里的核验报告：失败响应挂在顶层，成功路线随路线包一起返回。"""
    if not isinstance(response, dict):
        return None
    report = response.get("route_verification")
    if isinstance(report, dict):
        return report
    route = response.get("route")
    if isinstance(route, dict):
        nested = route.get("route_verification")
        if isinstance(nested, dict):
            return nested
    return None


def _record_experience_sample(query: str, response: dict) -> None:
    """把一次完成的编排沉淀为经验样本（best-effort）。

    只登记实际发生过工具调用的编排。成功判定从严：
    - 候选类结果：只要产出了候选即算成功（候选不涉及路线核验）；
    - 路线类结果：必须路线存在且核验状态为 pass；
    - 其余（澄清/闲聊/核验失败）：记为失败样本，蒸馏时据此拉低置信度。
    """
    if not isinstance(response, dict):
        return
    try:
        activities = ((response.get("provenance") or {}).get("tool_activities") or [])
        if not activities:
            return
        verification = _response_verification(response)
        status = verification.get("status") if isinstance(verification, dict) else None
        has_route = response.get("route") is not None
        success = bool(response.get("candidates")) or (has_route and status == "pass")
        experience.record_run(
            query, activities, success=success, verification_status=status,
        )
    except Exception:
        # 经验沉淀是旁路观测，任何失败都不得影响主链路响应
        logger.warning("经验沉淀失败（忽略）", exc_info=True)


def run_agent(query: str, context: dict = None, history: list = None,
              coord_start: dict = None, coord_end: dict = None,
              uid: str = None, travel_mode: str = None,
              coord_waypoints: list = None,
              proof_identity: dict = None,
              tool_trace_callback: Callable[[dict], None] | None = None) -> dict:
    """Agent 主循环。

    Returns:
        {
            "response_kind": "route" | "candidates" | "clarify" | "chat",
            "message": str,            # 给用户的最终自然语言答复
            "route": dict | None,      # plan_route/plan_via_route/plan_tour 的完整路径包
            "route_kind": "direct" | "via" | "tour" | None,
            "candidates": list | None, # search_poi_candidates 结果
            "clarify": {"question", "options"} | None,
            "turns": int,              # 实际 LLM 轮次（观测用）
        }

    Raises:
        PlannerError: LLM API 不可用（调用方落回旧管道）
    """
    # 溯源记录：与 tool_trace_callback 同源，另存一份 run 级 PROV 记录。
    # 这里是唯一的采集点，下游（PG 事件 / 管理端点）共用同一份数据。
    _external_tool_trace = tool_trace_callback
    _provenance = provenance.ProvenanceRecorder(
        query,
        uid=uid,
        model=getattr(config, "LLM_MODEL", None),
        temperature=0.0,
        tool_names=agent_tools.TOOL_NAMES,
        has_gps=isinstance(coord_start, dict) or isinstance(coord_end, dict),
        has_waypoints=isinstance(coord_waypoints, list) and bool(coord_waypoints),
        task_revision=(context or {}).get("task_revision")
        if isinstance(context, dict) else None,
    )
    # Keep weather consistent between weather and routing tools in this run.
    # Do not carry a live snapshot into a later user turn.
    run_weather = {}

    def tool_trace_callback(payload):
        try:
            payload = payload if isinstance(payload, dict) else {}
            _provenance.add_activity(
                payload.get("tool_name"),
                argument_keys=payload.get("argument_keys"),
                outcome=payload.get("outcome"),
                artifact_kind=payload.get("artifact_kind"),
            )
        except Exception:
            logger.warning("溯源活动登记失败（忽略）", exc_info=True)
        if _external_tool_trace is not None:
            _external_tool_trace(payload)

    def record_tool_result(name, args, result, artifact):
        artifact_kind = next(
            (kind for key, kind in (("route", "route"),
                                    ("candidates", "candidates"),
                                    ("clarify", "clarify"),
                                    ("suggestions", "suggestions"),
                                    ("route_comparison", "route_comparison"))
             if isinstance(artifact, dict) and key in artifact),
            None,
        )
        trace_item = {
                "tool_name": name,
                "argument_keys": sorted(args) if isinstance(args, dict) else [],
                "outcome": "tool_error" if isinstance(result, dict) and result.get("error")
                          else "success",
                "artifact_kind": artifact_kind,
        }
        if (name in {"plan_route", "plan_via_route", "plan_itinerary",
                     "plan_tour", "plan_multimodal_route"}
                and trace_item["outcome"] == "tool_error"
                and isinstance(result, dict)
                and result.get("error") == "route_not_found"
                and result.get("reason_code") == "graph_no_path"
                and isinstance(result.get("request_digest"), str)
                and re.fullmatch(r"[0-9a-f]{64}", result["request_digest"])):
            trace_item["error_code"] = "graph_no_path"
            trace_item["request_digest"] = result["request_digest"]
        if (name == "list_road_conditions" and trace_item["outcome"] == "success"
                and isinstance(result, dict)
                and isinstance(result.get("conditions"), list)):
            conditions = result["conditions"]
            ids = [item.get("id") for item in conditions
                   if isinstance(item, dict)]
            if (len(conditions) <= 128 and len(ids) == len(conditions)
                    and all(isinstance(value, str) and 0 < len(value) <= 80
                            for value in ids)
                    and len(set(ids)) == len(ids)):
                trace_item["road_event_ids"] = sorted(ids)
        tool_trace_callback(trace_item)

    def execute_tool(name, args, tool_context, *, record_trace=True):
        # This keyword is supplied only by trusted workflow code, separately
        # from browser context, uid and model-produced tool arguments.
        tool_context = dict(tool_context)
        if "_route_weather_snapshot" in run_weather:
            tool_context.setdefault("_route_weather_snapshot",
                                    run_weather["_route_weather_snapshot"])
        if isinstance(context, dict) and isinstance(context.get("task_revision"), int):
            tool_context.setdefault("task_revision", context["task_revision"])
        if proof_identity is not None:
            tool_context["_route_proof_identity"] = proof_identity
        try:
            result, artifact = agent_tools.execute_tool(name, args, tool_context)
        except Exception:
            if record_trace and tool_trace_callback:
                tool_trace_callback({
                    "tool_name": name,
                    "argument_keys": sorted(args) if isinstance(args, dict) else [],
                    "outcome": "exception",
                    "artifact_kind": None,
                })
            raise
        if "_route_weather_snapshot" in tool_context:
            run_weather["_route_weather_snapshot"] = tool_context["_route_weather_snapshot"]
        if record_trace and tool_trace_callback:
            record_tool_result(name, args, result, artifact)
        return result, artifact

    def route_weather(text):
        """Fetch explicitly requested weather before route tools share the snapshot."""
        if not re.search(r"天气|气温|现在.*(?:下雨|下雪)", text or ""):
            return None
        if re.search(r"(?:不用|无需|不要|不必)(?:再)?(?:查|看|问)?(?:天气|气温)", text or ""):
            return None
        result, _ = execute_tool("get_weather", {}, {"query": text, "uid": uid})
        return result if isinstance(result, dict) else {"error": "weather_unavailable"}

    def add_route_weather(response, weather):
        if weather is None:
            return response
        if not weather.get("error") and isinstance(weather.get("weather"), str) and weather["weather"]:
            description = weather["weather"]
            temperature = weather.get("temperature")
            if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
                description += f"、{temperature:g}°C"
            response["message"] += f" 天气参考：{description}。"
            response["weather"] = weather
        else:
            response["message"] += " 天气暂时获取不到。"
        if response.get("response_kind") == "clarify" and isinstance(response.get("clarify"), dict):
            response["clarify"]["question"] = response["message"]
        return response

    # 地图端已经把起点、终点和每个途经点解析成精确坐标，不应再让 LLM
    # 决定是否调用工具或重排用户选点。直接执行同一个 plan_via_route 工具，
    # 使地图多点规划不受模型超时、漏调工具或无工具文本回答影响。
    if ("地图标记" in (query or "") and isinstance(coord_start, dict)
            and isinstance(coord_end, dict) and isinstance(coord_waypoints, list)
            and coord_waypoints and len(coord_waypoints) <= agent_tools._MAX_VIA_POINTS):
        refs = [coord_start, *coord_waypoints, coord_end]
        if all(isinstance(ref, dict) and ref.get("lng") is not None and ref.get("lat") is not None
               for ref in refs):
            args = {
                "start": {"name": coord_start.get("name") or "地图起点", "type": "coord",
                          "lng": coord_start["lng"], "lat": coord_start["lat"]},
                "end": {"name": coord_end.get("name") or "地图终点", "type": "coord",
                        "lng": coord_end["lng"], "lat": coord_end["lat"]},
                "via_points": [
                    {"name": point.get("name") or f"地图途经点{index}", "type": "coord",
                     "lng": point["lng"], "lat": point["lat"]}
                    for index, point in enumerate(coord_waypoints, start=1)
                ],
                "mode": travel_mode if travel_mode in {"walk", "bike", "drive"} else "walk",
            }
            weather = route_weather(query)
            result, artifact = execute_tool(
                "plan_via_route", args, {"query": query, "uid": uid}
            )
            route = (artifact or {}).get("route") if isinstance(artifact, dict) else None
            if route:
                route["timings_ms"] = normalize_timings(route.get("timings_ms"), agent=0.0)
                return add_route_weather({
                    "response_kind": "route",
                    "message": _build_route_message(route) + _road_condition_feedback(route, query),
                    "route": route,
                    "route_kind": (artifact or {}).get("route_kind", "via"),
                    "candidates": None, "clarify": None, "suggestions": None, "turns": 0,
                }, weather)
            if isinstance(result, dict) and result.get("error") in {
                    "route_verification_failed", "route_verification_unknown"}:
                # Explicit map points must not be altered by an LLM repair path.
                return add_route_weather(_route_verification_failure_response(
                    result, message="你选定的地图点路线未通过独立核验，没有显示路线。",
                ), weather)
            return add_route_weather({
                "response_kind": "chat",
                "message": (result or {}).get("message", "多途经点路线暂时无法生成，请调整地点后重试。"),
                "route": None, "route_kind": None, "candidates": None,
                "clarify": None, "suggestions": None, "turns": 0,
            }, weather)

    # Resolve a deictic continuation only within the same unfinished task.
    # Chat history or an unrelated old route must never supply this endpoint.
    from agents.parser import detect_travel_mode
    if (not isinstance(coord_start, dict) and not _explicit_start_name(query)
            and not isinstance((context or {}).get("place_request"), dict)
            and not agent_tools.ordered_via_request(query)
            and not _named_via_request(query)
            and not re.search(r"先.+(?:然后|再|接着|最后)", query)):
        destination_match = re.match(
            r"^(?:我想|我要|请|帮我|麻烦)?\s*(?:带我去|导航到|前往|去|到)\s*([^，,。；;]+)",
            query.strip(),
        )
        destination_name = (re.split(r"(?=不走|避开|不要走|查天气|顺便|同时|起点)",
                                     destination_match.group(1), maxsplit=1)[0].strip()
                            if destination_match else "")
        destination, alternatives = (agent_tools.find_poi_ambiguous(destination_name)
                                     if destination_name else (None, []))
        if destination and not alternatives:
            mode = travel_mode if travel_mode in {"walk", "bike", "drive"} else "walk"
            detected_mode, explicit_mode = detect_travel_mode(query)
            if explicit_mode:
                mode = detected_mode
            constraints = {}
            if re.search(r"(?:不走|避开|不要走|绕开)(?:楼梯|台阶)", query):
                constraints["avoid_steps"] = True
            if re.search(r"避坡|少爬坡|不想爬坡|避开陡坡", query):
                constraints["avoid_slope"] = True
            pending = _place_request_state(
                query, pending_slot="start", selected_poi_id=destination["id"],
                constraints=constraints, mode=mode,
            )
            question = "你从哪里出发？可以点地图起点，或告诉我所在地点。"
            return {"response_kind": "clarify", "message": question,
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": {"question": question,
                                "options": ["地图上选起点", "告诉我所在地点"]},
                    "place_request": pending, "suggestions": None, "turns": 0}

    active_request = ((context or {}).get("active_task_request")
                      if isinstance(context, dict) else None)
    if (isinstance(active_request, str)
            and not (context or {}).get("place_request")
            and not (agent_tools.ordered_via_request(active_request)
                     or _named_via_request(active_request)
                     or re.search(r"先.+(?:然后|再|接着|最后)", active_request))
            and re.search(r"带我过去|带我去那(?:里|儿)|去那(?:里|儿)|到那(?:里|儿)", query)):
        destination_match = re.search(
            r"(?:目的地(?:是|为|设为|[:：])|导航到|前往|去|到)\s*([^，,。；;]+)",
            active_request,
        )
        destination_name = (re.split(r"(?:不走|避开|查天气|顺便)", destination_match.group(1))[0].strip()
                            if destination_match else "")
        destination, alternatives = (agent_tools.find_poi_ambiguous(destination_name)
                                     if destination_name else (None, []))
        if destination and not alternatives:
            start_name = _explicit_start_name(query) or _explicit_start_name(active_request)
            start_ref = ({"type": "poi", "name": start_name} if start_name else
                         {"type": "coord", "name": coord_start.get("name") or "我的位置",
                          "lng": coord_start["lng"], "lat": coord_start["lat"]}
                         if isinstance(coord_start, dict) else None)
            if not start_ref:
                question = "目的地已记住。请告诉我从哪里出发，或在地图上设置起点。"
                return {"response_kind": "clarify", "message": question,
                        "route": None, "route_kind": None, "candidates": None,
                        "clarify": {"question": question, "options": []},
                        "suggestions": None, "turns": 0}
            mode = travel_mode if travel_mode in {"walk", "bike", "drive"} else "walk"
            for text in (active_request, query):
                detected_mode, explicit_mode = detect_travel_mode(text)
                if explicit_mode:
                    mode = detected_mode
            route_args = {"start": start_ref,
                          "end": {"type": "poi", "poi_id": destination["id"],
                                  "name": destination["name"]}, "mode": mode}
            full_request = active_request + " " + query
            constraints = {}
            if re.search(r"(?:不走|避开|不要走|绕开)(?:楼梯|台阶)", full_request):
                constraints["avoid_steps"] = True
            if re.search(r"避坡|少爬坡|不想爬坡|避开陡坡", full_request):
                constraints["slope"] = "avoid"
            if re.search(r"(?:可以|允许)(?:走|经过)?(?:楼梯|台阶)|不(?:用|需要)避开(?:楼梯|台阶)", query):
                constraints.pop("avoid_steps", None)
            if re.search(r"不(?:用|需要)(?:再)?避(?:开)?(?:陡坡|坡)|可以(?:爬|走)(?:坡|陡坡)", query):
                constraints.pop("slope", None)
            if constraints:
                route_args["constraints"] = constraints
            if re.search(r"平坦优先|少爬坡|少走坡", full_request):
                route_args["weights"] = {"distance": 0.2, "slope": 0.6, "scenery": 0.2}
            elif re.search(r"风景优先|景观优先|看风景|风景好", full_request):
                route_args["weights"] = {"distance": 0.15, "slope": 0.15, "scenery": 0.7}
            if re.search(r"最短|赶时间|赶课|不绕路", query):
                route_args.pop("weights", None)
            result, artifact = execute_tool("plan_route", route_args,
                                            {"query": active_request + " " + query, "uid": uid})
            route = (artifact or {}).get("route") if isinstance(artifact, dict) else None
            if route:
                route["timings_ms"] = normalize_timings(route.get("timings_ms"), agent=0.0)
                return {"response_kind": "route",
                        "message": _build_route_message(route) + _road_condition_feedback(route, full_request),
                        "route": route, "route_kind": "direct", "candidates": None,
                        "clarify": None, "suggestions": None, "turns": 0}
            return {"response_kind": "chat", "message": (result or {}).get("message")
                    or "路线暂时无法核验，没有显示路线。", "route": None,
                    "route_kind": None, "candidates": None, "clarify": None,
                    "suggestions": None, "turns": 0}

    if (not (context or {}).get("place_request")
            and re.search(r"天气|气温|下雨|降雨", query)
            and not (isinstance(active_request, str)
                     and (agent_tools.ordered_via_request(active_request)
                          or _named_via_request(active_request)))
            and re.search(r"带我过去|带我去那(?:里|儿)|去那(?:里|儿)|到那(?:里|儿)", query)):
        start_name = _explicit_start_name(query)
        start_ref = ({"type": "poi", "name": start_name} if start_name else
                     {"type": "coord", "name": coord_start.get("name") or "我的位置",
                      "lng": coord_start["lng"], "lat": coord_start["lat"]}
                     if isinstance(coord_start, dict) else None)
        question = "你想去哪个地点？目的地明确后，我会从你说的起点规划路线。"
        pending = _place_request_state(query, pending_slot="destination", start=start_ref,
                                       mode=travel_mode if travel_mode in {"walk", "bike", "drive"}
                                       else "walk")
        return {"response_kind": "clarify", "message": question,
                "route": None, "route_kind": None, "candidates": None,
                "clarify": {"question": question, "options": []},
                "place_request": pending, "suggestions": None, "turns": 0}

    # A mode-only correction is an edit to the saved route, even if a weather
    # or POI turn appeared in between. Rebuild from the validated route state
    # so endpoints, ordered stops, hard constraints and strategy survive.
    from agents.context_builder import is_route_followup
    from agents.parser import detect_travel_mode
    mode_change = re.search(
        r"(?:切换到?|改成|改为|换成|改走).{0,5}(?:步行|走路|骑行|骑车|开车|驾车)",
        query or "",
    )
    prior_route = (context or {}).get("previous_route_state")
    # A mode edit may also replace an endpoint in the same utterance. Do not
    # consume a broader request with the mode-only shortcut: doing so would
    # silently retain the old endpoint or discard another requested change.
    endpoint_edits = list(re.finditer(
        r"(?:^|[，,；;])\s*(?:把|将)?(起点|终点)(?:改为|改成|换成|换为)\s*([^，,；;。]+)",
        query or "",
    ))
    unsupported_route_edit = re.search(
        r"(?:途经|途径|经过|台阶|楼梯|陡坡|策略|风景|平坦|最短).{0,8}"
        r"(?:改为|改成|换成|换为|新增|添加|删除|取消|去掉)", query or "",
    ) or re.search(
        r"(?:新增|添加|删除|取消|去掉|避开|不走|允许).{0,8}"
        r"(?:途经|途径|台阶|楼梯|陡坡)", query or "",
    )
    mentioned_endpoint_edits = list(re.finditer(
        r"(?:起点|终点).{0,3}(?:改为|改成|换成|换为)", query or "",
    ))
    if (mode_change and isinstance(prior_route, dict) and is_route_followup(query)
            and not isinstance((context or {}).get("place_request"), dict)
            and not unsupported_route_edit
            and len(mentioned_endpoint_edits) == len(endpoint_edits)):
        from agents.route_state import apply_change, validate_route_state
        from api.routes import _replan_tool_request

        mode, explicit = detect_travel_mode(query)
        if explicit:
            requested = apply_change(prior_route, {"travel_mode": mode})
            for edit in endpoint_edits:
                field = "start" if edit.group(1) == "起点" else "end"
                requested[field] = {"name": edit.group(2).strip(), "type": "poi"}
            requested = validate_route_state(requested)
            tool_name, args = _replan_tool_request(requested)
            weather = route_weather(query)
            result, artifact = execute_tool(
                tool_name, args,
                {"query": requested["original_query"], "uid": uid, "replan": True},
            )
            route = (artifact or {}).get("route") if isinstance(artifact, dict) else None
            if route:
                route["timings_ms"] = normalize_timings(route.get("timings_ms"), agent=0.0)
                return add_route_weather({"response_kind": "route",
                        "message": _build_route_message(route) + _road_condition_feedback(route, query),
                        "route": route, "route_kind": requested["route_kind"],
                        "candidates": None, "clarify": None, "suggestions": None, "turns": 0}, weather)
            if isinstance(result, dict) and result.get("error") in {
                    "route_verification_failed", "route_verification_unknown"}:
                return add_route_weather(_route_verification_failure_response(
                    result, message="切换出行方式后的路线未通过独立核验，没有显示路线。",
                ), weather)
            return add_route_weather({"response_kind": "chat",
                    "message": (result or {}).get("message") or "这条路线暂时无法按新出行方式规划。",
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": None, "suggestions": None, "turns": 0}, weather)

    # A spoken edit to an existing route must preserve its ordered stops and
    # untouched constraints. The LLM may otherwise issue a plausible direct
    # route after seeing only the endpoint and strategy in the short context.
    if isinstance(prior_route, dict) and is_route_followup(query):
        requested = _explicit_route_edit(query, prior_route)
        if requested is not None:
            from api.routes import _replan_tool_request

            tool_name, args = _replan_tool_request(requested)
            weather = route_weather(query)
            result, artifact = execute_tool(
                tool_name, args, {"query": query, "uid": uid, "replan": True},
            )
            route = (artifact or {}).get("route") if isinstance(artifact, dict) else None
            if route:
                route["timings_ms"] = normalize_timings(route.get("timings_ms"), agent=0.0)
                return add_route_weather({"response_kind": "route",
                        "message": _build_route_message(route) + _road_condition_feedback(route, query),
                        "route": route, "route_kind": requested["route_kind"],
                        "candidates": None, "clarify": None, "suggestions": None,
                        "turns": 0}, weather)
            if isinstance(result, dict) and result.get("error") in {
                    "route_verification_failed", "route_verification_unknown"}:
                return add_route_weather(_route_verification_failure_response(
                    result, message="修改后的路线未通过独立核验，没有显示路线。",
                ), weather)
            return add_route_weather({"response_kind": "chat",
                    "message": (result or {}).get("message") or "这条路线暂时无法按新要求规划。",
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": None, "suggestions": None, "turns": 0}, weather)

    # Explicit "via X to Y" requests have a required stop. Resolve that
    # structure before the LLM can accidentally use X as the start and return
    # a plausible-looking direct route that silently skips the stop.
    original_request = (context or {}).get("active_task_request") or query
    named_via = (agent_tools.ordered_via_request(original_request)
                 or _named_via_request(original_request))
    if named_via:
        via_names = named_via["via"] if isinstance(named_via["via"], list) else [named_via["via"]]
        revised_start = ((_explicit_start_name(query) or _continuation_start(query))
                         if original_request != query else "")
        start_name = revised_start or named_via["start"]
        if start_name in {"我这", "我这里", "我的位置", "当前位置"}:
            start_name = ""
        if not start_name and not coord_start:
            question = f"从哪里出发？我会经过{'、'.join(via_names)}，再到{named_via['end']}。"
            return {"response_kind": "clarify", "message": question, "route": None,
                    "route_kind": None, "candidates": None,
                    "clarify": {"question": question, "options": []},
                    "suggestions": None, "turns": 0}
        start_ref = ({"name": coord_start.get("name") or "我的位置", "type": "coord",
                      "lng": coord_start["lng"], "lat": coord_start["lat"]}
                     if not start_name and coord_start else {"name": start_name, "type": "poi"})
        full_request = str(original_request) + " " + str(query)
        constraints = {}
        if "避坡" in full_request or "陡坡" in full_request:
            constraints["slope"] = "avoid"
        if re.search(r"(?:不走|避开|不要走|绕开)(?:楼梯|台阶)", full_request):
            constraints["avoid_steps"] = True
        if re.search(r"不(?:用|需要)(?:再)?避(?:开)?(?:陡坡|坡)|可以(?:爬|走)(?:坡|陡坡)", query):
            constraints.pop("slope", None)
        if re.search(r"(?:可以|允许)(?:走|经过)?(?:楼梯|台阶)|不(?:用|需要)避开(?:楼梯|台阶)", query):
            constraints.pop("avoid_steps", None)
        mode = travel_mode if travel_mode in {"walk", "bike", "drive"} else "walk"
        for text in (original_request, query):
            detected_mode, explicit_mode = detect_travel_mode(text)
            if explicit_mode:
                mode = detected_mode
        args = {"start": start_ref, "end": {"name": named_via["end"], "type": "poi"},
                "via_points": [{"name": name, "type": "poi"} for name in via_names], "mode": mode}
        if constraints:
            args["constraints"] = constraints
        weather = route_weather(full_request)
        result, artifact = execute_tool(
            "plan_via_route", args, {"query": full_request, "uid": uid}
        )
        route = (artifact or {}).get("route") if isinstance(artifact, dict) else None
        if route:
            route["timings_ms"] = normalize_timings(route.get("timings_ms"), agent=0.0)
            return add_route_weather({"response_kind": "route",
                    "message": _build_route_message(route) + _road_condition_feedback(route, full_request),
                    "route": route, "route_kind": "via", "candidates": None,
                    "clarify": None, "suggestions": None, "turns": 0}, weather)
        if isinstance(result, dict) and result.get("error") in {
                "route_verification_failed", "route_verification_unknown"}:
            # Preserve the user's explicit stop order; fail closed rather than
            # asking the LLM to rewrite a deterministic named-via request.
            return add_route_weather(_route_verification_failure_response(
                result, message="按你指定的途经顺序规划的路线未通过独立核验，没有显示路线。",
            ), weather)
        question = (result or {}).get("message") or "途经点或终点暂时无法确认，请换个地点名称。"
        return add_route_weather({"response_kind": "clarify", "message": question, "route": None,
                "route_kind": None, "candidates": None,
                "clarify": {"question": question, "options": []},
                "suggestions": None, "turns": 0}, weather)

    # Activity choices and candidate IDs resume the same durable POI request.
    place_request = (context or {}).get("place_request") if isinstance(context, dict) else None
    place_request = (_apply_place_request_correction(place_request, query)
                     if isinstance(place_request, dict) else None)
    selected_poi_id = ((context or {}).get("selected_poi_id")
                       or (place_request or {}).get("selected_poi_id")) if isinstance(context, dict) else None
    if (not selected_poi_id and place_request
            and place_request.get("pending_slot") == "selection"):
        selected_poi_id = _selected_candidate_from_text(
            query, place_request.get("candidate_poi_ids") or [],
        )
    recommendation_query = original_request

    def current_coordinate_start():
        if not isinstance(coord_start, dict):
            return None
        try:
            lng, lat = float(coord_start.get("lng")), float(coord_start.get("lat"))
        except (TypeError, ValueError):
            return None
        if not (-180 <= lng <= 180 and -90 <= lat <= 90):
            return None
        return {"type": "coord", "name": coord_start.get("name") or "我的位置",
                "lng": lng, "lat": lat}

    def explicit_place_start(text):
        name = agent_tools.recommendation_origin_from_text(text)
        if not name:
            return None
        poi, alternatives = agent_tools.find_poi_ambiguous(name)
        if poi and not alternatives:
            return {"type": "poi", "name": poi["name"]}
        if alternatives:
            return {"ambiguous_name": name, "alternatives": alternatives[:5]}
        return {"type": "poi", "name": name}

    def with_next_place_stage(route_response: dict, destination: dict,
                              saved_request: dict | None, followups: list[dict]) -> dict:
        completed_legs = list((saved_request or {}).get("completed_legs") or [])
        completed_legs.append(deepcopy(route_response["route"]))
        if len(completed_legs) > 1:
            join_issue = _place_route_join_issue(completed_legs)
            if not join_issue:
                join_issue = _place_route_constraint_issue(completed_legs, (saved_request or {}).get("constraints") or {})
            if not join_issue:
                join_issue = _place_route_snapshot_issue(completed_legs)
            refreshed_route = None
            refresh_error = None
            if join_issue in {"两段路线使用的路网或路况版本不同", "当前路况已更新"}:
                refreshed_route, refresh_error = _replan_changed_place_route(
                    completed_legs, {"query": str((saved_request or {}).get("original_query") or "") + " " + query,
                                     "uid": uid, "task_constraints": (saved_request or {}).get("constraints") or {}})
                if refreshed_route is not None:
                    join_issue = None
            if join_issue:
                pending = _place_request_state(
                    "重新规划全程", pending_slot="route_retry", previous=saved_request,
                )
                pending["completed_legs"] = completed_legs
                route_response["response_kind"] = "clarify"
                route_response["route"] = None
                route_response["route_kind"] = None
                route_response["candidates"] = None
                route_response["needs_input"] = True
                route_response["place_request"] = pending
                question = (f"各段已单独规划，但{join_issue}。{refresh_error or ''}无法确认完整路线。"
                            "已保留你的行程，可以点击“重试”。")
                route_response["message"] = question
                route_response["clarify"] = {"question": question, "options": ["重试"]}
                return route_response
            route_response["route"] = refreshed_route or _combined_place_route(completed_legs)
            route_response["route_kind"] = (
                "multimodal" if len({leg.get("mode") for leg in completed_legs}) > 1
                else "via"
            )
            route_response["message"] = _build_route_message(route_response["route"]) + _road_condition_feedback(
                route_response["route"], str((saved_request or {}).get("original_query") or "") + " " + query,
            )
            if refreshed_route is not None:
                route_response["message"] += "路况发生变化，已按最新记录重新规划全程。"
            if route_response["route"].get("route_state_unavailable"):
                route_response["message"] += "各段方式或约束不同，请在分段路线查看详情；全程暂不支持一键切换策略。"
        if not followups:
            return route_response
        next_request = followups[0]
        next_query = str(next_request.get("query") or "").strip()
        named_final_id = next_request.get("destination_poi_id")
        if named_final_id:
            if named_final_id == (destination.get("id") or destination.get("poi_id")):
                route_response["message"] += "你选的前一地点就是指定终点，无须再走第二段。"
                route_response["needs_input"] = False
                route_response["place_request"] = None
                route_response["suggestions"] = None
                return route_response
            final_poi = next((poi for poi in agent_tools.load_pois()
                              if poi.get("id") == named_final_id), None)
            if not final_poi:
                return {**route_response, "response_kind": "chat", "route": None,
                        "route_kind": None, "message": "指定的终点已不在校园地点库中，无法显示全程路线。",
                        "needs_input": False, "place_request": None}
            final_args = {
                "start": {"type": "poi", "poi_id": destination.get("id") or destination.get("poi_id"),
                          "name": destination.get("name", "")},
                "end": {"type": "poi", "poi_id": named_final_id,
                        "name": final_poi.get("name", "")},
                "mode": (saved_request or {}).get("mode") or route_response["route"].get("mode") or "walk",
            }
            saved_constraints = (saved_request or {}).get("constraints") or {}
            try:
                final_constraints = _place_tool_constraints(saved_constraints)
            except ValueError:
                message = "任务约束无法确认，请保留需求后重试。"
                return {**route_response, "response_kind": "clarify", "route": None,
                        "route_kind": None, "message": message,
                        "clarify": {"question": message, "options": ["重试"]},
                        "needs_input": True, "place_request": saved_request, "candidates": None}
            if final_constraints:
                final_args["constraints"] = final_constraints
            first_strategy = ((route_response["route"].get("route_state") or {})
                              .get("strategy") or {}).get("name")
            final_strategy = first_strategy or (saved_request or {}).get("strategy") or detect_strategy_hint(
                str((saved_request or {}).get("original_query") or ""),
            )
            if final_strategy in {"shortest", "recommended", "scenery", "flat"}:
                final_args["strategy"] = final_strategy
            final_result, final_artifact = execute_tool(
                "plan_route", final_args, {"query": next_query, "uid": uid},
            )
            final_route = ((final_artifact or {}).get("route")
                           if isinstance(final_artifact, dict) else None)
            if not final_route:
                pending = _place_request_state("重试行程", pending_slot="selection", previous=saved_request)
                pending["selected_poi_id"] = destination.get("id") or destination.get("poi_id")
                if pending["selected_poi_id"] and pending["selected_poi_id"] not in pending["candidate_poi_ids"]:
                    pending["candidate_poi_ids"].append(pending["selected_poi_id"])
                message = (final_result or {}).get("message") or "终点路段暂时无法规划，未显示不完整的全程路线。"
                return {**route_response, "response_kind": "clarify", "route": None,
                        "route_kind": None, "message": message,
                        "clarify": {"question": message, "options": ["重试"]},
                        "needs_input": True, "place_request": pending, "candidates": None}
            full_legs = completed_legs + [final_route]
            join_issue = _place_route_join_issue(full_legs)
            if not join_issue:
                join_issue = _place_route_constraint_issue(full_legs, (saved_request or {}).get("constraints") or {})
            if not join_issue:
                join_issue = _place_route_snapshot_issue(full_legs)
            refreshed_route = None
            refresh_error = None
            if join_issue in {"两段路线使用的路网或路况版本不同", "当前路况已更新"}:
                refreshed_route, refresh_error = _replan_changed_place_route(
                    full_legs, {"query": str((saved_request or {}).get("original_query") or "") + " " + query,
                                "uid": uid, "task_constraints": (saved_request or {}).get("constraints") or {}})
                if refreshed_route is not None:
                    join_issue = None
            if join_issue:
                pending = _place_request_state("重新规划全程", pending_slot="route_retry", previous=saved_request)
                pending["completed_legs"] = full_legs
                message = f"各段路线已算出，但{join_issue}。{refresh_error or ''}无法确认完整路线。已保留你的行程，可以点击“重试”。"
                return {**route_response, "response_kind": "clarify", "route": None,
                        "route_kind": None,
                        "message": message, "clarify": {"question": message, "options": ["重试"]},
                        "needs_input": True, "place_request": pending, "candidates": None}
            full_route = refreshed_route or _combined_place_route(full_legs)
            return {**route_response, "response_kind": "route", "route": full_route,
                    "route_kind": "via", "message": _build_route_message(full_route) + _road_condition_feedback(
                        full_route, str((saved_request or {}).get("original_query") or "") + " " + query)
                        + ("路况发生变化，已按最新记录重新规划全程。" if refreshed_route is not None else ""),
                    "needs_input": False, "place_request": None, "clarify": None,
                    "suggestions": None}
        next_activities = list(next_request.get("activities") or [])
        next_categories = list(next_request.get("subcategories") or [])
        destination_name = str(destination.get("name") or "").strip()
        next_start = {"type": "poi", "name": destination_name,
                      "poi_id": destination.get("id") or destination.get("poi_id")}
        previous = {**(saved_request or {}), "original_query": next_query or "下一站",
                    "selected_poi_id": None, "candidate_poi_ids": []}
        pending = _place_request_state(
            next_query or "下一站", activities=next_activities,
            subcategories=next_categories, pending_slot="follow_up",
            start=next_start, follow_up_requests=followups[1:],
            previous=previous,
            mode=(saved_request or {}).get("mode") or route_response["route"].get("mode") or "walk",
            constraints=(saved_request or {}).get("constraints") or {},
            candidate_poi_ids=[],
        )
        if pending.get("strategy") is None:
            route_strategy = ((route_response["route"].get("route_state") or {})
                              .get("strategy") or {}).get("name")
            if route_strategy in {"shortest", "recommended", "scenery", "flat"}:
                pending["strategy"] = route_strategy
        pending["completed_legs"] = completed_legs
        route_response["needs_input"] = True
        route_response["place_request"] = pending
        if not next_query or not (next_activities or next_categories):
            pending["pending_slot"] = "destination"
            question = ("已规划到当前地点。你要回哪栋宿舍？请说具体楼栋名称。"
                        if re.search(r"宿舍|寝室", next_query) else
                        "已规划到当前地点。下一站要去哪里？请说具体地点名称。")
            route_response["clarify"] = {"question": question, "options": []}
            return route_response
        lookup_args = {
            "subcategories": next_categories, "activities": next_activities,
            "mode": pending["mode"], "limit": 4, "start": next_start,
            "max_distance_m": 10000,
        }
        lookup_constraints = {}
        if pending["constraints"].get("avoid_steps"):
            lookup_constraints["avoid_steps"] = True
        if pending["constraints"].get("avoid_slope"):
            lookup_constraints["slope"] = "avoid"
        if lookup_constraints:
            lookup_args["constraints"] = lookup_constraints
        lookup, artifact = execute_tool(
            "search_poi_candidates", lookup_args,
            {"query": next_query, "uid": uid},
        )
        candidates = ((artifact or {}).get("candidates")
                      if isinstance(artifact, dict) else None) or []
        if isinstance(lookup, dict) and lookup.get("error"):
            question = "第一段路线已生成。下一站的地点暂时查不到，请稍后重试或说明另一个地点。"
        elif not candidates:
            question = "第一段路线已生成。校园地点库暂未找到下一站的合适地点，请换个地点需求。"
        else:
            for candidate in candidates:
                candidate["place_id"] = candidate.get("poi_id")
                candidate["recommendation_start_name"] = destination_name
            pending["pending_slot"] = "selection"
            pending["candidate_poi_ids"] = [
                item["poi_id"] for item in candidates if item.get("poi_id")
            ]
            route_response["candidates"] = candidates
            question = f"下一步从{destination_name}出发找{next_query}，请选择地点。"
        route_response["clarify"] = {
            "question": question,
            "options": [] if candidates else ["重试查找"],
        }
        return route_response

    if (place_request and place_request.get("pending_slot") == "route_retry"
            and re.fullmatch(r"\s*(?:重试|重新规划|重新计算|重算)\s*[。！!]?\s*", query)):
        old_legs = place_request.get("completed_legs") or []
        route, error = _replan_changed_place_route(old_legs, {
            "query": place_request.get("original_query") or query, "uid": uid,
            "task_constraints": place_request.get("constraints") or {},
        })
        if not route:
            message = error or "当前无法可靠重算完整行程，请稍后重试。"
            return {"response_kind": "clarify", "message": message, "route": None,
                    "route_kind": None, "candidates": None, "needs_input": True,
                    "place_request": place_request, "clarify": {"question": message, "options": ["重试"]},
                    "suggestions": None, "turns": 0}
        end = (route.get("route_state") or {}).get("end") or {}
        destination = next((poi for poi in agent_tools.load_pois() if poi.get("id") == end.get("poi_id")), None)
        followups = list(place_request.get("follow_up_requests") or [])
        if followups and not destination:
            message = "已重算的终点无法与后续目标可靠关联，请重新选择地点。"
            return {"response_kind": "clarify", "message": message, "route": None,
                    "route_kind": None, "candidates": None, "needs_input": True,
                    "place_request": place_request, "clarify": {"question": message, "options": ["重试"]},
                    "suggestions": None, "turns": 0}
        response = {"response_kind": "route", "message": _build_route_message(route)
                    + "路况发生变化，已按最新记录重新规划全程。", "route": route, "route_kind": "via",
                    "candidates": None, "clarify": None, "place_request": None,
                    "needs_input": False, "suggestions": None, "turns": 0}
        if not followups:
            return response
        resumed = {**place_request, "completed_legs": [], "pending_slot": None}
        return with_next_place_stage(response, destination, resumed, followups)

    if place_request and place_request.get("pending_slot") == "destination" and not selected_poi_id:
        destination, alternatives = agent_tools.find_poi_ambiguous(query.strip())
        if destination and not alternatives:
            selected_poi_id = destination.get("id")
        elif alternatives:
            candidates = [{"poi_id": poi.get("id"), "place_id": poi.get("id"),
                           "name": poi.get("name"), "campus": poi.get("campus"),
                           "subcategory": poi.get("subcategory")}
                          for poi in alternatives[:5] if poi.get("id")]
            pending = _place_request_state(
                place_request.get("original_query") or query,
                pending_slot="selection", previous=place_request,
                candidate_poi_ids=[item["poi_id"] for item in candidates],
            )
            question = "找到了几个相近地点，请选择具体的终点。"
            return {"response_kind": "candidates", "message": question,
                    "route": None, "route_kind": None, "candidates": candidates,
                    "clarify": None, "place_request": pending,
                    "suggestions": None, "turns": 0}
        if not selected_poi_id:
            question = ("已更新出行条件。你要回哪栋宿舍？请说具体楼栋名称。"
                        if mode_change or re.search(r"台阶|楼梯|避坡|陡坡", query) else
                        "还没找到这个楼栋。请说宿舍的完整名称，例如“桂园6舍”。")
            return {"response_kind": "clarify", "message": question,
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": {"question": question, "options": []},
                    "place_request": place_request, "suggestions": None, "turns": 0}

    if selected_poi_id:
        if (place_request and place_request.get("candidate_poi_ids")
                and selected_poi_id not in place_request["candidate_poi_ids"]):
            return {"response_kind": "chat", "message": "这个候选不属于当前列表，请重新搜索后选择。",
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": None, "place_request": None, "suggestions": None, "turns": 0}
        selected_poi = next((poi for poi in agent_tools.load_pois()
                             if poi.get("id") == selected_poi_id), None)
        if not selected_poi:
            return {"response_kind": "chat", "message": "这个地点已不在当前主数据中，请重新搜索。",
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": None, "place_request": None, "suggestions": None, "turns": 0}
        start_ref = ((place_request or {}).get("start")
                     if (place_request or {}).get("pending_slot") == "destination" else
                     explicit_place_start(query) or (place_request or {}).get("start")
                     or current_coordinate_start())
        if isinstance(start_ref, dict) and start_ref.get("ambiguous_name"):
            choices = [f"{poi.get('name')}（{poi.get('campus') or '校内'}）"
                       for poi in start_ref["alternatives"]]
            pending = _place_request_state(query, pending_slot="start", start=None, previous=place_request,
                                           candidate_poi_ids=place_request.get("candidate_poi_ids", []))
            return {"response_kind": "clarify", "message": "这个起点有多个匹配地点，请选一个。",
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": {"question": "这个起点有多个匹配地点，请选一个。", "options": choices},
                    "place_request": pending, "suggestions": None, "turns": 0}
        if not start_ref:
            pending = _place_request_state(query, pending_slot="start", start=None, previous=place_request,
                                           selected_poi_id=selected_poi_id,
                                           candidate_poi_ids=place_request.get("candidate_poi_ids", [])
                                           if place_request else [])
            question = "你从哪里出发？可以点地图起点，或告诉我所在地点。"
            return {"response_kind": "clarify", "message": question, "route": None,
                    "route_kind": None, "candidates": None,
                    "clarify": {"question": question, "options": ["地图上选起点", "告诉我所在地点"]},
                    "place_request": pending, "suggestions": None, "turns": 0}
        route_args = {"start": start_ref,
                      "end": {"type": "poi", "poi_id": selected_poi_id,
                              "name": selected_poi.get("name", "")},
                      "mode": (place_request or {}).get("mode") or travel_mode or "walk"}
        saved_constraints = (place_request or {}).get("constraints") or {}
        try:
            route_constraints = _place_tool_constraints(saved_constraints)
        except ValueError:
            return {"response_kind": "clarify", "message": "任务约束无法确认，请保留需求后重试。",
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": {"question": "任务约束无法确认，请保留需求后重试。", "options": []},
                    "place_request": place_request, "needs_input": True, "suggestions": None, "turns": 0}
        if route_constraints:
            route_args["constraints"] = route_constraints
        full_request = str((place_request or {}).get("original_query") or "") + " " + str(query)
        route_strategy = (place_request or {}).get("strategy") or _explicit_place_strategy(full_request)
        if route_strategy:
            route_args["strategy"] = route_strategy
        result, artifact = execute_tool("plan_route", route_args,
                                        {"query": full_request, "uid": uid})
        route = (artifact or {}).get("route") if isinstance(artifact, dict) else None
        if route:
            route["timings_ms"] = normalize_timings(route.get("timings_ms"), agent=0.0)
            follow_up_requests = (place_request or {}).get("follow_up_requests") or []
            suggestions = []
            if follow_up_requests:
                next_request = follow_up_requests[0]
                follow_query = str(next_request.get("query") or "").strip()
                if follow_query:
                    continuation = f"从{selected_poi.get('name', '当前地点')}出发找{follow_query}"
                    current_mode = route_args["mode"]
                    if current_mode in {"bike", "drive"}:
                        continuation += "，" + {"bike": "骑行", "drive": "驾车"}[current_mode]
                    if saved_constraints.get("avoid_steps"):
                        continuation += "，不走台阶"
                    if saved_constraints.get("avoid_slope"):
                        continuation += "，避开陡坡"
                    suggestions.append({
                        "label": "下一步：找餐饮" if any(
                            value in (next_request.get("activities") or [])
                            for value in ("meal", "canteen_meal", "restaurant_meal", "fast_food")
                        ) else "继续下一步",
                        "query": continuation,
                    })
            response = {"response_kind": "route",
                        "message": _build_route_message(route) + _road_condition_feedback(route, full_request),
                        "route": route, "route_kind": "direct", "candidates": None,
                        "clarify": None, "place_request": None,
                        "suggestions": suggestions, "turns": 0}
            return with_next_place_stage(response, selected_poi, place_request,
                                         follow_up_requests)
        return {"response_kind": "chat", "message": (result or {}).get("message") or "这条路线暂时无法生成。",
                "route": None, "route_kind": None, "candidates": None,
                "clarify": None, "place_request": None, "suggestions": None, "turns": 0}

    if place_request and place_request.get("pending_slot") == "follow_up":
        original_request = place_request.get("original_query") or original_request
    if not place_request and not coord_end:
        start_ref = explicit_place_start(query)
        origin_clause = re.match(
            r"^(?:我(?:现在)?在|(?:我)?从|由)[^，,。；;]{1,40}?"
            r"(?:出发|附近|旁边|周边|周围|这边)[，,。；;]?",
            str(query or "").strip(),
        ) or re.match(
            r"^(?:我(?:现在)?在|(?:我)?从|由)[^，,。；;]{1,40}[，,。；;]",
            str(query or "").strip(),
        )
        tail = str(query or "")[origin_clause.end():] if origin_clause else ""
        wants_route = agent_tools.query_requests_route(tail) or bool(re.search(
            r"平坦(?:的)?路|平坦优先|风景优先|景观优先|少爬坡|少走坡|不走台阶",
            tail,
        ))
        named_destination = re.search(r"(?:到|去|前往|终点(?:是|为|设为|[:：]))\s*[^，,。；;]+", tail)
        if (origin_clause and start_ref and not start_ref.get("ambiguous_name")
                and wants_route and not named_destination
                and not agent_tools.resolve_activity_query(query)):
            detected_mode, explicit_mode = detect_travel_mode(query)
            constraints = {}
            if re.search(r"(?:不走|避开|不要走|绕开)(?:楼梯|台阶)", query):
                constraints["avoid_steps"] = True
            if re.search(r"避坡|少爬坡|不想爬坡|避开陡坡", query):
                constraints["avoid_slope"] = True
            pending = _place_request_state(
                query, pending_slot="destination", start=start_ref,
                mode=detected_mode if explicit_mode else travel_mode or "walk",
                constraints=constraints,
            )
            question = "你想去哪个地点？目的地明确后，我会从你说的起点规划路线。"
            return {"response_kind": "clarify", "message": question,
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": {"question": question, "options": []},
                    "place_request": pending, "suggestions": None, "turns": 0}
    activity_sequence = agent_tools.resolve_activity_sequence(original_request)
    sequence_followups = [
        {"query": item["query"],
         "activities": list(item["intent"].get("activities") or []),
         "subcategories": list(item["intent"].get("subcategories") or [])}
        for item in activity_sequence[1:]
    ]
    # A concrete last stop belongs to the ordered route, not to the fuzzy
    # recommendation for the first stop. Resolve only the final clause.
    ordered_parts = re.split(r"(?:然后|接着|随后|之后|最后|再去|再)", str(original_request or ""))
    named_final = None
    if len(ordered_parts) > 1:
        final_name = ordered_parts[-1].strip(" ，,。；;、 ")
        final_name = re.sub(r"^(?:到|去|前往)\s*", "", final_name)
        if final_name:
            resolved_final, final_alternatives = agent_tools.find_poi_ambiguous(final_name)
            if resolved_final and not final_alternatives:
                named_final = resolved_final
                named_followup = {
                    "query": final_name, "destination_poi_id": resolved_final["id"],
                    "activities": [], "subcategories": [],
                }
                if (sequence_followups and activity_sequence[-1]["query"].strip(" ，,。；;、 ")
                        == ordered_parts[-1].strip(" ，,。；;、 ")):
                    sequence_followups[-1] = named_followup
                elif activity_sequence or agent_tools.resolve_activity_query(ordered_parts[0]):
                    sequence_followups.append(named_followup)
    activity_intent = (activity_sequence[0]["intent"] if activity_sequence else
                       agent_tools.resolve_activity_query(original_request))
    if place_request and place_request.get("pending_slot") == "activity":
        answer_intent = agent_tools.resolve_activity_query(query)
        if not answer_intent:
            original_activity = agent_tools.resolve_activity_query(
                (place_request or {}).get("original_query") or original_request
            )
            options = (original_activity or {}).get("options") or [
                "篮球", "羽毛球", "乒乓球", "足球", "其他球类",
            ]
            question = (f"校园地点库暂未标出明确支持「{query.strip()}」的场地。"
                        "可以补充具体场馆名，或选择一个已收录的项目。")
            return {"response_kind": "clarify", "message": question, "route": None,
                    "route_kind": None, "candidates": None,
                    "clarify": {"question": question,
                                "options": options},
                    "place_request": _place_request_state(
                        query, pending_slot="activity", previous=place_request,
                    ),
                    "suggestions": None, "turns": 0}
        if answer_intent.get("status") == "needs_clarification":
            return {"response_kind": "clarify", "message": answer_intent["question"],
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": {"question": answer_intent["question"],
                                "options": answer_intent["options"]},
                    "place_request": _place_request_state(query, pending_slot="activity", previous=place_request),
                    "suggestions": None, "turns": 0}
        activity_intent = answer_intent
        recommendation_query = f"{original_request} {query}"
    elif (activity_intent and activity_intent.get("status") == "needs_clarification"
          and not (place_request and place_request.get("pending_slot") in {"start", "selection"})):
        start_ref = explicit_place_start(original_request)
        if start_ref and start_ref.get("ambiguous_name"):
            start_ref = None
        start_ref = start_ref or current_coordinate_start()
        constraints = {
            "avoid_steps": bool(re.search(r"(?:不走|避开|不要走|绕开)(?:楼梯|台阶)", original_request)),
            "avoid_slope": bool(re.search(r"避坡|少爬坡|不想爬坡|避开陡坡", original_request)),
        }
        pending = _place_request_state(original_request, activities=[], subcategories=[],
                                       pending_slot="activity", start=start_ref,
                                       constraints=constraints,
                                       follow_up_requests=sequence_followups,
                                       mode=travel_mode if travel_mode in {"walk", "bike", "drive"} else "walk")
        return {"response_kind": "clarify", "message": activity_intent["question"],
                "route": None, "route_kind": None, "candidates": None,
                "clarify": {"question": activity_intent["question"],
                            "options": activity_intent["options"]},
                "place_request": pending, "suggestions": None, "turns": 0}

    # 模糊的类别需求走确定性的候选检索，避免模型按熟悉度挑固定 POI。
    # 起点只取本轮坐标、用户本轮明确说出的地点，或正在回答的起点追问；不继承旧路线终点。
    if place_request and place_request.get("pending_slot") == "activity":
        recommendation_query = f"{original_request} {query}"
    recommendation_categories = list((activity_intent or {}).get("subcategories") or
                                     (place_request or {}).get("subcategories") or
                                     agent_tools.infer_recommendation_subcategories(recommendation_query) or [])
    recommendation_activities = list((activity_intent or {}).get("activities") or
                                     (place_request or {}).get("activities") or [])
    follow_up_requests = (place_request or {}).get("follow_up_requests") or sequence_followups
    recommendation_is_catalog = agent_tools.is_poi_catalog_query(recommendation_query)
    recommendation_has_target = (
        agent_tools._query_has_explicit_poi_destination(recommendation_query)
        and named_final is None
    )
    route_to_nearest = bool(
        agent_tools.query_requests_route(recommendation_query)
        and re.search(r"最近|离我最(?:近|方便)|距离最短", recommendation_query)
    )
    if ((recommendation_categories or recommendation_activities)
            and not recommendation_has_target
            and (not agent_tools.query_requests_route(recommendation_query)
                 or route_to_nearest or bool(activity_sequence and named_final))):
        start_ref = None
        explicit_origin = agent_tools.recommendation_origin_from_text(query)
        if explicit_origin:
            # 本轮文字明确给出的地点优先于页面附带的 GPS 或地图坐标。
            resolved_origin = explicit_place_start(query)
            if resolved_origin and not resolved_origin.get("ambiguous_name"):
                start_ref = resolved_origin
            elif resolved_origin and resolved_origin.get("alternatives"):
                choices = [f"{item.get('name')}（{item.get('campus') or '校内'}）"
                           for item in resolved_origin["alternatives"]]
                pending = _place_request_state(
                    recommendation_query, activities=recommendation_activities,
                    subcategories=recommendation_categories, pending_slot="start",
                    follow_up_requests=follow_up_requests, previous=place_request,
                    mode=travel_mode if travel_mode in {"walk", "bike", "drive"} else "walk",
                )
                question = "这个起点有多个匹配地点，请选一个。"
                return {"response_kind": "clarify", "message": question, "route": None,
                        "route_kind": None, "candidates": None,
                        "clarify": {"question": question, "options": choices},
                        "place_request": pending, "suggestions": None, "turns": 0}
        elif original_request != query and (
                not place_request or place_request.get("pending_slot") == "start"):
            # 只有当前问题确实在询问起点时，才把简短回答解析为新起点。
            # 例如球类回答“乒乓球”也能命中“乒乓球馆”，不能覆盖已保存起点。
            answered_origin, alternatives = agent_tools.find_poi_ambiguous(query.strip())
            if answered_origin and not alternatives:
                start_ref = {"type": "poi", "name": answered_origin["name"]}
        if start_ref is None and isinstance((place_request or {}).get("start"), dict):
            start_ref = place_request["start"]

        if start_ref is None and isinstance(coord_start, dict):
            try:
                lng, lat = float(coord_start.get("lng")), float(coord_start.get("lat"))
                if (-180 <= lng <= 180 and -90 <= lat <= 90):
                    start_ref = {"type": "coord", "name": coord_start.get("name") or "我的位置",
                                 "lng": lng, "lat": lat}
            except (TypeError, ValueError):
                start_ref = None

        if start_ref is None and not recommendation_is_catalog:
            question = "你从哪里出发？可以在地图上点起点，或告诉我所在的地点，我再按路网距离推荐。"
            options = ["地图上选起点", "告诉我所在地点"]
            pending = _place_request_state(
                recommendation_query, activities=recommendation_activities,
                subcategories=recommendation_categories, pending_slot="start",
                follow_up_requests=follow_up_requests, previous=place_request,
                mode=travel_mode if travel_mode in {"walk", "bike", "drive"} else "walk",
            )
            return {"response_kind": "clarify", "message": question,
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": {"question": question, "options": options},
                    "place_request": pending,
                    "suggestions": None, "turns": 0}

        recommendation_args = {
            "subcategories": recommendation_categories,
            "activities": recommendation_activities,
            "mode": (place_request or {}).get("mode") or
                    (travel_mode if travel_mode in {"walk", "bike", "drive"} else "walk"),
            "limit": _requested_recommendation_limit(
                recommendation_query, catalog=recommendation_is_catalog,
            ),
        }
        request_constraints = (place_request or {}).get("constraints") or {}
        constraint_query = str(recommendation_query) + " " + str(query)
        tool_constraints = {}
        if request_constraints.get("avoid_steps") or re.search(
            r"(?:不走|避开|不要走|绕开)(?:楼梯|台阶)", constraint_query
        ):
            tool_constraints["avoid_steps"] = True
        if request_constraints.get("avoid_slope") or re.search(
            r"避坡|少爬坡|不想爬坡|避开陡坡", constraint_query
        ):
            tool_constraints["slope"] = "avoid"
        if tool_constraints:
            recommendation_args["constraints"] = tool_constraints
        if start_ref and not recommendation_is_catalog:
            recommendation_args.update({"start": start_ref, "max_distance_m": 10000})
        result, artifact = execute_tool(
            "search_poi_candidates", recommendation_args,
            {"query": recommendation_query, "uid": uid},
        )
        candidates = (artifact or {}).get("candidates") if isinstance(artifact, dict) else None
        if result.get("error"):
            return {"response_kind": "chat", "message": result.get("message") or "附近地点暂时无法核实。",
                    "route": None, "route_kind": None, "candidates": None,
                    "clarify": None, "suggestions": None, "turns": 0}
        if not candidates:
            empty_message = ("校园地点主数据中暂未记录明确支持该用途的场所；这不等于现实中不存在。"
                             if recommendation_activities else
                             result.get("message") or "当前路网范围内没有找到合适的地点。")
            return {"response_kind": "chat",
                    "message": empty_message,
                    "route": None, "route_kind": None, "candidates": [],
                    "clarify": None, "place_request": None, "suggestions": None, "turns": 0}

        for candidate in candidates:
            candidate["place_id"] = candidate.get("poi_id")

        if recommendation_is_catalog:
            for candidate in candidates:
                candidate["recommendation_start_name"] = ""
            total_count = result.get("total_count")
            if isinstance(total_count, int) and total_count > len(candidates):
                summary = (
                    f"校园地点主数据中共有 {total_count} 个匹配地点，当前先列出 {len(candidates)} 个。"
                    "可按学部或细分类别缩小范围；开放和使用规则以现场或校方信息为准。"
                )
            else:
                summary = f"校园地点主数据中找到 {len(candidates)} 个匹配地点，已列出候选。开放和使用规则以现场或校方信息为准。"
            pending = _place_request_state(
                recommendation_query, activities=recommendation_activities,
                subcategories=recommendation_categories, pending_slot="selection",
                candidate_poi_ids=[item.get("poi_id") for item in candidates if item.get("poi_id")],
                follow_up_requests=follow_up_requests,
                previous=place_request,
                mode=travel_mode if travel_mode in {"walk", "bike", "drive"} else "walk",
            )
            return {"response_kind": "candidates", "message": summary,
                    "route": None, "route_kind": None, "candidates": candidates,
                    "clarify": None, "place_request": pending,
                    "suggestions": None, "turns": 0}

        if route_to_nearest:
            route_request = str(recommendation_query) + " " + str(query)
            selected_candidate = candidates[0]
            selected_endpoint = {"type": "poi", "name": selected_candidate.get("name", "")}
            if selected_candidate.get("poi_id"):
                selected_endpoint["poi_id"] = selected_candidate["poi_id"]
            route_args = {
                "start": start_ref,
                "end": selected_endpoint,
                "mode": recommendation_args["mode"],
            }
            if tool_constraints:
                route_args["constraints"] = dict(tool_constraints)
            if re.search(r"平坦优先|少爬坡|少走坡|避坡|不想爬坡", route_request):
                route_args["weights"] = {"distance": 0.2, "slope": 0.6, "scenery": 0.2}
                route_args.setdefault("constraints", {})["slope"] = "avoid"
            elif re.search(r"风景优先|景观优先|赏樱|看风景|风景好", route_request):
                route_args["weights"] = {"distance": 0.15, "slope": 0.15, "scenery": 0.7}
            if re.search(r"(?:不走|避开|不要走|绕开)(?:楼梯|台阶)", route_request):
                route_args["constraints"] = {**route_args.get("constraints", {}), "avoid_steps": True}
            weather = route_weather(route_request)
            route_result, route_artifact = execute_tool(
                "plan_route", route_args, {"query": route_request, "uid": uid},
            )
            route = ((route_artifact or {}).get("route")
                     if isinstance(route_artifact, dict) else None)
            if route:
                route["timings_ms"] = normalize_timings(route.get("timings_ms"), agent=0.0)
                response = {"response_kind": "route",
                            "message": _build_route_message(route) + _road_condition_feedback(route, route_request),
                            "route": route, "route_kind": "direct", "candidates": None,
                            "recommendation_candidates": candidates,
                            "clarify": None, "place_request": None, "suggestions": None, "turns": 0}
                saved_request = {**(place_request or {}), "mode": route_args["mode"],
                                 "constraints": {
                                     "avoid_steps": bool(route_args.get("constraints", {}).get("avoid_steps")),
                                     "avoid_slope": route_args.get("constraints", {}).get("slope") == "avoid",
                                 }}
                return add_route_weather(
                    with_next_place_stage(response, selected_candidate, saved_request,
                                          follow_up_requests), weather,
                )
            if route_result.get("error"):
                return add_route_weather({"response_kind": "chat",
                        "message": route_result.get("message") or "找到附近地点，但当前无法生成可靠路线。",
                        "route": None, "route_kind": None, "candidates": None,
                        "clarify": None, "place_request": None, "suggestions": None, "turns": 0}, weather)

        first = candidates[0]
        start_name = result.get("start_name") or start_ref.get("name") or "你选定的起点"
        if start_ref.get("type") == "poi":
            for candidate in candidates:
                candidate["recommendation_start_name"] = start_name
        raw_category = first.get("activity_labels") or first.get("subcategory_label") or first.get("subcategory")
        if isinstance(raw_category, (list, tuple)):
            category = "、".join(str(label).strip() for label in raw_category if str(label).strip())
        else:
            category = str(raw_category or "").strip()
        place_name = str(first.get("name", "候选地点"))
        if category and category in place_name:
            category = ""
        distance = first.get("distance_m")
        if distance is not None:
            category_suffix = f"（{category}）" if category else ""
            summary = (f"从{start_name}出发，我按当前路网估算距离排了 {len(candidates)} 个选项。"
                       f"最近的是{place_name}{category_suffix}，约 {round(float(distance))} 米。")
        else:
            summary = f"我按从{start_name}出发的路网距离排了 {len(candidates)} 个选项。"
        summary += "起点和地点到道路的接驳按直线估算，未核实实际通行；点选地点可继续规划路线。"
        pending = _place_request_state(
            recommendation_query, activities=recommendation_activities,
            subcategories=recommendation_categories, pending_slot="selection", start=start_ref,
            follow_up_requests=follow_up_requests,
            constraints=(place_request or {}).get("constraints") or {
                "avoid_steps": bool(re.search(r"(?:不走|避开|不要走|绕开)(?:楼梯|台阶)", recommendation_query)),
                "avoid_slope": bool(re.search(r"避坡|少爬坡|不想爬坡|避开陡坡", recommendation_query)),
            },
            candidate_poi_ids=[item.get("poi_id") for item in candidates if item.get("poi_id")],
            previous=place_request,
            mode=travel_mode if travel_mode in {"walk", "bike", "drive"} else "walk",
        )
        return {"response_kind": "candidates", "message": summary,
                "route": None, "route_kind": None, "candidates": candidates,
                "clarify": None, "place_request": pending, "suggestions": None, "turns": 0}

    if not config.DEEPSEEK_API_KEY:
        raise PlannerError("DEEPSEEK_API_KEY 未配置")

    started = time.monotonic()
    client = _make_client()
    messages = _build_messages(query, context, history, coord_start, coord_end,
                               uid=uid, travel_mode=travel_mode,
                               coord_waypoints=coord_waypoints)

    artifact_route, route_kind = None, None
    multiple_route_artifacts = False
    artifact_candidates, artifact_clarify = None, None
    artifact_route_comparison = None
    no_path_result = None
    artifact_suggestions = None
    weather_result = None
    weather_tool_failed = False
    road_condition_result = None
    road_condition_failed = False
    place_detail_results = {}
    resolved_poi_results = {}
    last_tool_results = {}
    weather_in_structured_answer = False
    executed_tool_names = set()
    final_message = ""
    turns = 0
    agent_elapsed_ms = 0.0
    verification_repair_attempts = 0
    verification_failure = None
    verification_diagnosis = None
    verification_retry_exhausted = False
    active_task_request = (context or {}).get("active_task_request")
    if not isinstance(active_task_request, str):
        active_task_request = ""
    # A clarification answer is still part of the original route request.
    preference_query = active_task_request or query
    if active_task_request and re.search(r"风景优先|景观优先|平坦优先|少爬坡|少走坡|赏樱|赏花", query):
        preference_query = query
    allow_preferences = route_preference_requested(preference_query, context, history)
    if active_task_request and re.search(r"最短|赶时间|赶课|快一点|快点|尽快|直接到|不绕路|别绕路", query):
        allow_preferences = False
    previous = (context or {}).get("previous_intent") or {}
    prev_constraints = previous.get("constraints") or (context or {}).get("constraints") or {}
    constraint_query = (active_task_request or "") + " " + query
    avoid_slope = prev_constraints.get("slope") == "avoid" or "避坡" in constraint_query or "陡坡" in constraint_query
    if re.search(r"(?:不(?:用|需要)(?:再)?避(?:开)?(?:陡坡|坡)|可以(?:爬|走)(?:坡|陡坡))", query):
        avoid_slope = False
    avoid_steps = bool(re.search(r"(?:不走|避开|不要走|绕开)(?:楼梯|台阶)", constraint_query))
    if re.search(r"(?:可以|允许)(?:走|经过)?(?:楼梯|台阶)|不(?:用|需要)避开(?:楼梯|台阶)", query):
        avoid_steps = False
    active_weights = None
    if allow_preferences:
        active_weights = previous.get("weights")
    route_tools = {"plan_route", "plan_via_route", "plan_tour", "plan_multimodal_route"}

    def _reflection_budget_available() -> bool:
        """剩余轮次与时间是否还够一次定向修正（反思-重规划）。"""
        if turns >= MAX_TURNS - 1:
            return False
        return (time.monotonic() - started) <= TIME_BUDGET_S - _REFLECTION_MIN_BUDGET_S

    def _with_provenance(response):
        """把 run 级 PROV 记录挂到响应内部字段（下游持久化后剥离，不外露）。"""
        if not isinstance(response, dict):
            return response
        try:
            verification = _response_verification(response)
            response["provenance"] = _provenance.finish(
                response_kind=response.get("response_kind"),
                artifact_kinds=[
                    kind for kind, present in (
                        ("route", response.get("route")),
                        ("candidates", response.get("candidates")),
                        ("clarify", response.get("clarify")),
                        ("suggestions", response.get("suggestions")),
                        ("route_comparison", response.get("route_comparison")),
                    ) if present
                ],
                verification_status=verification.get("status")
                if isinstance(verification, dict) else None,
                turns=response.get("turns"),
            )
        except Exception:
            logger.warning("溯源记录封口失败（忽略）", exc_info=True)
        _record_experience_sample(query, response)
        return response

    for _ in range(MAX_TURNS):
        if time.monotonic() - started > TIME_BUDGET_S:
            logger.warning("Agent 循环超时（%.1fs），基于已有结果收尾", time.monotonic() - started)
            break
        turns += 1

        try:
            llm_started = time.perf_counter()
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=agent_tools.TOOL_SCHEMAS,
                temperature=0.0,
                max_tokens=LLM_MAX_TOKENS,
            )
            agent_elapsed_ms += (time.perf_counter() - llm_started) * 1000
        except Exception as e:
            agent_elapsed_ms += (time.perf_counter() - llm_started) * 1000
            if verification_failure is not None:
                return _route_verification_failure_response(
                    verification_failure,
                    message="路线未通过独立核验，定向修正请求失败，没有显示路线。",
                    turns=turns,
                )
            raise PlannerError(f"LLM 调用失败: {type(e).__name__}: {e}") from e

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        # 无工具调用：LLM 直接给出最终答复，循环结束
        if not tool_calls:
            final_message = (msg.content or "").strip()
            break

        messages.append(msg.model_dump(exclude_none=True))

        add_verification_repair_instruction = False
        for call_index, tc in enumerate(tool_calls):
            name = tc.function.name
            is_verification_repair = False
            if name in route_tools and verification_failure is not None:
                if (verification_repair_attempts >= MAX_REFLECTION_ATTEMPTS
                        or not _reflection_budget_available()):
                    verification_retry_exhausted = True
                    result = {
                        "error": "route_verification_retry_exhausted",
                        "message": "定向修正机会已用完，本次路线调用未执行。",
                    }
                    artifact = None
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": agent_tools.result_to_json(result),
                    })
                    if tool_trace_callback:
                        tool_trace_callback({
                            "tool_name": name, "argument_keys": [],
                            "outcome": "skipped_after_route_verification_retry",
                            "artifact_kind": None,
                        })
                    for skipped_call in tool_calls[call_index + 1:]:
                        messages.append({
                            "role": "tool", "tool_call_id": skipped_call.id,
                            "content": agent_tools.result_to_json({
                                "error": "not_executed_after_route_verification_failure",
                                "message": "路线修正机会已用完，本轮后续工具调用未执行。",
                            }),
                        })
                        if tool_trace_callback:
                            tool_trace_callback({
                                "tool_name": skipped_call.function.name,
                                "argument_keys": [],
                                "outcome": "skipped_after_route_verification_retry",
                                "artifact_kind": None,
                            })
                    break
                verification_repair_attempts += 1
                is_verification_repair = True
            manual_start = _manual_map_start_ref(coord_start)
            # A new explicit origin in this turn supersedes the original task;
            # otherwise keep the original named origin through clarifications.
            explicit_start = (_explicit_start_name(query)
                              or _explicit_start_name(active_task_request))
            manual_start_applies = bool(
                manual_start and (not explicit_start or explicit_start == manual_start["name"])
            )
            args = {}
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                result = {"error": "invalid_args", "message": "参数不是合法 JSON，请修正后重试"}
                artifact = None
                if tool_trace_callback:
                    tool_trace_callback({
                        "tool_name": name, "argument_keys": [],
                        "outcome": "invalid_args", "artifact_kind": None,
                    })
            else:
                if (name == "ask_user" and manual_start_applies and isinstance(args, dict)
                        and _asks_for_origin(args.get("question", ""))):
                    # The selected map origin already answers this clarification. Return it
                    # to the model as tool context and let the bounded agent loop continue.
                    result = {
                        "message": "用户已经在地图上手动选定起点，精确坐标已随请求提供。"
                                   "请直接以该坐标作为起点继续处理，不要再次询问起点。"
                    }
                    artifact = None
                else:
                    if name in route_tools and isinstance(args, dict):
                        if manual_start_applies:
                            # Do not let the model omit the selected origin or replace it with
                            # a guessed POI; an explicitly named different origin remains intact.
                            args["start"] = manual_start
                        if not allow_preferences:
                            args.pop("weights", None)
                        elif active_weights and "weights" not in args:
                            args["weights"] = active_weights
                        elif args.get("weights"):
                            active_weights = args["weights"]
                        constraints = args.get("constraints")
                        if isinstance(constraints, dict):
                            avoid_slope = avoid_slope or constraints.get("slope") == "avoid"
                        if avoid_slope:
                            args["constraints"] = {**(constraints if isinstance(constraints, dict) else {}),
                                                   "slope": "avoid"}
                        if avoid_steps:
                            args["constraints"] = {**(args.get("constraints") or {}),
                                                   "avoid_steps": True}
                    try:
                        result, artifact = execute_tool(
                            name, args, {"query": constraint_query if name in route_tools else query,
                                         "uid": uid},
                            record_trace=False,
                        )
                    except Exception:
                        if tool_trace_callback:
                            tool_trace_callback({
                                "tool_name": name, "argument_keys": sorted(args),
                                "outcome": "exception", "artifact_kind": None,
                            })
                        raise
                if tool_trace_callback:
                    record_tool_result(name, args, result, artifact)

            executed_tool_names.add(name)
            tool_identity = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
            last_tool_results[(name, tool_identity)] = result
            if name == "get_place_details":
                place_detail_results[tool_identity] = result
            if name == "resolve_poi":
                resolved_poi_results[tool_identity] = result
            if name == "list_road_conditions":
                if (isinstance(result, dict) and not result.get("error")
                        and isinstance(result.get("conditions"), list)):
                    road_condition_result, road_condition_failed = result, False
                else:
                    road_condition_result, road_condition_failed = None, True

            if name == "get_weather" and isinstance(result, dict):
                if result.get("error"):
                    weather_result, weather_tool_failed = None, True
                elif isinstance(result.get("weather"), str) and result["weather"]:
                    weather_result, weather_tool_failed = result, False
                else:
                    weather_result, weather_tool_failed = None, True

            if (name in route_tools and artifact_route is not None
                    and isinstance(result, dict) and result.get("error")
                    and re.search(r"(?:先|首先).*?(?:再|然后|接着|随后|最后)",
                                  (active_task_request or "") + query)):
                multiple_route_artifacts = True
            if artifact:
                if "route" in artifact:
                    if artifact_route is not None and artifact["route"] is not None:
                        multiple_route_artifacts = True
                    artifact_route = artifact["route"]
                    route_kind = artifact.get("route_kind", "direct")
                if "candidates" in artifact:
                    artifact_candidates = artifact["candidates"]
                if "clarify" in artifact:
                    artifact_clarify = artifact["clarify"]
                if "suggestions" in artifact:
                    artifact_suggestions = artifact["suggestions"]
                if "route_comparison" in artifact:
                    artifact_route_comparison = artifact["route_comparison"]

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": agent_tools.result_to_json(result),
            })

            if (name == "plan_route" and verification_failure is None
                    and isinstance(result, dict)
                    and result.get("error") == "route_not_found"
                    and result.get("reason_code") == "graph_no_path"
                    and isinstance(result.get("route_request"), dict)):
                no_path_result = result

            if (name in route_tools and isinstance(result, dict)
                    and result.get("error") in {
                        "route_verification_failed", "route_verification_unknown",
                    }):
                artifact_route, route_kind = None, None
                artifact_clarify = None
                verification_failure = result
                verification_diagnosis = reflection.diagnose(
                    result.get("route_verification") or result
                )
                _provenance.add_influence(
                    verification_diagnosis.get("primary_conflict")
                )
                if (verification_repair_attempts >= MAX_REFLECTION_ATTEMPTS
                        or not _reflection_budget_available()
                        or verification_diagnosis.get("should_give_up")):
                    verification_retry_exhausted = True
                else:
                    add_verification_repair_instruction = True
                # Tool calls from one assistant response are a single batch. Once a
                # route fails verification, do not execute a later route from that
                # same response before the model has seen the failure. Complete the
                # tool-call protocol with explicit skipped results, then request one
                # targeted correction on the next model turn.
                for skipped_call in tool_calls[call_index + 1:]:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": skipped_call.id,
                        "content": agent_tools.result_to_json({
                            "error": "not_executed_after_route_verification_failure",
                            "message": "前一条路线未通过独立核验，本轮其余工具调用未执行。",
                        }),
                    })
                    if tool_trace_callback:
                        tool_trace_callback({
                            "tool_name": skipped_call.function.name,
                            "argument_keys": [],
                            "outcome": "skipped_after_route_verification_failure",
                            "artifact_kind": None,
                        })
                break

            if is_verification_repair:
                if not artifact_route and not artifact_clarify:
                    verification_retry_exhausted = True
                # A single correction call is the entire retry budget. Do not
                # let another route call in this model batch replace its result.
                for skipped_call in tool_calls[call_index + 1:]:
                    messages.append({
                        "role": "tool", "tool_call_id": skipped_call.id,
                        "content": agent_tools.result_to_json({
                            "error": "not_executed_after_route_verification_retry",
                            "message": "路线修正调用已完成，本轮后续工具调用未执行。",
                        }),
                    })
                    if tool_trace_callback:
                        tool_trace_callback({
                            "tool_name": skipped_call.function.name,
                            "argument_keys": [],
                            "outcome": "skipped_after_route_verification_retry",
                            "artifact_kind": None,
                        })
                break

        if add_verification_repair_instruction:
            messages.append({
                "role": "system",
                "content": reflection.build_reflection_message(
                    verification_diagnosis or {}, verification_repair_attempts + 1,
                ),
            })

        # 终止型工具：澄清提问立即结束
        if verification_retry_exhausted:
            break
        if (no_path_result is not None and artifact_route is None
                and artifact_clarify is None and artifact_route_comparison is None):
            break
        if artifact_clarify:
            break

        # 路线已产出 → 立即收尾，不再让 LLM 多跑一轮
        # （DeepSeek 在历史含 tool_call 时会把内部 DSML 工具调用格式当纯文本吐出）
        if artifact_route:
            break
        if artifact_route_comparison:
            break

    # ---- 收尾：组装响应 ----
    if verification_retry_exhausted:
        reason = reflection.summarize(verification_diagnosis or {})
        prefix = f"{reason}，" if reason else ""
        return _with_provenance(_route_verification_failure_response(
            verification_failure,
            message=f"{prefix}路线未通过独立核验且已无可执行的修正，没有展示路线。",
            turns=turns,
        ))

    if (no_path_result is not None and artifact_route is None
            and artifact_clarify is None and artifact_route_comparison is None):
        route_request = no_path_result["route_request"]
        start_name = str((route_request.get("start") or {}).get("name") or "起点")
        end_name = str((route_request.get("end") or {}).get("name") or "终点")
        mode_name = {"walk": "步行", "bike": "骑行", "drive": "驾车"}.get(
            route_request.get("travel_mode"), "当前出行方式",
        )
        hard_constraints = route_request.get("hard_constraints")
        avoid_steps = isinstance(hard_constraints, dict) and hard_constraints.get("avoid_steps") is True
        conditions = f"{mode_name}且避开已标注台阶" if avoid_steps else mode_name
        feedback = _road_condition_feedback({
            "no_route": True,
            "mode": route_request.get("travel_mode"),
            "road_conditions_applied": no_path_result.get("road_conditions_applied"),
            "road_condition_overview": no_path_result.get("road_condition_overview"),
        }, query)
        return _with_provenance({
            "response_kind": "no_route",
            "message": (f"按当前路网和已记录路况，从{start_name}到{end_name}"
                        f"在{conditions}条件下未找到可通行的完整路线。"
                        "现场通行状态未单独核实。" + feedback),
            "route": None, "route_kind": None,
            "route_state": route_request,
            "no_path_evidence": {"reason_code": "graph_no_path"},
            "candidates": None, "clarify": None, "suggestions": None,
            "turns": turns,
        })

    if multiple_route_artifacts and verification_failure is None:
        message = "完整行程暂未能确认，已计算的单段路线不会作为全程显示，请稍后重试。"
        return _with_provenance({
            "response_kind": "clarify", "message": message,
            "route": None, "route_kind": None, "candidates": None,
            "clarify": {"question": message, "options": []},
            "needs_input": True, "suggestions": None, "turns": turns,
        })

    if artifact_route:
        artifact_route["timings_ms"] = normalize_timings(
            artifact_route.get("timings_ms"), agent=agent_elapsed_ms
        )
    if artifact_clarify:
        return _with_provenance({
            "response_kind": "clarify",
            "message": _sanitize_message(final_message) or artifact_clarify["question"],
            "route": artifact_route, "route_kind": route_kind,
            "candidates": artifact_candidates,
            "clarify": artifact_clarify,
            "suggestions": artifact_suggestions,
            "route_verification": (verification_failure or {}).get("route_verification"),
            "turns": turns,
        })

    if artifact_route_comparison:
        return _with_provenance({
            "response_kind": "route_comparison",
            "message": _build_route_comparison_message(artifact_route_comparison),
            "route": None,
            "route_kind": None,
            "route_comparison": artifact_route_comparison,
            "candidates": None,
            "clarify": None,
            "suggestions": None,
            "turns": turns,
        })

    # A verification failure on the final available turn must not fall through
    # to the API's legacy parser fallback, which does not carry this proof report.
    if verification_failure is not None and artifact_route is None:
        reason = reflection.summarize(verification_diagnosis or {})
        prefix = f"{reason}，" if reason else ""
        return _with_provenance(_route_verification_failure_response(
            verification_failure,
            message=f"{prefix}本轮无法完成修正，没有展示路线。",
            turns=turns,
        ))

    # 防御：LLM 直出文本里若混入 DSML 标记，剥掉
    final_message = _sanitize_message(final_message)

    if not final_message:
        # 循环耗尽/超时但 LLM 没给结论：基于已有 artifact 兜底生成
        if artifact_route:
            final_message = _build_route_message(artifact_route)
        elif artifact_candidates:
            final_message = "帮你找到这些候选地点，选一个我帮你规划路线～"
        else:
            raise PlannerError("Agent 循环结束但没有产出任何结果")

    if road_condition_failed:
        unavailable = "本次路况查询失败，无法确认最新路况，请稍后重试。"
        pure_condition_query = (not artifact_route and not artifact_candidates
                                and executed_tool_names <= {"list_road_conditions", "get_weather"})
        final_message = (unavailable if pure_condition_query else _mixed_evidence_summary(
            unavailable, route=artifact_route, candidates=artifact_candidates,
            place_results=list(place_detail_results.values()), resolved_results=list(resolved_poi_results.values()),
            last_tool_results=last_tool_results, weather_result=weather_result,
            weather_failed=weather_tool_failed, tools_used=executed_tool_names, query=query))
        weather_in_structured_answer = not pure_condition_query and "get_weather" in executed_tool_names
        if pure_condition_query and "get_weather" in executed_tool_names:
            if weather_result:
                description = weather_result["weather"]
                temperature = weather_result.get("temperature")
                if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
                    description += f"、{temperature:g}°C"
                final_message += f"\n天气参考：{description}。"
                advice = weather_result.get("advice")
                if isinstance(advice, str) and advice.strip():
                    final_message += f"{advice.strip()}。"
            elif weather_tool_failed:
                final_message += "\n天气暂时获取不到。"
    elif road_condition_result is not None:
        condition_summary = _vision_condition_summary(road_condition_result, (active_task_request or "") + query)
        if (condition_summary and road_condition_result.get("conditions") == []
                and isinstance(artifact_route, dict)
                and ((type(artifact_route.get("road_conditions_applied")) is int
                      and artifact_route["road_conditions_applied"] > 0)
                     or bool(artifact_route.get("road_condition_overview")))):
            condition_summary = "查询与路线规划时的路况记录不一致，可能发生了变化，请重新查询。"
        if condition_summary:
            pure_condition_query = (not artifact_route and not artifact_candidates
                                    and executed_tool_names <= {"list_road_conditions", "get_weather"})
            final_message = (condition_summary if pure_condition_query else _mixed_evidence_summary(
                condition_summary, route=artifact_route, candidates=artifact_candidates,
                place_results=list(place_detail_results.values()), resolved_results=list(resolved_poi_results.values()),
                last_tool_results=last_tool_results, weather_result=weather_result,
                weather_failed=weather_tool_failed, tools_used=executed_tool_names, query=query))
            weather_in_structured_answer = not pure_condition_query and "get_weather" in executed_tool_names
            if pure_condition_query and "get_weather" in executed_tool_names:
                if weather_result:
                    description = weather_result["weather"]
                    temperature = weather_result.get("temperature")
                    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
                        description += f"、{temperature:g}°C"
                    final_message += f"\n天气参考：{description}。"
                    advice = weather_result.get("advice")
                    if isinstance(advice, str) and advice.strip():
                        final_message += f"{advice.strip()}。"
                elif weather_tool_failed:
                    final_message += "\n天气暂时获取不到。"

    delivered_weather = weather_result if weather_in_structured_answer else None
    if artifact_route and not weather_in_structured_answer and re.search(r"天气|气温|现在.*(?:下雨|下雪)",
                                    (active_task_request or "") + query):
        if weather_result:
            description = weather_result["weather"]
            temperature = weather_result.get("temperature")
            if isinstance(temperature, (int, float)):
                description += f"、{temperature:g}°C"
            final_message += f" 天气参考：{description}。"
            delivered_weather = weather_result
        elif weather_tool_failed:
            final_message += " 天气暂时获取不到；本次路线没有依据天气调整。"
        else:
            final_message += " 本次未取得天气信息；路线仅依据现有道路数据。"

    if artifact_route:
        final_message += _road_condition_feedback(
            artifact_route, (active_task_request or "") + query,
        )

    response_kind = "chat"
    if artifact_route:
        response_kind = "route"
    elif artifact_candidates:
        response_kind = "candidates"

    return _with_provenance({
        "response_kind": response_kind,
        "message": final_message,
        "route": artifact_route,
        "weather": delivered_weather,
        "route_kind": route_kind,
        "candidates": artifact_candidates,
        "clarify": None,
        "suggestions": artifact_suggestions,
        "turns": turns,
    })
