"""珞珈智行 — Agent 工具的能力说明层（GeoSkill 式技能规范）。

工具 schema 里的 description 只服务单次 function calling；
本模块给每个工具补一张结构化的「能力说明卡」，说明能力的结构与边界
（能做什么、需要什么、产出什么、何时该用、何时**不该**用、与谁易混），
再把卡片渲染成一张紧凑说明表注入 system 上下文，让模型不必靠语义猜工具。

设计取舍：
- 只描述能力，不承担计算与决策；不修改 agents/tools.py 的 schema 文本。
- 与 agents/tools.py 解耦（本模块不 import tools），覆盖性由
  validate_skill_coverage() 在测试中延迟校验，避免循环导入。
"""

from __future__ import annotations

# 能力说明卡的合法产出类型（与 planner 的 artifact_kind 一致）
VALID_PRODUCES = {None, "route", "candidates", "clarify", "suggestions", "route_comparison"}

VALID_CATEGORIES = {"地点", "路线", "环境", "交互"}

TOOL_SKILLS = {
    "resolve_poi": {
        "category": "地点",
        "capability": "把用户口中的地名解析成校内主数据里的确定地点",
        "requires": "一个地名或别名",
        "produces": None,
        "use_when": "地名不确定或需要确认是否存在（如「七舍」「珞珈山」）",
        "do_not_use_when": "用户要的是按用途找地点（用 search_poi_candidates）；"
                           "已经拿到 poi_id 时不要重复解析",
        "confusable_with": ["get_place_details", "search_poi_candidates"],
    },
    "get_place_details": {
        "category": "地点",
        "capability": "返回地点的类别、别名、来源与核实状态",
        "requires": "一个地点名",
        "produces": None,
        "use_when": "需要来源、别名、开放/无障碍等属性，或要说明哪些字段未知",
        "do_not_use_when": "只需要位置或要规划路线；工具返回 unknown 的字段不得用常识补全",
        "confusable_with": ["resolve_poi"],
    },
    "search_poi_candidates": {
        "category": "地点",
        "capability": "按类别/用途/关键词检索候选，并按本轮起点的路网距离排序",
        "requires": "推荐附近地点时必须给本轮起点 start；纯目录列举可不给",
        "produces": "candidates",
        "use_when": "用户只有目的没有具体地点（「想吃饭」「想喝咖啡」「去打球」）",
        "do_not_use_when": "已有明确地点名（用 resolve_poi/plan_route）；"
                           "问「步行十分钟内能去哪」用 find_reachable_places",
        "confusable_with": ["find_reachable_places", "resolve_poi"],
    },
    "find_reachable_places": {
        "category": "地点",
        "capability": "在已标记可通行路网上按距离或时间预算检索可达地点",
        "requires": "本轮起点 start；距离与时间预算二选一",
        "produces": "candidates",
        "use_when": "用户问「步行十分钟内有什么食堂」「附近能去哪」",
        "do_not_use_when": "只需列出某类地点（用 search_poi_candidates）；"
                           "末端接驳未核实，不得宣称为入口已确认可达",
        "confusable_with": ["search_poi_candidates"],
    },
    "compare_routes": {
        "category": "路线",
        "capability": "在同一数据/天气/路况快照下对比最短、风景、平坦三种策略",
        "requires": "明确的起点与终点",
        "produces": "route_comparison",
        "use_when": "用户明确要求比较不同走法或三种偏好",
        "do_not_use_when": "用户已在表达偏好并要一条路线（直接 plan_route）；"
                           "本工具只比较，不切换地图当前路线",
        "confusable_with": ["plan_route"],
    },
    "plan_route": {
        "category": "路线",
        "capability": "规划起点到终点的多因素最优路径并做独立核验",
        "requires": "起点与终点都明确",
        "produces": "route",
        "use_when": "普通 A→B 出行，含「避开陡坡」「不走台阶」等硬约束",
        "do_not_use_when": "有途经点（用 plan_via_route）；有总时间预算（用 plan_itinerary）；"
                           "多点游览（用 plan_tour）；分段换乘（用 plan_multimodal_route）",
        "confusable_with": ["plan_via_route", "compare_routes"],
    },
    "plan_via_route": {
        "category": "路线",
        "capability": "规划按顺序经过若干途经点的路径，途经点可点名、给类别或给坐标",
        "requires": "起点与终点；建议给出至少一个途经点（点名 / 类别 / 坐标三选一）",
        "produces": "route",
        "use_when": "出现「顺便/路上/先去 A 再去 B」等顺路或有序多点需求",
        "do_not_use_when": "用户给了总时间预算与停留（用 plan_itinerary）；"
                           "不得臆造时间预算、停留或返程",
        "confusable_with": ["plan_itinerary", "plan_route"],
    },
    "plan_itinerary": {
        "category": "路线",
        "capability": "按时间预算规划多站行程，计入每站停留并报告超预算差额",
        "requires": "起点、有序站点、明确的 time_budget_min",
        "produces": "route",
        "use_when": "用户明确给出总时间、停留地点或「逛几个地方再回到某处」",
        "do_not_use_when": "用户没给时间预算时不得自行编造；"
                           "单纯先到、再到用 plan_via_route",
        "confusable_with": ["plan_via_route", "plan_tour"],
    },
    "plan_multimodal_route": {
        "category": "路线",
        "capability": "规划含 2~4 段、每段独立出行方式的分段换乘路径",
        "requires": "起点与至少两段 legs（每段有 mode 与 end）",
        "produces": "route",
        "use_when": "用户明确说「先骑车到 X 再步行到 Y」这类换乘需求",
        "do_not_use_when": "普通单方式路径（用 plan_route）；"
                           "不要仅因起终点附近可能是步行道就调它",
        "confusable_with": ["plan_route"],
    },
    "plan_tour": {
        "category": "路线",
        "capability": "规划多点游览路线，自动选点排序，生成环线或开放线",
        "requires": "无必填参数：起点可选（不传时以重要度最高的景点为锚），"
                    "主题、季节、点名必经站也均为可选",
        "produces": "route",
        "use_when": "「逛遍全校」「推荐一条赏樱路线」「带我转转」等无明确终点需求；"
                    "用户没说从哪出发也直接规划，不必先反问位置",
        "do_not_use_when": "用户点名游览地点时必须按 poi_names 全量纳入，"
                           "不得退化成泛景点环线或用其他景点替代",
        "confusable_with": ["plan_itinerary", "search_poi_candidates"],
    },
    "get_weather": {
        "category": "环境",
        "capability": "返回武汉实时天气及其对出行的影响",
        "requires": "无参数",
        "produces": None,
        "use_when": "用户问天气、要不要带伞、适不适合出门",
        "do_not_use_when": "不得凭记忆回答天气；天气失败时静默跳过，不阻断路线规划",
        "confusable_with": ["list_road_conditions"],
    },
    "list_road_conditions": {
        "category": "环境",
        "capability": "返回当前生效的封路/施工/积水/事故/活动管制（绑定具体路段）",
        "requires": "无参数",
        "produces": None,
        "use_when": "用户问哪里封路、哪段施工，或判断管制是否影响路线",
        "do_not_use_when": "不得凭印象说「能走」；步行同样会绕封闭路段",
        "confusable_with": ["get_weather"],
    },
    "ask_user": {
        "category": "交互",
        "capability": "向用户提问澄清缺失或歧义的关键信息",
        "requires": "一个具体问题，可选 2~5 个选项",
        "produces": "clarify",
        "use_when": "缺起点、地名有多个候选、需求存在歧义",
        "do_not_use_when": "信息已足够时不得反问；调用后本轮对话立即结束，"
                           "不要在同一批次里再调其他工具",
        "confusable_with": ["suggest_followup"],
    },
    "suggest_followup": {
        "category": "交互",
        "capability": "给出 2~3 个可一键继续的后续问法",
        "requires": "2~3 条 {label, query}",
        "produces": "suggestions",
        "use_when": "非路线规划场景（地点介绍、天气查询）答复完成后",
        "do_not_use_when": "路线规划成功后系统自动附加跟进建议，不要重复调用",
        "confusable_with": ["ask_user"],
    },
}


_MANIFEST_HEADER = (
    "工具能力说明（能力边界优先于工具名直觉；下表与工具 schema 同时生效）："
)

_MANIFEST_COLUMNS = "工具｜类别｜何时用｜何时不要用"

_manifest_cache = None


def build_skill_manifest() -> str:
    """渲染紧凑的能力说明表；进程内缓存。"""
    global _manifest_cache
    if _manifest_cache is not None:
        return _manifest_cache
    lines = [_MANIFEST_HEADER, _MANIFEST_COLUMNS]
    for name, skill in TOOL_SKILLS.items():
        lines.append(
            f"- {name}｜{skill['category']}｜{skill['use_when']}"
            f"｜{skill['do_not_use_when']}"
        )
    _manifest_cache = "\n".join(lines)
    return _manifest_cache


def reset_manifest_cache():
    """仅供测试使用：清空渲染缓存。"""
    global _manifest_cache
    _manifest_cache = None


def validate_skill_coverage() -> list:
    """校验能力卡与工具注册表一致；返回问题列表（空表示通过）。

    延迟导入 agents.tools，避免模块级循环依赖。
    """
    from agents import tools as agent_tools

    problems = []
    registered = set(agent_tools.TOOL_NAMES)
    declared = set(TOOL_SKILLS)
    for missing in sorted(registered - declared):
        problems.append(f"缺少能力卡: {missing}")
    for orphan in sorted(declared - registered):
        problems.append(f"能力卡无对应工具: {orphan}")

    for name, skill in TOOL_SKILLS.items():
        if skill.get("category") not in VALID_CATEGORIES:
            problems.append(f"{name}: 非法 category {skill.get('category')!r}")
        if skill.get("produces") not in VALID_PRODUCES:
            problems.append(f"{name}: 非法 produces {skill.get('produces')!r}")
        for key in ("capability", "requires", "use_when", "do_not_use_when"):
            if not str(skill.get(key) or "").strip():
                problems.append(f"{name}: 缺少 {key}")
        for other in skill.get("confusable_with") or []:
            if other not in registered:
                problems.append(f"{name}: confusable_with 指向未知工具 {other}")
    return problems
