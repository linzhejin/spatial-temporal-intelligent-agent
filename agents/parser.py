import json
import logging
import re
from pathlib import Path
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, ValidationError

from config import DEEPSEEK_API_KEY, OPENAI_BASE_URL, LLM_MODEL
from agents.preferences import detect_strategy_hint, route_preference_requested

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

FEW_SHOT_EXAMPLES = [
    (
        "从牌坊到樱顶",
        '{"task_type":"path_planning","start":{"name":"牌坊","type":"poi"},"end":{"name":"樱顶","type":"poi"},"constraints":{"distance":"medium","slope":"normal","scenery":"normal"},"strategy_hint":"shortest","weights":null,"mode":"walk","input_method":"nl","ambiguity":null}',
    ),
    (
        "从牌坊骑车到樱顶",
        '{"task_type":"path_planning","start":{"name":"牌坊","type":"poi"},"end":{"name":"樱顶","type":"poi"},"constraints":{"distance":"medium","slope":"normal","scenery":"normal"},"strategy_hint":"shortest","weights":null,"mode":"bike","input_method":"nl","ambiguity":null}',
    ),
    (
        "开车从信息学部到文理学部教五",
        '{"task_type":"path_planning","start":{"name":"信息学部","type":"poi"},"end":{"name":"教五","type":"poi"},"constraints":{"distance":"medium","slope":"normal","scenery":"normal"},"strategy_hint":"shortest","weights":null,"mode":"drive","input_method":"nl","ambiguity":null}',
    ),
    (
        "从牌坊到樱顶，避开陡坡",
        '{"task_type":"path_planning","start":{"name":"牌坊","type":"poi"},"end":{"name":"樱顶","type":"poi"},"constraints":{"distance":"medium","slope":"avoid","scenery":"normal"},"strategy_hint":"flat","weights":null,"mode":"walk","input_method":"nl","ambiguity":null}',
    ),
    (
        "我第一次来武大，想看樱花和老图书馆，但膝盖不好",
        '{"task_type":"path_planning","start":null,"end":null,"constraints":{"distance":"medium","slope":"avoid","scenery":"high"},"strategy_hint":"flat","weights":null,"mode":"walk","input_method":"nl","ambiguity":"请指定起点"}',
    ),
    (
        "从梅园到桂园，最短路径",
        '{"task_type":"path_planning","start":{"name":"梅园","type":"poi"},"end":{"name":"桂园","type":"poi"},"constraints":{"distance":"short","slope":"normal","scenery":"normal"},"strategy_hint":"shortest","weights":null,"mode":"walk","input_method":"nl","ambiguity":null}',
    ),
    (
        "带朋友逛，从教五去图书馆，走风景好的路",
        '{"task_type":"path_planning","start":{"name":"教五","type":"poi"},"end":{"name":"图书馆","type":"poi"},"constraints":{"distance":"medium","slope":"normal","scenery":"high"},"strategy_hint":"scenery","weights":null,"mode":"walk","input_method":"nl","ambiguity":null}',
    ),
    (
        "樱顶在哪里",
        '{"task_type":"poi_query","start":{"name":"樱顶","type":"poi"},"end":null,"constraints":{"distance":"medium","slope":"normal","scenery":"normal"},"weights":null,"mode":"walk","input_method":"nl","ambiguity":null}',
    ),
    (
        "你能做什么",
        '{"task_type":"help","start":null,"end":null,"constraints":{"distance":"medium","slope":"normal","scenery":"normal"},"weights":null,"mode":"walk","input_method":"nl","ambiguity":null}',
    ),
    (
        "推荐几个景点",
        '{"task_type":"help","start":null,"end":null,"constraints":{"distance":"medium","slope":"normal","scenery":"normal"},"weights":null,"mode":"walk","input_method":"nl","ambiguity":null}',
    ),
    (
        "今天天气怎么样",
        '{"task_type":"chat","start":null,"end":null,"constraints":{"distance":"medium","slope":"normal","scenery":"normal"},"weights":null,"mode":"walk","input_method":"nl","ambiguity":null}',
    ),
]


class PoiRef(BaseModel):
    name: str
    type: Literal["poi", "coord"]
    coordinates: Optional[dict] = None


class Constraints(BaseModel):
    distance: Literal["short", "medium", "relaxed"]
    slope: Literal["avoid", "normal", "any"]
    scenery: Literal["high", "normal", "any"]


class TaskIntent(BaseModel):
    task_type: Literal["path_planning", "poi_query", "help", "unknown", "chat"]
    start: Optional[PoiRef] = None
    end: Optional[PoiRef] = None
    constraints: Constraints
    mode: Literal["walk", "bike", "drive"] = "walk"
    weights: Optional[dict] = None
    input_method: Literal["nl", "shortcut", "map_click"] = "nl"
    ambiguity: Optional[str] = None
    weight_source: Optional[Literal["explicit_nl", "shortcut", "default"]] = None
    strategy_hint: Optional[Literal["shortest", "recommended", "scenery", "flat", "custom"]] = None


# 出行方式关键词（与 spatial.routing.TRAVEL_MODES 对应）：
#   bike  骑行/骑车/自行车/单车/共享单车…
#   drive 自驾/驾车/开车/打车/坐车/乘车…（"校车""公交车"等不含精确词，不误判；"车"单字不判）
#   walk  步行/走路/散步/徒步/走着…
# 检测顺序：bike → drive → walk（bike 与 drive 关键词互不重叠）；都不命中返回 ("walk", False)
_BIKE_MODE_PATTERN = re.compile(r"骑行|骑车|骑自行车|骑单车|单车|自行车|共享单车|骑车去|骑个车")
_DRIVE_MODE_PATTERN = re.compile(r"自驾|驾车|开车|打车|坐车|乘车|开车去|驾个车|开车来")
_WALK_MODE_PATTERN = re.compile(r"步行|走路|散步|徒步|走着|走过去|走路去")


def detect_travel_mode(query: str) -> tuple[str, bool]:
    """从自然语言中检测出行方式，返回 (mode, explicit)。

    mode ∈ {"walk", "bike", "drive"}；explicit=True 表示用户显式提到了出行方式
    （关键词后处理纠偏优先级高于 LLM），False 表示未提及，调用方应保留默认/继承值。
    """
    if not query:
        return ("walk", False)
    if _BIKE_MODE_PATTERN.search(query):
        return ("bike", True)
    if _DRIVE_MODE_PATTERN.search(query):
        return ("drive", True)
    if _WALK_MODE_PATTERN.search(query):
        return ("walk", True)
    return ("walk", False)


_system_prompt_cache = None


def _load_system_prompt() -> str:
    """加载解析 prompt 模板（两阶段架构：LLM 只做原文名提取，不再注入 POI 列表；
    地名与 POI 库的匹配由后端 _normalize_poi_refs 完成，避免 320+ POI 撑爆 prompt）。"""
    global _system_prompt_cache
    if _system_prompt_cache is not None:
        return _system_prompt_cache
    prompt_path = PROMPTS_DIR / "parse_system.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        _system_prompt_cache = f.read()
    return _system_prompt_cache


def _build_messages(query: str, context: Optional[dict] = None, use_few_shot: bool = True) -> list[dict]:
    """组装 LLM 消息序列。

    多轮上下文（2026-09 重构）以真实 user/assistant 消息流注入，LLM 可据此做
    指代消解（"那里/那个食堂/第一条"）；规划槽位（start/end/constraints/
    weights/ambiguity）以结构化 JSON 单独注入，保证槽位承接不依赖模型自由发挥。
    兼容旧版 context.history=[{query: ...}] 形态。
    """
    system_prompt = _load_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    if use_few_shot:
        for shot_query, shot_output in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": shot_query})
            messages.append({"role": "assistant", "content": shot_output})

    if context:
        # 1) 历史对话消息流（最近若干轮，按角色还原）
        history = context.get("history") or []
        for turn in history:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role == "user" and content:
                messages.append({"role": "user", "content": str(content)})
            elif role == "assistant" and content:
                text = str(content)
                entities = turn.get("entities")
                if entities:
                    text += f"\n[本轮涉及地点/状态: {json.dumps(entities, ensure_ascii=False)}]"
                messages.append({"role": "assistant", "content": text})
            elif turn.get("query"):  # 旧版 history 形态兼容
                messages.append({"role": "user", "content": str(turn["query"])})

        # 2) 当前规划槽位（结构化、确定性，供"换偏好/补起终点"承接）
        slot = {}
        for key in ("start", "end", "constraints", "weights", "last_ambiguity"):
            if context.get(key) is not None:
                slot[key] = context[key]
        prev = context.get("previous_intent")
        if not slot and isinstance(prev, dict):
            for key in ("start", "end", "constraints", "weights", "ambiguity", "mode"):
                if prev.get(key) is not None:
                    slot[key] = prev[key]
        if slot:
            messages.append({
                "role": "user",
                "content": "当前规划状态（系统记录，仅供本轮承接与指代消解，不要复述）: "
                           + json.dumps(slot, ensure_ascii=False),
            })
            messages.append({
                "role": "assistant",
                "content": "好的，我已结合当前规划状态理解你的下一句需求。",
            })

    messages.append({"role": "user", "content": query})
    return messages


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("无法从响应中提取 JSON", text, 0)


def _fallback_task_intent(query: str) -> TaskIntent:
    return TaskIntent(
        task_type="path_planning",
        start=None,
        end=None,
        constraints=Constraints(distance="medium", slope="normal", scenery="normal"),
        mode="walk",
        weights=None,
        input_method="nl",
        ambiguity="解析失败，请尝试更明确的描述",
    )


def parse_query(query: str, context: Optional[dict] = None, use_few_shot: bool = True) -> TaskIntent:
    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY 未配置，使用规则兜底解析")
        return _t011_post_process(
            _annotate_weight_source(_fallback_task_intent(query)), query, context
        )

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai SDK 未安装，使用规则兜底解析")
        return _t011_post_process(
            _annotate_weight_source(_fallback_task_intent(query)), query, context
        )

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=OPENAI_BASE_URL,
        default_headers={"Accept-Encoding": "gzip, deflate"},
        # read 25s：DeepSeek 常见响应 10~25s，10s 会频繁超时导致"无返回"；
        # 2 次重试最坏 ~52s，仍在 gunicorn --timeout 60 之内。
        timeout=httpx.Timeout(connect=5.0, read=25.0, write=10.0, pool=5.0),
    )

    messages = _build_messages(query, context, use_few_shot=use_few_shot)

    last_error = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            raw_text = response.choices[0].message.content
            data = _extract_json(raw_text)
            return _t011_post_process(_annotate_weight_source(TaskIntent(**data)), query, context)
        except ValidationError as e:
            last_error = e
            logger.warning(f"Pydantic 校验失败 (attempt {attempt + 1}): {e}")
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"JSON 解析失败 (attempt {attempt + 1}): {e}")
        except Exception as e:
            last_error = e
            logger.warning(f"LLM 调用失败 (attempt {attempt + 1}): {type(e).__name__}: {e}")

        if attempt < 2:
            messages.append({
                "role": "user",
                "content": "上一次输出格式有误，请重新输出正确的 JSON 格式。",
            })

    logger.error(f"解析全部失败，使用兜底方案。最后错误: {last_error}")
    return _t011_post_process(_annotate_weight_source(_fallback_task_intent(query)), query, context)


def _annotate_weight_source(intent: TaskIntent) -> TaskIntent:
    """根据输入来源和 weights 状态，打标 weight_source 字段（DEC-013）。

    优先级规则（架构审核 N6）：
      - input_method=shortcut                  → shortcut     （快捷按钮预设）
      - input_method=nl 且 weights != null     → explicit_nl  （NL 明确表达偏好）
      - input_method=nl 且 weights == null     → default      （无偏好，用默认权重）
    """
    if intent.input_method == "shortcut":
        intent.weight_source = "shortcut"
    elif intent.input_method == "nl":
        if intent.weights is not None:
            intent.weight_source = "explicit_nl"
        else:
            intent.weight_source = "default"
    else:
        intent.weight_source = "default"
    return intent


# ====== T-011 append-only 扩展：验收 5~11 辅助函数 ======
# 不动原 parse_query 函数体的 LLM 调用逻辑，新增规则兜底 + 后处理

DEFAULT_CONSTRAINTS_DICT = {
    "distance": "medium",
    "slope": "normal",
    "scenery": "normal",
}

UNKNOWN_GUIDE_TEXT = (
    "抱歉，我只能回答武大校园内的路径规划和景点信息查询问题哦。"
    "可以告诉我你想从哪走到哪，或者问「樱顶在哪」查询景点介绍~"
)

EXTERNAL_GUIDE_TEXT = (
    "这个地点不在武汉大学校园内哦～我只熟悉武大文理学部、工学部、信息学部"
    "三个学部的道路和地点，换个校内目的地试试吧？比如「从珞珈门到樱花大道」😊"
)

HELP_GUIDE_AMBIGUITY = None

# 从 POI 库（data/pois.json，320+ 条）提取名称 + 别名，用于规则匹配
def _build_poi_name_index() -> tuple[set, dict]:
    """返回 (全部名称集合小写索引)。名称含主名与别名。"""
    try:
        from spatial.poi import load_pois
        pois = load_pois()
    except Exception:
        pois = []
    names_set = set()
    names_lower = {}
    for p in pois:
        name = p.get("name", "")
        if not name:
            continue
        names_set.add(name)
        names_lower.setdefault(name.lower(), name)
        for alias in p.get("aliases", []) or []:
            if alias:
                names_set.add(alias)
                names_lower.setdefault(alias.lower(), name)
    return names_set, names_lower


_POI_NAMES_SET, _POI_NAMES_LOWER = _build_poi_name_index()

# POI 查询关键词模式（T-011 验收 7）
_POI_QUERY_PATTERNS = [
    r"^(.+?)在(哪|哪里|哪儿|什么地方|哪块)",
    r"^(.+?)介?绍(一下)?$",
    r"^(.+?)是(什么|啥|咋样|怎么样)",
    r"^(.+?)的?(位置|地址|坐标|在哪)",
    r"^找(.+?)$",
    r"^查(一?下)?(.+?)$",
    r"^(.+?)怎么走$",   # "X怎么走" 无起点 → poi_query
    r"^(.+?)怎么去$",   # "X怎么去" 无起点 → poi_query
    r"^(.+?)(?:怎么样|如何|好玩吗|值得去吗)$",   # "X怎么样" 评价类 → poi_query
]

# Help 查询关键词（T-011 验收 8）
_HELP_PATTERNS = [
    r"你能做什么|你会什么|有什么功能|功能介绍|介?绍(一下)?你|怎么(用|玩|操作|使用)",
    r"推?荐(几个|一些|一下)?景点|有哪些(景点|好玩的|好看的|地方)|有什么(景点|好玩的|好看的|地方)",
    r"推荐|help|帮助|功能|怎么用|说明|guide",
    r"武大(有什么|有啥)(景点|好玩的|地方)",
    r"想去赏樱|想去赏花|想去赏景",   # 想赏樱/花/景 → help（推荐赏花地点）
    r"想看.*(?:樱花|最美的景|最美的花|最美的校园)(?!.*路线)",   # 想看风景 → help（除非含路线）
]

# 明显无关的闲聊关键词 —— 命中即 chat（交给 DeepSeek 对话模型闲聊，而非 unknown 报错）
# 注意: 不含 "你好" "谢谢" 等礼貌用语，它们常作为路径 query 的前后缀
_UNRELATED_KEYWORDS = [
    "天气", "下雨", "温度", "湿度", "刮风",
    "吃饭", "餐厅", "外卖", "食堂推荐", "吃什么", "好吃", "美食",
    "酒店", "住宿", "订房", "订酒店",
    "电影", "唱歌", "逛街", "购物",
    "笑话", "故事", "你是谁", "再见",
    "讲个", "写个", "翻译", "代码", "编程",
    "股票", "基金", "理财",
    "高考", "考研", "分数", "分数线", "招生",
]

# 路径语义关键词 — 当 query 同时命中 _UNRELATED_KEYWORDS 和这些词时，
# 说明用户可能在礼貌语或补充说明中提到了无关词，但主体仍是路径请求
_PATH_SEMANTIC_KEYWORDS = [
    "从", "到", "去", "走", "路线", "路径", "怎么走", "避开", "风景", "平坦",
    "最短", "陡坡", "逛", "出发", "规划",
]

# 路径提取时应忽略的非地名前缀（语气词/否定词/修饰词），
# 避免 "想去X" 被误判为 start="想"、"不对去X" 被误判为 start="不对"
_NON_PLACE_PREFIXES = {
    "想", "要", "不", "别", "就", "先", "再", "改", "换",
    "不对", "不是", "不想", "不要", "想要", "打算", "准备",
    "可以", "还能", "还是", "改成", "换成", "然后", "顺便",
}

# 校园闲聊关键词 — 命中则 task_type=chat（不是 unknown）
_CAMPUS_CHAT_KEYWORDS = [
    "樱花", "🌸", "开了吗", "花期", "枫叶", "银杏", "桂花", "梅花",
    "食堂", "好吃", "美食", "餐厅",
    "教学楼", "教室", "自习", "座位",
    "操场", "体育馆", "游泳", "篮球", "足球",
    "校车", "公交", "怎么坐车", "交通",
    "宿舍", "寝室", "住宿",
    "学长", "学姐", "新生", "入学", "报到",
    "社团", "活动", "讲座", "演出",
    "武大", "珞珈山", "校园", "学校",
    "上课", "下课", "课程", "选修",
    "附近", "周边", "购物", "超市",
    "几点", "时间", "开门", "关门", "开放",
    "人多吗", "拥挤", "排队",
    "拍照", "打卡",
    "天气", "热", "冷", "下雨",
    # 礼貌用语/寒暄 —— 走闲聊而非 unknown
    "你好", "您好", "嗨", "哈喽", "hello", "hi", "在吗",
    "谢谢", "感谢", "谢啦", "辛苦了", "早上好", "下午好", "晚上好",
]


def _is_explicit_poi_query(q: str) -> bool:
    """判断 query 是否是明确的 POI 查询模式（"X在哪/是什么/怎么走"等）。

    这些模式优先级高于闲聊检测：即使 X 含"樱花/珞珈山"等闲聊词，
    "樱花大道在哪里"、"珞珈山是什么" 也应判为 poi_query 而非 chat。
    """
    q = q.strip()
    if re.match(r"^(.+?)(?:怎么走|怎么去|去哪|在哪里|在哪儿|在哪|怎么样|如何|好玩吗|值得去吗)$", q):
        return True
    for pattern in _POI_QUERY_PATTERNS:
        if re.match(pattern, q):
            return True
    return False


def _rule_based_classify(query: str) -> dict:
    """规则优先兜底分类（T-011 验收 7/8/10）。

    当 LLM 未配置 / 调用失败 / 输出不稳定时，
    用关键词规则识别 task_type。返回 dict：
      {
        "task_type": "path_planning" | "poi_query" | "help" | "unknown",
        "start_name": str | None,
        "end_name": str | None,
      }
    """
    q = query.strip()
    q_lower = q.lower()

    # Step 0: 校外单位查询（华师/华科/武体等）→ unknown + 边界提示
    try:
        from spatial.poi import _is_external_query
        if _is_external_query(q):
            return {
                "task_type": "unknown",
                "start_name": None,
                "end_name": None,
                "external": True,
            }
    except Exception:
        pass

    # Step 0.5: 校园闲聊检测 — 含校园关键词但不含路径语义，且不是明确的 POI 查询
    has_path_semantics_0 = any(kw in q for kw in _PATH_SEMANTIC_KEYWORDS)
    if not has_path_semantics_0 and not _is_explicit_poi_query(q):
        for kw in _CAMPUS_CHAT_KEYWORDS:
            if kw in q:
                return {
                    "task_type": "chat",
                    "start_name": None,
                    "end_name": None,
                }

    # Step 1: 先匹配明确无关词（天气/股票/笑话等）
    # 命中走 chat（交给 DeepSeek 对话模型闲聊），而非 unknown 报错
    # 但如果 query 同时含路径语义关键词，跳过（礼貌用语 + 路径请求不误判）
    has_path_semantics = any(kw in q for kw in _PATH_SEMANTIC_KEYWORDS)
    if not has_path_semantics:
        for kw in _UNRELATED_KEYWORDS:
            if kw in q:
                return {
                    "task_type": "chat",
                    "start_name": None,
                    "end_name": None,
                }

    # Step 1.5: "X附近有什么" → poi_query（在 help 之前，因为更具体）
    m_nearby = re.match(r"^(.+?)附近有(?:什么|哪些)", q)
    if m_nearby:
        candidate = m_nearby.group(1).strip(" 的地得了吗啊呀你我他她它，。！？、")
        matched_poi = _fuzzy_match_poi_name(candidate)
        if matched_poi:
            return {
                "task_type": "poi_query",
                "start_name": matched_poi,
                "end_name": None,
            }

    # Step 2: 匹配 Help / 功能说明（T-011 验收 8）。含路径语义时继续向下
    # 提取起终点，避免“推荐路线”中的“推荐”抢先把规划请求判成帮助。
    if not has_path_semantics:
        for pattern in _HELP_PATTERNS:
            if re.search(pattern, q, flags=re.IGNORECASE):
                return {
                    "task_type": "help",
                    "start_name": None,
                    "end_name": None,
                }

    # Step 3: 匹配 "我在X" → 声明起点，path_planning
    at_pattern = re.match(r"我?在(.+?)(?:附近|周围)?$", q)
    if at_pattern:
        candidate = at_pattern.group(1).strip(" 的地得了吗啊呀你我他她它，。！？、")
        matched = _fuzzy_match_poi_name(candidate)
        if matched and len(candidate) <= 6:
            return {
                "task_type": "path_planning",
                "start_name": matched,
                "end_name": None,
            }

    # Step 3.5: "去X" 开头的目的地请求 → path_planning（缺起点，由上层引导/上下文承接）
    m_go = re.match(r"^(?:我(?:想|要|打算|准备)|想去|要去|打算|准备)?去(.+)$", q)
    if m_go:
        end_candidate = m_go.group(1).strip(" 的地得了吗啊呀你我他她它，。！？、那边")
        if end_candidate and not q.startswith("去哪"):
            matched_end = _fuzzy_match_poi_name(end_candidate)
            end_name = matched_end or (
                end_candidate if 2 <= len(end_candidate) <= 12
                and end_candidate not in _NON_PLACE_PREFIXES else None
            )
            if end_name:
                return {
                    "task_type": "path_planning",
                    "start_name": None,
                    "end_name": end_name,
                }

    # Step 4: 匹配 POI 纯查询（T-011 验收 7）—— 不包含 A→B 路径语义
    # "X怎么走/去哪" 无起点 → poi_query（只有孤立的目的地问路）
    m_x_how = re.match(r"^(.+?)(?:怎么走|怎么去|去哪|在哪里|在哪儿|在哪)$", q)
    if m_x_how:
        candidate = m_x_how.group(1).strip(" 的地得了吗啊呀你我他她它，。！？、")
        # 只有当候选不含路径结构（从/到）且能匹配到单个 POI 时才判为 poi_query
        has_path_structure = any(w in candidate for w in _PATH_SEMANTIC_KEYWORDS)
        if not has_path_structure:
            matched_poi = _fuzzy_match_poi_name(candidate)
            if matched_poi:
                return {
                    "task_type": "poi_query",
                    "start_name": matched_poi,
                    "end_name": None,
                }

    has_path_word = any(w in q for w in ["到", "去", "往", "走", "出发", "从", "路线", "路径"])
    if not has_path_word:
        for pattern in _POI_QUERY_PATTERNS:
            m = re.match(pattern, q)
            if m:
                candidate = m.group(1).strip(" 的地得了吗啊呀你我他她它")
                matched_poi = _fuzzy_match_poi_name(candidate)
                if matched_poi:
                    return {
                        "task_type": "poi_query",
                        "start_name": matched_poi,
                        "end_name": None,
                    }
        # 纯 POI 名（无动词）也视为 poi_query
        pure_name = _fuzzy_match_poi_name(q)
        if pure_name and len(q) <= 8:
            return {
                "task_type": "poi_query",
                "start_name": pure_name,
                "end_name": None,
            }

    # Step 5: 路径提取兜底 — "从X到Y"、"X去Y" 等模式
    path_patterns = [
        re.compile(r"从(.+?)(?:出发|走|去|到)(.+)(?:怎么走|路线|怎么去|的路线)?$"),
        re.compile(r"(.+?)去(.+)(?:怎么走|路线|怎么去)?$"),
        re.compile(r"(.+?)到(.+)(?:怎么走|路线|怎么去)?$"),
    ]
    for pat in path_patterns:
        m = pat.search(q)
        if m:
            start_candidate = m.group(1).strip(" 的地得了吗啊呀你我他她它，。！？、")
            end_candidate = m.group(2).strip(" 的地得了吗啊呀你我他她它，。！？、")
            start_match = _fuzzy_match_poi_name(start_candidate)
            end_match = _fuzzy_match_poi_name(end_candidate)
            if start_match and end_match and start_match != end_match:
                return {
                    "task_type": "path_planning",
                    "start_name": start_match,
                    "end_name": end_match,
                }
            elif start_match and not end_match:
                # 终点未识别但用户明确说了，保留原名（后续 get_poi 找不到时引导）；
                # 但语气词/否定词（如"想去X"的"想"）不保留为地名
                end_name = end_candidate if (
                    end_candidate and len(end_candidate) <= 12
                    and end_candidate not in _NON_PLACE_PREFIXES
                ) else None
                return {
                    "task_type": "path_planning",
                    "start_name": start_match,
                    "end_name": end_name,
                }
            elif end_match and not start_match:
                # 起点未识别但用户明确说了，保留原名（后续 get_poi 找不到时引导）；
                # 但语气词/否定词（如"想去X"的"想"）不保留为地名
                start_name = start_candidate if (
                    start_candidate and len(start_candidate) <= 12
                    and start_candidate not in _NON_PLACE_PREFIXES
                ) else None
                return {
                    "task_type": "path_planning",
                    "start_name": start_name,
                    "end_name": end_match,
                }
            break  # first matching pattern wins

    # Step 5: 其他 — 保留 path_planning 默认（让原 parse_query / LLM 继续处理）
    return {
        "task_type": None,  # None 表示不覆盖，走原逻辑
        "start_name": None,
        "end_name": None,
    }


def _fuzzy_match_poi_name(candidate: str) -> str | None:
    """按字符串包含关系 + 大小写近似匹配 POI 名（规则兜底用，确定性匹配）。

    匹配优先级 (降序):
      1. 精确名称匹配
      2. 精确别名匹配（通过 WHU_POIS 中的 name 字段）
      3. 候选是 name 的前缀 或 name 是候选的前缀
      4. 子串包含
    同优先级内按 name 字母序解耦，保证确定性。
    """
    if not candidate:
        return None
    cand = candidate.strip()
    # 过短/语气词前缀（"想""不对"等）不参与模糊匹配，避免单字子串误命中长名
    if len(cand) < 2 or cand in _NON_PLACE_PREFIXES:
        return None
    # 精确匹配名称（别名命中时也返回其所属规范名，保证两阶段归一化一致）
    # 注意：必须先于宽泛区域词保护——片区中心点 POI（如"梅园"区域质心）
    # 以精确同名入库，精确命中应直接返回；保护只针对子串/前缀扩散匹配。
    cand_low = cand.lower()
    if cand in _POI_NAMES_SET or cand_low in _POI_NAMES_LOWER:
        return _POI_NAMES_LOWER.get(cand_low, cand)
    # 宽泛区域词（桂园/信息学部等）若非精确区域点，不做子串/前缀匹配
    try:
        from spatial.poi import _BROAD_AREA_TERMS
        if cand in _BROAD_AREA_TERMS:
            return None
    except Exception:
        pass
    # 分级模糊匹配（确定性排序）
    prefix_matches = []
    substring_matches = []
    for name in sorted(_POI_NAMES_SET):  # 按字母序遍历保证确定性
        if name.startswith(cand) or cand.startswith(name):
            prefix_matches.append(name)
        elif cand in name or name in cand:
            substring_matches.append(name)
    # 前缀匹配优先，然后子串匹配；各集合内"更短名称"优先（更短 = 更具体的实体，
    # 如"工学部集贸市场"应命中市场本身而非市场内的小吃店），同长按字母序保证确定性
    if prefix_matches:
        prefix_matches.sort(key=lambda n: (len(n), n))
        return _POI_NAMES_LOWER.get(prefix_matches[0].lower(), prefix_matches[0])
    if substring_matches:
        substring_matches.sort(key=lambda n: (len(n), n))
        return _POI_NAMES_LOWER.get(substring_matches[0].lower(), substring_matches[0])
    return None


def _normalize_poi_refs(intent: TaskIntent) -> TaskIntent:
    """两阶段识别·第二阶段：把 LLM/规则提取的原始地名匹配到 POI 库规范名（DEC 两阶段架构）。

    - start/end 的 name 通过 spatial.poi.find_poi（别名 + 模糊）映射为规范名
    - 匹配失败时保留原文名（后续 route 层 get_poi 找不到会给用户引导，不在此拦截）
    - 仅处理 type="poi" 的引用，coord 类型不动
    """
    if intent.task_type not in ("path_planning", "poi_query"):
        return intent

    from spatial.poi import find_poi_ambiguous

    for ref in (intent.start, intent.end):
        if ref is None or ref.type != "poi" or not ref.name:
            continue
        # 同分歧义（如「三教」三个学部各有一个）时保留原名，交由对话层请用户消歧，不静默错配
        matched, _alternatives = find_poi_ambiguous(ref.name)
        if matched and matched.get("name"):
            ref.name = matched["name"]
    return intent


def _apply_priority_logic(intent: TaskIntent, weight_source_hint: str | None = None) -> TaskIntent:
    """显式打标优先级来源（T-011 验收 6 · 架构审核 N6）。

    优先级顺序：explicit_nl > shortcut > default
    - NL 明确写了偏好（weights != null） → explicit_nl
    - 快捷按钮 mode                     → shortcut
    - 其他 fallback / 无偏好              → default
    """
    if weight_source_hint:
        intent.weight_source = weight_source_hint
        return intent
    # 默认已经由 _annotate_weight_source 打标
    if intent.weight_source is None:
        intent = _annotate_weight_source(intent)
    return intent


# 多轮延续语标记：命中说明本轮是在上一轮基础上调整（换偏好/继续问），
# 才允许在未提取到新地点时沿用 context 的起终点。
_CONTINUATION_MARKERS = [
    "换", "改", "调", "避开", "缩短", "延长", "加", "减", "别走", "不走",
    "这条", "刚才", "继续", "重新", "还是", "一样", "同样", "原来的",
    "风景", "坡", "快", "慢", "远", "近", "起点", "终点",
]


def _merge_context_with_intent(intent: TaskIntent, context: dict | None, query: str) -> TaskIntent:
    """多轮上下文承接：本轮缺 start/end 时复用 context 中的起终点（T-011 验收 9）。

    context 结构示例（前端从 localStorage 的 whu_walker:context:{sid} 透传）：
      {
        "start": {"name": "牌坊", "type": "poi"},
        "end":   {"name": "樱顶",  "type": "poi"},
        "constraints": {...},
        "weights": {...},
        "previous_intent": {...TaskIntent snapshot...}
      }

    守卫（防止多轮污染，2026-09 反馈：前面对话的 POI 被静默当作本轮起终点）：
    必须满足其一才承接：
      a) 本轮提取到至少一个新地点（用户明确在说路径的一部分）
      b) 本轮含多轮延续语（换/改/避开/这条/刚才…，说明在调整上一轮）
    否则（如"皇冠幸福里怎么样"这类新话题）不承接，交由上层引导或闲聊。
    """
    if not context:
        return intent
    if intent.task_type not in ("path_planning", "poi_query"):
        return intent

    has_new_place = intent.start is not None or intent.end is not None
    has_continuation = any(m in query for m in _CONTINUATION_MARKERS)
    if not has_new_place and not has_continuation:
        return intent

    # 缺 start → 补 context.start
    if intent.start is None and isinstance(context.get("start"), dict):
        ctx_start = context["start"]
        if ctx_start.get("name"):
            intent.start = PoiRef(
                name=ctx_start["name"],
                type=ctx_start.get("type", "poi"),
                coordinates=ctx_start.get("coordinates"),
            )
    # 缺 end → 补 context.end（poi_query 不需要 end）
    if intent.task_type == "path_planning" and intent.end is None and isinstance(context.get("end"), dict):
        ctx_end = context["end"]
        if ctx_end.get("name"):
            intent.end = PoiRef(
                name=ctx_end["name"],
                type=ctx_end.get("type", "poi"),
                coordinates=ctx_end.get("coordinates"),
            )
    # 出行方式承接：本轮未显式提到出行方式（无 bike/drive/walk 关键词）时，
    # 沿用上一轮 previous_intent 的 mode（与起终点承接共用同一守卫，防止跨话题污染）
    _nl_mode, mode_explicit = detect_travel_mode(query)
    if not mode_explicit:
        prev_intent = context.get("previous_intent") or {}
        prev_mode = prev_intent.get("mode")
        if prev_mode in ("walk", "bike", "drive"):
            intent.mode = prev_mode

    # 清除已补全字段的 ambiguity 提示
    if intent.start is not None and intent.end is not None and intent.ambiguity in (
        "请指定起点", "请指定终点", "请指定起点和终点"
    ):
        intent.ambiguity = None
    return intent


def _resolve_ambiguity_completion(intent: TaskIntent, query: str, context: dict | None) -> TaskIntent:
    """ambiguity 补全：上轮返回 ambiguity 提示，本轮用户单 POI 名视为补字段（T-011 验收 11）。

    场景：
      - 上轮 ambiguity="请指定起点"，本轮用户回复"牌坊" → start=牌坊，保留 context 中的 end/constraints/weights
      - 上轮 ambiguity="请指定终点"，本轮用户回复"樱顶" → end=樱顶，保留 context 中的 start/constraints/weights
    """
    if not context:
        return intent

    prev = context.get("previous_intent") or {}
    prev_ambi = prev.get("ambiguity") or context.get("last_ambiguity") or ""
    if not prev_ambi:
        return intent

    # 本轮 query 解析出的纯 POI 名
    matched_poi = _fuzzy_match_poi_name(query.strip())
    if not matched_poi:
        return intent

    prev_start = prev.get("start")
    prev_end = prev.get("end")
    prev_constraints = prev.get("constraints") or DEFAULT_CONSTRAINTS_DICT
    prev_weights = prev.get("weights")

    merged = False
    if "请指定起点" in prev_ambi and prev_end is not None:
        # 上轮缺起点，本轮补的是起点
        intent.start = PoiRef(name=matched_poi, type="poi")
        intent.end = PoiRef(
            name=prev_end["name"],
            type=prev_end.get("type", "poi"),
            coordinates=prev_end.get("coordinates"),
        )
        merged = True
    elif "请指定终点" in prev_ambi and prev_start is not None:
        # 上轮缺终点，本轮补的是终点
        intent.start = PoiRef(
            name=prev_start["name"],
            type=prev_start.get("type", "poi"),
            coordinates=prev_start.get("coordinates"),
        )
        intent.end = PoiRef(name=matched_poi, type="poi")
        merged = True

    if merged:
        # 保留上轮 constraints / weights，防止 LLM 重新识别为空
        try:
            intent.constraints = Constraints(**prev_constraints)
        except (ValidationError, TypeError):
            pass
        if prev_weights is not None:
            intent.weights = prev_weights
        intent.ambiguity = None
        intent.task_type = "path_planning"

    return intent


def _t011_post_process(
    intent: TaskIntent,
    query: str,
    context: dict | None,
    shortcut_mode_hint: bool = False,
) -> TaskIntent:
    """T-011 统一后处理入口（append-only 在 parse_query 返回前调用，不修改原 LLM 流程）。

    处理顺序（符合优先级规则）：
      1. 规则兜底分类（验收 7/8/10）—— LLM 误判时强覆盖 task_type
      2. ambiguity 补全（验收 11）—— 用户单 POI 名补缺字段
      3. 多轮上下文承接（验收 9）—— 缺 start/end 从 context 复用
      4. 优先级打标（验收 6）—— weight_source 字段
      5. unknown 兜底文案（验收 10）—— task_type=unknown 时填充引导语
    """
    # 1. 规则分类优先覆盖（LLM 没判对 / fallback 默认 intent 时用规则纠正）
    classified = _rule_based_classify(query)
    if classified["task_type"] == "poi_query":
        intent.task_type = "poi_query"
        intent.start = PoiRef(name=classified["start_name"], type="poi")
        intent.end = None
        intent.ambiguity = None
    elif classified["task_type"] == "help":
        intent.task_type = "help"
        intent.start = None
        intent.end = None
        intent.ambiguity = None
    elif classified["task_type"] == "unknown":
        intent.task_type = "unknown"
        intent.start = None
        intent.end = None
        intent.ambiguity = EXTERNAL_GUIDE_TEXT if classified.get("external") else UNKNOWN_GUIDE_TEXT
    elif classified["task_type"] == "chat":
        intent.task_type = "chat"
        intent.start = None
        intent.end = None
        intent.ambiguity = None
    elif classified["task_type"] == "path_planning":
        # 正则兜底：从 query 中提取到起终点，覆盖 LLM fallback 的空值；
        # 同时纠正 LLM 误判的 task_type（如 "我在珞珈门" 被 LLM 判成 poi_query）
        intent.task_type = "path_planning"
        if classified["start_name"]:
            intent.start = PoiRef(name=classified["start_name"], type="poi")
        if classified["end_name"]:
            intent.end = PoiRef(name=classified["end_name"], type="poi")
        # 两个都补全了 → 清除 ambiguity
        if intent.start is not None and intent.end is not None and intent.ambiguity:
            intent.ambiguity = None

    # 1.5 两阶段·地名归一化（LLM/规则输出的原文名 → POI 库规范名）
    intent = _normalize_poi_refs(intent)

    # 2. ambiguity 补全（T-011 验收 11）—— 在 context 合并之前判断是否命中补全语义
    intent = _resolve_ambiguity_completion(intent, query, context)

    # 3. 多轮 context 承接（T-011 验收 9）—— 缺 start/end 时用 context 补齐
    intent = _merge_context_with_intent(intent, context, query)

    # 3.5 出行方式关键词纠偏：NL 显式提到骑行/驾车/步行时强制覆盖（优先级高于 LLM
    #     与上下文继承，与项目"关键词后处理纠偏"约定一致）；未提及时保留 LLM/继承值
    kw_mode, mode_explicit = detect_travel_mode(query)
    if mode_explicit:
        intent.mode = kw_mode

    # 多轮只改出行方式时，保留上一轮明确的避坡/景观硬约束。
    if context and route_preference_requested(query, context):
        prev = context.get("previous_intent") or {}
        prev_constraints = prev.get("constraints") or context.get("constraints") or {}
        if isinstance(prev_constraints, dict):
            if prev_constraints.get("slope") in ("avoid", "prefer") and intent.constraints.slope == "normal":
                intent.constraints.slope = prev_constraints["slope"]
            if prev_constraints.get("scenery") == "high" and intent.constraints.scenery == "normal":
                intent.constraints.scenery = "high"
        if intent.weights is None and prev.get("weights"):
            intent.weights = prev["weights"]

    # 解析层只输出策略证据。固定策略的数值由 routing_policy 统一负责；仅当
    # 用户明确给出三维比例时保留 weights 作为 custom 策略输入。
    if intent.task_type == "path_planning":
        intent.strategy_hint = detect_strategy_hint(query, context)
        if intent.strategy_hint != "custom":
            intent.weights = None

    # 4. 优先级打标（T-011 验收 6）
    if shortcut_mode_hint or intent.input_method == "shortcut":
        intent = _apply_priority_logic(intent, weight_source_hint="shortcut")
    elif intent.strategy_hint in ("recommended", "scenery", "flat", "custom"):
        intent = _apply_priority_logic(intent, weight_source_hint="explicit_nl")
    else:
        intent = _apply_priority_logic(intent, weight_source_hint="default")

    # 5. unknown 兜底：如果 task_type=unknown 且 ambiguity 空，补引导语
    if intent.task_type == "unknown" and not intent.ambiguity:
        intent.ambiguity = UNKNOWN_GUIDE_TEXT

    # 6. help / unknown / chat 类型的 constraints 保证合法（T-011 验收 12）
    if intent.task_type in ("help", "unknown", "chat"):
        try:
            intent.constraints = Constraints(**DEFAULT_CONSTRAINTS_DICT)
        except (ValidationError, TypeError):
            pass

    # 7. path_planning 的 ambiguity 规范化：
    #    规则兜底/上下文承接已补全起终点 → 清除；仍缺某端 → 给出对应引导，
    #    避免残留"解析失败"文案（与实际状态不符，且会污染下一轮补全判断）
    if intent.task_type == "path_planning":
        if intent.start is not None and intent.end is not None:
            intent.ambiguity = None
        elif intent.ambiguity in ("解析失败，请尝试更明确的描述", None, ""):
            if intent.start is None and intent.end is None:
                intent.ambiguity = "解析失败，请尝试更明确的描述"
            elif intent.start is None:
                intent.ambiguity = "请指定起点"
            else:
                intent.ambiguity = "请指定终点"

    return intent
