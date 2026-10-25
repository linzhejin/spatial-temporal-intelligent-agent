"""决定是否允许偏好权重，避免模型或历史画像替普通通勤添加游览偏好。"""

import re


_PREFERENCE = re.compile(
    r"风景|景观|看景|赏景|赏樱|赏花|看花|看樱花|观景|拍照|最美|漂亮|"
    r"游览|旅游|游客|逛|散步|走走|沿湖|沿着湖|树荫|阴凉|少晒|遮阳|"
    r"爬坡|陡坡|坡道|平坦|平路|台阶|楼梯|省力|轻松|费力|膝盖|轮椅|无障碍|"
    r"锻炼|跑步|爬山|多走|远一点|景色好|不想上坡|不要上坡"
)
_DISTANCE_ONLY = re.compile(r"最短|赶时间|赶课|快一点|快点|尽快|直接到|不绕路|别绕路|默认路线")
_FOLLOWUP = re.compile(r"^(还是|换|改|那|继续|然后|从那里|从这|骑车|骑行|开车|驾车|步行|走路)")
_NEGATED_SCENERY = re.compile(r"(?:不看|不要看|不用看|不考虑|无所谓)(?:风景|景观|景色|景)")
_LEISURE = re.compile(r"游览|逛|散步|赏樱|赏花|拍照|打卡|带朋友|推荐.{0,6}路线|游玩|参观")
_SCENERY = re.compile(r"风景|景观|临湖|林荫|赏樱|赏花|拍照|打卡")
_FLAT = re.compile(r"平坦|平路|少爬坡|少走坡|不爬坡|避开陡坡|省力|无障碍")
_CUSTOM_WEIGHTS = re.compile(
    r"(?:距离.{0,8}坡度.{0,8}风景|distance.{0,8}slope.{0,8}scenery)"
    r".{0,12}(?:按|权重|比例)?\s*\d+(?:\.\d+)?\s*[:：/]\s*"
    r"\d+(?:\.\d+)?\s*[:：/]\s*\d+(?:\.\d+)?",
    re.IGNORECASE,
)


def route_preference_requested(query: str, context: dict = None, history: list = None) -> bool:
    """只对显式偏好及其后续修改开放权重；新 A→B 行程不继承游览偏好。"""
    text = _NEGATED_SCENERY.sub("", query or "").strip()
    if re.search(r"从.{1,20}(?:到|去).{1,20}", text):
        # 新的完整 A→B 行程不继承上一轮的游览偏好。
        explicit = _PREFERENCE.search(text)
        return bool(explicit)
    if _PREFERENCE.search(text):
        return True
    if _DISTANCE_ONLY.search(text):
        return False
    context = context if isinstance(context, dict) else {}
    previous = context.get("previous_intent") or {}
    if not isinstance(previous, dict):
        previous = {}
    turns = history or context.get("history") or []
    clarification = any(isinstance(t, dict) and t.get("role") == "assistant"
                        and any(x in str(t.get("content", "")) for x in ("起点", "从哪里", "出发"))
                        for t in turns[-2:])
    if not _FOLLOWUP.search(text) and not previous.get("ambiguity") and not clarification:
        return False
    for turn in reversed(turns):
        if not isinstance(turn, dict):
            continue
        prior = turn.get("content") if turn.get("role") == "user" else turn.get("query")
        if not prior or prior == query:
            continue
        # 递归只传更早的历史，让连续修改也能追溯到原始偏好。
        return route_preference_requested(str(prior), history=turns[:turns.index(turn)])
    constraints = previous.get("constraints") or context.get("constraints") or {}
    return constraints.get("slope") in ("avoid", "prefer") or constraints.get("scenery") == "high"


def detect_strategy_hint(query: str, context: dict = None) -> str:
    """将自然语言证据映射为策略名称，不在解析层生成数值权重。

    出行方式单独切换时沿用上一轮策略；新的普通 A→B 请求始终回到最短路。
    """
    text = _NEGATED_SCENERY.sub("", query or "").strip()
    if _CUSTOM_WEIGHTS.search(text):
        return "custom"
    if _DISTANCE_ONLY.search(text):
        return "shortest"
    if _FLAT.search(text):
        return "flat"
    if _SCENERY.search(text):
        return "scenery"
    if _LEISURE.search(text):
        return "recommended"

    context = context if isinstance(context, dict) else {}
    previous = context.get("previous_intent") or {}
    if not isinstance(previous, dict):
        previous = {}
    previous_hint = previous.get("strategy_hint") or context.get("strategy_hint")
    if previous_hint in {"shortest", "recommended", "scenery", "flat", "custom"}:
        # 只有延续语才继承，避免一段新行程被旧偏好污染。
        is_new_trip = bool(re.search(r"从.{1,20}(?:到|去).{1,20}", text))
        if not is_new_trip and _FOLLOWUP.search(text):
            return previous_hint
    return "shortest"
