"""Validate a model-proposed radius against numeric evidence, not intent keywords."""
from __future__ import annotations

import re

_DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千点半]+)"
_DISTANCE = re.compile(rf"(?<![\d.])({_NUMBER})\s*(公里|千米|km|米|m)(半)?(?![a-z])", re.I)
_DURATION = re.compile(rf"(?<![\d.])({_NUMBER})\s*(小时|钟头|分钟|分(?!米))(半)?", re.I)
_MULTIPLIER = re.compile(rf"(?<![\d.])({_NUMBER})\s*倍")
_EXPANSION = re.compile(r"扩(?:大|宽|展|张|增)|放(?:大|宽)|拓宽|加大|增大|变大|"
                        r"(?:更|再|稍)?(?:远|宽|大)(?:一点|一些|点|些)|"
                        r"(?:范围|半径).{0,5}(?:大|宽|广|远)")
_NEGATED = re.compile(r"(?:不要|不用|无需|不能|别|勿|不再|没必要|不必|不可|不)\s*(?:再|往|向|去)?\s*$")


def _number(value: str) -> float:
    if value == "半":
        return 0.5
    try:
        return float(value)
    except ValueError:
        pass
    if "点" in value:
        integer, fraction = value.split("点", 1)
        if not fraction or any(char not in _DIGITS for char in fraction):
            raise ValueError("invalid decimal")
        return _number(integer or "零") + float("0." + "".join(str(_DIGITS[char]) for char in fraction))
    if all(char in _DIGITS for char in value):
        return float("".join(str(_DIGITS[char]) for char in value))
    total = 0
    digit = None
    last_unit = 0
    zero_after_unit = False
    for char in value:
        if char in _DIGITS:
            digit = _DIGITS[char]
            if digit == 0 and last_unit:
                zero_after_unit = True
        elif char in "十百千":
            unit = {"十": 10, "百": 100, "千": 1000}[char]
            if last_unit and unit >= last_unit:
                raise ValueError("invalid number order")
            total += (1 if digit is None else digit) * unit
            digit, last_unit, zero_after_unit = None, unit, False
        else:
            raise ValueError("invalid number")
    if digit is not None:
        total += digit * (last_unit // 10 if last_unit >= 100 and not zero_after_unit else 1)
    return float(total)


def radius_supported_by_message(message: str, proposed_m: int, previous_m: int) -> bool:
    """Accept explicit distance, additive change or multiplier from the raw user text.

    The model still decides which condition the number describes; this function
    checks that a changed numeric radius has a possible user-supplied source.
    """
    for match in _DISTANCE.finditer(message or ""):
        try:
            value = _number(match.group(1)) + (0.5 if match.group(3) else 0)
        except ValueError:
            continue
        meters = value * (1000 if match.group(2).lower() in {"公里", "千米", "km"} else 1)
        if any(abs(proposed_m - candidate) < 0.001 for candidate in (meters, previous_m + meters, previous_m - meters)):
            return True
    for match in _MULTIPLIER.finditer(message or ""):
        try:
            if abs(proposed_m - previous_m * _number(match.group(1))) < 0.001:
                return True
        except ValueError:
            continue
    return False


def explicit_search_radius(message: str):
    """Return an unambiguous radius and whether a separate equal distance exists.

    A bare distance can describe a route, so require range/radius wording or
    a bounded nearby-place search. Relative changes remain model-led.
    """
    message = message or ""
    radii, other_distances = [], []
    for match in _DISTANCE.finditer(message):
        try:
            value = _number(match.group(1)) + (0.5 if match.group(3) else 0)
        except ValueError:
            continue
        meters = value * (1000 if match.group(2).lower() in {"公里", "千米", "km"} else 1)
        prefix = re.split(r"[，,、。；;！？!?]", message[:match.start()])[-1][-12:]
        suffix = re.split(r"[，,、。；;！？!?]", message[match.end():])[0][:6]
        marked = bool(re.search(r"(?:范围|半径).{0,8}$", prefix) or re.match(r"\s*(?:的)?范围", suffix)
                      or (re.search(r"找|搜索|检索|查询|附近|周边|周围|旁边", prefix)
                          and re.match(r"\s*(?:以?内|之内)", suffix)))
        route_distance = re.search(
            r"(?:路线(?:长度|距离)|(?:路程|行程)(?:长度|距离)?)(?:(?!搜索|查找|附近|周边|半径).)*$"
            r"|(?:步行|走路|骑行|骑车|行驶|开车)(?:(?:距离|路程|最多|最长|不超过|在|为|\s))*$", prefix)
        lower_bound = re.search(r"至少|不少于|不小于|大于|多于|(?<!不)超过", prefix)
        negated = re.search(r"不用|不要|不必|无需|取消|撤销|不再", prefix)
        relative_change = (re.search(r"扩大|放大|增加|增大|缩小|减少|再加|加大|(?<!增)加|减", prefix)
                           and not re.search(r"(?:到|至|为)\s*$", prefix))
        if marked and not route_distance and not lower_bound and not relative_change and not negated:
            radii.append(meters)
        else:
            other_distances.append(meters)
    unique = set(radii)
    if len(unique) != 1:
        return None, False
    radius = unique.pop()
    if not 100 <= radius <= 5000 or int(radius) != radius:
        return None, False
    return int(radius), radius in other_distances


def distance_mentions_m(message: str):
    """Extract explicit distances for checking delivered text against saved limits."""
    values = []
    for match in _DISTANCE.finditer(message or ""):
        try:
            value = _number(match.group(1)) + (0.5 if match.group(3) else 0)
        except ValueError:
            continue
        values.append(value * (1000 if match.group(2).lower() in {"公里", "千米", "km"} else 1))
    return values


def explicit_travel_time_limit(message: str):
    """Extract a clearly stated upper bound, excluding dwell and departure times."""
    message = message or ""
    limits = set()
    for match in _DURATION.finditer(message):
        prefix = re.split(r"[，,、。；;！？!?]", message[:match.start()])[-1][-16:]
        suffix = re.split(r"[，,、。；;！？!?]", message[match.end():])[0][:8]
        if re.search(r"停留|等待|休息|逛|游览|用餐|吃饭|排队|办完|完成|不用|不必|无需|取消|撤销|不再", prefix):
            continue
        limit_before = re.search(r"(?:不超过|不多于|最多|最长|至多|上限)(?:(?:走|步行|骑行|行驶|耗时|用时|为|到|在|\s))*$"
                                 r"|(?:用时|时长|时间).{0,6}(?:到|至)\s*$", prefix)
        limit_after = re.match(r"\s*(?:以?内|之内|可达|能到|到达)", suffix)
        relative = (re.search(r"增加|减少|再多|再少", prefix)
                    and not re.search(r"(?:到|至)\s*$", prefix))
        if not (limit_before or limit_after) or relative:
            continue
        try:
            value = _number(match.group(1)) + (0.5 if match.group(3) else 0)
        except ValueError:
            continue
        seconds = value * (3600 if match.group(2) in {"小时", "钟头"} else 60)
        if 0 < seconds <= 10800 and int(seconds) == seconds:
            limits.add(int(seconds))
    return next(iter(limits)) if len(limits) == 1 else None


def expansion_supported_by_message(message: str) -> bool:
    """Require an affirmative user expression before doubling a search radius."""
    message = message or ""
    for match in _EXPANSION.finditer(message):
        prefix = message[max(0, match.start() - 8):match.start()]
        if prefix.endswith("能不能") or prefix.endswith("可不可以"):
            return True
        if not _NEGATED.search(prefix):
            return True
    return False
