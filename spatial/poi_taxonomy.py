"""Controlled activity vocabulary used to filter, explain, and recommend POIs."""

from __future__ import annotations

import re
from typing import Iterable


ACTIVITY_SUBCATEGORIES = {
    # These are the existing master-data categories. The separate activity
    # field narrows the result without rewriting stable type/subcategory values.
    "basketball": ("field", "sports_field", "court"),
    "badminton": ("field", "sports_field", "court", "gym"),
    "table_tennis": ("court", "sports_centre", "gym"),
    "football": ("field", "sports_field"),
    "volleyball": ("field", "sports_field", "court", "gym"),
    "tennis": ("field", "sports_field", "court"),
    "running": ("field", "sports_field"),
    "fitness": ("gym", "sports_centre", "sports_other"),
    "swimming": ("pool",),
    "meal": ("canteen", "restaurant", "fastfood"),
    "canteen_meal": ("canteen",),
    "restaurant_meal": ("restaurant",),
    "fast_food": ("fastfood",),
    "coffee": ("coffee",),
    "tea_drink": ("tea_drink",),
    "study": ("library", "study_other"),
    "shopping": ("supermarket",),
    "medical": ("hospital",),
    "package_pickup": ("service",),
    "postal_service": ("post",),
    "view_lake": ("lake",),
    "view_sakura": ("sakura",),
}

ACTIVITY_LABELS = {
    "basketball": "篮球场", "badminton": "羽毛球场", "table_tennis": "乒乓球馆",
    "football": "足球场", "volleyball": "排球场", "tennis": "网球场",
    "running": "跑步场地", "fitness": "健身场馆", "swimming": "游泳场馆",
    "meal": "餐饮", "canteen_meal": "食堂", "restaurant_meal": "餐厅",
    "fast_food": "快餐小吃", "coffee": "咖啡", "tea_drink": "茶饮",
    "study": "学习场所", "shopping": "购物", "medical": "医疗服务",
    "package_pickup": "快递服务", "postal_service": "邮政服务",
    "view_lake": "湖景", "view_sakura": "赏樱",
}

_BALL_OPTIONS = ("篮球", "羽毛球", "乒乓球", "足球", "其他球类")
_SPORT_OPTIONS = ("跑步", "打球", "健身", "其他运动")


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def resolve_activity_query(query: str) -> dict | None:
    """Resolve an explicit use case, or ask only for a materially missing one."""
    text = _compact(query)
    if not text:
        return None

    # Named food venue types take precedence over generic meal verbs.
    if "食堂" in text:
        return {"status": "resolved", "activities": ["canteen_meal"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["canteen_meal"])}
    if any(term in text for term in ("餐厅", "饭店", "聚餐")):
        return {"status": "resolved", "activities": ["restaurant_meal"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["restaurant_meal"])}
    if any(term in text for term in ("快餐", "小吃", "炸鸡", "汉堡")):
        return {"status": "resolved", "activities": ["fast_food"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["fast_food"])}
    if any(term in text for term in ("咖啡", "coffee", "拿铁", "美式")):
        activities = ["coffee"]
        if any(term in text for term in ("奶茶", "茶饮", "果茶", "果汁", "饮料")):
            activities.append("tea_drink")
        return {"status": "resolved", "activities": activities,
                "subcategories": list(dict.fromkeys(
                    sub for activity in activities for sub in ACTIVITY_SUBCATEGORIES[activity]))}
    if any(term in text for term in ("奶茶", "茶饮", "果茶", "果汁", "喝饮料", "饮品")):
        return {"status": "resolved", "activities": ["tea_drink"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["tea_drink"])}
    if any(term in text for term in (
            "美食", "吃饭", "吃点", "吃什么", "吃东西", "觅食", "饿了",
            "午饭", "午餐", "晚饭", "晚餐", "早餐", "早饭", "夜宵", "餐饮", "好吃")):
        return {"status": "resolved", "activities": ["meal"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["meal"])}

    sport_rules = (
        ("basketball", ("打篮球", "篮球场", "篮球馆", "篮球")),
        ("badminton", ("打羽毛球", "羽毛球")),
        ("table_tennis", ("打乒乓球", "乒乓球", "打乒乓", "乒乓")),
        ("football", ("踢足球", "足球场", "踢球")),
        ("volleyball", ("打排球", "排球")),
        ("tennis", ("打网球", "网球场", "网球")),
        ("swimming", ("游泳", "游个泳")),
        ("running", ("跑步", "慢跑", "跑个步", "晨跑", "夜跑", "跑操")),
        ("fitness", ("健身", "力量训练", "练器械")),
    )
    for activity, terms in sport_rules:
        if any(term in text for term in terms):
            return {"status": "resolved", "activities": [activity],
                    "subcategories": list(ACTIVITY_SUBCATEGORIES[activity])}

    ball_request = any(term in text for term in ("打球", "球场", "球馆"))
    if ball_request:
        return {"status": "needs_clarification", "question": "想打什么球？",
                "options": list(_BALL_OPTIONS)}
    if any(term in text for term in ("运动", "锻炼")):
        return {"status": "needs_clarification", "question": "想做哪种运动？",
                "options": list(_SPORT_OPTIONS)}

    if any(term in text for term in (
            "图书馆", "自习", "看书", "复习", "备考", "写作业",
            "学习的地方", "找地方学习", "学习场所", "想学习", "安静的地方")):
        return {"status": "resolved", "activities": ["study"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["study"])}
    if any(term in text for term in (
            "超市", "便利店", "买东西", "买水", "买瓶水", "买饮料", "买瓶饮料",
            "购买饮用水", "零食", "文具", "购物")):
        return {"status": "resolved", "activities": ["shopping"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["shopping"])}
    if any(term in text for term in ("医院", "看病", "看医生", "买药")):
        return {"status": "resolved", "activities": ["medical"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["medical"])}
    if any(term in text for term in (
            "取快递", "寄快递", "取件", "快递站", "快递点", "快递柜", "快递服务")):
        return {"status": "resolved", "activities": ["package_pickup"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["package_pickup"])}
    if any(term in text for term in ("邮局", "邮政")):
        return {"status": "resolved", "activities": ["postal_service"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["postal_service"])}
    if any(term in text for term in ("湖景", "看湖", "湖边", "东湖边")):
        return {"status": "resolved", "activities": ["view_lake"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["view_lake"])}
    if any(term in text for term in ("赏樱", "樱花")):
        return {"status": "resolved", "activities": ["view_sakura"],
                "subcategories": list(ACTIVITY_SUBCATEGORIES["view_sakura"])}
    return None


def resolve_activity_sequence(query: str) -> list[dict]:
    """Keep explicit ordered place needs in requests such as ``打球再吃饭``."""
    parts = re.split(r"(?:然后|接着|随后|之后|最后|再去|再)", str(query or ""))
    resolved = []
    for part in parts:
        text = str(part or "").strip(" ，,。；;、")
        intent = resolve_activity_query(text)
        if intent:
            resolved.append({"query": text, "intent": intent})
        elif re.search(r"(?:宿舍|寝室)", text):
            # A return-to-dorm stage is a real destination requirement, but
            # the user's particular dorm is not inferable from the POI type.
            resolved.append({"query": text, "intent": {
                "status": "needs_destination", "activities": [], "subcategories": [],
            }})
    return resolved if len(resolved) > 1 else []


def activities_for_poi(poi: dict) -> set[str]:
    values = poi.get("activities") or []
    return {str(value) for value in values if str(value) in ACTIVITY_LABELS}


def matches_activities(poi: dict, activities: Iterable[str] | None) -> bool:
    requested = {str(value) for value in activities or [] if value}
    if not requested:
        return True
    supported = activities_for_poi(poi)
    # Generic meal intent includes all meal venues; specific intent must keep
    # canteens, restaurants and fast food distinct.
    if "meal" in requested:
        requested.update({"canteen_meal", "restaurant_meal", "fast_food"})
    return bool(supported & requested)


_SOURCE_REVIEWED_SPORTS = {
    "poi_077": ("swimming", "poi name identifies a swimming pool"),
    "poi_078": ("volleyball", "poi name identifies volleyball ground"),
    "poi_079": ("badminton", "poi name identifies badminton court"),
    "poi_080": ("running", "named athletic ground and aliases identify the running track"),
    "poi_081": ("fitness", "sports hall subcategory; no ball sport is claimed"),
    "poi_082": ("running", "poi name identifies an athletic ground"),
    "poi_083": ("basketball", "poi name and OSM reviewed sports feature identify basketball"),
    "poi_084": ("tennis", "poi name and OSM reviewed sports feature identify tennis"),
    "poi_085": ("football", "poi name and aliases identify the football ground"),
    "poi_086": (
        ["fitness", "badminton", "table_tennis"],
        "武汉大学学生资助管理中心《生活在珞珈丨第7期：运动场地大盘点——文理学部》"
        "（2024-04-24）记载风雨体育馆设羽毛球、乒乓球场地；场馆当前开放和预约情况未核实；"
        "https://zzgl.whu.edu.cn/info/1017/1028.htm",
    ),
    "poi_204": (
        ["fitness", "badminton", "table_tennis", "volleyball"],
        "武汉大学学生资助管理中心《生活在珞珈丨第8期：运动场地大盘点——工学部》"
        "（2024-04-24）记载工学部体育馆设羽毛球、乒乓球和排球场地；场馆当前开放和预约情况未核实；"
        "https://zzgl.whu.edu.cn/info/1017/1029.htm",
    ),
    "poi_205": ("running", "aliases identify the athletics ground"),
    "poi_206": ("basketball", "poi name identifies basketball court"),
    "poi_257": (
        ["fitness", "badminton"],
        "武汉大学学生资助管理中心《生活在珞珈丨第9期：运动场地大盘点——信息学部》"
        "（2024-04-25）明确记载星湖体育馆内设羽毛球场；实时开放和预约安排未核实；"
        "https://zzgl.whu.edu.cn/info/1017/1030.htm",
    ),
    "poi_258": (None, "sports field has no supported specific activity tag"),
    "poi_259": ("fitness", "sports hall subcategory; no ball sport is claimed"),
    "poi_260": (
        ["running", "football", "volleyball"],
        "武汉大学学生资助管理中心《生活在珞珈丨第9期：运动场地大盘点——信息学部》"
        "（2024-04-25）记载：信息学部田径场又称竹园操场/信操，场区包含足球场和排球场；"
        "https://zzgl.whu.edu.cn/info/1017/1030.htm",
    ),
    "poi_261": ("swimming", "poi name and OSM source identify swimming pool"),
    "poi_262": ("basketball", "poi name identifies basketball court"),
    "poi_418": ("fitness", "sports hall subcategory; no ball sport is claimed"),
    "poi_431": ("fitness", "sports hall subcategory; no ball sport is claimed"),
    "poi_432": ("volleyball", "OSM reviewed source names a volleyball court"),
    "poi_445": (None, "sports venue has no supported specific activity tag"),
    "poi_446": ("tennis", "OSM reviewed source names a tennis court"),
    "poi_447": ("badminton", "OSM reviewed source names a badminton court"),
    "poi_448": (None, "multi-purpose sports centre; no specific activity is verified"),
}


def audit_poi_activities(pois: list[dict]) -> list[dict]:
    """Return per-POI activity tags plus an auditable basis for every decision."""
    reviewed = []
    seen_ids = set()
    for poi in pois:
        poi_id = str(poi.get("id") or "")
        if not poi_id or poi_id in seen_ids:
            raise ValueError("every POI must have a unique nonempty id")
        seen_ids.add(poi_id)
        subcategory = str(poi.get("subcategory") or "")
        place_type = str(poi.get("type") or poi.get("category") or "")
        name_text = " ".join([str(poi.get("name") or ""), *map(str, poi.get("aliases") or [])])
        activities: list[str] = []
        basis = ""
        if place_type == "sports":
            source_ids = {str(source.get("id") or "") for source in poi.get("source_refs") or []}
            source_activity = {
                "node/2177869899": ("table_tennis", "OSM source names a table-tennis facility in 信息学部"),
                "node/10742578168": ("badminton", "OSM source names a badminton court in 信息学部"),
            }
            activity_entry = next((entry for source_id, entry in source_activity.items()
                                   if source_id in source_ids), None)
            if activity_entry is None:
                activity_entry = _SOURCE_REVIEWED_SPORTS.get(poi_id)
            if activity_entry is None:
                raise ValueError(f"sports POI has no reviewed activity disposition: {poi_id}")
            activity, basis = activity_entry
            if activity:
                activities = [activity] if isinstance(activity, str) else list(activity)
            else:
                basis = "reviewed: " + basis
        else:
            category_activity = {
                "canteen": ("canteen_meal", "category=canteen"),
                "restaurant": ("restaurant_meal", "category=restaurant"),
                "fastfood": ("fast_food", "category=fastfood"),
                "coffee": ("coffee", "category=coffee"),
                "tea_drink": ("tea_drink", "category=tea_drink"),
                "library": ("study", "category=library"),
                "supermarket": ("shopping", "category=supermarket"),
                "hospital": ("medical", "category=hospital"),
                "post": ("postal_service", "category=post"),
                "lake": ("view_lake", "category=lake"),
                "sakura": ("view_sakura", "category=sakura"),
            }.get(subcategory)
            if category_activity:
                activities, basis = [category_activity[0]], "category_supported: " + category_activity[1]
            elif re.search(r"快递|取件|菜鸟驿站", name_text):
                activities, basis = ["package_pickup"], "name_supported: parcel service name"
            else:
                basis = "reviewed: no specific activity supported by current POI evidence"
        reviewed.append({"poi_id": poi_id, "activities": activities, "activity_basis": basis,
                         "activity_review_status": (
                             "source_supported" if basis.startswith(("category_supported", "name_supported"))
                             or place_type == "sports" and activities else
                             "category_supported" if activities else
                             "reviewed_no_supported_activity")})
    missing_sports = set(_SOURCE_REVIEWED_SPORTS) - seen_ids
    if missing_sports:
        raise ValueError(f"reviewed sports POI IDs are missing from the master: {sorted(missing_sports)}")
    return reviewed
