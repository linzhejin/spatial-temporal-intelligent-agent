"""
POI 加载与匹配模块

从 data/pois.json 加载 POI 数据，支持模糊搜索和按类型过滤。
当 JSON 文件不可用时，回退到 config.WHU_POIS。
"""

import json
import math
import os
import re
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Optional

_POIS_CACHE: list = []
_LOADED = False
_PENDING_POI_QUERY_NAMES: set | None = None
_SOURCE_BACKED_POI_IDS: set | None = None
_SOURCE_BACKED_POI_REFS: dict[str, list] | None = None

# 搜索频率统计（用于途经点重要度排序）：{poi_id: count}
# 冷启动时无真实数据，重要度靠景观分 + 类型加权兜底
_SEARCH_STATS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "poi_search_stats.json"
)
_search_stats: dict = {}
_stats_dirty = False


def _load_search_stats() -> dict:
    global _search_stats
    if _search_stats:
        return _search_stats
    try:
        if os.path.exists(_SEARCH_STATS_PATH):
            with open(_SEARCH_STATS_PATH, "r", encoding="utf-8") as f:
                _search_stats = json.load(f)
    except (json.JSONDecodeError, IOError):
        _search_stats = {}
    return _search_stats


def _record_search(poi_id: str):
    """命中 POI 时累加搜索次数，服务重启不丢失（落盘到 poi_search_stats.json）。"""
    global _stats_dirty
    if not poi_id:
        return
    stats = _load_search_stats()
    stats[poi_id] = stats.get(poi_id, 0) + 1
    _stats_dirty = True
    # 直接落盘，简单可靠；多 worker 偶发竞争不影响精度
    try:
        with open(_SEARCH_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)
        _stats_dirty = False
    except IOError:
        pass


# 类型加权：景观/校门 > 教学楼/食堂/体育 > 服务 > 宿舍
_TYPE_BONUS = {
    "scenery": 5.0,
    "gate": 4.0,
    "study": 2.0,
    "dining": 2.0,
    "sports": 2.0,
    "service": 1.0,
    "dorm": 0.0,
}


def importance_score(poi: dict) -> float:
    """POI 重要度 = 搜索次数×1 + 景观分×2 + 类型加权。

    用于路线途经点筛选：重要的 POI（樱花大道/樱顶/牌坊）优先显示，
    普通宿舍即使离路线近也不抢镜。
    """
    poi_id = poi.get("id", "")
    count = _load_search_stats().get(poi_id, 0)
    scenery = float(poi.get("scenery_score", 3))
    type_bonus = _TYPE_BONUS.get(poi.get("type", ""), 1.0)
    return count * 1.0 + scenery * 2.0 + type_bonus


def _pois_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "data", "pois.json")


def _load_from_json() -> list:
    path = _pois_file_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        defaults = data.get("provenance_defaults", {})
        pois = data.get("pois", [])
        for poi in pois:
            for key, value in defaults.items():
                poi.setdefault(key, deepcopy(value))
        return pois
    except (json.JSONDecodeError, IOError):
        return []


def _load_from_config() -> list:
    try:
        from config import WHU_POIS
    except ImportError:
        return []

    pois = []
    for poi_id, info in WHU_POIS.items():
        pois.append({
            "id": poi_id,
            "name": info.get("name", poi_id),
            "aliases": [poi_id],
            "coordinates": {"lng": info["lon"], "lat": info["lat"]},
            "type": info.get("type", "landmark"),
            "description": info.get("desc", ""),
            "scenery_score": info.get("scenery_score", 3),
            "source_refs": [],
            "verification_status": "legacy_unverified",
        })
    return pois


def load_pois() -> list:
    """
    加载 POI 数据（带缓存，仅首次调用读文件）。

    优先从 data/pois.json 加载，失败则回退到 config.WHU_POIS。

    Returns:
        POI 对象列表，每个 POI 包含 id, name, aliases, coordinates, type, description 等字段
    """
    global _POIS_CACHE, _LOADED
    if _LOADED:
        return _POIS_CACHE

    pois = _load_from_json()
    if not pois:
        pois = _load_from_config()
    # Preserve reports in the source file, but do not recommend or route to
    # a location the user reports absent until the record has been reviewed.
    pois = [poi for poi in pois if poi.get("recommendation_status") != "reported_absent_pending_review"]
    _POIS_CACHE = pois
    _LOADED = True
    return pois


def reload_pois() -> list:
    """强制重新加载 POI 数据（忽略缓存）。"""
    global _LOADED, _PENDING_POI_QUERY_NAMES, _SOURCE_BACKED_POI_IDS, _SOURCE_BACKED_POI_REFS
    _LOADED = False
    _PENDING_POI_QUERY_NAMES = None
    _SOURCE_BACKED_POI_IDS = None
    _SOURCE_BACKED_POI_REFS = None
    return load_pois()


def _pending_osm_poi_names() -> set:
    """Names of OSM POIs explicitly held for review, not safe to fuzzy-route to."""
    global _PENDING_POI_QUERY_NAMES
    if _PENDING_POI_QUERY_NAMES is not None:
        return _PENDING_POI_QUERY_NAMES
    names = set()
    # Guard exact names and aliases of reported-absent entries against fuzzy
    # replacement by an unrelated venue after they are removed from search.
    for poi in _load_from_json():
        if poi.get("recommendation_status") == "reported_absent_pending_review":
            for label in [poi.get("name"), *(poi.get("aliases") or [])]:
                normalized = re.sub(r"\s+", "", str(label or "")).casefold()
                if normalized:
                    names.add(normalized)
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    try:
        with open(os.path.join(data_dir, "campus_review_decisions.json"),
                  "r", encoding="utf-8") as handle:
            review = json.load(handle)
        with open(os.path.join(data_dir, "pois_osm_candidates.json"),
                  "r", encoding="utf-8") as handle:
            candidate_doc = json.load(handle)
        pending_ids = {
            row.get("source_id") for row in review.get("poi_decisions", [])
            if row.get("source") == "OpenStreetMap"
            and row.get("decision") == "pending_review"
        }
        for candidate in candidate_doc.get("candidates", []):
            if candidate.get("osm_id") not in pending_ids:
                continue
            candidate_name = re.sub(r"\s+", "", str(candidate.get("name") or "")).casefold()
            if candidate_name:
                names.add(candidate_name)
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        pass  # Keep user-reported names even when the separate OSM review is unavailable.
    _PENDING_POI_QUERY_NAMES = names
    return names


def _normalize_poi_query(name: str) -> str:
    return re.sub(r"\s+", "", str(name or "")).casefold()


def has_source_backed_record(poi: dict) -> bool:
    """Whether a POI has a record-specific source in the local/open catalog.

    Historical entries use a file-level provenance default. They do not pass
    this check unless a source is attached to the POI or an OSM source record
    was explicitly matched to its stable POI ID.
    """
    if not isinstance(poi, dict) or poi.get("recommendation_status") == "reported_absent_pending_review":
        return False
    status = str(poi.get("verification_status") or "legacy_unverified")
    if poi.get("source_refs") and status in {"source_only", "institution_verified", "field_verified"}:
        return True
    poi_id = str(poi.get("id") or poi.get("poi_id") or "")
    if not poi_id:
        return False
    global _SOURCE_BACKED_POI_IDS, _SOURCE_BACKED_POI_REFS
    if _SOURCE_BACKED_POI_IDS is None:
        _SOURCE_BACKED_POI_IDS = set()
        _SOURCE_BACKED_POI_REFS = {}
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        try:
            with open(os.path.join(data_dir, "pois_osm_candidates.json"),
                      "r", encoding="utf-8") as handle:
                candidates = json.load(handle).get("candidates", [])
            with open(os.path.join(data_dir, "campus_review_decisions.json"),
                      "r", encoding="utf-8") as handle:
                review_rows = json.load(handle).get("poi_decisions", [])
            decisions = {
                row.get("source_id"): row.get("decision") for row in review_rows
                if row.get("source") == "OpenStreetMap" and row.get("source_id")
            }
            accepted_decisions = {
                "existing_name_match", "included", "alias_or_source_enrichment", "campus_corrected",
            }
            for candidate in candidates:
                refs = candidate.get("source_refs") or []
                decision = decisions.get(candidate.get("osm_id"))
                if (candidate.get("verification_status") == "source_only" and refs
                        and decision in accepted_decisions):
                    for value in candidate.get("matched_poi_ids") or []:
                        if value:
                            key = str(value)
                            _SOURCE_BACKED_POI_IDS.add(key)
                            _SOURCE_BACKED_POI_REFS.setdefault(key, []).extend(ref for ref in refs
                                                                              if ref not in _SOURCE_BACKED_POI_REFS.get(key, []))
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            pass
    return poi_id in _SOURCE_BACKED_POI_IDS


def source_refs_for_poi(poi: dict) -> list:
    refs = list(poi.get("source_refs") or []) if isinstance(poi, dict) else []
    if refs or not has_source_backed_record(poi):
        return refs
    poi_id = str(poi.get("id") or poi.get("poi_id") or "")
    return list((_SOURCE_BACKED_POI_REFS or {}).get(poi_id, []))


def filter_reported_absent_provider_results(candidates: list[dict], *, max_distance_m: float = 100.0) -> list[dict]:
    """Do not reintroduce an individually reported missing branch from a live provider.

    A match requires a name/alias identity and a nearby recorded location.
    A provider's generic campus suffix is compatible with a more specific
    reported branch at that location; explicitly different branches stay separate.
    """
    held = [row for row in _load_from_json()
            if row.get("recommendation_status") == "reported_absent_pending_review"]
    if not held:
        return list(candidates or [])

    def identity(row):
        def normalize(value):
            return _normalize_num(re.sub(r"[\s\u3000\-—_()（）·]", "", value)).casefold()

        labels, businesses, campus_businesses = set(), set(), set()
        campus_suffixes = ("武汉大学校内店", "武汉大学校园店", "武汉大学店",
                           "武大校内店", "武大校园店", "武大店")
        for value in [row.get("name"), *(row.get("aliases") or [])]:
            text = str(value or "").strip()
            if not text:
                continue
            normalized = normalize(text)
            labels.add(normalized)
            named_branch = re.fullmatch(r"(.+?)[(（][^()（）]+[)）]", text)
            businesses.add(normalize(named_branch.group(1)) if named_branch else normalized)
            for suffix in campus_suffixes:
                if normalized.endswith(suffix) and len(normalized) > len(suffix):
                    business = normalized[:-len(suffix)]
                    campus_businesses.add(business)
                    businesses.add(business)
                    break
        return labels, businesses, campus_businesses

    def point(row):
        coord = row.get("coordinates") or {}
        try:
            lng = float(coord.get("lng", row.get("lon", row.get("lng"))))
            lat = float(coord.get("lat", row.get("lat")))
            if math.isfinite(lng) and math.isfinite(lat):
                return lng, lat
        except (TypeError, ValueError):
            pass
        return None

    held_rows = [(identity(row), point(row)) for row in held]
    kept = []
    for candidate in candidates or []:
        candidate_labels, _, candidate_campus = identity(candidate)
        candidate_point = point(candidate)
        excluded = False
        if candidate_point:
            lng, lat = candidate_point
            for (held_labels, held_businesses, _), held_point in held_rows:
                same_identity = (candidate_labels.intersection(held_labels)
                                 or candidate_campus.intersection(held_businesses))
                if not same_identity or not held_point:
                    continue
                old_lng, old_lat = held_point
                mean_lat = math.radians((lat + old_lat) / 2)
                distance = math.hypot((lng - old_lng) * 111320 * math.cos(mean_lat),
                                      (lat - old_lat) * 110540)
                if distance <= max_distance_m:
                    excluded = True
                    break
        if not excluded:
            kept.append(candidate)
    return kept


def _exact_master_poi_matches(name: str, pois: list) -> list:
    query = _normalize_poi_query(name)
    if not query:
        return []
    matches = []
    for poi in pois:
        labels = [poi.get("name", ""), *poi.get("aliases", [])]
        if any(_normalize_poi_query(label) == query for label in labels):
            matches.append(poi)
    return matches


def _is_pending_poi_query(name: str) -> bool:
    query = _normalize_poi_query(name)
    return bool(query and any(candidate_name in query
                              for candidate_name in _pending_osm_poi_names()))


def list_pois() -> list:
    """
    返回所有 POI 列表。

    Returns:
        POI 对象列表
    """
    return load_pois()


def get_pois_by_type(poi_type: str) -> list:
    """
    按类型过滤 POI。

    Args:
        poi_type: POI 类型，如 "scenery", "study", "landmark"

    Returns:
        匹配该类型的 POI 列表
    """
    pois = load_pois()
    return [p for p in pois if p.get("type") == poi_type]


_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_NUM_RUN_RE = re.compile(r"[零一二两三四五六七八九十]+")


def _cn_run_to_int(tok: str):
    """中文数字串转整数：'七'->7、'十四'->14、'二十三'->23、'十'->10。"""
    if "十" in tok:
        left, _, right = tok.partition("十")
        tens = _CN_DIGITS[left] if left in _CN_DIGITS else 1
        ones = _CN_DIGITS[right] if right in _CN_DIGITS else 0
        return tens * 10 + ones
    if len(tok) == 1:
        return _CN_DIGITS.get(tok)
    return None


def _normalize_num(s: str) -> str:
    """中文数字→阿拉伯数字归一化（组合数词正确处理），让「十四舍」和「14舍」能匹配上。"""
    def repl(m):
        v = _cn_run_to_int(m.group(0))
        return str(v) if v is not None else m.group(0)
    return _CN_NUM_RUN_RE.sub(repl, s)

# 宽泛区域词：这些词是学部/园区名，不应作为子串匹配到具体教学楼/食堂
# 例如"信息学部"不应匹配到"信息学部第一教学楼"
_BROAD_AREA_TERMS = {
    "工学部", "文理学部", "信息学部", "医学部",
    "湖滨", "枫园", "梅园", "桂园", "樱园", "星湖",
    "国际园区", "西区", "东区", "南区", "北区",
}


def _is_broad_area_term(name: str) -> bool:
    """判断输入是否是宽泛区域词（学部名/园区名）。"""
    return name.strip() in _BROAD_AREA_TERMS


# 外校/外部单位标志词：用户查询指向校外单位时，不应匹配任何武大POI
_EXTERNAL_TERMS_RE = re.compile(
    r"华中师范|华师|师范大学|理工大学|理工大|华中科技|华科|"
    r"武汉体育学院|体育学院|武体|"
    r"职业技术|电力职|中国科学院|中科院|卓刀泉中学|附属中学|附属小学"
)


def _is_external_query(name: str) -> bool:
    """查询是否指向校外单位（含外校标志且不是在问武大）。"""
    if _EXTERNAL_TERMS_RE.search(name):
        return "武汉大学" not in name and "武大" not in name
    return False


def _digits_aligned(short: str, long: str) -> bool:
    """short 是 long 的子串时，检查数字边界：
    若 short 首/尾为数字，long 中对应位置的前/后字符不能也是数字
    （防止"1教"误匹配"11教学楼"、"7舍"误匹配"17舍"）。"""
    idx = long.find(short)
    if idx < 0:
        return False
    if short[0].isdigit() and idx > 0 and long[idx - 1].isdigit():
        return False
    if short[-1].isdigit() and idx + len(short) < len(long) and long[idx + len(short)].isdigit():
        return False
    return True


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    if a_lower == b_lower:
        return 1.0
    # 宽泛词保护：输入是学部/园区名时，不允许通过子串匹配到更长的POI名
    # 例如"信息学部"不应子串匹配到"信息学部第一教学楼"
    if _is_broad_area_term(a_lower) and len(b_lower) > len(a_lower):
        return 0.0
    if b_lower in a_lower and _digits_aligned(b_lower, a_lower):
        return 0.9
    if a_lower in b_lower and _digits_aligned(a_lower, b_lower):
        return 0.85
    # 数字归一化后再比一次（中文数字 vs 阿拉伯数字）
    a_n = _normalize_num(a_lower)
    b_n = _normalize_num(b_lower)
    if a_n != a_lower or b_n != b_lower:
        if a_n == b_n:
            return 1.0
        if b_n in a_n and _digits_aligned(b_n, a_n):
            return 0.9
        if a_n in b_n and _digits_aligned(a_n, b_n):
            return 0.85
        r = SequenceMatcher(None, a_n, b_n).ratio()
        if r > 0:
            return r
    return SequenceMatcher(None, a_lower, b_lower).ratio()


def find_poi(name: str, min_score: float = 0.6) -> Optional[dict]:
    """
    根据名称模糊匹配单个 POI。

    依次匹配：精确名称 > 别名精确 > 模糊匹配（名称+别名）。
    返回相似度最高且超过阈值的 POI。

    Args:
        name: 用户输入的 POI 名称
        min_score: 最低相似度阈值（0-1），默认 0.6

    Returns:
        匹配到的 POI 对象，未匹配返回 None
    """
    pois = load_pois()
    if _is_external_query(name) or not name.strip():
        return None
    exact_matches = _exact_master_poi_matches(name, pois)
    if exact_matches:
        _record_search(exact_matches[0].get("id", ""))
        return exact_matches[0]
    if _is_pending_poi_query(name):
        return None
    # 过短输入（≤2字）提高阈值：精确名/别名命中为1.0不受影响，模糊近似（如"扬波门"误中"凌波门"）被拦
    threshold = max(min_score, 0.75) if len(name.strip()) <= 2 else min_score
    best_poi = None
    best_score = 0.0

    for poi in pois:
        name_score = _similarity(name, poi["name"])
        if name_score > best_score:
            best_score = name_score
            best_poi = poi

        for alias in poi.get("aliases", []):
            alias_score = _similarity(name, alias)
            if alias_score > best_score:
                best_score = alias_score
                best_poi = poi

    if best_score >= threshold:
        _record_search(best_poi.get("id", ""))
        return best_poi
    return None


def find_poi_candidates(name: str, limit: int = 5, min_score: float = 0.3) -> list:
    """
    根据名称模糊匹配多个候选 POI（用于消歧）。

    Args:
        name: 用户输入的 POI 名称
        limit: 最多返回候选数
        min_score: 最低相似度阈值

    Returns:
        [(poi, score), ...] 按相似度降序排列
    """
    pois = load_pois()
    if _is_external_query(name) or not name.strip():
        return []
    exact_matches = _exact_master_poi_matches(name, pois)
    if exact_matches:
        return [(poi, 1.0) for poi in exact_matches[:limit]]
    if _is_pending_poi_query(name):
        return []
    scored = []

    for poi in pois:
        score = max(
            _similarity(name, poi["name"]),
            max((_similarity(name, a) for a in poi.get("aliases", [])), default=0.0),
        )
        if score >= min_score:
            scored.append((poi, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def _flatten_poi(poi: dict) -> dict:
    coords = poi.get("coordinates", {})
    return {
        "id": poi.get("id", ""),
        "name": poi.get("name", ""),
        "type": poi.get("type", "landmark"),
        "category": poi.get("category", poi.get("type", "landmark")),
        "subcategory": poi.get("subcategory", ""),
        "is_minor": bool(poi.get("is_minor", False)),
        "campus": poi.get("campus", ""),
        "lat": coords.get("lat", 0),
        "lon": coords.get("lng", 0),
        "coordinates": {"lat": coords.get("lat", 0), "lng": coords.get("lng", 0)},
        "description": poi.get("description", ""),
        "aliases": poi.get("aliases", []),
        "season_tags": poi.get("season_tags", []),
        "scenery_score": poi.get("scenery_score", 3),
        "source_refs": poi.get("source_refs", []),
        "verification_status": poi.get("verification_status", "legacy_unverified"),
        "coordinate_verification_status": poi.get("coordinate_verification_status", "unknown"),
        "opening_hours": poi.get("opening_hours", "unknown"),
        # Keep the audited use tags across the shared search/ranking boundary.
        # Candidate tools filter the flattened records, so dropping these here
        # silently made every activity-specific query look empty.
        "activities": list(poi.get("activities") or []),
        "activity_basis": poi.get("activity_basis", ""),
        "activity_review_status": poi.get("activity_review_status", "unknown"),
        "activity_taxonomy_version": poi.get("activity_taxonomy_version"),
    }


def get_poi(name: str, fuzzy: bool = True) -> Optional[dict]:
    if fuzzy:
        poi = find_poi(name)
    else:
        pois = load_pois()
        poi = None
        for p in pois:
            if p.get("name", "") == name or name in p.get("aliases", []):
                poi = p
                break

    if poi is None:
        return None
    return _flatten_poi(poi)


def find_poi_ambiguous(name: str, min_score: float = 0.6):
    """歧义感知的 POI 匹配（对话入口用）。

    返回 (poi_or_None, alternatives)：
      - 精确/唯一最高命中 → (poi, [])
      - 多个不同 POI 模糊同分（如「三教」在文理学部/工学部/信息学部各有一个）
        → (None, [poi, ...])，供对话层请用户消歧，避免静默错配
      - 无命中 / 校外查询 → (None, [])
    返回的 POI 均为 _flatten_poi 后的对外结构。
    """
    if _is_external_query(name) or not name.strip():
        return None, []
    threshold = max(min_score, 0.75) if len(name.strip()) <= 2 else min_score
    scored = find_poi_candidates(name, limit=20, min_score=min(0.3, threshold))
    if not scored:
        return None, []
    top_score = scored[0][1]
    if top_score < threshold:
        return None, []
    if top_score >= 0.99:
        return _flatten_poi(scored[0][0]), []
    ties = [p for p, s in scored if s >= top_score - 0.005]
    if len(ties) > 1:
        return None, [_flatten_poi(p) for p in ties]
    return _flatten_poi(scored[0][0]), []


def search_pois(keyword: str, poi_type: str = None, season: str = None,
                subcategory: str = None, include_minor: bool = True,
                limit: int = 50) -> list:
    """模糊检索 POI，支持 type / subcategory / season 筛选。

    include_minor=False 时过滤小店铺（is_minor 标记），用于泛推荐场景；
    用户明确按类别检索时保持 True，保证"想喝咖啡"能命中瑞幸等小店。
    """
    try:
        candidate_limit = max(1, min(int(limit), len(load_pois())))
    except (TypeError, ValueError):
        candidate_limit = 50
    candidates = find_poi_candidates(keyword, limit=candidate_limit, min_score=0.3)
    results = []
    for poi, score in candidates:
        flat = _flatten_poi(poi)
        if poi_type and flat["type"] != poi_type:
            continue
        if subcategory and flat["subcategory"] != subcategory:
            continue
        if not include_minor and flat["is_minor"]:
            continue
        if season and season not in flat.get("season_tags", []):
            continue
        results.append(flat)
    return results


def search_by_category(subcategory: str = None, poi_type: str = None,
                       season: str = None, include_minor: bool = True,
                       limit: int = 20) -> list:
    """按类别全量检索（目标型需求"想吃饭/想喝咖啡"用），按重要度降序。"""
    pois = load_pois()
    results = []
    for poi in pois:
        flat = _flatten_poi(poi)
        if poi_type and flat["type"] != poi_type:
            continue
        if subcategory and flat["subcategory"] != subcategory:
            continue
        if not include_minor and flat["is_minor"]:
            continue
        if season and season not in flat.get("season_tags", []):
            continue
        results.append((flat, importance_score(flat)))
    results.sort(key=lambda x: x[1], reverse=True)
    return [flat for flat, _ in results[:limit]]


def list_all_pois(poi_type: str = None, season: str = None) -> list:
    pois = load_pois()
    results = []
    for poi in pois:
        flat = _flatten_poi(poi)
        if poi_type and flat["type"] != poi_type:
            continue
        if season and season not in flat.get("season_tags", []):
            continue
        results.append(flat)
    return results
