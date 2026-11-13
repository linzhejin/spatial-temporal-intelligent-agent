"""Model-led dialogue, backed by the existing accepted task and spatial tools.

The model owns interpretation and recommendation. This module owns identity,
delivery alignment and bounded execution; it does not classify user sentences.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

import config


def model_dialogue_enabled() -> bool:
    return os.getenv("AGENT_DIALOGUE_MODE", "model").strip().lower() != "legacy"


def _function(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(required),
                       "additionalProperties": False}}}


_OPERATIONS = ("place_search", "route", "itinerary", "route_comparison", "weather", "campus_answer")


def tool_schemas():
    from agents.tools import TOOL_SCHEMAS, _ENDPOINT_SCHEMA, _PREFERENCE_SCHEMA
    from agents.state_models import HardConstraints

    clearable_fields = ["budget_per_person", "limit", "requirements", "area", "keywords",
                        "exclude_terms", "web_page_read_requested", "start", "search_anchor",
                        "strategy", "activities", "subcategories", "selected_poi_id", "tour_stops", "tour_scope"]
    clearable_fields.extend(f"constraints.{field}" for field in HardConstraints.model_fields)
    schemas = deepcopy(TOOL_SCHEMAS)
    search = next(item["function"] for item in schemas if item["function"]["name"] == "search_poi_candidates")
    search["description"] = ("按模型理解的关键词、区域、类别/用途检索地图或校园目录中的候选记录。"
                              "目录匹配不证明现场存在、营业或入口可通行。无需出发点也能查询校园店铺。"
                              "有参照点可计算距离；返回检索结果，由你选择推荐；不把最近当作最好。")
    search["parameters"]["properties"].update({
        "area": {"type": "string", "description": "地点所在区域，例如武汉大学、梅园、信息学部"},
        "keyword": {"type": "string", "description": "具体店名或检索词，例如火锅、肯德基"},
        "search_anchor": _ENDPOINT_SCHEMA,
        "max_distance_m": {"type": "number", "minimum": 100, "maximum": 10000},
        "exclude_terms": {"type": "array", "items": {"type": "string"}},
        "candidate_names": {"type": "array", "maxItems": 5,
                             "items": {"type": "string", "maxLength": 160},
                             "description": "仅填网页来源中明确提到、需要用地图核对的门店名称"},
    })
    for item in schemas:
        name = item["function"]["name"]
        if name == "resolve_poi":
            item["function"]["description"] = ("核对具体地名并查询校园/在线目录记录；多分店返回候选，不擅自选定。"
                "名称和坐标匹配不代表现场存在、营业或入口可通行。")
        elif name == "plan_tour":
            item["function"]["description"] += (
                "实际出行须先通过 update_task 保存用户明确的出发点；只提及想去的景点不代表从第一个景点出发。"
                "用户明确只要景点间的独立参考路线时，可在任务中设置 tour_scope=independent。")
        elif name in {"get_weather", "plan_route"}:
            parameters = item["function"]["parameters"]
            parameters["properties"]["end_turn_if_success"] = {
                "type": "boolean",
                "description": "只有当前用户请求仅需完成这一项、没有其他子任务且无需详细解释时填 true；否则填 false。服务器只在工具成功并且没有其他未完成工具时才结束本轮。",
            }
            required = parameters.setdefault("required", [])
            if "end_turn_if_success" not in required:
                required.append("end_turn_if_success")
    schemas.extend([
        _function("update_task", "保存你理解的完整需求或本轮修订。只更新传入字段；继续任务保留其余条件。参数明确时，可在同一响应中与后续工具一起调用，服务器会先处理更新。已接受的字段不要重复提交。",
            {"relation": {"type": "string", "enum": ["new", "continue", "resume_route"]},
             "task_id": {"type": "string", "description": "续接或恢复时填写服务器提供的任务 ID"},
             "task_revision": {"type": "integer", "minimum": 0, "description": "所选任务的当前版本"},
             "operation": {"type": "string", "enum": list(_OPERATIONS),
                           "description": "本轮语义任务类型；按主要用户目标选择"},
             "goal": {"type": "string"},
             "start": {**_ENDPOINT_SCHEMA, "description": "实际出发地点，即用户表示当前所在或从哪里出发的位置。不能用用户另行指定的搜索区域代替"},
             "tour_stops": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 160},
                            "description": "用户明确要游览的地点名称，完整保存；不是用户的出发点。只在用户修改地点时替换。"},
             "tour_scope": {"type": "string", "enum": ["from_user", "independent"],
                            "description": "实际个人出行填 from_user；仅用户明确要求不考虑本人出发位置、只看景点间参考路线时填 independent。"},
             "search_anchor": {**_ENDPOINT_SCHEMA, "description": "用户希望在哪个地点附近或区域内搜索的参照点。用户另行说了搜索地点时必须保留该地点，不能复制start来替代；只有用户未给搜索地点时才可省略"},
             "search_radius_m": {"type": "integer", "minimum": 100, "maximum": 5000},
             "expand_search_radius": {"type": "boolean", "description": "用户只说扩大范围且未指定幅度时设为 true；服务器翻倍，最高 5 公里"},
             "mode": {"type": "string", "enum": ["walk", "bike", "drive"]},
             "strategy": {"type": "string", "enum": ["shortest", "recommended", "flat", "scenery"]},
             "requirements": {"type": "array", "items": {"type": "string"}, "description": "完整保留否定、用途及其他条件"},
             "area": {"type": "string", "description": "检索区域文字，例如梅园、桂园或武汉大学"},
             "keywords": {"type": "array", "items": {"type": "string"}},
             "exclude_terms": {"type": "array", "items": {"type": "string"}},
             "web_research_requested": {"type": "boolean", "description": "网页资料能明显补足本轮需求时设为true，例如店铺详情、校园攻略或地图目录可能不全的地点；普通附近推荐在地图候选足够时可不查网页"},
             "web_page_read_requested": {"type": "boolean", "description": "仅当用户明确需要核对门店级详情（如人均、营业时间或近期评价）时设为true；一般附近推荐只需搜索摘要和地图匹配，不必读取网页正文"},
             "candidate_action": {"type": "string", "enum": ["recommend", "next_batch"],
                                  "description": "只有地点筛选条件与半径完全不变时才设为next_batch；任何范围、类别、关键词、区域或要求变化都用recommend并重新检索"},
             "activities": {"type": "array", "items": {"type": "string"}},
             "subcategories": {"type": "array", "items": {"type": "string"}},
             "budget_per_person": {"type": "number", "exclusiveMinimum": 0},
             "limit": {"type": "integer", "minimum": 1, "maximum": 20},
             "constraints": {**_PREFERENCE_SCHEMA["constraints"], "description":
                "必须核对用户已明确提出的道路、距离和时间硬约束并填写对应字段；"
                "仅在 requirements 或 goal 中写文字不会使路径工具执行这些约束。"
                "本轮没有新的硬约束时传空对象，已接受的硬约束仍会保留。"},
             "clear_fields": {"type": "array", "items": {"type": "string", "enum": clearable_fields},
                               "description": "仅在用户明确取消时清除已有条件；仅可使用枚举中的字段名。修改路线偏好只更新strategy，不清除硬约束；constraints.slope 不是可清除字段"}},
            ("relation", "operation", "constraints")),
        _function("present_places", "提交最终推荐。仅允许已查询的地点ID；顺序由你决定。聊天只写一句中性简短引导，不要把用户的筛选条件总结成候选事实。卡片理由只引用地点记录或有效网页来源直接支持的特点；服务器会清除证据不支持的理由。没有可靠特点时填空字符串。source_ids只允许使用网页工具返回的source_id；地图记录的source/data_source值不是source_id，没有网页来源时省略该字段。",
            {"introduction": {"type": "string", "maxLength": 500},
             "places": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
                 "type": "object", "properties": {"poi_id": {"type": "string"},
                     "reason": {"type": "string", "maxLength": 36,
                                "description": "仅填记录中直接支持的简短特点；不重复距离；无可靠特点填空字符串"},
                     "source_ids": {"type": "array", "items": {"type": "string"},
                                    "description": "仅复制search_web/read_web_page返回的source_id；不要填地图source或data_source；没有网页来源时省略"}},
                 "required": ["poi_id", "reason"], "additionalProperties": False}}}, ("introduction", "places")),
        _function("search_web", "检索校园生活/商户资料，返回带来源与日期的文本；不是已核实的地点或坐标。",
                  {"query": {"type": "string"}}, ("query",)),
        _function("read_web_page", "读取本轮搜索返回的某个来源，核对内容、日期与分店。",
                  {"source_id": {"type": "string"}}, ("source_id",)),
        _function("browse_candidates", "读取已接受候选池中的具体地点。条件未变而要换一批时优先用update_task(candidate_action=next_batch)，其candidate_page已含地点；仅在需要分页读取时调用本工具。若本轮首个操作就是读取候选池，必须提供服务器列出的task_id与task_revision，服务器会校验版本并绑定该任务；搜索条件变化后不得读取旧候选池。默认只显示尚未展示的地点。",
                  {"task_id": {"type": "string", "description": "本轮尚未绑定任务时，填写服务器列出的候选任务ID"},
                   "task_revision": {"type": "integer", "minimum": 0, "description": "本轮尚未绑定任务时，填写该任务当前版本"},
                   "offset": {"type": "integer", "minimum": 0},
                   "limit": {"type": "integer", "minimum": 1, "maximum": 12},
                   "include_shown": {"type": "boolean"}}),
        _function("conclude_without_places", "已查看候选但没有适合本轮条件的地点时，说明无法推荐的原因；不要据此声称候选池为空。",
                  {"reason": {"type": "string", "minLength": 1, "maxLength": 1000}}, ("reason",)),
        _function("read_result", "读取本会话历史结果引用；路线几何与候选按页返回。",
                  {"run_id": {"type": "string"},
                   "section": {"type": "string", "enum": ["route_summary", "route_geometry", "candidates"]},
                   "offset": {"type": "integer", "minimum": 0},
                   "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, ("run_id", "section")),
    ])
    return schemas


def _compact(value):
    """Keep complete JSON and useful facts; omit geometry rather than cut JSON."""
    if isinstance(value, list):
        return [_compact(item) for item in value]
    if isinstance(value, dict):
        omitted = {"recommended", "shortest", "recommended_edge_ids", "shortest_edge_ids",
                   "steps", "pois", "geometry", "route_reuse_proof", "reuse_certificate"}
        return {key: _compact(item) for key, item in value.items() if key not in omitted}
    return value


_MODEL_PLACE_FIELDS = (
    "poi_id", "name", "type", "type_name", "category", "subcategory", "campus",
    "aliases", "description", "activities", "activity_basis", "source", "data_source",
    "verification_status", "coordinate_verification_status", "provider_rating",
    "provider_cost", "address", "business_area", "opening_hours", "observed_at",
    "distance_m", "network_distance_m", "distance_basis", "distance_reference_name",
    "estimated_duration_min", "access_link_verified", "step_access_status", "accessibility",
)


def _model_task_view(task):
    """Keep shown candidates and task state; fetch unshown records on demand."""
    item = deepcopy(task)
    state = item.get("state") if isinstance(item, dict) else None
    pending = state.get("place_request") if isinstance(state, dict) else None
    if isinstance(pending, dict):
        pool = pending.pop("candidate_pool", None) or []
        shown = set((pending.get("semantic_request") or {}).get("shown_ids") or [])
        pending["shown_candidates"] = [
            {key: row[key] for key in _MODEL_PLACE_FIELDS if key in row}
            for row in pool if isinstance(row, dict) and row.get("poi_id") in shown
        ]
        pending["candidate_pool_count"] = len(pool)
        pending["unshown_candidate_count"] = sum(1 for row in pool if isinstance(row, dict) and row.get("poi_id") not in shown)
        pending["candidate_pool_ref"] = "browse_candidates"
    return _compact(item)


def _model_tool_content(name, answer):
    """Send recommendation-relevant facts to the model without repeating full POI geometry."""
    if name == "search_web" and isinstance(answer, dict):
        result = {key: answer[key] for key in ("status", "error", "message", "count") if key in answer}
        rows = answer.get("results")
        if isinstance(rows, list):
            result["results"] = [
                {key: str(row.get(key) or "")[:500] if key == "excerpt" else row.get(key)
                 for key in ("source_id", "title", "site_name", "url", "excerpt", "published_at")
                 if key in row}
                for row in rows[:4] if isinstance(row, dict)
            ]
        return json.dumps(result, ensure_ascii=False, default=str)
    if name == "read_web_page" and isinstance(answer, dict):
        result = {key: answer[key] for key in (
            "source_id", "title", "url", "error", "message", "trust_note",
        ) if key in answer}
        if isinstance(answer.get("content"), str):
            result["content"] = answer["content"][:4000]
        return json.dumps(result, ensure_ascii=False, default=str)
    if name not in {"search_poi_candidates", "find_reachable_places"} or not isinstance(answer, dict):
        return json.dumps(_compact(answer), ensure_ascii=False, default=str)

    result = {key: answer[key] for key in (
        "status", "error", "message", "count", "online_status", "search_area",
        "search_keyword", "start_name", "max_distance_m", "max_time_min",
    ) if key in answer}
    result["evidence_scope"] = "地图/校园目录匹配只证明返回了候选记录，不证明现场存在、营业或入口可通行。"
    rows = answer.get("candidates")
    if isinstance(rows, list):
        result["candidates"] = []
        for row in rows[:12]:
            if not isinstance(row, dict):
                continue
            item = {key: row[key] for key in _MODEL_PLACE_FIELDS if key in row}
            for key in ("name", "type", "type_name", "category", "subcategory", "campus", "source",
                        "data_source", "verification_status", "coordinate_verification_status", "address",
                        "business_area", "distance_basis", "distance_reference_name", "step_access_status",
                        "accessibility", "opening_hours", "provider_rating", "provider_cost"):
                if isinstance(item.get(key), str):
                    item[key] = item[key][:240]
            for key in ("description", "activity_basis"):
                if isinstance(item.get(key), str):
                    item[key] = item[key][:500]
            if isinstance(item.get("aliases"), list):
                item["aliases"] = [str(value)[:120] for value in item["aliases"][:5]]
            if isinstance(item.get("activities"), list):
                item["activities"] = [str(value)[:120] for value in item["activities"][:8]]
            result["candidates"].append(item)
        total = answer.get("count")
        if not isinstance(total, int):
            total = len(rows)
        result["candidate_batch_truncated"] = len(rows) > len(result["candidates"])
        result["candidate_batch_has_more"] = total > len(result["candidates"])
    return json.dumps(result, ensure_ascii=False, default=str)


def _resume_constraints(saved, route_constraints):
    """Reconcile route-tool preferences with the durable hard-constraint schema."""
    from agents.state_models import HardConstraints

    if not isinstance(route_constraints, dict):
        raise ValueError("invalid route constraints")
    current = HardConstraints.model_validate(saved or {}).model_dump()
    route = HardConstraints.model_validate({
        "avoid_slope": route_constraints.get("avoid_slope", route_constraints.get("slope") == "avoid"),
        "avoid_steps": route_constraints.get("avoid_steps", route_constraints.get("steps") == "avoid"),
        "require_verified_accessibility": route_constraints.get("require_verified_accessibility", False),
        "max_distance_m": route_constraints.get("max_distance_m"),
        "max_duration_s": route_constraints.get("max_duration_s"),
        "excluded_edge_ids": route_constraints.get("excluded_edge_ids", []),
    }).model_dump()
    for field in ("avoid_slope", "avoid_steps", "require_verified_accessibility"):
        current[field] = current[field] or route[field]
    for field in ("max_distance_m", "max_duration_s"):
        if current[field] is None:
            current[field] = route[field]
    current["excluded_edge_ids"] = list(dict.fromkeys(current["excluded_edge_ids"] + route["excluded_edge_ids"]))
    return HardConstraints.model_validate(current).model_dump()


def _normalize_claim(value):
    return "".join(character for character in str(value or "").lower()
                   if character.isalnum() or "\u3400" <= character <= "\u9fff")


def _reason_is_supported(reason, place, sources=()):
    """Accept a card reason only when each short claim appears in returned evidence."""
    if re.search(r"\d+(?:\.\d+)?\s*(?:米|公里|分钟)", reason):
        return False
    normalized_reason = _normalize_claim(reason)
    if any(normalized_reason == _normalize_claim(place.get(field)) for field in (
            "name", "type_name", "category", "subcategory", "subcategory_label") if place.get(field)):
        return False
    evidence = []

    def collect(value):
        if isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                collect(nested)
        elif isinstance(value, (str, int, float)):
            evidence.append(_normalize_claim(value))

    for field in ("name", "description", "activities", "activity_basis", "address",
                  "business_area", "accessibility"):
        collect(place.get(field))
    for source in sources:
        if isinstance(source, dict):
            for field in ("title", "site_name", "excerpt", "content"):
                collect(source.get(field))

    clauses = [item.strip() for item in re.split(r"[，,、；;。！？\n]+", reason) if item.strip()]
    if not clauses:
        return True
    for clause in clauses:
        claim = clause
        for prefix in ("位于", "坐落于", "处于", "地址在", "在"):
            if claim.startswith(prefix):
                claim = claim[len(prefix):]
                break
        for suffix in ("方向", "一带", "附近", "周边"):
            if claim.endswith(suffix):
                claim = claim[:-len(suffix)]
                break
        normalized = _normalize_claim(claim)
        if not normalized or not any(normalized in item for item in evidence):
            return False
    return True


def _task_with_unconfirmed_tour_origin(task):
    """Do not revive an old implicit tour anchor as the user's departure."""
    item = deepcopy(task)
    state = item.get("state") if isinstance(item, dict) else None
    pending = state.get("place_request") if isinstance(state, dict) else None
    intent = pending.get("semantic_request") if isinstance(pending, dict) else None
    if not isinstance(intent, dict) or intent.get("tour_origin_confirmed"):
        return item
    route = item.get("route_state") or intent.get("route_to_modify") or {}
    tour = route.get("tour") if isinstance(route, dict) else None
    places = tour.get("pois") if isinstance(tour, dict) else None
    origin = pending.get("start")
    first = places[0] if isinstance(places, list) and places else None
    provenance = state.get("field_provenance") or {}
    has_source = any(key in provenance for key in ("place_request.start", "route_spec.start"))
    if (isinstance(route, dict) and route.get("route_kind") == "tour"
            and isinstance(origin, dict) and isinstance(first, dict)
            and origin.get("name") and origin.get("name") == first.get("name") and not has_source):
        pending.pop("start", None)
        pending["pending_slot"] = "start"
        intent["tour_origin_unconfirmed"] = True
        if isinstance(state.get("route_spec"), dict):
            state["route_spec"]["start"] = None
    return item


def _missing_tour_stops(route, requested, tools):
    """Compare model-saved destinations with the verified route's actual stops."""
    state = route.get("route_state") if isinstance(route, dict) else None
    state = state if isinstance(state, dict) else {}
    refs = []

    def add(ref):
        if isinstance(ref, str):
            refs.append({"name": ref})
        elif isinstance(ref, dict):
            if ref.get("type") == "multi" and isinstance(ref.get("points"), list):
                for item in ref["points"]:
                    add(item)
            elif ref.get("name") or ref.get("poi_id") or ref.get("id"):
                refs.append(ref)

    for field in ("start", "end", "via"):
        add(state.get(field))
    for item in (state.get("tour") or {}).get("pois") or []:
        add(item)
    for item in (state.get("itinerary") or {}).get("stops") or []:
        add(item.get("poi") or item if isinstance(item, dict) else item)
    for leg in state.get("legs") or []:
        if isinstance(leg, dict):
            add(leg.get("start"))
            add(leg.get("end"))
            add(leg.get("via"))
    add(route.get("start_name"))
    add(route.get("end_name"))

    def identity(name):
        if not isinstance(name, str) or not name.strip():
            return None
        poi, _ = tools.find_poi_ambiguous(name.strip())
        return (poi or {}).get("id")

    known = [(str(ref.get("name") or "").strip(), ref.get("poi_id") or ref.get("id")) for ref in refs]
    known = [(name, identifier or identity(name)) for name, identifier in known]
    missing = []
    for name in requested:
        target = str(name).strip()
        target_id = identity(target)
        if not any(target == actual or (target_id and target_id == actual_id) for actual, actual_id in known):
            missing.append(target)
    return missing


def _same_accepted_location(first, second, tools):
    """Compare trusted point references without guessing an ambiguous identity."""
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    if first == second:
        return True
    if first.get("type") == second.get("type") == "coord":
        def coordinates(ref):
            nested = ref.get("coordinates") if isinstance(ref.get("coordinates"), dict) else {}
            lng = ref.get("lng", nested.get("lng", nested.get("longitude")))
            lat = ref.get("lat", nested.get("lat", nested.get("latitude")))
            return (lng, lat) if all(type(value) in (int, float) for value in (lng, lat)) else None

        left, right = coordinates(first), coordinates(second)
        return left is not None and right is not None and left == right
    if first.get("poi_id") and second.get("poi_id"):
        return str(first["poi_id"]) == str(second["poi_id"])

    def identity(ref):
        if ref.get("poi_id"):
            return str(ref["poi_id"])
        name = ref.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        poi, alternatives = tools.find_poi_ambiguous(name.strip())
        return (poi or {}).get("id") if not alternatives else None

    first_name, second_name = first.get("name"), second.get("name")
    if (isinstance(first_name, str) and isinstance(second_name, str)
            and first_name.strip() and first_name.strip() == second_name.strip()):
        return True
    left, right = identity(first), identity(second)
    return bool(left and right and left == right)


def _search_radius_is_explicit(pending, intent, query):
    from agents.distance_evidence import radius_supported_by_message
    radius = pending.get("search_radius_m") or 1000
    return (intent.get("search_radius_explicit") is True or radius != 1000
            or radius_supported_by_message(pending.get("original_query", ""), radius, 1000)
            or radius_supported_by_message(query, radius, 1000))


def _anchor_only_embedded_in_origin(query, origin, anchor):
    """Do not turn part of a stated origin into a separate center by itself."""
    if not isinstance(origin, dict) or not isinstance(anchor, dict):
        return False
    normalize = lambda value: re.sub(r"\s+", "", str(value or "")).casefold()
    start_name, center_name, message = (normalize(origin.get("name")), normalize(anchor.get("name")),
                                        normalize(query))
    return bool(start_name and center_name and center_name != start_name and center_name in start_name
                and start_name in message and center_name not in message.replace(start_name, ""))


def _plain_nearby_requirement(requirement, location, tools):
    if not isinstance(requirement, str):
        return False
    match = re.fullmatch(r"\s*(.+?)\s*(?:附近|周边|周围|旁边|一带)\s*", requirement)
    return bool(match and _same_accepted_location(location, {"type": "poi", "name": match.group(1).strip()}, tools))


def _plain_activity_requirement(requirement):
    """Identify a standalone activity, never a clause containing other constraints."""
    if not isinstance(requirement, str) or not re.fullmatch(
            r"\s*(?:寻找|推荐|想找|想去|找|去|吃|喝|打|踢)?"
            r"(?:咖啡|奶茶|茶饮|果茶|吃饭|用餐|餐饮|餐厅|饭店|食堂|快餐|小吃|"
            r"篮球|羽毛球|乒乓球|足球|排球|网球|跑步|游泳|健身|自习|学习|购物|医疗|快递|邮政)"
            r"(?:店|馆|厅|点|场地|场|服务|的地方|场所)?\s*", requirement):
        return set()
    from spatial.poi_taxonomy import resolve_activity_query
    parsed = resolve_activity_query(requirement) or {}
    return set(parsed.get("activities") or [])


def _plain_stop_requirement(requirement, location, tools):
    if not isinstance(requirement, str):
        return False
    match = re.fullmatch(r"\s*(?:先|再|最后|然后|随后|接着)?(?:去|到|前往|抵达)(.+?)\s*", requirement)
    return bool(match and _same_accepted_location(location, {"type": "poi", "name": match.group(1).strip()}, tools))


def _plain_ordered_stops(requirement):
    """Read a pure stop ordering only; never discard a clause with other constraints."""
    if not isinstance(requirement, str):
        return []
    arrow = re.fullmatch(r"\s*(?:顺序|路线顺序|次序)\s*[:：]\s*(.+?)\s*", requirement)
    if arrow:
        names = [part.strip() for part in re.split(r"\s*(?:→|->|➡|➜)\s*", arrow.group(1))]
    else:
        parts = re.split(r"\s*[，,、]?\s*(?:再|然后|接着|最后)(?:去|到|前往)\s*", requirement.strip())
        if not 2 <= len(parts) <= 8:
            return []
        first = re.fullmatch(r"(?:先|首先)(?:去|到|前往)(.+)", parts[0])
        if not first:
            return []
        names = [first.group(1).strip(), *(part.strip() for part in parts[1:])]
    if not 2 <= len(names) <= 8:
        return []
    if any(not name or len(name) > 80 or re.search(r"[，,。；;！？!?、]|避开|不要|必须|分钟|公里|台阶", name)
           for name in names):
        return []
    return names


def _separate_order_groups(requirements):
    """Collect consecutive standalone first/next stop requirements."""
    groups, current = [], []

    def finish():
        if len(current) >= 2:
            groups.append(([item[1] for item in current], [item[0] for item in current]))

    for requirement in requirements:
        match = (re.fullmatch(r"\s*(先|首先|再|然后|接着|最后)(?:去|到|前往)([^，,。；;！？!?、]{1,80})\s*", requirement)
                 if isinstance(requirement, str) else None)
        if not match or re.search(r"避开|不要|必须|分钟|公里|台阶", match.group(2)):
            finish()
            current = []
        elif match.group(1) in {"先", "首先"}:
            finish()
            current = [(requirement, match.group(2).strip())]
        elif current:
            current.append((requirement, match.group(2).strip()))
    finish()
    return groups


def _user_supplied_current_position(query, endpoint, tools):
    """Require an explicit origin or arrival before moving a former endpoint to start."""
    message = str(query or "")
    if re.search(r"没(?:有)?到|未到|尚未到", message):
        return False
    name = str(endpoint.get("name") or "") if isinstance(endpoint, dict) else ""
    names = [name] if name else []
    poi, alternatives = tools.find_poi_ambiguous(name) if name else (None, None)
    if poi and not alternatives:
        for alias in poi.get("aliases") or []:
            resolved, ambiguous = tools.find_poi_ambiguous(alias)
            if resolved and not ambiguous and resolved.get("id") == poi.get("id"):
                names.append(alias)
    for name in names:
        for match in re.finditer(r"(?:从|我在|我们在|现在在|现在就在|起点是|当前位置是)\s*" + re.escape(name), message):
            if not re.search(r"(?:不|别|不要|没有|未)$", message[max(0, match.start() - 4):match.start()]):
                return True
    return bool(re.search(r"(?:已经|刚刚|刚|我|我们).{0,10}(?:到了|到达|抵达|走完|到.{0,20}了)", message))


def _unconfirmed_previous_end_as_current(query, question, accepted_start, route, tools):
    """Do not turn a former destination into the user's current location by implication."""
    if not isinstance(route, dict) or not isinstance(route.get("end"), dict) or not isinstance(question, str):
        return False
    end = route["end"]
    if accepted_start and _same_accepted_location(accepted_start, end, tools):
        return False
    match = re.search(r"从\s*(.{1,60}?)(?:往|出发|继续|步行|走|去|前往)", question)
    if not match or not _same_accepted_location({"type": "poi", "name": match.group(1).strip()}, end, tools):
        return False
    if _user_supplied_current_position(query, end, tools):
        return False
    return True


_BALL_ACTIVITIES = {"basketball", "badminton", "table_tennis", "football", "volleyball", "tennis"}


def _expand_unqualified_sport_categories(args, pending, query):
    """Keep catalogue labels from silently narrowing a generic sport request."""
    from spatial.poi_taxonomy import ACTIVITY_SUBCATEGORIES

    activities = args.get("activities") or pending.get("activities") or []
    categories = args.get("subcategories") or ([args["subcategory"]] if args.get("subcategory") else [])
    if (not isinstance(activities, list) or not activities or not all(isinstance(x, str) for x in activities)
            or not set(activities) <= _BALL_ACTIVITIES or not isinstance(categories, list) or not categories
            or not all(isinstance(x, str) for x in categories)):
        return
    allowed = list(dict.fromkeys(category for activity in activities
                                 for category in ACTIVITY_SUBCATEGORIES[activity]))
    if not set(categories) <= set(allowed):
        return
    intent = pending.get("semantic_request") or {}
    evidence = [query, pending.get("original_query"), *(intent.get("requirements") or []),
                *(args.get("requirements") or [])]
    endpoints = [args.get("start"), args.get("search_anchor"),
                 pending.get("start"), pending.get("search_anchor")]
    for value in evidence:
        if not isinstance(value, str):
            continue
        for endpoint in endpoints:
            if isinstance(endpoint, dict) and isinstance(endpoint.get("name"), str):
                value = value.replace(endpoint["name"], "")
        if re.search(r"室内|室外|户外|露天|体育馆|球馆|运动馆|场馆|操场|运动场|田径场", value):
            return
    args["subcategories"] = allowed
    args.pop("subcategory", None)


def _ambiguous_ball_request(message):
    """Identify an unspecified ball activity in the request, not its origin name."""
    from spatial.poi_taxonomy import resolve_activity_query
    text = str(message or "")
    # A generic venue catalogue may intentionally include different courts;
    # a request to play a ball sport needs the particular activity clarified.
    if not re.search(r"打(?:个)?球|球类运动", text):
        return None
    if re.search(r"(?:球|球场|球馆).{0,8}(?:都可以|都行|不限|都看看|随便|任意)"
                 r"|(?:随便|任意|不限).{0,8}(?:球|球场|球馆)", text):
        return None
    text = re.sub(r"^.{0,100}?出发", "", text)
    parsed = []
    for part in re.split(r"[，,。；;！？!?]", text):
        fragments = [part, part.rsplit("找", 1)[-1], part.rsplit("推荐", 1)[-1]]
        parsed.extend(intent for fragment in fragments if (intent := resolve_activity_query(fragment)))
    if any(set(intent.get("activities") or []) & _BALL_ACTIVITIES for intent in parsed):
        return None
    return next((intent for intent in parsed if intent.get("status") == "needs_clarification"
                 and "球" in str(intent.get("question") or "")), None)


def _ambiguous_route_reference(ref, query, tools):
    """Find an ambiguous user-mentioned suffix without guessing a POI identity."""
    if not isinstance(ref, dict) or ref.get("type") == "coord" or not isinstance(ref.get("name"), str):
        return None
    name, message = ref["name"].strip(), re.sub(r"\s+", "", str(query or ""))
    if len(name) < 2 or not message:
        return None
    identifier = ref.get("poi_id")
    resolved, _ = tools.find_poi_ambiguous(name)
    identifier = identifier or (resolved or {}).get("id")
    raw = next((row for row in tools.load_pois() if row.get("id") == identifier), {}) if identifier else {}
    labels = [name, raw.get("name"), *(raw.get("aliases") or [])]
    for label in labels:
        label = re.sub(r"\s+", "", str(label or ""))
        if len(label) < 2 or label not in message:
            continue
        matched, alternatives = tools.find_poi_ambiguous(label)
        if matched and not alternatives and matched.get("id") == identifier:
            return None
    compact_name = re.sub(r"\s+", "", name)
    for length in range(len(compact_name), 1, -1):
        term = compact_name[-length:]
        if term not in message:
            continue
        matched, alternatives = tools.find_poi_ambiguous(term)
        if matched and not alternatives and matched.get("id") == identifier:
            return None
        names = list(dict.fromkeys(str(row.get("name") or "").strip() for row in alternatives or []))
        names = [item for item in names if item]
        if len(names) >= 2:
            return term, names[:5]
    return None


def _normalize_coarse_place_type(args, tools):
    """A catalogue type in a subcategory field should retain its type meaning."""
    schema = next(item["function"]["parameters"]["properties"]["poi_type"]
                  for item in tools.TOOL_SCHEMAS if item["function"]["name"] == "search_poi_candidates")
    types = set(schema["enum"])
    subcategories = set(tools._SUBCATEGORY_LABELS)
    subcategories.update(row.get("subcategory") for row in tools.load_pois() if row.get("subcategory"))
    subcategories.update(sub for values in tools.ACTIVITY_SUBCATEGORIES.values() for sub in values)
    values = args.get("subcategories") if isinstance(args.get("subcategories"), list) else []
    single = args.get("subcategory")
    coarse = {value for value in [*values, single] if isinstance(value, str)
              and value in types and value not in subcategories}
    if not coarse:
        return None
    if len(coarse) != 1 or args.get("poi_type") not in (None, next(iter(coarse))):
        return {"error": "conflicting_place_categories", "message": "地点大类条件不一致，请核对查询条件。"}
    category = next(iter(coarse))
    args["poi_type"] = category
    if "subcategories" in args:
        args["subcategories"] = [item for item in values if item != category]
    if single == category:
        args.pop("subcategory", None)
    return None


class Dialogue:
    def __init__(self, query, *, context=None, history=None, coord_start=None, coord_end=None,
                 coord_waypoints=None, travel_mode=None, uid=None, execute_tool, client_factory, trace,
                 bind_task=None, read_result=None, deadline_monotonic=None):
        self.query, self.context, self.uid = query, context or {}, uid
        self.requested_mode = travel_mode
        self.bind_task = bind_task
        self.read_result = read_result
        self.bound_task = None
        self.modified_fields = set()
        self.deadline_monotonic = deadline_monotonic if deadline_monotonic is not None else time.monotonic() + 75
        self.available_tasks = {str(item.get("task_id")): _task_with_unconfirmed_tour_origin(item)
                                for item in self.context.get("available_tasks", [])
                                if isinstance(item, dict) and item.get("task_id")}
        self.execute_tool, self.client_factory, self.trace = execute_tool, client_factory, trace
        self.pending = deepcopy(self.context.get("place_request") or {})
        self.pending.setdefault("request_id", str(uuid4()))
        self.pending.setdefault("original_query", query[:2000])
        self.pending.setdefault("mode", travel_mode or "walk")
        self.pending.setdefault("search_radius_m", 1000)
        self.pending.setdefault("constraints", {})
        self._prior_ball_activities = set(self.pending.get("activities") or []) & _BALL_ACTIVITIES
        self.intent = self.pending.setdefault("semantic_request", {}) or {}
        self.pending["semantic_request"] = self.intent
        self.sources = {item["source_id"]: item for item in self.intent.get("sources", [])
                        if isinstance(item, dict) and item.get("source_id")}
        self.known = {item["poi_id"]: item for item in [
                          *(self.pending.get("candidate_pool") or []),
                          *(self.pending.get("candidate_places") or []),
                          self.pending.get("selected_place"),
                      ]
                      if isinstance(item, dict) and item.get("poi_id")}
        self._resolved_this_turn = {}
        selected_id = self.context.get("selected_poi_id")
        accepted_ids = set(self.pending.get("candidate_poi_ids") or [])
        if selected_id in accepted_ids and selected_id in self.known:
            self.pending["selected_poi_id"] = selected_id
            self.pending["selected_place"] = deepcopy(self.known[selected_id])
        self.turn_action = "recommend"
        self.candidates, self.route, self.route_kind, self.comparison = None, None, None, None
        self.route_is_independent_tour = False
        self.weather = None
        self.route_tool_failure = None
        self.no_path, self.clarify, self.suggestions = None, None, None
        self.last_search = {}
        self._empty_search_keys = set()
        self._map_search_completed = False
        self.web_search_count = 0
        self.web_fetch_count = 0
        self.place_search_count = 0
        self._place_search_limit = 8
        self._scope_recheck_granted = False
        self.place_service_error = None
        self._expanded_this_turn = False
        self.turns = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.original_coords = [point for point in [coord_start, coord_end, *(coord_waypoints or [])] if point]
        observation = self.context.get("location_observation")
        if isinstance(observation, dict):
            from datetime import datetime, timezone
            try:
                observed = datetime.fromisoformat(str(observation.get("observed_at")).replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - observed).total_seconds()
                if (0 <= age <= 120 and 0 < float(observation.get("accuracy_m", 0)) <= 100
                        and observation.get("crs") == "WGS84"):
                    self.original_coords.append({"lng": observation["lng"], "lat": observation["lat"]})
            except (ValueError, TypeError, KeyError):
                pass
        self.coord_start, self.coord_end, self.coord_waypoints = coord_start, coord_end, coord_waypoints
        self.previous_route = self.context.get("previous_route_state")
        self.spatial_snapshot = None
        trusted_task = deepcopy(self.pending)
        if trusted_task.get("candidate_pool"):
            trusted_task["candidate_pool_count"] = len(trusted_task["candidate_pool"])
            trusted_task.pop("candidate_pool", None)
        trusted = {"active_task": None if self.bind_task else _compact(trusted_task), "previous_route": _compact(self.previous_route),
                   "route_context_uncertain": bool(self.context.get("route_context_uncertain")),
                   "map_start": coord_start, "map_end": coord_end, "map_waypoints": coord_waypoints,
                   "selected_poi_id": self.context.get("selected_poi_id"), "travel_mode": travel_mode,
                   "available_tasks": [_model_task_view(task) for task in self.available_tasks.values()],
                   "message_records": _compact(self.context.get("message_records") or []),
                   "input_action": self.context.get("input_action"),
                   "location_observation": self.context.get("location_observation")}
        self.messages = []
        transcript = [{"role": item["role"], "content": str(item["content"])}
                      for item in (history or self.context.get("history") or [])
                      if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and item.get("content")]
        self.messages = [*self.messages,
                         {"role": "system", "content": (Path(__file__).parent / "prompts" / "dialogue_system.txt").read_text(encoding="utf-8")},
                         *transcript,
                         {"role": "system", "content": "服务器提供的已接受状态（历史并不代表用户当前的位置）：\n" + json.dumps(trusted, ensure_ascii=False)},
                         {"role": "user", "content": query}]

    def update(self, args):
        from agents.state_models import HardConstraints, PlaceRequestState
        relation = args.get("relation")
        if relation not in {"new", "continue", "resume_route"}:
            raise ValueError("relation must be new, continue or resume_route")
        if self.bind_task and self.bound_task is not None and relation == "new":
            raise ValueError("a run cannot replace its already bound task")
        if self.bind_task and self.bound_task is None and relation != "new":
            chosen = self.available_tasks.get(str(args.get("task_id") or ""))
            if not chosen or type(args.get("task_revision")) is not int or chosen.get("revision") != args["task_revision"]:
                raise ValueError("task_version_invalid: 请选择服务器提供的任务及当前版本")
            state = chosen.get("state") or {}
            self.previous_route = deepcopy(chosen.get("route_state"))
            self.pending = deepcopy(state.get("place_request") or {})
            self._prior_ball_activities = set(self.pending.get("activities") or []) & _BALL_ACTIVITIES
            self.pending.setdefault("request_id", str(uuid4()))
            self.pending.setdefault("original_query", self.query[:2000])
            self.pending.setdefault("mode", "walk")
            self.pending.setdefault("constraints", {})
            self.intent = self.pending.setdefault("semantic_request", {}) or {}
            self.pending["semantic_request"] = self.intent
            self.sources = {item["source_id"]: item for item in self.intent.get("sources", [])
                            if isinstance(item, dict) and item.get("source_id")}
            self.known = {item["poi_id"]: item for item in [*(self.pending.get("candidate_pool") or []),
                                                             *(self.pending.get("candidate_places") or []),
                                                             self.pending.get("selected_place")]
                          if isinstance(item, dict) and item.get("poi_id")}
            self.known.update(deepcopy(self._resolved_this_turn))
            selected = self.context.get("selected_poi_id")
            if (selected and str(args.get("task_id")) == str(self.context.get("selected_task_id"))
                    and selected in (self.pending.get("candidate_poi_ids") or []) and selected in self.known):
                self.pending["selected_poi_id"] = selected
                self.pending["selected_place"] = deepcopy(self.known[selected])
        elif self.bind_task and self.bound_task is not None and args.get("task_id") not in (None, self.bound_task["task_id"]):
            raise ValueError("a run cannot change its bound task")
        pending = deepcopy(self.pending)
        if relation == "new":
            self.known = deepcopy(self._resolved_this_turn)
            self._prior_ball_activities = set()
            pending = {"request_id": str(uuid4()), "original_query": self.query[:2000],
                       "mode": self.requested_mode or self.pending.get("mode", "walk"),
                       "constraints": {}, "semantic_request": {}, "search_radius_m": 1000}
        if relation == "resume_route":
            if not self.previous_route or self.context.get("route_context_uncertain"):
                raise ValueError("prior_route_unavailable: 请确认要修改的路线")
            state = self.previous_route
            pending_intent = pending.get("semantic_request") or {}
            independent_tour = ((pending_intent.get("tour_scope") == "independent" and not pending.get("start"))
                                or pending_intent.get("tour_origin_unconfirmed") is True)
            pending.update({"mode": state["travel_mode"],
                            "constraints": _resume_constraints(pending.get("constraints"), state["hard_constraints"])})
            if not independent_tour:
                pending["start"] = deepcopy(state["start"])
            strategy = state.get("strategy") or {}
            pending["strategy"] = strategy.get("name", "shortest") if isinstance(strategy, dict) else strategy
            pending.setdefault("semantic_request", {})["route_to_modify"] = deepcopy(state)
        intent = pending.setdefault("semantic_request", {}) or {}
        _expand_unqualified_sport_categories(args, pending, self.query)
        pruned_requirement = False
        previous_center = pending.get("search_anchor") or pending.get("start")
        next_center = args.get("search_anchor")
        if previous_center and next_center and isinstance(intent.get("requirements"), list):
            from agents import tools
            if (not _same_accepted_location(previous_center, next_center, tools)
                    and not _anchor_only_embedded_in_origin(self.query, pending.get("start"), next_center)):
                explicitly_retained = args.get("requirements") if isinstance(args.get("requirements"), list) else []
                original_requirements = intent["requirements"]
                intent["requirements"] = [item for item in original_requirements if item in explicitly_retained
                                          or not _plain_nearby_requirement(item, previous_center, tools)]
                pruned_requirement = intent["requirements"] != original_requirements
        previous_activities = set(pending.get("activities") or [])
        next_activities = args.get("activities")
        removed_activities = (previous_activities - set(next_activities)
                              if isinstance(next_activities, list) else set())
        next_categories = args.get("subcategories")
        changed_categories = (isinstance(next_categories, list) and bool(next_categories)
                              and set(next_categories) != set(pending.get("subcategories") or []))
        previous_route = intent.get("route_to_modify") or {}
        previous_via = previous_route.get("via") if isinstance(previous_route, dict) else None
        if isinstance(previous_via, dict) and previous_via.get("type") == "multi":
            prior_stops = previous_via.get("points") or []
        else:
            prior_stops = [previous_via] if isinstance(previous_via, dict) else []
        next_stops = args.get("tour_stops")
        removed_stops = []
        if isinstance(next_stops, list) and all(isinstance(item, str) for item in next_stops):
            from agents import tools
            removed_stops = [stop for stop in prior_stops if isinstance(stop, dict) and not any(
                _same_accepted_location(stop, {"type": "poi", "name": name}, tools) for name in next_stops)]

        proposed_requirements = args.get("requirements") if isinstance(args.get("requirements"), list) else []
        prior_groups = _separate_order_groups(intent.get("requirements") or [])
        proposed_groups = _separate_order_groups(proposed_requirements)
        new_orderings = [_plain_ordered_stops(item) for item in proposed_requirements]
        new_orderings = [order for order in new_orderings if order] + [names for names, _ in proposed_groups]
        if len(new_orderings) > 1:
            from agents import tools
            prior_orderings = ([_plain_ordered_stops(item) for item in intent.get("requirements") or []] +
                               [names for names, _ in prior_groups])
            same_sequence = lambda left, right: len(left) == len(right) and all(
                _same_accepted_location({"type": "poi", "name": a}, {"type": "poi", "name": b}, tools)
                for a, b in zip(left, right))
            fresh = []
            for order in new_orderings:
                if not any(same_sequence(order, prior) for prior in prior_orderings if prior) and not any(
                        same_sequence(order, other) for other in fresh):
                    fresh.append(order)
            new_orderings = fresh if len(fresh) == 1 else []

        def order_replaced(old):
            if not old:
                return False
            from agents import tools
            same = lambda left, right: _same_accepted_location(
                {"type": "poi", "name": left}, {"type": "poi", "name": right}, tools)
            for order in new_orderings:
                if len(old) != len(order) or all(same(a, b) for a, b in zip(old, order)):
                    continue
                remaining = list(order)
                for name in old:
                    index = next((i for i, candidate in enumerate(remaining) if same(name, candidate)), None)
                    if index is None:
                        break
                    remaining.pop(index)
                else:
                    return True
            return False

        def replaced_order(requirement):
            return order_replaced(_plain_ordered_stops(requirement))

        stale_split_order = {item for names, items in prior_groups if order_replaced(names)
                             for item in items if item not in proposed_requirements}
        order_changed = bool(stale_split_order) or any(replaced_order(item) for item in intent.get("requirements") or [])

        def obsolete_requirement(requirement):
            if requirement in stale_split_order or replaced_order(requirement):
                return True
            activity = _plain_activity_requirement(requirement)
            if activity and activity <= removed_activities:
                return True
            if activity and changed_categories:
                if isinstance(next_activities, list) and activity.intersection(next_activities):
                    return False
                from spatial.poi_taxonomy import ACTIVITY_SUBCATEGORIES
                if all(ACTIVITY_SUBCATEGORIES.get(item) and
                       set(ACTIVITY_SUBCATEGORIES[item]).isdisjoint(next_categories) for item in activity):
                    return True
            if removed_stops:
                from agents import tools
                return any(_plain_stop_requirement(requirement, stop, tools) for stop in removed_stops)
            return False

        def revise_requirements(requirements):
            revised = []
            for requirement in requirements:
                if obsolete_requirement(requirement):
                    continue
                if isinstance(requirement, str):
                    qualifier, separator, target = requirement.rpartition("的")
                    if (separator and qualifier.strip() and
                            re.fullmatch(r"[^，,。；;！？!?]{1,200}", qualifier.strip()) and
                            obsolete_requirement(target)):
                        requirement = qualifier.strip()
                if requirement not in revised:
                    revised.append(requirement)
            return revised

        if isinstance(intent.get("requirements"), list) and (removed_activities or removed_stops or changed_categories or order_changed):
            original_requirements = intent["requirements"]
            intent["requirements"] = revise_requirements(original_requirements)
            pruned_requirement |= intent["requirements"] != original_requirements
        explicit_radius, separate_equal_distance = None, False
        if args.get("operation", intent.get("operation")) == "place_search":
            from agents.distance_evidence import explicit_search_radius, explicit_travel_time_limit
            explicit_radius, separate_equal_distance = explicit_search_radius(self.query)
            if explicit_radius is not None:
                args["search_radius_m"] = explicit_radius
            explicit_duration = explicit_travel_time_limit(self.query)
            if explicit_duration is not None:
                args.setdefault("constraints", {})
                if isinstance(args["constraints"], dict):
                    args["constraints"]["max_duration_s"] = explicit_duration
        changed_search = relation == "new"
        for field in ("operation", "goal", "requirements", "area", "keywords", "exclude_terms",
                      "web_research_requested", "web_page_read_requested", "budget_per_person", "limit",
                      "tour_stops", "tour_scope"):
            if field in args:
                value = args[field]
                if field == "operation" and (not isinstance(value, str) or value not in _OPERATIONS):
                    raise ValueError("invalid operation")
                if field in {"requirements", "keywords", "exclude_terms"}:
                    if not isinstance(value, list) or len(value) > 30 or any(not isinstance(x, str) or len(x) > 500 for x in value):
                        raise ValueError(f"invalid {field}")
                    if field == "requirements" and (removed_activities or removed_stops or changed_categories or order_changed):
                        value = revise_requirements(value)
                    if (relation != "new" and field in {"requirements", "exclude_terms"}
                            and field not in args.get("clear_fields", [])):
                        value = list(dict.fromkeys([*(intent.get(field) or []), *value]))
                if field == "tour_stops":
                    if (not isinstance(value, list) or len(value) > 8 or
                            any(not isinstance(x, str) or not x.strip() or len(x) > 160 for x in value)):
                        raise ValueError("invalid tour_stops")
                    value = list(dict.fromkeys(x.strip() for x in value))
                if field == "tour_scope" and value not in {"from_user", "independent"}:
                    raise ValueError("invalid tour_scope")
                if field == "goal" and (not isinstance(value, str) or not 0 < len(value) <= 2000):
                    raise ValueError("invalid goal")
                if field == "limit" and (type(value) is not int or not 1 <= value <= 20):
                    raise ValueError("invalid limit")
                if field in {"web_research_requested", "web_page_read_requested"} and type(value) is not bool:
                    raise ValueError(f"invalid {field}")
                if field == "budget_per_person" and (type(value) not in (int, float) or not 0 < value <= 100000):
                    raise ValueError("invalid budget_per_person")
                changed_search |= field not in {"goal", "operation"} and value != intent.get(field)
                intent[field] = deepcopy(value)
        if "web_research_requested" not in args and relation == "new":
            intent["web_research_requested"] = False
        if "web_page_read_requested" not in args and relation == "new":
            intent["web_page_read_requested"] = False
        anchor_note = None
        ignored_anchor = False
        for field in ("start", "search_anchor", "mode", "strategy", "activities", "subcategories"):
            if field in args:
                if field in {"start", "search_anchor"} and args[field] is None:
                    raise ValueError("clear an endpoint only through clear_fields")
                if field in {"start", "search_anchor"} and args[field] is not None:
                    args[field] = self._endpoint(args[field], allow_pending=relation != "new")
                if field == "start" and args[field] is not None and self.coord_start is None:
                    from agents import tools
                    prior_routes = [self.previous_route, self.intent.get("route_to_modify"),
                                    *(task.get("route_state") for task in self.available_tasks.values())]
                    for route in prior_routes:
                        if (isinstance(route, dict) and isinstance(route.get("start"), dict)
                                and isinstance(route.get("end"), dict)
                                and not _same_accepted_location(route["start"], route["end"], tools)
                                and _same_accepted_location(args[field], route["end"], tools)
                                and not _user_supplied_current_position(self.query, route["end"], tools)):
                            raise ValueError("unconfirmed_current_position: 用户尚未确认到达原终点，请保留原起点或询问")
                if field in {"start", "search_anchor"}:
                    from agents import tools
                    if (field == "search_anchor" and not _same_accepted_location(pending.get(field), args[field], tools)
                            and _anchor_only_embedded_in_origin(self.query, pending.get("start"), args[field])):
                        ignored_anchor = True
                        anchor_note = "搜索中心仅出现在出发地点名称中；已保留原搜索中心。"
                        continue
                    changed_search |= not _same_accepted_location(pending.get(field), args[field], tools)
                elif field in {"mode", "activities", "subcategories"}:
                    changed_search |= args[field] != pending.get(field)
                pending[field] = deepcopy(args[field])
                if field == "activities" and args[field] and pending.get("pending_slot") == "activity":
                    pending["pending_slot"] = None
                if field == "start":
                    if pending.get("pending_slot") == "start":
                        pending["pending_slot"] = None
                    intent["tour_origin_confirmed"] = True
                    intent.pop("tour_origin_unconfirmed", None)
                    if intent.get("tour_scope") == "independent" and "tour_scope" not in args:
                        intent["tour_scope"] = "from_user"
        changed_search_without_radius = changed_search
        radius_before_update = pending.get("search_radius_m") or 1000
        radius_note = None
        ignored_radius = False
        ignored_expansion = False
        expanded_this_update = False
        if "search_radius_m" in args:
            radius = args["search_radius_m"]
            if type(radius) is not int or not 100 <= radius <= 5000:
                raise ValueError("invalid search radius")
            previous_radius = pending.get("search_radius_m") or 1000
            from agents.distance_evidence import radius_supported_by_message
            supported_radius = radius_supported_by_message(self.query, radius, previous_radius)
            if radius != previous_radius and not supported_radius:
                ignored_radius = True
                radius_note = "本轮没有支持该数值的用户距离条件，已保留原半径；需要扩大可使用 expand_search_radius。"
            else:
                changed_search |= previous_radius != radius
                pending["search_radius_m"] = radius
                if supported_radius:
                    intent["search_radius_explicit"] = True
        if ("search_radius_m" not in args or ignored_radius) and args.get("expand_search_radius") is True:
            from agents.distance_evidence import expansion_supported_by_message
            if not expansion_supported_by_message(self.query):
                ignored_expansion = True
                radius_note = "本轮没有明确要求扩大范围，已保留原搜索半径。"
            elif not self._expanded_this_turn:
                radius = min(5000, (pending.get("search_radius_m") or 1000) * 2)
                changed_search |= pending.get("search_radius_m") != radius
                pending["search_radius_m"] = radius
                intent["search_radius_explicit"] = True
                expanded_this_update = True
        if "constraints" in args:
            previous_constraints = HardConstraints.model_validate(pending.get("constraints") or {}).model_dump()
            incoming = dict(args["constraints"])
            if (explicit_radius is not None and not separate_equal_distance
                    and incoming.get("max_distance_m") == explicit_radius):
                incoming.pop("max_distance_m")
            if "slope" in incoming:
                slope = incoming.pop("slope")
                if slope not in {"avoid", "normal", "prefer"}:
                    raise ValueError("invalid slope preference")
                if slope == "avoid":
                    incoming["avoid_slope"] = True
            for field in ("distance", "scenery"):
                incoming.pop(field, None)
            for field, value in list(incoming.items()):
                if previous_constraints.get(field) not in (None, False, []) and value in (None, False, []):
                    incoming.pop(field)
            pending["constraints"] = HardConstraints.model_validate({**previous_constraints, **incoming}).model_dump()
            constraints_changed = pending["constraints"] != previous_constraints
            changed_search |= constraints_changed
            changed_search_without_radius |= constraints_changed
        for field in args.get("clear_fields", []):
            if field in {"budget_per_person", "limit", "requirements", "area", "keywords", "exclude_terms",
                         "web_page_read_requested", "tour_stops", "tour_scope"}:
                if field not in args:
                    intent.pop(field, None)
                changed_search = True
                changed_search_without_radius = True
            elif field in {"start", "search_anchor", "strategy", "activities", "subcategories"}:
                if field not in args:
                    pending.pop(field, None)
                changed_search = True
                changed_search_without_radius = True
            elif field == "selected_poi_id":
                pending.pop("selected_poi_id", None)
                pending.pop("selected_place", None)
            elif isinstance(field, str) and field.startswith("constraints."):
                constraint = field.removeprefix("constraints.")
                defaults = HardConstraints().model_dump()
                if constraint not in defaults:
                    raise ValueError("unsupported clear field")
                current = HardConstraints.model_validate(pending.get("constraints") or {}).model_dump()
                current[constraint] = defaults[constraint]
                pending["constraints"] = current
                changed_search = True
                changed_search_without_radius = True
            else:
                raise ValueError("unsupported clear field")
        candidate_action = args.get("candidate_action") or ("recommend" if changed_search else self.turn_action)
        if candidate_action not in {"recommend", "next_batch"}:
            raise ValueError("invalid candidate_action")
        normalized_candidate_action = candidate_action == "next_batch" and (relation != "continue" or changed_search)
        if normalized_candidate_action:
            candidate_action = "recommend"
        next_known, next_last_search = self.known, self.last_search
        if changed_search:
            self._empty_search_keys.clear()
            self._map_search_completed = False
            selected_id = pending.get("selected_poi_id")
            selected_place = self.known.get(selected_id) if selected_id else None
            if selected_place:
                pending["selected_place"] = deepcopy(selected_place)
            expanding_same_search = (relation != "new" and not changed_search_without_radius
                                     and (pending.get("search_radius_m") or 1000) > radius_before_update)
            if not expanding_same_search:
                intent["shown_ids"] = []
            pending.update({"candidate_poi_ids": [], "candidate_places": [], "candidate_pool": [],
                            "result_dependencies": None})
            keep_ids = set(self._resolved_this_turn)
            if selected_id:
                keep_ids.add(selected_id)
            for endpoint in (pending.get("start"), pending.get("search_anchor")):
                if isinstance(endpoint, dict) and endpoint.get("poi_id"):
                    keep_ids.add(endpoint["poi_id"])
            next_known = {identifier: deepcopy(self.known[identifier])
                          for identifier in keep_ids if identifier in self.known}
            next_last_search = {}
        pending["semantic_request"] = intent
        if candidate_action == "next_batch":
            from spatial.poi import same_display_place
            shown = set(intent.get("shown_ids") or [])
            pool = pending.get("candidate_pool") or []
            shown_records = [row for row in pool if isinstance(row, dict) and row.get("poi_id") in shown]
            remaining = [row for row in pool if isinstance(row, dict) and row.get("poi_id") not in shown
                         and not any(same_display_place(row, previous) for previous in shown_records)]
            next_last_search = {"status": "ok", "online_status": "ok",
                                "candidates": deepcopy(remaining), "count": len(remaining)}
        PlaceRequestState.model_validate(pending)
        grant_scope_recheck = (changed_search and self.place_search_count >= self._place_search_limit
                               and not self._scope_recheck_granted)
        changed_fields = ["place_request." + ("search_radius_m" if key == "expand_search_radius" else key)
                          for key in args if key not in {"relation", "task_id", "task_revision", "clear_fields", "candidate_action"}
                          and not (key == "search_radius_m" and ignored_radius)
                          and not (key == "search_anchor" and ignored_anchor)
                          and not (key == "expand_search_radius" and ignored_expansion)]
        changed_fields.extend("place_request." + field for field in args.get("clear_fields", []) if isinstance(field, str))
        if pruned_requirement and "place_request.requirements" not in changed_fields:
            changed_fields.append("place_request.requirements")
        if self.bind_task and self.bound_task is None:
            self.bound_task = self.bind_task(relation=relation, task_id=args.get("task_id"),
                                             base_revision=args.get("task_revision"), place_request=pending,
                                             changed_fields=changed_fields)
        if expanded_this_update:
            self._expanded_this_turn = True
        if grant_scope_recheck:
            self._place_search_limit += 1
            self._scope_recheck_granted = True
        self.known, self.last_search = next_known, next_last_search
        self.turn_action = candidate_action
        if candidate_action == "next_batch":
            self.place_search_count = 2
        self.modified_fields.update(changed_fields)
        self.pending, self.intent = pending, intent
        compact_pending = deepcopy(pending)
        compact_pending.pop("candidate_pool", None)
        response = {"accepted_task": _compact(compact_pending)}
        if candidate_action == "recommend" and changed_search:
            response["search_required"] = True
        if radius_note:
            response["radius_note"] = radius_note
        if anchor_note:
            response["search_anchor_note"] = anchor_note
        if self.bound_task:
            response.update({"task_id": self.bound_task["task_id"], "task_revision": self.bound_task["task_revision"]})
        if candidate_action == "next_batch":
            response["candidate_page"] = json.loads(_model_tool_content("search_poi_candidates", self.last_search))
            page = response["candidate_page"]
            page["offset"] = 0
            page["next_offset"] = len(page.get("candidates") or [])
            page["exhausted"] = page["next_offset"] >= page["count"]
        return response

    def _endpoint(self, ref, *, allow_pending=True):
        """Never accept invented coordinates or use ambiguous provider identities."""
        from agents import tools
        if not isinstance(ref, dict):
            raise ValueError("missing_endpoint")
        if ref.get("type") == "coord":
            allowed = list(self.original_coords)
            states = [self.previous_route or {}]
            if allow_pending:
                states.append(self.pending)
            for state in states:
                allowed.extend(item for item in (state.get("start"), state.get("end")) if isinstance(item, dict))
            if not any(all(ref.get(key) == point.get(key) for key in ("lng", "lat")) for point in allowed):
                raise ValueError("untrusted_coordinates: 请解析实际地点或使用用户地图选点")
            return ref
        identifier = ref.get("poi_id")
        if identifier and str(identifier).startswith("amap:"):
            known = self.known.get(identifier)
            if not known:
                raise ValueError("unknown_place: 先检索或解析该门店")
            refreshed = tools.amap_poi.lookup_by_id(identifier)
            matched = tools.amap_poi.choose_match(known.get("name"), [refreshed] if refreshed else [], known=known)
            if not matched or not tools.filter_reported_absent_provider_results([matched]):
                raise ValueError("place_changed: 门店身份或位置未能重新核对，请重新查询")
            self.known[identifier] = {**known, **matched}
            return {"type": "poi", "poi_id": identifier, "name": matched["name"]}
        if identifier:
            poi = next((item for item in tools.load_pois() if item.get("id") == identifier), None)
            if not poi:
                raise ValueError("unknown_place_id: 请使用已返回的地点 ID，或只填地点名称后调用 resolve_poi")
            if ref.get("name"):
                resolved, _ = tools.find_poi_ambiguous(ref["name"])
                if resolved and resolved.get("id") != identifier:
                    raise ValueError("place_identity_mismatch: 地点名称与 ID 不一致")
            return {"type": "poi", "poi_id": identifier, "name": poi.get("name")}
        return ref

    def spatial(self, name, args):
        from agents import tools
        args = deepcopy(args)
        if name == "ask_user" and _unconfirmed_previous_end_as_current(
                self.query, args.get("question"), self.pending.get("start"),
                self.intent.get("route_to_modify") or self.previous_route, tools):
            return {"error": "unconfirmed_current_position", "message":
                    "用户尚未确认到达原终点，不能把它当作当前位置。请保留原路线；如需要新位置，请先询问。"}
        if name in {"search_poi_candidates", "find_reachable_places"}:
            if self.intent.get("operation") == "place_search" and not self._prior_ball_activities:
                ambiguous = _ambiguous_ball_request(self.query)
                if ambiguous:
                    self.pending["activities"] = []
                    self.pending["pending_slot"] = "activity"
                    self.clarify = {"question": ambiguous["question"], "options": ambiguous.get("options") or [],
                                    "slot": "sport"}
                    self.modified_fields.update({"place_request.activities", "place_request.pending_slot"})
                    return {**self.clarify, "needs_input": True}
            if self.place_service_error:
                return {"error": "place_service_unavailable", "message": "地点服务本轮已报告故障，请稍后重试。"}
            if (self.place_search_count >= 4 and self.intent.get("operation") == "place_search"
                    and self.last_search.get("count") == 0 and self.last_search.get("candidates") == []
                    and self.last_search.get("online_status") == "ok" and not self.last_search.get("error")
                    and not self.last_search.get("has_more")
                    and not self.pending.get("candidate_pool")):
                return {"error": "empty_search_retry_limit", "message":
                        "当前条件已多次查询且暂无候选；请如实说明，或继续处理其他尚未完成的要求。"}
            if (name == "search_poi_candidates" and args.get("area") and self.intent.get("area")
                    and args["area"] != self.intent["area"]):
                return {"error": "task_update_required", "message": "检索区域与已接受任务不一致，请核对后重试。"}
            if self.place_search_count >= self._place_search_limit:
                return {"error": "place_search_limit",
                        "message": "本轮地点检索次数已达上限，请根据已有结果回答。"}
        context = {"query": self.query, "uid": self.uid, "_model_dialogue": True,
                   "_semantic_request": self.intent, "_known_places": self.known}
        if name in {"search_poi_candidates", "find_reachable_places"}:
            for field in ("start", "search_anchor", "mode", "activities", "subcategories"):
                if self.pending.get(field) and field not in args:
                    args[field] = deepcopy(self.pending[field])
            if name == "search_poi_candidates" and args.get("search_anchor"):
                accepted_center = self.pending.get("search_anchor") or self.pending.get("start")
                if accepted_center and not _same_accepted_location(accepted_center, args["search_anchor"], tools):
                    return {"error": "task_update_required", "message": "请先更新任务中的搜索中心。"}
            for field in ("start", "search_anchor"):
                if args.get(field):
                    args[field] = self._endpoint(args[field])
            if "keyword" not in args and self.intent.get("keywords"):
                args["keyword"] = " ".join(self.intent["keywords"])
            category_error = _normalize_coarse_place_type(args, tools)
            if category_error:
                return category_error
            _expand_unqualified_sport_categories(args, self.pending, self.query)
            args.setdefault("exclude_terms", self.intent.get("exclude_terms", []))
            args.setdefault("limit", 20)
            if name == "search_poi_candidates" and self.pending.get("search_radius_m"):
                radius = self.pending["search_radius_m"]
                duration = (self.pending.get("constraints") or {}).get("max_duration_s")
                anchor, actual_start = self.pending.get("search_anchor"), self.pending.get("start")
                same_center = not anchor or (actual_start and _same_accepted_location(anchor, actual_start, tools))
                time_only = (duration and same_center
                             and not _search_radius_is_explicit(self.pending, self.intent, self.query))
                if time_only:
                    mode = args.get("mode") or self.pending.get("mode", "walk")
                    distance_bound = math.ceil(float(duration) * tools.MODE_SPEEDS_KMH.get(mode, 4.5) * 1000 / 3600)
                    if distance_bound > 10000:
                        return {"error": "reachability_incomplete", "message": "时间范围过大，暂无法完整检索，请缩小范围。"}
                    args["max_distance_m"] = max(100, distance_bound)
                else:
                    if args.get("max_distance_m") not in (None, radius):
                        return {"error": "task_update_required", "message": "请先更新任务中的搜索半径。"}
                    args["max_distance_m"] = radius
            if name == "find_reachable_places" and self.intent.get("operation") == "place_search":
                anchor, start = self.pending.get("search_anchor"), args.get("start")
                accepted_start = self.pending.get("start")
                if accepted_start and start and not _same_accepted_location(accepted_start, start, tools):
                    return {"error": "task_update_required", "message":
                            "可达时间须从已接受的实际出发点计算；搜索中心不同时请使用 search_poi_candidates。"}
                if anchor and start and not _same_accepted_location(anchor, start, tools):
                    return {"error": "search_center_mismatch", "message":
                            "搜索中心与出发点不同，请使用 search_poi_candidates 按已接受的搜索中心检索。"}
                # A model-invented time budget must not silently widen the
                # accepted nearby-search radius. A saved user time limit is
                # a separate, explicit reachability request.
                duration = (self.pending.get("constraints") or {}).get("max_duration_s")
                radius = self.pending.get("search_radius_m")
                if radius and not duration:
                    proposed = args.get("max_distance_m")
                    if not isinstance(proposed, (int, float)) or isinstance(proposed, bool):
                        proposed = radius
                    args["max_distance_m"] = min(radius, proposed)
                    args.pop("max_time_min", None)
                elif radius and duration:
                    args.pop("max_distance_m", None)
                    args["max_time_min"] = float(duration) / 60.0
                    if _search_radius_is_explicit(self.pending, self.intent, self.query):
                        context["_search_scope_distance_m"] = radius
            if self.pending.get("constraints"):
                args["constraints"] = self._constraints(args.get("constraints"))
        route_tools = {"plan_route", "plan_via_route", "plan_itinerary", "plan_multimodal_route", "plan_tour", "compare_routes"}
        if name in route_tools:
            if name == "plan_tour":
                stops = self.intent.get("tour_stops")
                if not stops and isinstance(args.get("poi_names"), list):
                    names = args["poi_names"]
                    if len(names) <= 8 and all(isinstance(p, str) and p.strip() and len(p) <= 160 for p in names):
                        stops = list(dict.fromkeys(p.strip() for p in names))
                        if stops:
                            self.intent["tour_stops"] = stops
                if stops:
                    args["poi_names"] = deepcopy(stops)
            personal_itinerary = (self.intent.get("operation") == "itinerary"
                                  and self.intent.get("tour_scope") != "independent"
                                  and name != "compare_routes")
            if personal_itinerary:
                accepted_start = self.pending.get("start") or (
                    {"type": "coord", **self.coord_start} if self.coord_start else None)
                if not accepted_start:
                    self.pending["pending_slot"] = "start"
                    if args.get("start"):
                        message = "请先用 update_task 保存用户明确给出的出发点；若未给出，询问用户，不要把游览地点当作出发点。"
                        self.route_tool_failure = {"code": "task_update_required", "message": message}
                        return {"error": "task_update_required", "message": message}
                    question = "你们从哪里出发？可以告诉我地点，或在地图上选择起点。"
                    self.clarify = {"question": question, "options": [], "slot": "start"}
                    return {"question": question, "slot": "start", "needs_input": True}
                args["start"] = deepcopy(accepted_start)
                self.intent["tour_origin_confirmed"] = True
                self.intent.pop("tour_origin_unconfirmed", None)
            accepted_start = self.pending.get("start")
            supplied_start = args.get("start")
            if (accepted_start and supplied_start and self.intent.get("tour_scope") != "independent"
                    and not _same_accepted_location(accepted_start, supplied_start, tools)):
                message = "路线起点与已接受任务不一致。若用户改变了出发点，请先更新任务；否则保留原起点。"
                self.route_tool_failure = {"code": "task_update_required", "message": message}
                return {"error": "task_update_required", "message": message}
            if "start" not in args:
                args["start"] = self.pending.get("start") or ({"type": "coord", **self.coord_start} if self.coord_start else None)
            if "end" not in args and self.coord_end:
                args["end"] = {"type": "coord", **self.coord_end}
            for field in ("start", "end"):
                if args.get(field):
                    args[field] = self._endpoint(args[field])
            if name == "plan_route" and self.intent.get("operation") == "route":
                for field, slot in (("start", "start"), ("end", "destination")):
                    ref = args.get(field)
                    prior = (self.previous_route or {}).get(field) if isinstance(self.previous_route, dict) else None
                    selected = self.pending.get("selected_place") if field == "end" else None
                    if ((prior and _same_accepted_location(prior, ref, tools)) or
                            (selected and _same_accepted_location(selected, ref, tools))):
                        continue
                    ambiguous = _ambiguous_route_reference(ref, self.query, tools)
                    if ambiguous:
                        term, options = ambiguous
                        question = "“" + term + "”可能指多个地点，你想选哪一个？"
                        self.clarify = {"question": question, "options": options, "slot": slot}
                        self.pending["pending_slot"] = slot
                        if slot == "start":
                            self.pending.pop("start", None)
                        return {"question": question, "options": options, "slot": slot, "needs_input": True}
            if self.intent.get("operation") == "place_search" and name in {"plan_route", "plan_via_route"}:
                end = args.get("end")
                selected = self.pending.get("selected_place")
                scoped = self.pending.get("candidate_pool") or []
                if end and not (selected and _same_accepted_location(selected, end, tools)) and not any(
                        _same_accepted_location(place, end, tools) for place in scoped if isinstance(place, dict)):
                    return {"error": "candidate_scope_unverified", "message":
                            "该地点尚未通过当前范围的候选检索；请先展示核验过的候选，或更新用户指定的路线任务。"}
            args.setdefault("mode", self.pending.get("mode", "walk"))
            args["constraints"] = self._constraints(args.get("constraints"))
            if self.pending.get("strategy"):
                args.setdefault("strategy", self.pending["strategy"])
            # Named online identities must resolve through this run's server-side
            # evidence, never through model coordinates or a stale local store.
            context["_known_places"] = self.known
        needs_route_snapshot = (name in route_tools or name == "find_reachable_places"
                               or name == "search_poi_candidates"
                               and bool(args.get("start") or args.get("search_anchor")))
        if needs_route_snapshot:
            if self.spatial_snapshot is None:
                snapshot, error = tools._prepare_route_context({})
                if error:
                    return error
                self.spatial_snapshot = snapshot
            context.update(self.spatial_snapshot)
        search_key = None
        if name in {"search_poi_candidates", "find_reachable_places"}:
            search_args = {key: value for key, value in args.items() if key != "limit"}
            semantics = {key: self.intent.get(key) for key in
                         ("operation", "area", "keywords", "exclude_terms", "requirements", "shown_ids")}
            search_key = json.dumps([name, search_args, semantics, context.get("_search_scope_distance_m")],
                                    ensure_ascii=False, sort_keys=True, default=str)
            if search_key in self._empty_search_keys:
                return {"error": "duplicate_empty_search", "message":
                        "相同条件的查询已完成且没有候选；请如实说明，或处理其他尚未完成的要求。"}
            self.place_search_count += 1
        result, artifact = self.execute_tool(name, args, context)
        if not isinstance(result, dict):
            raise ValueError("invalid_tool_result")
        artifact = artifact or {}
        if (name in route_tools and result.get("error")
                and (name != "compare_routes" or self.route_tool_failure is None)):
            self.route_tool_failure = {"code": str(result["error"]),
                                       "message": str(result.get("message") or "路线服务未完成计算。")}
        observed = list(artifact.get("candidates") or result.get("candidates") or [])
        if isinstance(result.get("poi"), dict):
            observed.append(result["poi"])
        for row in observed:
            identifier = row.get("poi_id") or row.get("id")
            if identifier:
                row = {**row, "poi_id": identifier}
                self.known[identifier] = row
                if name == "resolve_poi" and not result.get("error"):
                    self._resolved_this_turn[identifier] = deepcopy(row)
        if name in {"search_poi_candidates", "find_reachable_places"}:
            self.last_search = result
            if (search_key and not result.get("error") and result.get("count") == 0
                    and result.get("candidates") == [] and not result.get("has_more")
                    and result.get("online_status") in (None, "ok", "not_requested")):
                self._empty_search_keys.add(search_key)
            if result.get("candidates") and result.get("data_version") and result.get("road_condition_version"):
                from agents.result_maintenance import record_candidate_snapshot
                record_candidate_snapshot(self.pending, result)
            provider_status = result.get("online_status")
            if provider_status == "ok":
                self._map_search_completed = True
            if (provider_status and provider_status not in {"ok", "not_requested", "query_required", "center_unavailable"}
                    and not result.get("candidates")):
                self.place_service_error = provider_status
            if self.intent.get("operation") == "place_search" and result.get("candidates"):
                pool = {row["poi_id"]: deepcopy(row) for row in self.pending.get("candidate_pool") or []
                        if isinstance(row, dict) and row.get("poi_id")}
                for row in result.get("candidates") or []:
                    if isinstance(row, dict) and row.get("poi_id"):
                        pool[row["poi_id"]] = deepcopy(row)
                self.pending["candidate_pool"] = list(pool.values())[:20]
            if name == "search_poi_candidates":
                if result.get("search_area"):
                    self.intent["area"] = result["search_area"]
                if result.get("search_keyword"):
                    self.intent["last_search_keyword"] = result["search_keyword"]
        route = artifact.get("route")
        if isinstance(route, dict):
            if (route.get("route_verification") or {}).get("status") != "pass":
                return {"error": "route_verification_failed", "message": "路线未通过核验，不可展示，请检查参数与约束。"}
            if self.intent.get("operation") == "itinerary" and self.intent.get("tour_stops"):
                missing = _missing_tour_stops(route, self.intent["tour_stops"], tools)
                if missing:
                    message = "路线尚未包含这些已保存的游览地点：" + "、".join(missing) + "。请保留原起点并重新规划完整路线。"
                    self.route_tool_failure = {"code": "itinerary_stops_missing", "message": message}
                    return {"error": "itinerary_stops_missing", "message": message, "missing_stops": missing}
            self.route, self.route_kind = route, artifact.get("route_kind") or "direct"
            self.route_is_independent_tour = (self.intent.get("tour_scope") == "independent"
                                              and not self.pending.get("start") and not self.coord_start)
            self.route_tool_failure = None
            state = route.get("route_state") or {}
            if not self.route_is_independent_tour:
                self.pending["start"] = deepcopy(state.get("start") or args.get("start"))
            end_id = (args.get("end") or {}).get("poi_id")
            if name in {"plan_route", "plan_via_route"} and end_id in self.known:
                self.pending["selected_poi_id"] = end_id
                self.pending["selected_place"] = deepcopy(self.known[end_id])
            self.intent["route_to_modify"] = deepcopy(state)
        if isinstance(artifact.get("route_comparison"), dict):
            self.comparison = artifact["route_comparison"]
            if not result.get("error"):
                self.route_tool_failure = None
        if result.get("reason_code") == "graph_no_path":
            self.no_path = result
        if name == "get_weather" and not result.get("error"):
            self.weather = deepcopy(result)
        if artifact.get("clarify"):
            self.clarify = artifact["clarify"]
            if (self.clarify.get("slot") in {"sport", "activity"} and not self._prior_ball_activities
                    and _ambiguous_ball_request(self.query)):
                self.pending["activities"] = []
                self.pending["pending_slot"] = "activity"
                self.modified_fields.update({"place_request.activities", "place_request.pending_slot"})
            if self.intent.get("operation") == "itinerary" and self.clarify.get("slot") == "start":
                self.pending["pending_slot"] = "start"
        if artifact.get("suggestions"):
            self.suggestions = artifact["suggestions"]
        return result

    def _constraints(self, provided=None):
        constraints = dict(provided or {})
        saved = self.pending.get("constraints") or {}
        for key, value in saved.items():
            if value not in (None, False, []):
                constraints["slope" if key == "avoid_slope" else key] = "avoid" if key == "avoid_slope" else value
        return constraints

    def browse_candidates(self, args):
        if self.bind_task and self.bound_task is None:
            task_id = args.get("task_id")
            task_revision = args.get("task_revision")
            chosen = self.available_tasks.get(str(task_id or ""))
            if (not chosen or type(task_revision) is not int
                    or chosen.get("revision") != task_revision):
                raise ValueError("task_version_invalid: 请选择服务器提供的候选任务及当前版本")
            state = chosen.get("state") or {}
            place_request = state.get("place_request") or {}
            operation = (place_request.get("semantic_request") or {}).get("operation")
            if not isinstance(operation, str) or operation not in _OPERATIONS:
                raise ValueError("candidate_task_requires_explicit_update: 请先用update_task选择任务")
            self.update({"relation": "continue", "task_id": task_id,
                         "task_revision": task_revision, "operation": operation,
                         "candidate_action": "next_batch"})
        offset = args.get("offset", 0)
        limit = args.get("limit", 12)
        include_shown = args.get("include_shown", False)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 12:
            raise ValueError("invalid candidate page")
        if type(include_shown) is not bool:
            raise ValueError("include_shown must be a boolean")
        shown = set(self.intent.get("shown_ids") or [])
        all_rows = [row for row in self.pending.get("candidate_pool") or [] if isinstance(row, dict)]
        from spatial.poi import same_display_place
        shown_records = [row for row in all_rows if row.get("poi_id") in shown]
        pool = [row for row in all_rows if include_shown or (row.get("poi_id") not in shown
                and not any(same_display_place(row, previous) for previous in shown_records))]
        page = pool[offset:offset + limit]
        return {"status": "ok", "candidates": deepcopy(page), "count": len(pool),
                "offset": offset, "next_offset": offset + len(page),
                "exhausted": offset + len(page) >= len(pool)}

    def present(self, args):
        from agents.result_maintenance import record_candidate_snapshot
        places = args.get("places")
        max_recommended = min(self.intent.get("limit", 4), 4)
        if not isinstance(places, list) or not 1 <= len(places) <= max_recommended:
            raise ValueError("recommendation_count: 必须遵守用户限定的数量")
        ids = [item.get("poi_id") for item in places if isinstance(item, dict)]
        if len(ids) != len(places) or len(ids) != len(set(ids)):
            raise ValueError("duplicate_place")
        shown = set(self.intent.get("shown_ids", []))
        scoped = (self.intent.get("operation") == "place_search"
                  and bool(self.pending.get("search_anchor") or self.pending.get("start")))
        scoped_records = {row.get("poi_id"): row for row in self.pending.get("candidate_pool") or []
                          if isinstance(row, dict) and row.get("poi_id")}
        from spatial.poi import same_display_place
        shown_records = [row for key, row in scoped_records.items() if key in shown]
        chosen_records = []
        for identifier in ids:
            if identifier not in self.known:
                raise ValueError("unknown_place: 只能展示已查得地点")
            if identifier in shown:
                raise ValueError("already_shown: 换批不得重复推荐，耗尽时请明确说明")
            if scoped and identifier not in scoped_records:
                raise ValueError("candidate_scope_unverified: 请先按当前搜索中心和范围检索，再推荐该地点")
            row = scoped_records.get(identifier) or self.known[identifier]
            if any(same_display_place(row, previous) for previous in shown_records):
                raise ValueError("already_shown: 换批不得重复推荐，耗尽时请明确说明")
            if any(same_display_place(row, previous) for previous in chosen_records):
                raise ValueError("duplicate_place: 同一地点不能重复展示")
            chosen_records.append(row)
        introduction = str(args.get("introduction") or "").strip()
        if len(introduction) > 500:
            raise ValueError("recommendation_intro_too_long: 请简明回答，用户要求详细介绍时再展开")
        prose = introduction + " ".join(str(item.get("reason", "")) for item in places)
        if any(row.get("name") and row["name"] in prose for key, row in self.known.items() if key not in ids):
            raise ValueError("unselected_place_in_text: 文字不可另推卡片之外的地点")
        candidates = []
        for item in places:
            row = deepcopy(scoped_records.get(item["poi_id"]) if scoped else self.known[item["poi_id"]])
            reason = str(item.get("reason") or "").strip()
            if len(reason) > 36:
                raise ValueError("recommendation_reason_too_long: 请用一句短话概括")
            source_ids = item.get("source_ids") or []
            if any(source_id not in self.sources for source_id in source_ids):
                raise ValueError("unknown_source")
            if reason and not _reason_is_supported(
                    reason, row, [self.sources[source_id] for source_id in source_ids]):
                reason = ""
            row.update({"recommendation_reason": reason, "source_ids": source_ids})
            candidates.append(row)
        self.candidates = candidates
        self.pending.update({"pending_slot": "selection", "candidate_poi_ids": ids,
                             "candidate_places": deepcopy(candidates)})
        self.intent["shown_ids"] = list(dict.fromkeys([*self.intent.get("shown_ids", []), *ids]))
        if self.last_search.get("data_version"):
            record_candidate_snapshot(self.pending, self.last_search)
        return introduction

    def result(self, message, *, failed=False):
        from agents.planner import _sanitize_message
        if failed and self.candidates:
            message = "结果已保留，可以选择地点或换一批。"
        elif failed and self.route:
            message = "路线已保留，请稍后重试。"
        question = (self.clarify or {}).get("question") or str(message or "")
        if (not failed and not self.candidates and not self.route and not self._prior_ball_activities
                and self.intent.get("operation") == "place_search" and _ambiguous_ball_request(self.query)
                and re.search(r"(?:哪|什么).{0,5}球|球.{0,5}(?:哪|什么)", question)):
            if not self.clarify:
                self.clarify = {"question": question, "options": [], "slot": "sport"}
            else:
                self.clarify.setdefault("slot", "sport")
            self.pending["activities"] = []
            self.pending["pending_slot"] = "activity"
            self.modified_fields.update({"place_request.activities", "place_request.pending_slot"})
        if self.route_is_independent_tour and not ("参考" in message and "出发" in message):
            message = "这是景点间的参考路线，未包含你的出发路段。" + str(message or "")
        self.intent["sources"] = list(self.sources.values())[-12:]
        self.pending["semantic_request"] = self.intent
        awaiting_selection = bool(self.candidates and self.intent.get("operation") == "place_search"
                                  and self.pending.get("pending_slot") == "selection")
        kind = ("candidates" if awaiting_selection else "route" if self.route else
                "route_comparison" if self.comparison else "candidates" if self.candidates else
                "clarify" if self.clarify else "chat")
        result = {"response_kind": kind, "message": _sanitize_message(message), "turns": self.turns,
                  "route": None if awaiting_selection else self.route, "route_kind": self.route_kind, "candidates": self.candidates,
                  "clarify": self.clarify, "suggestions": self.suggestions, "route_comparison": self.comparison,
                  "weather": self.weather,
                  "place_request": self.pending, "sources": list(self.sources.values()),
                  "updated_task_fields": sorted(self.modified_fields),
                  "model_status": "unavailable" if failed else "ok", "model_usage": dict(self.usage)}
        if self.route and not awaiting_selection:
            result["completed_place_request"] = deepcopy(self.pending)
            result["place_request"] = None
        if failed:
            result["execution_status"] = "partial" if (self.route or self.candidates
                or self.comparison and not self.route_tool_failure) else "failed"
        return result

    def run(self):
        from agents.dialogue_loop import run_dialogue_loop
        return run_dialogue_loop(self)


def run_dialogue(query, **kwargs):
    return Dialogue(query, **kwargs).run()
