"""Model-led dialogue, backed by the existing accepted task and spatial tools.

The model owns interpretation and recommendation. This module owns identity,
delivery alignment and bounded execution; it does not classify user sentences.
"""
from __future__ import annotations

from copy import deepcopy
import json
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


_COMMON_DIALOGUE_TOOLS = {
    "update_task", "present_places", "resolve_poi", "get_place_details",
    "search_web", "read_web_page", "ask_user", "suggest_followup",
}
_PLACE_SEARCH_TOOLS = _COMMON_DIALOGUE_TOOLS | {
    "search_poi_candidates", "find_reachable_places",
}
_ROUTE_DIALOGUE_TOOLS = _COMMON_DIALOGUE_TOOLS | {
    "find_reachable_places", "plan_route", "plan_via_route", "get_weather",
    "list_road_conditions",
}
_OPERATION_TOOL_NAMES = {
    "place_search": _PLACE_SEARCH_TOOLS,
    "route": _ROUTE_DIALOGUE_TOOLS,
    "itinerary": _ROUTE_DIALOGUE_TOOLS | {
        "search_poi_candidates", "plan_itinerary", "plan_tour", "plan_multimodal_route",
    },
    "route_comparison": _ROUTE_DIALOGUE_TOOLS | {"compare_routes"},
    "weather": _COMMON_DIALOGUE_TOOLS | {"get_weather"},
    "campus_answer": _COMMON_DIALOGUE_TOOLS | {"search_poi_candidates"},
}


def tool_schemas(operation=None, *, allow_place_search=True, ready_to_present=False,
                 web_research_requested=False, web_search_available=True,
                 web_fetch_available=True, web_map_search_pending=False,
                 selected_place=False, route_preplanned=False):
    from agents.tools import TOOL_SCHEMAS, _ENDPOINT_SCHEMA, _PREFERENCE_SCHEMA
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
        if item["function"]["name"] == "resolve_poi":
            item["function"]["description"] = ("核对具体地名并查询校园/在线目录记录；多分店返回候选，不擅自选定。"
                "名称和坐标匹配不代表现场存在、营业或入口可通行。")
    schemas.extend([
        _function("update_task", "保存你理解的完整需求或本轮修订。只更新传入字段；继续任务保留其余条件。",
            {"relation": {"type": "string", "enum": ["new", "continue", "resume_route"]},
             "operation": {"type": "string", "enum": list(_OPERATION_TOOL_NAMES),
                           "description": "本轮语义任务类型；按主要用户目标选择"},
             "goal": {"type": "string"},
             "start": {**_ENDPOINT_SCHEMA, "description": "实际出发地点，不能用搜索中心代替"},
             "search_anchor": {**_ENDPOINT_SCHEMA, "description": "附近查询参照点，可与起点不同"},
             "mode": {"type": "string", "enum": ["walk", "bike", "drive"]},
             "strategy": {"type": "string", "enum": ["shortest", "recommended", "flat", "scenery"]},
             "requirements": {"type": "array", "items": {"type": "string"}, "description": "完整保留否定、用途及其他条件"},
             "area": {"type": "string", "description": "检索区域文字，例如梅园、桂园或武汉大学"},
             "keywords": {"type": "array", "items": {"type": "string"}},
             "exclude_terms": {"type": "array", "items": {"type": "string"}},
             "web_research_requested": {"type": "boolean", "description": "用户请求商户/餐饮推荐、攻略或地图目录可能不全的地点时设为true；地图只能匹配目录中的名称与坐标记录，不能证明现场存在或营业；网页用于发现候选和补充有来源的描述"},
             "web_page_read_requested": {"type": "boolean", "description": "仅当用户明确需要核对门店级详情（如人均、营业时间或近期评价）时设为true；一般附近推荐只需搜索摘要和地图匹配，不必读取网页正文"},
             "candidate_action": {"type": "string", "enum": ["recommend", "next_batch"],
                                  "description": "用户要求换一批/换一家且继续当前地点条件时设为next_batch；筛选条件变化时设为recommend"},
             "activities": {"type": "array", "items": {"type": "string"}},
             "subcategories": {"type": "array", "items": {"type": "string"}},
             "budget_per_person": {"type": "number", "exclusiveMinimum": 0},
             "limit": {"type": "integer", "minimum": 1, "maximum": 20},
             "constraints": _PREFERENCE_SCHEMA["constraints"],
             "clear_fields": {"type": "array", "items": {"type": "string"},
                              "description": "仅在用户明确取消时清除已有字段或条件，例如budget_per_person"}},
            ("relation", "operation")),
        _function("present_places", "提交最终推荐。仅允许已查询的地点ID；顺序由你决定，理由与来源会和卡片一起展示。只能称地图/校园目录中匹配到的候选记录，不能称为实际存在、实地核实或正在营业。",
            {"introduction": {"type": "string"},
             "places": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
                 "type": "object", "properties": {"poi_id": {"type": "string"},
                     "reason": {"type": "string"}, "source_ids": {"type": "array", "items": {"type": "string"}}},
                 "required": ["poi_id", "reason"], "additionalProperties": False}}}, ("introduction", "places")),
        _function("search_web", "检索校园生活/商户资料，返回带来源与日期的文本；不是已核实的地点或坐标。",
                  {"query": {"type": "string"}}, ("query",)),
        _function("read_web_page", "读取本轮搜索返回的某个来源，核对内容、日期与分店。",
                  {"source_id": {"type": "string"}}, ("source_id",)),
    ])
    allowed = _OPERATION_TOOL_NAMES.get(operation) if isinstance(operation, str) else None
    if allowed and operation == "place_search" and web_research_requested and web_search_available:
        allowed = {"search_web"}
    elif allowed and operation == "place_search" and web_research_requested and web_fetch_available:
        allowed = {"read_web_page"}
    elif allowed and operation == "place_search" and web_map_search_pending:
        allowed = {"search_poi_candidates"}
    elif allowed and operation == "place_search" and ready_to_present:
        allowed = {"present_places"}
    elif allowed and operation == "place_search" and not allow_place_search:
        allowed = allowed - {"search_poi_candidates", "find_reachable_places", "resolve_poi"}
    if allowed and selected_place:
        allowed = {"update_task", "plan_route", "plan_via_route", "plan_itinerary",
                   "get_weather", "list_road_conditions"}
    if allowed and route_preplanned:
        allowed = {"get_weather", "list_road_conditions"}
    return [item for item in schemas if item["function"]["name"] in allowed] if allowed else schemas


def _compact(value):
    """Keep complete JSON and useful facts; omit geometry rather than cut JSON."""
    if isinstance(value, list):
        return [_compact(item) for item in value[:20]]
    if isinstance(value, dict):
        omitted = {"recommended", "shortest", "recommended_edge_ids", "shortest_edge_ids",
                   "steps", "pois", "geometry", "route_reuse_proof", "reuse_certificate"}
        return {key: _compact(item) for key, item in value.items() if key not in omitted}
    return value[:7000] if isinstance(value, str) else value


_MODEL_PLACE_FIELDS = (
    "poi_id", "name", "type", "type_name", "category", "subcategory", "campus",
    "aliases", "description", "activities", "activity_basis", "source", "data_source",
    "verification_status", "coordinate_verification_status", "provider_rating",
    "provider_cost", "address", "business_area", "opening_hours", "observed_at",
    "distance_m", "network_distance_m", "distance_basis", "distance_reference_name",
    "estimated_duration_min", "access_link_verified", "step_access_status", "accessibility",
)


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


def _safe_candidate_prose(value):
    """Do not let model prose turn a provider directory match into field verification."""
    text = str(value or "")
    text = re.sub(
        r"(?:地图|高德)(?:地图)?(?:目录)?(?:已|已经)?(?:核对|核验|确认)(?:到|了)?(?:的)?(?:实际|真实)(?:地点|门店|店铺|商户)",
        "地图目录中匹配到的候选记录", text,
    )
    text = re.sub(
        r"(?:这些|上述|以上)(?:店铺|门店|商户|地点)?(?:都|均)?(?:是|为)(?:地图|高德)?(?:目录中)?(?:真实|实际)(?:存在的)?(?:地点|店铺|门店|商户)",
        "这些是目录匹配候选，现场存在情况未核实", text,
    )
    text = re.sub(
        r"(?:已|已经)?(?:实地)?(?:核实|确认)(?:了)?(?:这家|该|上述|这些)?(?:门店|店铺|商户|地点)?(?:的)?(?:真实|确实|实际)?(?:存在|营业|开业)",
        "对应目录记录已匹配，现场存在与营业情况未核实", text,
    )
    return text


def _web_search_query(intent):
    """Build a bounded search query from the model's accepted structured request."""
    parts = ["武汉大学", intent.get("area")]
    keywords = intent.get("keywords")
    if isinstance(keywords, list):
        parts.extend(keywords[:8])
    requirements = intent.get("requirements")
    if isinstance(requirements, list):
        parts.extend(requirements[:4])
    normalized = []
    seen = set()
    for value in parts:
        text = " ".join(str(value or "").split())
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            normalized.append(text)
    return " ".join(normalized)[:128]


_NEXT_BATCH_COMMANDS = {
    "换一批", "换一批吧", "换一家", "换一家吧", "再换一批", "再换一批吧",
    "再换一家", "再换一家吧", "再来一批", "再来一批吧", "再来几个",
    "还有别的吗", "还有其他的吗", "还有其他选择吗", "再推荐几个",
}


class Dialogue:
    def __init__(self, query, *, context=None, history=None, coord_start=None, coord_end=None,
                 coord_waypoints=None, travel_mode=None, uid=None, execute_tool, client_factory, trace):
        self.query, self.context, self.uid = query, context or {}, uid
        self.requested_mode = travel_mode
        self.execute_tool, self.client_factory, self.trace = execute_tool, client_factory, trace
        self.pending = deepcopy(self.context.get("place_request") or {})
        self.pending.setdefault("request_id", str(uuid4()))
        self.pending.setdefault("original_query", query[:2000])
        self.pending.setdefault("mode", travel_mode or "walk")
        self.pending.setdefault("constraints", {})
        self.intent = self.pending.setdefault("semantic_request", {}) or {}
        self.pending["semantic_request"] = self.intent
        self.sources = {item["source_id"]: item for item in self.intent.get("sources", [])
                        if isinstance(item, dict) and item.get("source_id")}
        self.known = {item["poi_id"]: item for item in [
                          *(self.pending.get("candidate_pool") or []),
                          *(self.pending.get("candidate_places") or []),
                      ]
                      if isinstance(item, dict) and item.get("poi_id")}
        selected_id = self.context.get("selected_poi_id")
        accepted_ids = set(self.pending.get("candidate_poi_ids") or [])
        if selected_id in accepted_ids and selected_id in self.known:
            self.pending["selected_poi_id"] = selected_id
            complex_follow_up = (self.intent.get("operation") in {"itinerary", "route_comparison"}
                                 or bool(self.pending.get("completed_legs") or self.pending.get("follow_up_requests")))
            if not complex_follow_up:
                self.intent["operation"] = "route"
                self.direct_route_selection = True
            else:
                self.direct_route_selection = False
        else:
            self.pending.pop("selected_poi_id", None)
            self.direct_route_selection = False
        self.turn_action = "recommend"
        self._intent_resolved = not bool(self.context.get("place_request")) or self.direct_route_selection
        self.route_preplanned = False
        self.fresh = set()
        self.candidates, self.route, self.route_kind, self.comparison = None, None, None, None
        self.weather = None
        self.no_path, self.clarify, self.suggestions = None, None, None
        self.last_search = {}
        self.web_search_count = 0
        self.web_fetch_count = 0
        self.place_search_count = 0
        self._last_task_update_changed_search = False
        self.turns = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.original_coords = [point for point in [coord_start, coord_end, *(coord_waypoints or [])] if point]
        self.coord_start, self.coord_end, self.coord_waypoints = coord_start, coord_end, coord_waypoints
        self.previous_route = self.context.get("previous_route_state")
        self.spatial_snapshot = None
        trusted_task = deepcopy(self.pending)
        if trusted_task.get("candidate_pool"):
            trusted_task["candidate_pool_count"] = len(trusted_task["candidate_pool"])
            trusted_task.pop("candidate_pool", None)
        trusted = {"active_task": _compact(trusted_task), "previous_route": _compact(self.previous_route),
                   "route_context_uncertain": bool(self.context.get("route_context_uncertain")),
                   "map_start": coord_start, "map_end": coord_end, "map_waypoints": coord_waypoints,
                   "selected_poi_id": self.pending.get("selected_poi_id"), "travel_mode": travel_mode}
        if self.pending.get("selected_poi_id"):
            self.messages = [{"role": "system", "content": (
                "本轮有已接受的结构化地点选择。按用户原有出发点直接调用 plan_route，"
                "终点使用 selected_poi_id 对应候选；不要再次搜索或 resolve_poi。") }]
        else:
            self.messages = []
        transcript = [{"role": item["role"], "content": str(item["content"])[:4000]}
                      for item in (history or self.context.get("history") or [])[-12:]
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
        pending = deepcopy(self.pending)
        if relation == "new":
            pending = {"request_id": str(uuid4()), "original_query": self.query[:2000],
                       "mode": self.requested_mode or self.pending.get("mode", "walk"),
                       "constraints": {}, "semantic_request": {}}
        if relation == "resume_route":
            if not self.previous_route or self.context.get("route_context_uncertain"):
                raise ValueError("prior_route_unavailable: 请确认要修改的路线")
            state = self.previous_route
            pending.update({"start": deepcopy(state["start"]), "mode": state["travel_mode"],
                            "constraints": deepcopy(state["hard_constraints"])})
            strategy = state.get("strategy") or {}
            pending["strategy"] = strategy.get("name", "shortest") if isinstance(strategy, dict) else strategy
            pending.setdefault("semantic_request", {})["route_to_modify"] = deepcopy(state)
        intent = pending.setdefault("semantic_request", {}) or {}
        changed_search = relation == "new"
        for field in ("operation", "goal", "requirements", "area", "keywords", "exclude_terms",
                      "web_research_requested", "web_page_read_requested", "budget_per_person", "limit"):
            if field in args:
                value = args[field]
                if field == "operation" and (not isinstance(value, str) or value not in _OPERATION_TOOL_NAMES):
                    raise ValueError("invalid operation")
                if field in {"requirements", "keywords", "exclude_terms"}:
                    if not isinstance(value, list) or len(value) > 30 or any(not isinstance(x, str) or len(x) > 500 for x in value):
                        raise ValueError(f"invalid {field}")
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
        for field in ("start", "search_anchor", "mode", "strategy", "activities", "subcategories"):
            if field in args:
                if field in {"start", "search_anchor"} and args[field] is not None:
                    args[field] = self._endpoint(args[field], allow_pending=relation != "new")
                changed_search |= args[field] != pending.get(field)
                pending[field] = deepcopy(args[field])
        if "constraints" in args:
            previous_constraints = HardConstraints.model_validate(pending.get("constraints") or {}).model_dump()
            incoming = dict(args["constraints"])
            if "slope" in incoming:
                incoming["avoid_slope"] = incoming.pop("slope") == "avoid"
            for field in ("distance", "scenery"):
                incoming.pop(field, None)
            pending["constraints"] = HardConstraints.model_validate({**previous_constraints, **incoming}).model_dump()
            changed_search |= pending["constraints"] != previous_constraints
        for field in args.get("clear_fields", []):
            if field in {"budget_per_person", "limit", "requirements", "area", "keywords", "exclude_terms",
                         "web_page_read_requested"}:
                intent.pop(field, None)
                changed_search = True
            elif field in {"start", "search_anchor", "strategy"}:
                pending.pop(field, None)
                changed_search = True
            else:
                raise ValueError("unsupported clear field")
        self._last_task_update_changed_search = changed_search
        if changed_search:
            intent["shown_ids"] = []
            pending.update({"candidate_poi_ids": [], "candidate_places": [], "candidate_pool": [],
                            "result_dependencies": None})
            pending.pop("selected_poi_id", None)
            self.fresh.clear()
            self.last_search = {}
        pending["semantic_request"] = intent
        candidate_action = args.get("candidate_action", "recommend")
        if candidate_action not in {"recommend", "next_batch"}:
            raise ValueError("invalid candidate_action")
        if candidate_action == "next_batch" and (relation != "continue" or changed_search):
            raise ValueError("next_batch requires unchanged place search conditions")
        self.turn_action = candidate_action
        self._intent_resolved = True
        if candidate_action == "next_batch":
            shown = set(intent.get("shown_ids") or [])
            pool = pending.get("candidate_pool") or []
            remaining = [row for row in pool if row.get("poi_id") not in shown]
            self.last_search = {"status": "ok", "online_status": "ok",
                                "candidates": deepcopy(remaining), "count": len(remaining)}
            self.place_search_count = 2
            if not remaining:
                self.clarify = {
                    "question": "这次检索到的候选已经展示完了。要扩大搜索范围，还是换一种地点类型？",
                    "options": ["扩大搜索范围", "换一种地点类型"], "slot": "selection",
                }
        PlaceRequestState.model_validate(pending)
        self.pending, self.intent = pending, intent
        compact_pending = deepcopy(pending)
        compact_pending.pop("candidate_pool", None)
        return {"accepted_task": _compact(compact_pending)}

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
        return ref

    def spatial(self, name, args):
        from agents import tools
        args = deepcopy(args)
        if name == "resolve_poi" and self.intent.get("operation") == "place_search":
            requested_count = min(self.intent.get("limit", 4), 4)
            if len(self.last_search.get("candidates") or []) >= requested_count:
                return {"error": "candidate_search_complete",
                        "message": "地图已返回足够候选，无需再次核验这些门店；请基于现有候选提交 present_places。"}
        if name in {"search_poi_candidates", "find_reachable_places"}:
            requested_count = min(self.intent.get("limit", 4), 4)
            previous_candidates = self.last_search.get("candidates") or []
            if len(previous_candidates) >= requested_count:
                return {"error": "candidate_search_complete",
                        "message": "已有足够候选，请按现有地点资料筛选并立即提交 present_places。价格或独立门店信息没有来源时可说明未核实。"}
            if self.place_search_count >= 2:
                return {"error": "place_search_limit",
                        "message": "本轮已完成两次地点检索；请基于已有候选回答，未证实的事实说明未知。"}
            self.place_search_count += 1
        context = {"query": self.query, "uid": self.uid, "_model_dialogue": True,
                   "_semantic_request": self.intent, "_known_places": self.known}
        if name in {"search_poi_candidates", "find_reachable_places"}:
            for field in ("start", "search_anchor", "mode", "activities", "subcategories"):
                if self.pending.get(field) and field not in args:
                    args[field] = deepcopy(self.pending[field])
            for field in ("start", "search_anchor"):
                if args.get(field):
                    args[field] = self._endpoint(args[field])
            if "keyword" not in args and self.intent.get("keywords"):
                args["keyword"] = " ".join(self.intent["keywords"])
            args.setdefault("exclude_terms", self.intent.get("exclude_terms", []))
            args.setdefault("limit", 20)
            if self.pending.get("constraints"):
                args["constraints"] = self._constraints(args.get("constraints"))
        route_tools = {"plan_route", "plan_via_route", "plan_itinerary", "plan_multimodal_route", "plan_tour", "compare_routes"}
        if name in route_tools:
            if "start" not in args:
                args["start"] = self.pending.get("start") or ({"type": "coord", **self.coord_start} if self.coord_start else None)
            if "end" not in args and self.coord_end:
                args["end"] = {"type": "coord", **self.coord_end}
            for field in ("start", "end"):
                if args.get(field):
                    args[field] = self._endpoint(args[field])
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
        result, artifact = self.execute_tool(name, args, context)
        if not isinstance(result, dict):
            raise ValueError("invalid_tool_result")
        artifact = artifact or {}
        observed = list(artifact.get("candidates") or result.get("candidates") or [])
        if isinstance(result.get("poi"), dict):
            observed.append(result["poi"])
        for row in observed:
            identifier = row.get("poi_id") or row.get("id")
            if identifier:
                row = {**row, "poi_id": identifier}
                self.known[identifier] = row
                self.fresh.add(identifier)
        if name in {"search_poi_candidates", "find_reachable_places"}:
            self.last_search = result
            if self.intent.get("operation") == "place_search":
                self.pending["candidate_pool"] = [
                    deepcopy(row) for row in (result.get("candidates") or [])
                    if isinstance(row, dict) and row.get("poi_id")
                ][:20]
            if name == "search_poi_candidates":
                if result.get("search_area"):
                    self.intent["area"] = result["search_area"]
                if result.get("search_keyword"):
                    self.intent["last_search_keyword"] = result["search_keyword"]
        route = artifact.get("route")
        if isinstance(route, dict):
            if (route.get("route_verification") or {}).get("status") != "pass":
                return {"error": "route_verification_failed", "message": "路线未通过核验，不可展示，请检查参数与约束。"}
            self.route, self.route_kind = route, artifact.get("route_kind") or "direct"
            state = route.get("route_state") or {}
            self.pending["start"] = deepcopy(state.get("start") or args.get("start"))
            self.intent["route_to_modify"] = deepcopy(state)
        if isinstance(artifact.get("route_comparison"), dict):
            self.comparison = artifact["route_comparison"]
        if result.get("reason_code") == "graph_no_path":
            self.no_path = result
        if name == "get_weather" and not result.get("error"):
            self.weather = deepcopy(result)
        if artifact.get("clarify"):
            self.clarify = artifact["clarify"]
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
        for identifier in ids:
            if identifier not in self.known:
                raise ValueError("unknown_place: 只能展示已查得地点")
            if identifier in shown:
                raise ValueError("already_shown: 换批不得重复推荐，耗尽时请明确说明")
        introduction = _safe_candidate_prose(str(args.get("introduction") or "")[:2000])
        prose = introduction + " ".join(str(item.get("reason", "")) for item in places)
        if any(row.get("name") and row["name"] in prose for key, row in self.known.items() if key not in ids):
            raise ValueError("unselected_place_in_text: 文字不可另推卡片之外的地点")
        candidates, paragraphs = [], [introduction]
        for number, item in enumerate(places, 1):
            row = deepcopy(self.known[item["poi_id"]])
            reason = _safe_candidate_prose(str(item.get("reason") or "")[:1500])
            source_ids = item.get("source_ids") or []
            if any(source_id not in self.sources for source_id in source_ids):
                raise ValueError("unknown_source")
            row.update({"recommendation_reason": reason, "source_ids": source_ids})
            candidates.append(row)
            paragraphs.append(f"{number}. **{row['name']}**：{reason}")
        self.candidates = candidates
        self.pending.update({"pending_slot": "selection", "candidate_poi_ids": ids,
                             "candidate_places": deepcopy(candidates)})
        self.intent["shown_ids"] = list(dict.fromkeys([*self.intent.get("shown_ids", []), *ids]))[-100:]
        if self.last_search.get("data_version"):
            record_candidate_snapshot(self.pending, self.last_search)
        return "\n\n".join(paragraphs)

    def result(self, message, *, failed=False):
        from agents.planner import _sanitize_message
        if failed and not (self.route or self.comparison or self.candidates):
            question = "刚才这一步暂时没有完成，我保留了当前需求和已确认条件。你想按原条件继续，还是调整区域或类型？"
            self.clarify = {"question": question,
                            "options": ["按原条件继续", "调整区域或类型"], "slot": "other"}
            message = question
        elif failed and self.candidates:
            message = "我保留了这批已核对候选。你可以先选择其中一个，或说“换一批”继续。"
        elif failed and self.route:
            message = "路线结果已保留，你可以先查看路线；如果还需要说明，我可以继续补充。"
        self.intent["sources"] = list(self.sources.values())[-12:]
        self.pending["semantic_request"] = self.intent
        kind = "route" if self.route else "route_comparison" if self.comparison else "candidates" if self.candidates else "clarify" if self.clarify else "chat"
        if self.candidates:
            message = _safe_candidate_prose(message)
            scope_note = "地点依据：以上为地图/校园目录匹配结果，未现场核实存在性、入口或营业状态。"
            if scope_note not in message:
                message += "\n\n" + scope_note
        if self.sources:
            source_list = " · ".join(f"[{source['source_id']}]({source['url']})" for source in self.sources.values())
            message += "\n\n参考来源：" + source_list
        result = {"response_kind": kind, "message": _sanitize_message(message), "turns": self.turns,
                  "route": self.route, "route_kind": self.route_kind, "candidates": self.candidates,
                  "clarify": self.clarify, "suggestions": self.suggestions, "route_comparison": self.comparison,
                  "weather": self.weather,
                  "place_request": self.pending, "sources": list(self.sources.values()),
                  "model_status": "unavailable" if failed else "ok", "model_usage": dict(self.usage)}
        if self.route:
            result["completed_place_request"] = deepcopy(self.pending)
            result["place_request"] = None
        if failed:
            result["execution_status"] = "partial" if self.route or self.comparison or self.candidates else "needs_input"
            if result["execution_status"] == "needs_input":
                result["needs_input"] = True
        return result

    def run(self):
        from agents import web_evidence
        started = time.monotonic()
        max_seconds = min(120.0, max(20.0, float(os.getenv("AGENT_DIALOGUE_TIMEOUT_S", "90"))))
        compact_query = re.sub(r"[\s，。！？、,.!?;；:：]+", "", str(self.query or "")).casefold()
        if (compact_query in _NEXT_BATCH_COMMANDS
                and self.intent.get("operation") == "place_search"
                and self.pending.get("pending_slot") == "selection"
                and not self.direct_route_selection):
            shown = set(self.intent.get("shown_ids") or [])
            remaining = []
            seen = set()
            for row in self.pending.get("candidate_pool") or []:
                poi_id = row.get("poi_id") if isinstance(row, dict) else None
                if poi_id and poi_id not in shown and poi_id not in seen:
                    remaining.append(row)
                    seen.add(poi_id)
            if not remaining:
                self.clarify = {
                    "question": "这次检索到的候选已经展示完了。要扩大附近范围，还是换一种地点类型？",
                    "options": ["扩大搜索范围", "换一种地点类型"], "slot": "selection",
                }
                return self.result(self.clarify["question"])
            limit = min(self.intent.get("limit", 4), 4)
            selected = remaining[:limit]
            self.last_search = {"status": "ok", "online_status": "ok",
                                "candidates": deepcopy(remaining), "count": len(remaining)}
            self.turn_action = "next_batch"
            message = self.present({
                "introduction": "继续按原条件展示本次检索中尚未出现的候选。",
                "places": [{"poi_id": row["poi_id"],
                            "reason": "来自本次地点目录检索的其他候选记录。"}
                           for row in selected],
            })
            return self.result(message)
        if self.direct_route_selection:
            selected_id = self.pending["selected_poi_id"]
            selected = self.known[selected_id]
            route_result = self.spatial("plan_route", {
                "end": {"type": "poi", "poi_id": selected_id, "name": selected.get("name")},
            })
            if not self.route:
                message = ((self.clarify or {}).get("question")
                           or route_result.get("message")
                           or "暂时无法生成到所选地点的可靠路线，请检查起点或稍后重试。")
                return self.result(message, failed=bool(route_result.get("error")))
            route_summary = {key: self.route.get(key) for key in (
                "start_name", "end_name", "mode", "distance_m", "duration_min",
                "recommended_length_m", "route_verification", "endpoint_access",
                "road_condition_overview", "strategy",
            ) if key in self.route}
            self.messages.append({"role": "system", "content": (
                "地图工具已按服务端接受的地点 ID 完成路线计算。仅依据以下路线结果作简洁说明；"
                "不得修改距离、时间、起终点或核验状态。路线摘要："
                + json.dumps(_compact(route_summary), ensure_ascii=False, default=str)
            )})
            self.route_preplanned = True
        try:
            if not config.DEEPSEEK_API_KEY:
                raise RuntimeError("model_not_configured")
            client = self.client_factory()
        except Exception:
            return self.result("对话模型暂时不可用，已保留本轮需求，请稍后重试。", failed=True)
        for _ in range(12):
            if time.monotonic() - started > max_seconds or self.usage["total_tokens"] > 40000:
                return self.result("本轮查询达到执行上限，已保留得到的信息和任务，可以继续或缩小检索范围。", failed=True)
            operation = self.intent.get("operation")
            requested_count = min(self.intent.get("limit", 4), 4)
            candidate_count = len(self.last_search.get("candidates") or [])
            web_research_requested = self.intent.get("web_research_requested") is True
            page_read_requested = self.intent.get("web_page_read_requested") is True
            web_lookup_complete = (web_research_requested and self.web_search_count >= 1
                                   and (not page_read_requested or self.web_fetch_count >= 1 or not self.sources))
            web_results_exhausted = (web_lookup_complete and self.place_search_count >= 1
                                     and ("limit" not in self.intent or self.place_search_count >= 2))
            if operation == "place_search" and not candidate_count and (
                    self.place_search_count >= 2
                    or web_results_exhausted):
                online_status = self.last_search.get("online_status")
                message = ("地图地点服务暂时不可用，请稍后重试。"
                           if online_status not in (None, "ok", "not_requested")
                           else "这次地图检索没有找到可核对的地点；可以换一个区域或餐饮类型再试。")
                return self.result(message)
            self.turns += 1
            # The existing lease-aware callback also detects cancellation between
            # model turns; do not swallow errors raised by the workflow callback.
            self.trace({"tool_name": "model_dialogue", "argument_keys": [], "outcome": "success", "artifact_kind": None})
            web_candidates_ready = (web_lookup_complete and self.place_search_count >= 1
                                    and candidate_count > 0 and "limit" not in self.intent)
            ready_to_present = (candidate_count >= requested_count or self.place_search_count >= 2
                                or web_candidates_ready)
            required_tool = None
            try:
                web_map_search_pending = (web_research_requested and self.web_search_count >= 1
                    and (not page_read_requested or self.web_fetch_count >= 1 or not self.sources)
                    and self.place_search_count < 2
                    and (self.place_search_count == 0
                         or "limit" in self.intent and candidate_count < requested_count))
                schemas = tool_schemas(operation, allow_place_search=(
                        self.place_search_count < 2 and candidate_count < requested_count),
                        ready_to_present=ready_to_present,
                        web_research_requested=web_research_requested,
                        web_search_available=self.web_search_count < 1,
                        web_fetch_available=bool(self.sources) and page_read_requested and self.web_fetch_count < 1,
                        web_map_search_pending=web_map_search_pending,
                        selected_place=bool(self.pending.get("selected_poi_id")),
                        route_preplanned=self.route_preplanned)
                request = {
                    "model": config.LLM_MODEL,
                    "messages": deepcopy(self.messages),
                    "tools": schemas,
                    "temperature": 0.2,
                    "max_tokens": 2600,
                }
                if not self._intent_resolved:
                    intent_tools = tool_schemas(None)
                    request["tools"] = [item for item in intent_tools
                                        if item["function"]["name"] in {"update_task", "ask_user"}]
                    request["tool_choice"] = "auto"
                elif operation not in _OPERATION_TOOL_NAMES:
                    request["tools"] = [item for item in schemas
                                        if item["function"]["name"] == "update_task"]
                    request["tool_choice"] = {
                        "type": "function", "function": {"name": "update_task"},
                    }
                elif self.turn_action == "next_batch" and operation == "place_search":
                    required_tool = "present_places" if candidate_count else None
                    request["tools"] = [item for item in tool_schemas(None)
                                        if item["function"]["name"] == "present_places"]
                    if required_tool:
                        request["tool_choice"] = {
                            "type": "function", "function": {"name": required_tool},
                        }
                elif operation == "place_search" and web_research_requested:
                    if self.web_search_count < 1:
                        required_tool = "search_web"
                    elif self.sources and page_read_requested and self.web_fetch_count < 1:
                        required_tool = "read_web_page"
                    elif web_map_search_pending:
                        required_tool = "search_poi_candidates"
                    elif ready_to_present:
                        required_tool = "present_places"
                    request["tool_choice"] = {
                        "type": "function", "function": {"name": required_tool},
                    }
                elif operation == "place_search" and ready_to_present:
                    request["tool_choice"] = {
                        "type": "function", "function": {"name": "present_places"},
                    }
                available_tool_names = {
                    item["function"]["name"] for item in request["tools"]
                }
                response = client.chat.completions.create(**request)
            except Exception:
                return self.result("对话模型暂时没有完成回答，已保留本轮需求和查得信息，请重试。", failed=True)
            usage = getattr(response, "usage", None)
            for field in self.usage:
                self.usage[field] += int(getattr(usage, field, 0) or 0)
            choices = getattr(response, "choices", None)
            if not isinstance(choices, (list, tuple)) or not choices or getattr(choices[0], "message", None) is None:
                return self.result("对话模型返回了空结果，已保留本轮需求，请重试。", failed=True)
            msg = choices[0].message
            calls = getattr(msg, "tool_calls", None) or []
            if not calls and not self._intent_resolved:
                self.clarify = {
                    "question": "你是想继续刚才的地点搜索，还是想换一个区域或类型？",
                    "options": ["继续刚才的搜索", "换区域或类型"], "slot": "other",
                }
                return self.result(self.clarify["question"])
            if (operation == "place_search" and web_research_requested
                    and required_tool == "search_poi_candidates"):
                expected_calls = [call for call in calls
                                  if call.function.name == required_tool]
                if expected_calls:
                    # A strict retrieval phase accepts only the requested spatial
                    # tool. Extra model-proposed task edits cannot revise a search
                    # after its web evidence has already been gathered.
                    calls = expected_calls[:1]
                else:
                    # If the model declines the required map step, still ground the
                    # current request against the map before asking it to select.
                    try:
                        search_result = self.spatial("search_poi_candidates", {})
                        self.trace({"tool_name": "search_poi_candidates",
                                    "argument_keys": [],
                                    "outcome": "tool_error" if search_result.get("error") else "success",
                                    "artifact_kind": "candidates"})
                        self.messages.append({"role": "system", "content": (
                            "模型本轮没有提交地图检索参数，服务端已按已接受任务完成地图检索。"
                            "请只从返回的地点 ID 中筛选推荐；检索结果：\n"
                            + _model_tool_content("search_poi_candidates", search_result)
                        )})
                    except (ValueError, TypeError, KeyError) as exc:
                        return self.result("地图地点检索暂时没有完成，请稍后重试。", failed=True)
                    continue
            if (operation == "place_search" and web_research_requested
                    and required_tool == "present_places"):
                expected_calls = [call for call in calls
                                  if call.function.name == required_tool]
                if not expected_calls:
                    return self.result("我已核对地图候选，但本轮没有完成推荐整理，请重试。", failed=True)
                calls = expected_calls[:1]
            if not calls:
                text = str(msg.content or "").strip()
                return self.result(text or "本轮没有形成完整回答，请重试。", failed=not bool(text))
            record = {"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}} for call in calls]}
            if getattr(msg, "reasoning_content", None) is not None:
                record["reasoning_content"] = msg.reasoning_content
            self.messages.append(record)
            final = None
            task_update_applied = False
            search_intent_changed = False
            auto_search_done = False
            # Commit changed intent before any spatial call from the same model
            # response, so stale arguments cannot search the previous request.
            ordered_calls = sorted(calls[:8], key=lambda call: call.function.name != "update_task")
            for call in ordered_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be an object")
                    if name not in available_tool_names:
                        answer = {"error": "tool_not_available",
                                  "message": "当前阶段未开放此工具，请先完成任务信息提交后再继续。"}
                    elif name == "update_task":
                        if task_update_applied:
                            answer = {"error": "task_already_committed",
                                      "message": "本轮任务已保存，请基于当前任务继续。"}
                        else:
                            answer = self.update(args)
                            if not answer.get("error"):
                                task_update_applied = True
                            search_intent_changed = (
                                not answer.get("error")
                                and self._last_task_update_changed_search
                                and self.intent.get("operation") == "place_search"
                            )
                    elif name == "present_places":
                        final = self.present(args)
                        answer = {"accepted": True, "poi_ids": self.pending["candidate_poi_ids"]}
                    elif name in {"search_web", "read_web_page"}:
                        counter_name = "web_search_count" if name == "search_web" else "web_fetch_count"
                        if getattr(self, counter_name) >= 1:
                            answer = {"error": "web_evidence_limit", "message": "本轮网页检索次数已达上限，请基于已查来源回答。"}
                        else:
                            setattr(self, counter_name, getattr(self, counter_name) + 1)
                            answer = web_evidence.execute(name, args, self.sources,
                                                          original_question=self.query)
                    elif (name in {"search_poi_candidates", "find_reachable_places"}
                          and search_intent_changed
                          and (auto_search_done or self.intent.get("web_research_requested") is True)):
                        answer = {"error": "search_superseded",
                                  "message": "任务条件已更新，将按新条件完成网页查证和地图检索；请勿使用旧参数。"}
                    else:
                        answer = self.spatial(name, args)
                except (ValueError, TypeError, KeyError) as exc:
                    answer = {"error": "invalid_tool_request", "message": str(exc)[:300]}
                if name in {"update_task", "present_places", "search_web", "read_web_page"}:
                    self.trace({"tool_name": name, "argument_keys": sorted(args) if isinstance(args, dict) else [],
                                "outcome": "tool_error" if answer.get("error") else "success", "artifact_kind": None})
                self.messages.append({"role": "tool", "tool_call_id": call.id,
                                      "content": _model_tool_content(name, answer)})
                if (name == "update_task" and not answer.get("error")
                        and search_intent_changed and not auto_search_done
                        and self.place_search_count < 2):
                    if self.turn_action == "next_batch":
                        auto_search_done = True
                    elif self.intent.get("web_research_requested") is True:
                        if self.web_search_count < 1:
                            self.web_search_count = 1
                            query = _web_search_query(self.intent)
                            search_result = web_evidence.execute(
                                "search_web", {"query": query}, self.sources,
                                original_question=self.query,
                            )
                            self.trace({"tool_name": "search_web", "argument_keys": ["query"],
                                        "outcome": "tool_error" if search_result.get("error") else "success",
                                        "artifact_kind": None})
                            self.messages.append({"role": "system", "content": (
                                "服务端已按已接受任务调用 DeepSeek 网页搜索。网页只能提供名称和描述线索；"
                                "不能证明门店位置或现场情况。搜索结果：\n"
                                + _model_tool_content("search_web", search_result)
                            )})
                            source_rows = search_result.get("results") or []
                            source_id = next((row.get("source_id") for row in source_rows
                                              if isinstance(row, dict) and row.get("source_id")), None)
                            if source_id and self.intent.get("web_page_read_requested") is True:
                                self.web_fetch_count = 1
                                fetch_result = web_evidence.execute(
                                    "read_web_page", {"source_id": source_id}, self.sources,
                                    original_question=self.query,
                                )
                                self.trace({"tool_name": "read_web_page",
                                            "argument_keys": ["source_id"],
                                            "outcome": "tool_error" if fetch_result.get("error") else "success",
                                            "artifact_kind": None})
                                self.messages.append({"role": "system", "content": (
                                    "服务端读取了一个搜索来源；内容仍需由地图核对地点身份和位置。来源正文：\n"
                                    + _model_tool_content("read_web_page", fetch_result)
                                )})
                        auto_search_done = True
                    else:
                        search_result = self.spatial("search_poi_candidates", {})
                        self.messages.append({"role": "system", "content": (
                            "服务端已按刚保存的任务调用地图地点检索。只能从以下返回的 poi_id 中提交推荐；"
                            "来源没有提供的事实应说明未知。检索结果：\n"
                            + _model_tool_content("search_poi_candidates", search_result)
                        )})
                        auto_search_done = True
                    auto_search_done = True
                if name == "present_places" and not answer.get("error"):
                    break
            if final is not None:
                return self.result(final)
            if self.clarify:
                return self.result(self.clarify.get("question", "请补充具体要求。"))
        return self.result("本轮检索达到调用上限，已保留需求，请继续说明最重要的条件。", failed=True)


def run_dialogue(query, **kwargs):
    return Dialogue(query, **kwargs).run()
