"""Model-led dialogue, backed by the existing accepted task and spatial tools.

The model owns interpretation and recommendation. This module owns identity,
delivery alignment and bounded execution; it does not classify user sentences.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
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
        elif name in {"get_weather", "plan_route"}:
            item["function"]["description"] += (
                "如果本轮只有这一项任务且无需继续判断，必须在同一批工具调用中同时调用 finish_turn；"
                "不要等结果返回后再额外生成复述轮。"
            )
    schemas.extend([
        _function("update_task", "保存你理解的完整需求或本轮修订。只更新传入字段；继续任务保留其余条件。参数明确时，可在同一响应中与后续工具一起调用，服务器会先处理更新。已接受的字段不要重复提交。",
            {"relation": {"type": "string", "enum": ["new", "continue", "resume_route"]},
             "task_id": {"type": "string", "description": "续接或恢复时填写服务器提供的任务 ID"},
             "task_revision": {"type": "integer", "minimum": 0, "description": "所选任务的当前版本"},
             "operation": {"type": "string", "enum": list(_OPERATIONS),
                           "description": "本轮语义任务类型；按主要用户目标选择"},
             "goal": {"type": "string"},
             "start": {**_ENDPOINT_SCHEMA, "description": "实际出发地点，不能用搜索中心代替"},
             "search_anchor": {**_ENDPOINT_SCHEMA, "description": "附近查询参照点，可与起点不同"},
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
                                  "description": "用户要求换一批/换一家且继续当前地点条件时设为next_batch；筛选条件变化时设为recommend"},
             "activities": {"type": "array", "items": {"type": "string"}},
             "subcategories": {"type": "array", "items": {"type": "string"}},
             "budget_per_person": {"type": "number", "exclusiveMinimum": 0},
             "limit": {"type": "integer", "minimum": 1, "maximum": 20},
             "constraints": _PREFERENCE_SCHEMA["constraints"],
             "clear_fields": {"type": "array", "items": {"type": "string"},
                               "description": "仅在用户明确取消时清除已有条件；路线硬约束使用 constraints.avoid_slope、constraints.avoid_steps、constraints.max_distance_m 等字段名"}},
            ("relation", "operation")),
        _function("present_places", "提交最终推荐。仅允许已查询的地点ID；顺序由你决定。聊天只写简短概括；每张卡片附一句简短理由。source_ids 只用于后台追溯，不写进聊天文字。",
            {"introduction": {"type": "string", "maxLength": 500},
             "places": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
                 "type": "object", "properties": {"poi_id": {"type": "string"},
                     "reason": {"type": "string", "maxLength": 36}, "source_ids": {"type": "array", "items": {"type": "string"}}},
                 "required": ["poi_id", "reason"], "additionalProperties": False}}}, ("introduction", "places")),
        _function("search_web", "检索校园生活/商户资料，返回带来源与日期的文本；不是已核实的地点或坐标。",
                  {"query": {"type": "string"}}, ("query",)),
        _function("read_web_page", "读取本轮搜索返回的某个来源，核对内容、日期与分店。",
                  {"source_id": {"type": "string"}}, ("source_id",)),
        _function("browse_candidates", "查看已接受候选池中具体地点及分页状态；默认只显示尚未展示的地点。",
                  {"offset": {"type": "integer", "minimum": 0},
                   "limit": {"type": "integer", "minimum": 1, "maximum": 12},
                   "include_shown": {"type": "boolean"}}),
        _function("conclude_without_places", "已查看候选但没有适合本轮条件的地点时，说明无法推荐的原因；不要据此声称候选池为空。",
                  {"reason": {"type": "string", "minLength": 1, "maxLength": 1000}}, ("reason",)),
        _function("read_result", "读取本会话历史结果引用；路线几何与候选按页返回。",
                  {"run_id": {"type": "string"},
                   "section": {"type": "string", "enum": ["route_summary", "route_geometry", "candidates"]},
                   "offset": {"type": "integer", "minimum": 0},
                   "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, ("run_id", "section")),
        _function("finish_turn", "单独查询天气或规划单条路线，且无需继续判断时，必须与 get_weather/plan_route 放在同一批工具调用中。服务器会在结果成功后交付简短结论。天气加路况、多个路线、需要解释细节或交付地点卡片时不要调用。",
                  {}),
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
        self.available_tasks = {str(item.get("task_id")): item for item in self.context.get("available_tasks", [])
                                if isinstance(item, dict) and item.get("task_id")}
        self.execute_tool, self.client_factory, self.trace = execute_tool, client_factory, trace
        self.pending = deepcopy(self.context.get("place_request") or {})
        self.pending.setdefault("request_id", str(uuid4()))
        self.pending.setdefault("original_query", query[:2000])
        self.pending.setdefault("mode", travel_mode or "walk")
        self.pending.setdefault("search_radius_m", 1000)
        self.pending.setdefault("constraints", {})
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
        selected_id = self.context.get("selected_poi_id")
        accepted_ids = set(self.pending.get("candidate_poi_ids") or [])
        if selected_id in accepted_ids and selected_id in self.known:
            self.pending["selected_poi_id"] = selected_id
            self.pending["selected_place"] = deepcopy(self.known[selected_id])
        self.turn_action = "recommend"
        self.candidates, self.route, self.route_kind, self.comparison = None, None, None, None
        self.weather = None
        self.route_tool_failure = None
        self.no_path, self.clarify, self.suggestions = None, None, None
        self.last_search = {}
        self.web_search_count = 0
        self.web_fetch_count = 0
        self.place_search_count = 0
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
            selected = self.context.get("selected_poi_id")
            if (selected and str(args.get("task_id")) == str(self.context.get("selected_task_id"))
                    and selected in (self.pending.get("candidate_poi_ids") or []) and selected in self.known):
                self.pending["selected_poi_id"] = selected
                self.pending["selected_place"] = deepcopy(self.known[selected])
        elif self.bind_task and self.bound_task is not None and args.get("task_id") not in (None, self.bound_task["task_id"]):
            raise ValueError("a run cannot change its bound task")
        pending = deepcopy(self.pending)
        if relation == "new":
            self.known = {}
            pending = {"request_id": str(uuid4()), "original_query": self.query[:2000],
                       "mode": self.requested_mode or self.pending.get("mode", "walk"),
                       "constraints": {}, "semantic_request": {}, "search_radius_m": 1000}
        if relation == "resume_route":
            if not self.previous_route or self.context.get("route_context_uncertain"):
                raise ValueError("prior_route_unavailable: 请确认要修改的路线")
            state = self.previous_route
            pending.update({"start": deepcopy(state["start"]), "mode": state["travel_mode"],
                            "constraints": _resume_constraints(pending.get("constraints"), state["hard_constraints"])})
            strategy = state.get("strategy") or {}
            pending["strategy"] = strategy.get("name", "shortest") if isinstance(strategy, dict) else strategy
            pending.setdefault("semantic_request", {})["route_to_modify"] = deepcopy(state)
        intent = pending.setdefault("semantic_request", {}) or {}
        changed_search = relation == "new"
        for field in ("operation", "goal", "requirements", "area", "keywords", "exclude_terms",
                      "web_research_requested", "web_page_read_requested", "budget_per_person", "limit"):
            if field in args:
                value = args[field]
                if field == "operation" and (not isinstance(value, str) or value not in _OPERATIONS):
                    raise ValueError("invalid operation")
                if field in {"requirements", "keywords", "exclude_terms"}:
                    if not isinstance(value, list) or len(value) > 30 or any(not isinstance(x, str) or len(x) > 500 for x in value):
                        raise ValueError(f"invalid {field}")
                    if (relation != "new" and field in {"requirements", "exclude_terms"}
                            and field not in args.get("clear_fields", [])):
                        value = list(dict.fromkeys([*(intent.get(field) or []), *value]))
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
                if field in {"start", "search_anchor"} and args[field] is None:
                    raise ValueError("clear an endpoint only through clear_fields")
                if field in {"start", "search_anchor"} and args[field] is not None:
                    args[field] = self._endpoint(args[field], allow_pending=relation != "new")
                if field in {"start", "search_anchor", "mode", "activities", "subcategories"}:
                    changed_search |= args[field] != pending.get(field)
                pending[field] = deepcopy(args[field])
        changed_search_without_radius = changed_search
        radius_before_update = pending.get("search_radius_m") or 1000
        radius_note = None
        ignored_radius = False
        expanded_this_update = False
        if "search_radius_m" in args:
            radius = args["search_radius_m"]
            if type(radius) is not int or not 100 <= radius <= 5000:
                raise ValueError("invalid search radius")
            previous_radius = pending.get("search_radius_m") or 1000
            from agents.distance_evidence import radius_supported_by_message
            if radius != previous_radius and not radius_supported_by_message(self.query, radius, previous_radius):
                ignored_radius = True
                radius_note = "本轮没有支持该数值的用户距离条件，已保留原半径；需要扩大可使用 expand_search_radius。"
            else:
                changed_search |= previous_radius != radius
                pending["search_radius_m"] = radius
        if ("search_radius_m" not in args or ignored_radius) and args.get("expand_search_radius") is True:
            if not self._expanded_this_turn:
                radius = min(5000, (pending.get("search_radius_m") or 1000) * 2)
                changed_search |= pending.get("search_radius_m") != radius
                pending["search_radius_m"] = radius
                expanded_this_update = True
        if "constraints" in args:
            previous_constraints = HardConstraints.model_validate(pending.get("constraints") or {}).model_dump()
            incoming = dict(args["constraints"])
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
                         "web_page_read_requested"}:
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
        if candidate_action == "next_batch" and (relation != "continue" or changed_search):
            raise ValueError("next_batch requires unchanged place search conditions")
        next_known, next_last_search = self.known, self.last_search
        if changed_search:
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
            next_known = {selected_id: selected_place} if selected_place else {}
            next_last_search = {}
        pending["semantic_request"] = intent
        if candidate_action == "next_batch":
            shown = set(intent.get("shown_ids") or [])
            pool = pending.get("candidate_pool") or []
            remaining = [row for row in pool if row.get("poi_id") not in shown]
            next_last_search = {"status": "ok", "online_status": "ok",
                                "candidates": deepcopy(remaining), "count": len(remaining)}
        PlaceRequestState.model_validate(pending)
        changed_fields = ["place_request." + ("search_radius_m" if key == "expand_search_radius" else key)
                          for key in args if key not in {"relation", "task_id", "task_revision", "clear_fields", "candidate_action"}
                          and not (key == "search_radius_m" and ignored_radius)]
        changed_fields.extend("place_request." + field for field in args.get("clear_fields", []) if isinstance(field, str))
        if self.bind_task and self.bound_task is None:
            self.bound_task = self.bind_task(relation=relation, task_id=args.get("task_id"),
                                             base_revision=args.get("task_revision"), place_request=pending,
                                             changed_fields=changed_fields)
        if expanded_this_update:
            self._expanded_this_turn = True
        self.known, self.last_search = next_known, next_last_search
        self.turn_action = candidate_action
        if candidate_action == "next_batch":
            self.place_search_count = 2
        self.modified_fields.update(changed_fields)
        self.pending, self.intent = pending, intent
        compact_pending = deepcopy(pending)
        compact_pending.pop("candidate_pool", None)
        response = {"accepted_task": _compact(compact_pending)}
        if radius_note:
            response["radius_note"] = radius_note
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
        if name in {"search_poi_candidates", "find_reachable_places"}:
            if self.place_service_error:
                return {"error": "place_service_unavailable", "message": "地点服务本轮已报告故障，请稍后重试。"}
            if (name == "search_poi_candidates" and args.get("area") and self.intent.get("area")
                    and args["area"] != self.intent["area"]):
                return {"error": "task_update_required", "message": "检索区域与已接受任务不一致，请核对后重试。"}
            if self.place_search_count >= 8:
                return {"error": "place_search_limit",
                        "message": "本轮地点检索次数已达上限，请根据已有结果回答。"}
            self.place_search_count += 1
        context = {"query": self.query, "uid": self.uid, "_model_dialogue": True,
                   "_semantic_request": self.intent, "_known_places": self.known}
        if name in {"search_poi_candidates", "find_reachable_places"}:
            for field in ("start", "search_anchor", "mode", "activities", "subcategories"):
                if self.pending.get(field) and field not in args:
                    args[field] = deepcopy(self.pending[field])
            if name == "search_poi_candidates" and self.pending.get("search_anchor"):
                if args.get("search_anchor") != self.pending["search_anchor"]:
                    return {"error": "task_update_required", "message": "请先更新任务中的搜索中心。"}
            for field in ("start", "search_anchor"):
                if args.get(field):
                    args[field] = self._endpoint(args[field])
            if "keyword" not in args and self.intent.get("keywords"):
                args["keyword"] = " ".join(self.intent["keywords"])
            args.setdefault("exclude_terms", self.intent.get("exclude_terms", []))
            args.setdefault("limit", 20)
            if name == "search_poi_candidates" and self.pending.get("search_radius_m"):
                radius = self.pending["search_radius_m"]
                if args.get("max_distance_m") not in (None, radius):
                    return {"error": "task_update_required", "message": "请先更新任务中的搜索半径。"}
                args["max_distance_m"] = radius
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
        if name in {"search_poi_candidates", "find_reachable_places"}:
            self.last_search = result
            if result.get("candidates") and result.get("data_version") and result.get("road_condition_version"):
                from agents.result_maintenance import record_candidate_snapshot
                record_candidate_snapshot(self.pending, result)
            provider_status = result.get("online_status")
            if (provider_status and provider_status not in {"ok", "not_requested", "query_required", "center_unavailable"}
                    and not result.get("candidates")):
                self.place_service_error = provider_status
            if self.intent.get("operation") == "place_search" and result.get("candidates"):
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
            self.route_tool_failure = None
            state = route.get("route_state") or {}
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
        offset = args.get("offset", 0)
        limit = args.get("limit", 12)
        include_shown = args.get("include_shown", False)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 12:
            raise ValueError("invalid candidate page")
        if type(include_shown) is not bool:
            raise ValueError("include_shown must be a boolean")
        shown = set(self.intent.get("shown_ids") or [])
        pool = [row for row in self.pending.get("candidate_pool") or []
                if isinstance(row, dict) and (include_shown or row.get("poi_id") not in shown)]
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
        for identifier in ids:
            if identifier not in self.known:
                raise ValueError("unknown_place: 只能展示已查得地点")
            if identifier in shown:
                raise ValueError("already_shown: 换批不得重复推荐，耗尽时请明确说明")
        introduction = str(args.get("introduction") or "").strip()
        if len(introduction) > 500:
            raise ValueError("recommendation_intro_too_long: 请简明回答，用户要求详细介绍时再展开")
        prose = introduction + " ".join(str(item.get("reason", "")) for item in places)
        if any(row.get("name") and row["name"] in prose for key, row in self.known.items() if key not in ids):
            raise ValueError("unselected_place_in_text: 文字不可另推卡片之外的地点")
        candidates = []
        for item in places:
            row = deepcopy(self.known[item["poi_id"]])
            reason = str(item.get("reason") or "").strip()
            if len(reason) > 36:
                raise ValueError("recommendation_reason_too_long: 请用一句短话概括")
            source_ids = item.get("source_ids") or []
            if any(source_id not in self.sources for source_id in source_ids):
                raise ValueError("unknown_source")
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
        self.intent["sources"] = list(self.sources.values())[-12:]
        self.pending["semantic_request"] = self.intent
        kind = "route" if self.route else "route_comparison" if self.comparison else "candidates" if self.candidates else "clarify" if self.clarify else "chat"
        result = {"response_kind": kind, "message": _sanitize_message(message), "turns": self.turns,
                  "route": self.route, "route_kind": self.route_kind, "candidates": self.candidates,
                  "clarify": self.clarify, "suggestions": self.suggestions, "route_comparison": self.comparison,
                  "weather": self.weather,
                  "place_request": self.pending, "sources": list(self.sources.values()),
                  "updated_task_fields": sorted(self.modified_fields),
                  "model_status": "unavailable" if failed else "ok", "model_usage": dict(self.usage)}
        if self.route:
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
