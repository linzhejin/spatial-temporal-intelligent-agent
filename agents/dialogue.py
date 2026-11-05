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


def tool_schemas():
    from agents.tools import TOOL_SCHEMAS, _ENDPOINT_SCHEMA, _PREFERENCE_SCHEMA
    schemas = deepcopy(TOOL_SCHEMAS)
    search = next(item["function"] for item in schemas if item["function"]["name"] == "search_poi_candidates")
    search["description"] = ("按模型理解的关键词、区域、类别/用途检索真实地点。无需出发点也能查询校园店铺。"
                              "有参照点可计算距离；返回检索结果，由你选择推荐；不把最近当作最好。")
    search["parameters"]["properties"].update({
        "area": {"type": "string", "description": "地点所在区域，例如武汉大学、梅园、信息学部"},
        "keyword": {"type": "string", "description": "具体店名或检索词，例如火锅、肯德基"},
        "search_anchor": _ENDPOINT_SCHEMA,
        "max_distance_m": {"type": "number", "minimum": 100, "maximum": 10000},
        "exclude_terms": {"type": "array", "items": {"type": "string"}},
    })
    for item in schemas:
        if item["function"]["name"] == "resolve_poi":
            item["function"]["description"] = "核对具体地名，查询本地校园设施及在线商户；多分店返回候选，不擅自选定。"
    schemas.extend([
        _function("update_task", "保存你理解的完整需求或本轮修订。只更新传入字段；继续任务保留其余条件。",
            {"relation": {"type": "string", "enum": ["new", "continue", "resume_route"]},
             "goal": {"type": "string"},
             "start": {**_ENDPOINT_SCHEMA, "description": "实际出发地点，不能用搜索中心代替"},
             "search_anchor": {**_ENDPOINT_SCHEMA, "description": "附近查询参照点，可与起点不同"},
             "mode": {"type": "string", "enum": ["walk", "bike", "drive"]},
             "strategy": {"type": "string", "enum": ["shortest", "recommended", "flat", "scenery"]},
             "requirements": {"type": "array", "items": {"type": "string"}, "description": "完整保留否定、用途及其他条件"},
             "area": {"type": "string", "description": "检索区域文字，例如梅园、桂园或武汉大学"},
             "keywords": {"type": "array", "items": {"type": "string"}},
             "exclude_terms": {"type": "array", "items": {"type": "string"}},
             "activities": {"type": "array", "items": {"type": "string"}},
             "subcategories": {"type": "array", "items": {"type": "string"}},
             "budget_per_person": {"type": "number", "exclusiveMinimum": 0},
             "limit": {"type": "integer", "minimum": 1, "maximum": 20},
             "constraints": _PREFERENCE_SCHEMA["constraints"],
             "clear_fields": {"type": "array", "items": {"type": "string"},
                              "description": "仅在用户明确取消时清除已有字段或条件，例如budget_per_person"}}, ("relation",)),
        _function("present_places", "提交最终推荐。仅允许已查询的地点ID；顺序由你决定，理由与来源会和卡片一起展示。",
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
    return schemas


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
    if name not in {"search_poi_candidates", "find_reachable_places"} or not isinstance(answer, dict):
        return json.dumps(_compact(answer), ensure_ascii=False, default=str)

    result = {key: answer[key] for key in (
        "status", "error", "message", "count", "online_status", "search_area",
        "search_keyword", "start_name", "max_distance_m", "max_time_min",
    ) if key in answer}
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
        self.known = {item["poi_id"]: item for item in self.pending.get("candidate_places", [])
                      if isinstance(item, dict) and item.get("poi_id")}
        selected_id = self.context.get("selected_poi_id") or self.pending.get("selected_poi_id")
        accepted_ids = set(self.pending.get("candidate_poi_ids") or [])
        if selected_id in accepted_ids and selected_id in self.known:
            self.pending["selected_poi_id"] = selected_id
        else:
            self.pending.pop("selected_poi_id", None)
        self.fresh = set()
        self.candidates, self.route, self.route_kind, self.comparison = None, None, None, None
        self.weather = None
        self.no_path, self.clarify, self.suggestions = None, None, None
        self.last_search = {}
        self.web_search_count = 0
        self.web_fetch_count = 0
        self.place_search_count = 0
        self.turns = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.original_coords = [point for point in [coord_start, coord_end, *(coord_waypoints or [])] if point]
        self.coord_start, self.coord_end, self.coord_waypoints = coord_start, coord_end, coord_waypoints
        self.previous_route = self.context.get("previous_route_state")
        self.spatial_snapshot = None
        trusted = {"active_task": _compact(self.pending), "previous_route": _compact(self.previous_route),
                   "route_context_uncertain": bool(self.context.get("route_context_uncertain")),
                   "map_start": coord_start, "map_end": coord_end, "map_waypoints": coord_waypoints,
                   "selected_poi_id": self.pending.get("selected_poi_id"), "travel_mode": travel_mode}
        transcript = [{"role": item["role"], "content": str(item["content"])[:4000]}
                      for item in (history or self.context.get("history") or [])[-12:]
                      if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and item.get("content")]
        self.messages = [{"role": "system", "content": (Path(__file__).parent / "prompts" / "dialogue_system.txt").read_text(encoding="utf-8")},
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
        for field in ("goal", "requirements", "area", "keywords", "exclude_terms", "budget_per_person", "limit"):
            if field in args:
                value = args[field]
                if field in {"requirements", "keywords", "exclude_terms"}:
                    if not isinstance(value, list) or len(value) > 30 or any(not isinstance(x, str) or len(x) > 500 for x in value):
                        raise ValueError(f"invalid {field}")
                if field == "goal" and (not isinstance(value, str) or not 0 < len(value) <= 2000):
                    raise ValueError("invalid goal")
                if field == "limit" and (type(value) is not int or not 1 <= value <= 20):
                    raise ValueError("invalid limit")
                if field == "budget_per_person" and (type(value) not in (int, float) or not 0 < value <= 100000):
                    raise ValueError("invalid budget_per_person")
                changed_search |= field != "goal" and value != intent.get(field)
                intent[field] = deepcopy(value)
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
            if field in {"budget_per_person", "limit", "requirements", "area", "keywords", "exclude_terms"}:
                intent.pop(field, None)
                changed_search = True
            elif field in {"start", "search_anchor", "strategy"}:
                pending.pop(field, None)
                changed_search = True
            else:
                raise ValueError("unsupported clear field")
        if changed_search:
            intent["shown_ids"] = []
            pending.update({"candidate_poi_ids": [], "candidate_places": [], "result_dependencies": None})
            pending.pop("selected_poi_id", None)
            self.fresh.clear()
        pending["semantic_request"] = intent
        PlaceRequestState.model_validate(pending)
        self.pending, self.intent = pending, intent
        return {"accepted_task": _compact(pending)}

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
        if name in {"search_poi_candidates", "find_reachable_places"}:
            if self.place_search_count >= 6:
                return {"error": "place_search_limit",
                        "message": "本轮地点检索次数已达上限，请缩小区域或继续说明最重要的条件。"}
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
        introduction = str(args.get("introduction") or "")[:2000]
        prose = introduction + " ".join(str(item.get("reason", "")) for item in places)
        if any(row.get("name") and row["name"] in prose for key, row in self.known.items() if key not in ids):
            raise ValueError("unselected_place_in_text: 文字不可另推卡片之外的地点")
        candidates, paragraphs = [], [introduction]
        for number, item in enumerate(places, 1):
            row = deepcopy(self.known[item["poi_id"]])
            reason = str(item.get("reason") or "")[:1500]
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
        self.intent["sources"] = list(self.sources.values())[-12:]
        self.pending["semantic_request"] = self.intent
        kind = "route" if self.route else "route_comparison" if self.comparison else "candidates" if self.candidates else "clarify" if self.clarify else "chat"
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
            result["execution_status"] = "partial" if self.route or self.candidates else "failed"
        return result

    def run(self):
        from agents import web_evidence
        started = time.monotonic()
        max_seconds = min(120.0, max(20.0, float(os.getenv("AGENT_DIALOGUE_TIMEOUT_S", "90"))))
        try:
            if not config.DEEPSEEK_API_KEY:
                raise RuntimeError("model_not_configured")
            client = self.client_factory()
        except Exception:
            return self.result("对话模型暂时不可用，已保留本轮需求，请稍后重试。", failed=True)
        for _ in range(12):
            if time.monotonic() - started > max_seconds or self.usage["total_tokens"] > 40000:
                return self.result("本轮查询达到执行上限，已保留得到的信息和任务，可以继续或缩小检索范围。", failed=True)
            self.turns += 1
            # The existing lease-aware callback also detects cancellation between
            # model turns; do not swallow errors raised by the workflow callback.
            self.trace({"tool_name": "model_dialogue", "argument_keys": [], "outcome": "success", "artifact_kind": None})
            try:
                response = client.chat.completions.create(
                    model=config.LLM_MODEL, messages=deepcopy(self.messages), tools=tool_schemas(),
                    temperature=0.2, max_tokens=2600,
                )
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
            if not calls:
                text = str(msg.content or "").strip()
                return self.result(text or "本轮没有形成完整回答，请重试。", failed=not bool(text))
            record = {"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}} for call in calls]}
            if getattr(msg, "reasoning_content", None) is not None:
                record["reasoning_content"] = msg.reasoning_content
            self.messages.append(record)
            final = None
            for call in calls[:8]:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be an object")
                    if name == "update_task":
                        answer = self.update(args)
                    elif name == "present_places":
                        final = self.present(args)
                        answer = {"accepted": True, "poi_ids": self.pending["candidate_poi_ids"]}
                    elif name in {"search_web", "read_web_page"}:
                        counter_name = "web_search_count" if name == "search_web" else "web_fetch_count"
                        if getattr(self, counter_name) >= 3:
                            answer = {"error": "web_evidence_limit", "message": "本轮网页检索次数已达上限，请基于已查来源回答。"}
                        else:
                            setattr(self, counter_name, getattr(self, counter_name) + 1)
                            answer = web_evidence.execute(name, args, self.sources,
                                                          original_question=self.query)
                    else:
                        answer = self.spatial(name, args)
                except (ValueError, TypeError, KeyError) as exc:
                    answer = {"error": "invalid_tool_request", "message": str(exc)[:300]}
                if name in {"update_task", "present_places", "search_web", "read_web_page"}:
                    self.trace({"tool_name": name, "argument_keys": sorted(args) if isinstance(args, dict) else [],
                                "outcome": "tool_error" if answer.get("error") else "success", "artifact_kind": None})
                self.messages.append({"role": "tool", "tool_call_id": call.id,
                                      "content": _model_tool_content(name, answer)})
                if name == "present_places" and not answer.get("error"):
                    break
            if final is not None:
                return self.result(final)
            if self.clarify:
                return self.result(self.clarify.get("question", "请补充具体要求。"))
        return self.result("本轮检索达到调用上限，已保留需求，请继续说明最重要的条件。", failed=True)


def run_dialogue(query, **kwargs):
    return Dialogue(query, **kwargs).run()
