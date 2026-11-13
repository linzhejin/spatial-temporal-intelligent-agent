"""Evidence-based checks for expected clarification and blocked routes."""
from __future__ import annotations

import math
import re
import unicodedata

from experiments.complex_multiturn_dev.observation import _id


_ACTIVITY_EVIDENCE = {
    "coffee": {"coffee"},
    "meal": {"meal", "eat", "eating", "food", "dining", "restaurant_meal", "fast_food", "canteen_meal"},
    "basketball": {"basketball"},
    "badminton": {"badminton"},
}
_ACTIVITY_TERMS = {"coffee": r"咖啡|coffee", "meal": r"吃饭|用餐|就餐|餐饮|餐厅|食堂|快餐|小吃",
                   "basketball": r"篮球", "badminton": r"羽毛球"}


def _provider_record_matches(candidate, raw):
    ident = str(candidate.get("poi_id") or "")
    if not ident.startswith("amap:") or ident[5:] != str(raw.get("id") or ""):
        return False
    if str(candidate.get("name") or "").strip() != str(raw.get("name") or "").strip():
        return False
    try:
        lng, lat = (float(item) for item in str(raw.get("location") or "").split(","))
        return abs(float(candidate["lng"]) - lng) < 1e-5 and abs(float(candidate["lat"]) - lat) < 1e-5
    except (ValueError, KeyError, TypeError):
        return False


def _provider_supports(activity, raw):
    text = str(raw.get("type") or "") + " " + str(raw.get("name") or "")
    code = str(raw.get("typecode") or "")
    if activity == "coffee":
        return "咖啡" in text or code.startswith("0505")
    if activity == "meal":
        return any(word in text for word in ("中餐", "餐厅", "快餐", "食堂", "小吃", "外国餐厅")) or code.startswith(("0501", "0502", "0503"))
    return {"basketball": "篮球", "badminton": "羽毛球"}.get(activity, "\0") in text


def inspect_provider_type_denial(text, candidates, provider_rows):
    """Detect claims that an identified provider record lacks a type it names."""
    issues = []
    sentences = re.split(r"[。；;！!？?\n]", str(text or ""))
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        name, ident = str(candidate.get("name") or "").strip(), candidate.get("poi_id")
        if not name or not ident:
            continue
        for raw in provider_rows or []:
            if not _provider_record_matches(candidate, raw):
                continue
            provider_type = str(raw.get("type") or "")
            for activity, term in _ACTIVITY_TERMS.items():
                if not re.search(term, provider_type, re.I):
                    continue
                denial = re.compile(r"(?:没有|未有|未查到|没查到|未找到|没找到|缺少|暂无|无)"
                                    r".{0,18}(?:" + term + r").{0,12}(?:信息|记录|类型|标注|资料)", re.I)
                if any(name in sentence and denial.search(sentence) for sentence in sentences):
                    issues.append(f"provider_type_denied:{ident}:{activity}")
    return list(dict.fromkeys(issues))


def inspect_unconfirmed_position(user_text, reply, previous_route, lookup):
    """Detect a stated move to the former endpoint when no arrival was supplied."""
    if not isinstance(previous_route, dict):
        return []
    end = _id(previous_route.get("end"), lookup)
    start = _id(previous_route.get("start"), lookup)
    if not end or end == start:
        return []
    text = str(reply or "")
    match = re.search(r"(?:接着|继续|现在|已经).{0,8}从\s*(.{1,60}?)(?:往|出发|继续|步行|走|去|前往)", text)
    if not match or _id(match.group(1).strip(), lookup) != end:
        return []
    message = str(user_text or "")
    names = [name for name, identifier in lookup.get("names", {}).items() if identifier == end and len(name) >= 2]
    explicitly_named = any(name in message and re.search(r"(?:从|在|到达|抵达)\s*" + re.escape(name), message)
                           for name in names)
    arrived = (not re.search(r"没(?:有)?到|未到|尚未到", message) and re.search(
        r"(?:已经|刚刚|刚|现在|目前).{0,8}(?:到|抵达|走完)|(?:到了|抵达了|走完了)", message))
    return [] if explicitly_named or arrived else ["previous_destination_as_unconfirmed_location"]


def inspect_saved_stop_orders(requirements, expected_stops, lookup):
    """Flag a saved, identifiable stop sequence that contradicts the task order."""
    if not isinstance(expected_stops, list) or len(expected_stops) < 2:
        return []
    sequences, split_group = [], []

    def finish():
        if len(split_group) >= 2:
            sequences.append(list(split_group))

    for requirement in requirements or []:
        text = str(requirement or "").strip()
        arrow = re.fullmatch(r"(?:顺序|路线顺序|次序)\s*[:：]\s*(.+)", text)
        if arrow:
            finish()
            split_group = []
            sequences.append([item.strip() for item in re.split(r"\s*(?:→|->|➡|➜)\s*", arrow.group(1))])
            continue
        pieces = re.findall(r"(?:先|首先|再|然后|接着|最后)(?:去|到|前往)([^，,。；;！？!?、]+)", text)
        if len(pieces) >= 2:
            finish()
            split_group = []
            sequences.append([item.strip() for item in pieces])
            continue
        one = re.fullmatch(r"(先|首先|再|然后|接着|最后)(?:去|到|前往)([^，,。；;！？!?、]+)", text)
        if not one:
            finish()
            split_group = []
        elif one.group(1) in {"先", "首先"}:
            finish()
            split_group = [one.group(2).strip()]
        elif split_group:
            split_group.append(one.group(2).strip())
    finish()
    for sequence in sequences:
        positions = [expected_stops.index(identifier) for item in sequence
                     if (identifier := _id(item, lookup)) in expected_stops]
        if len(positions) >= 2 and len(set(positions)) == len(positions) and positions != sorted(positions):
            return ["stale_stop_order_retained"]
    return []


def _same_visible_place(first, second):
    """Independently compare same-named displayed points, without using product identity code."""
    def name(row):
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(row.get("name") or ""))).casefold()
    if not isinstance(first, dict) or not isinstance(second, dict) or not name(first) or name(first) != name(second):
        return False
    types = [row.get("type") or row.get("category") for row in (first, second)]
    if all(types) and types[0] != types[1]:
        return False
    systems = [str(row.get("coordinate_system") or row.get("coordinate_crs") or "GCJ-02").upper()
               for row in (first, second)]
    if systems[0] != systems[1] or systems[0] not in {"GCJ-02", "WGS-84", "WGS84"}:
        return False
    try:
        points = []
        for row in (first, second):
            coords = row.get("coordinates") if isinstance(row.get("coordinates"), dict) else {}
            lng = float(coords.get("lng", row.get("lon", row.get("lng"))))
            lat = float(coords.get("lat", row.get("lat")))
            if not math.isfinite(lng) or not math.isfinite(lat):
                return False
            points.append((math.radians(lng), math.radians(lat)))
        (lng1, lat1), (lng2, lat2) = points
        a = math.sin((lat1 - lat2) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng1 - lng2) / 2) ** 2
        return 2 * 6371000 * math.asin(min(1.0, math.sqrt(a))) <= 5.0
    except (TypeError, ValueError):
        return False


def inspect_candidate_semantics(rule, candidates, activities, local_pois, provider_rows, previously_shown, *,
                                requirements=None, previously_shown_rows=None):
    report = {"failures": [], "unknowns": []}
    intended = rule.get("activity")
    accepted = {str(item) for item in activities or []}
    allowed = _ACTIVITY_EVIDENCE.get(intended, set())
    concrete = set().union(*_ACTIVITY_EVIDENCE.values())
    positive = set()
    for requirement in requirements or []:
        text = str(requirement or "")
        for activity, pattern in _ACTIVITY_TERMS.items():
            for match in re.finditer(pattern, text, re.I):
                if not re.search(r"不要|不用|不喝|不吃|不打|不再|取消|排除|别", text[max(0, match.start() - 5):match.start()]):
                    positive.add(activity)
    if intended and accepted and not accepted.intersection(allowed):
        if accepted.intersection(concrete):
            report["failures"].append("activity_state_conflict")
        else:
            report["unknowns"].append("activity_state_unverified")
    elif intended and not accepted and intended not in positive:
        report["unknowns"].append("activity_state_unverified")
    forbidden = rule.get("forbidden_activity")
    if forbidden and (accepted.intersection(_ACTIVITY_EVIDENCE.get(forbidden, {forbidden})) or forbidden in positive):
        report["failures"].append("forbidden_activity_retained")
    ids = [row.get("poi_id") for row in candidates if isinstance(row, dict) and row.get("poi_id")]
    if len(ids) != len(candidates) or len(ids) != len(set(ids)):
        report["failures"].append("candidate_identity_invalid_or_duplicate")
    if rule.get("must_not_repeat") and set(ids).intersection(previously_shown):
        report["failures"].append("candidate_repeated")
    if rule.get("must_not_repeat") and any(_same_visible_place(candidate, previous)
            for candidate in candidates if isinstance(candidate, dict)
            for previous in previously_shown_rows or []):
        report["failures"].append("candidate_repeated_identity")
    if any(_same_visible_place(candidate, previous) for index, candidate in enumerate(candidates)
           if isinstance(candidate, dict) for previous in candidates[:index] if isinstance(previous, dict)):
        report["failures"].append("candidate_duplicate_identity")
    for candidate in candidates:
        ident = candidate.get("poi_id") if isinstance(candidate, dict) else None
        if not intended or not ident:
            continue
        local = local_pois.get(ident)
        supported = bool(local and set(local.get("activities") or []).intersection(allowed))
        if str(ident).startswith("amap:"):
            matching = [raw for raw in provider_rows if _provider_record_matches(candidate, raw)]
            if not supported:
                supported = any(_provider_supports(intended, raw) for raw in matching)
            for raw in matching:
                code, kind = str(raw.get("typecode") or ""), candidate.get("type")
                if (code.startswith("08") and kind == "dining") or (code.startswith("05") and kind == "sports"):
                    report["failures"].append("candidate_type_conflict:" + str(ident))
                    break
        if not supported:
            report["unknowns"].append("candidate_activity_unverified:" + str(ident))
    return report


def inspect_route_distance_requirement(rule, observed):
    result = {"failures": [], "unknowns": []}
    if rule.get("route_distance_absent"):
        if "route_max_distance_m" not in observed:
            result["unknowns"].append("route_distance_unverified")
        elif observed["route_max_distance_m"] is not None:
            result["failures"].append("unrequested_route_distance")
    return result


def inspect_stale_center_requirement(previous, current, requirements, lookup):
    if not previous or not current or previous == current:
        return []
    for requirement in requirements or []:
        match = re.fullmatch(r"\s*(.+?)\s*(?:附近|周边|周围|旁边|一带)\s*", str(requirement or ""))
        if match and _id(match.group(1).strip(), lookup) == previous:
            return ["previous_search_center_retained_in_requirements"]
    return []


def inspect_time_only_scope(observed, calls, lookup):
    """Verify the executed search did not impose an unrequested radius."""
    speeds = {"walk": 4.5, "bike": 14.0, "drive": 25.0}
    mode, duration, origin = observed.get("mode"), observed.get("max_duration_s"), observed.get("origin")
    if mode not in speeds or type(duration) not in (int, float) or duration <= 0 or not origin:
        return None
    valid, uncertain = False, False
    for call in calls or []:
        name, args, result = call.get("tool"), call.get("args") or {}, call.get("result") or {}
        if name not in {"search_poi_candidates", "find_reachable_places"} or result.get("error"):
            continue
        valid = True
        if _id(args.get("start"), lookup) != origin or args.get("mode") != mode:
            return False
        anchor = args.get("search_anchor")
        if anchor and _id(anchor, lookup) != origin:
            return False
        constraints = args.get("constraints") or {}
        if constraints.get("max_distance_m") is not None or constraints.get("max_duration_s") != duration:
            return False
        if name == "search_poi_candidates":
            bound = max(100, math.ceil(duration * speeds[mode] * 1000 / 3600))
            distance = args.get("max_distance_m")
            if type(distance) not in (int, float) or distance < bound:
                return False
        else:
            if args.get("max_time_min") != duration / 60:
                return False
            scope = call.get("trusted_search_scope")
            if not isinstance(scope, dict):
                uncertain = True
            elif scope.get("_search_scope_distance_m") is not None:
                return False
    return None if not valid or uncertain else True


def inspect_empty_search(rule, observed, result, calls, lookup):
    from spatial.poi_taxonomy import ACTIVITY_SUBCATEGORIES
    text = str(result.get("assistant_text") or result.get("message") or "")
    if re.search(r"(?:说明|表明|意味着|证明)(?:(?!可能|或许|无法确认|不能确定|不确定).){0,35}"
                 r"(?:路网|地图|数据).{0,15}(?:不足|缺失|不全|不完整|不完善|有误)", text):
        return {"verdict": "fail", "reason": "empty_search_unverified_data_cause"}
    if result.get("candidates") or not re.search(r"未找到|没找到|没查到|未查到|暂未查到|未检索到|没检索到|"
            r"(?:未|没有)返回.{0,16}(?:候选|结果|记录)|暂无.{0,8}(?:候选|结果)|没有.{0,5}(?:查到|找到)", text):
        return {"verdict": "unknown", "reason": "empty_search_message_unverified"}
    if rule.get("must_not_repeat"):
        return {"verdict": "unknown", "reason": "exhaustion_of_unshown_candidates_unverified"}
    expected_center = observed.get("search_anchor") or observed.get("origin")
    expected_activity = rule.get("activity")
    accepted_activities = _ACTIVITY_EVIDENCE.get(expected_activity, set())
    keywords = {"basketball": ("篮球",), "badminton": ("羽毛球",), "coffee": ("咖啡",),
                "meal": ("餐饮", "餐厅", "吃饭", "食堂", "快餐", "用餐")}
    for call in reversed(calls or []):
        args, answer = call.get("args") or {}, call.get("result") or {}
        center = _id(args.get("search_anchor") or args.get("start"), lookup)
        activities = set(args.get("activities") or [])
        categories = set(args.get("subcategories") or ([args["subcategory"]] if args.get("subcategory") else []))
        full_categories = set(ACTIVITY_SUBCATEGORIES.get(expected_activity, ()))
        category_coverage = not categories or bool(full_categories and full_categories.issubset(categories))
        keyword = str(args.get("keyword") or "")
        keyword_supported = not keyword or any(word in keyword for word in keywords.get(expected_activity, ()))
        if (call.get("tool") in {"search_poi_candidates", "find_reachable_places"}
                and expected_center and center == expected_center
                and args.get("max_distance_m") == observed.get("radius_m")
                and args.get("mode") == observed.get("mode")
                and activities.intersection(accepted_activities) and keyword_supported and category_coverage
                and (not observed.get("avoid_steps") or (args.get("constraints") or {}).get("avoid_steps") is True)
                and answer.get("error") is None and answer.get("count") == 0 and not answer.get("has_more")
                and answer.get("online_status") in {None, "ok", "not_requested"}):
            return {"verdict": "pass", "reason": "matching_empty_tool_search_and_cautious_message"}
    return {"verdict": "unknown", "reason": "empty_search_scope_or_category_unverified"}


def inspect_start_clarification(turn, result):
    expected = any(check.get("field") == "clarification_slot" and check.get("op") == "eq"
                   and check.get("value") == "start" for check in turn.get("checks") or [])
    if not expected:
        return {"verdict": "unknown", "reason": "clarification_not_specified"}
    if result.get("response_kind") == "route" or result.get("recommended_edge_ids"):
        return {"verdict": "fail", "reason": "route_delivered_without_required_start"}
    text = str(result.get("assistant_text") or result.get("message") or "")
    clarification = result.get("clarify") if isinstance(result.get("clarify"), dict) else {}
    question = str(clarification.get("question") or text)
    asks_origin = (re.search(r"出发|起点|位置|所在|从哪|在哪", text) is not None
                   and re.search(r"哪|什么|请(?:先|告诉|提供)|告诉我|选择", text) is not None)
    if (text and asks_origin and question and clarification.get("slot") in (None, "start")
            and result.get("response_kind") in {"clarify", "chat"}):
        return {"verdict": "pass", "reason": "visible_question_requests_start"}
    return {"verdict": "unknown", "reason": "start_question_not_independently_confirmed"}


def inspect_destination_clarification(result):
    if result.get("response_kind") == "route" or result.get("recommended_edge_ids"):
        return {"verdict": "fail", "reason": "ambiguous_destination_chosen_without_clarification"}
    text = str(result.get("assistant_text") or result.get("message") or "")
    clarification = result.get("clarify") if isinstance(result.get("clarify"), dict) else {}
    if (result.get("response_kind") in {"clarify", "chat"} and
            re.search(r"哪(?:个|一|座|处|里)|哪个|哪边|哪个校区", text) and
            (clarification.get("slot") in (None, "destination") or clarification.get("options"))):
        return {"verdict": "pass", "reason": "visible_question_requests_destination"}
    return {"verdict": "unknown", "reason": "destination_clarification_unverified"}


def inspect_sport_clarification(result):
    if result.get("candidates") or result.get("response_kind") == "route" or result.get("recommended_edge_ids"):
        return {"verdict": "fail", "reason": "ambiguous_sport_candidates_before_clarification"}
    text = str(result.get("assistant_text") or result.get("message") or "")
    clarification = result.get("clarify") if isinstance(result.get("clarify"), dict) else {}
    if (result.get("response_kind") in {"clarify", "chat"} and
            re.search(r"(?:哪|什么).{0,5}球|球.{0,5}(?:哪|什么)", text) and
            clarification.get("slot") in (None, "sport", "activity")):
        request = result.get("place_request") or result.get("completed_place_request")
        if not isinstance(request, dict) or not isinstance(request.get("activities"), list):
            return {"verdict": "unknown", "reason": "unconfirmed_sport_state_unverified"}
        if request["activities"]:
            return {"verdict": "fail", "reason": "unconfirmed_sport_saved_as_selected"}
        return {"verdict": "pass", "reason": "visible_question_requests_ball_sport"}
    return {"verdict": "unknown", "reason": "sport_clarification_unverified"}


def inspect_expected_no_path(fixture, item, result, lookup):
    if (fixture.get("kind") != "blocked" or
            (fixture.get("active_reference") or {}).get("verdict") != "infeasible"):
        return {"verdict": "unknown", "reason": "independent_no_path_reference_missing"}
    proof = item.get("event_proof") or {}
    expected_ids = proof.get("active_ids")
    consumed = item.get("consumed_event_ids") or []
    if (proof.get("stage") != "active" or not expected_ids or not consumed
            or any(row != expected_ids for row in consumed)):
        return {"verdict": "unknown", "reason": "active_event_consumption_unverified"}
    matching = False
    for call in item.get("tool_observations") or []:
        args, answer = call.get("args") or {}, call.get("result") or {}
        if (call.get("tool") in {"plan_route", "plan_via_route", "plan_itinerary", "plan_tour"}
                and _id(args.get("start"), lookup) == fixture.get("start_poi_id")
                and _id(args.get("end"), lookup) == fixture.get("end_poi_id")
                and args.get("mode") == fixture.get("mode") and answer.get("reason_code") == "graph_no_path"):
            matching = True
            break
    text = str(result.get("assistant_text") or result.get("message") or "")
    if (matching and result.get("failure_code") == "route_not_found"
            and re.search(r"无法|不能|不可|不通|没法|未能|走不通", text)):
        return {"verdict": "pass", "reason": "matching_endpoints_event_and_no_path_message"}
    return {"verdict": "unknown", "reason": "no_path_tool_arguments_or_message_unverified"}
