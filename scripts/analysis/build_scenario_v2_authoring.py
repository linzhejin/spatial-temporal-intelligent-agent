"""Create 120 reviewable campus task cards without asserting empirical gold."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POIS = ROOT / "data/pois.json"
DEFAULT_GRAPH = ROOT / "data/whu_road_network.graphml"
DEFAULT_OVERRIDES = ROOT / "data/edge_overrides.json"
DEFAULT_REVIEWS = ROOT / "data/campus_review_decisions.json"
DEFAULT_OUTPUT = ROOT / "experiments/datasets/scenarios_test_v2_draft.json"

_ANCHORS = (
    ("文理学部", "poi_139", "poi_070", "poi_015", "poi_068"),
    ("工学部", "poi_226", "poi_075", "poi_016", "poi_201"),
    ("信息学部", "poi_449", "poi_307", "poi_240", "poi_253"),
)

# One source-backed start/end pair per family and campus. This is an authoring
# alternative, not adjudicated gold or a claim about real-world access.
_DIVERSE_ANCHORS = (
    ("文理学部", "poi_015", "poi_068",
     ("poi_139", "poi_031", "poi_030", "poi_137", "poi_086", "poi_025", "poi_068", "poi_152"),
     ("poi_070", "poi_069", "poi_082", "poi_153", "poi_079", "poi_074", "poi_067", "poi_141")),
    ("工学部", "poi_016", "poi_201",
     ("poi_226", "poi_181", "poi_182", "poi_227", "poi_432", "poi_040", "poi_201", "poi_230"),
     ("poi_075", "poi_202", "poi_447", "poi_160", "poi_016", "poi_295", "poi_226", "poi_227")),
    ("信息学部", "poi_240", "poi_253",
     ("poi_449", "poi_258", "poi_238", "poi_296", "poi_261", "poi_245", "poi_307", "poi_424"),
     ("poi_307", "poi_254", "poi_446", "poi_297", "poi_237", "poi_261", "poi_449", "poi_296")),
)

_T1 = (
    ("explicit", "从{start}到{end}怎么走？"),
    ("avoid_steps", "从{start}步行到{end}，别走台阶。"),
    ("one_via", "从{start}先去{library}，再去{end}。"),
    ("two_via", "从{start}经过{library}和{other_food}，最后到{end}。"),
    ("bike", "从{start}骑车到{end}。"),
    ("drive", "从{start}驾车到{end}，不可通行就明确告诉我。"),
    ("scenic", "从{start}到{end}，优先看风景。"),
    ("flat", "从{start}到{end}，尽量平坦。"),
)
_T2 = (
    ("meal", "我在{start}，找附近能吃正餐的地方。"),
    ("canteen", "我在{start}，找最近的食堂。"),
    ("coffee", "我在{start}，附近哪里能喝咖啡？"),
    ("basketball", "我在{start}，想打篮球。"),
    ("running", "我在{start}，找适合跑步的地方。"),
    ("study", "我在{start}，想找有依据能自习的地方。"),
    ("shopping", "我在{start}，附近能买东西吗？"),
    ("pickup", "我在{start}，找能取快递的地方。"),
)
_T2_ACTIVITIES = {
    "meal": ["canteen_meal", "restaurant_meal", "fast_food"],
    "canteen": ["canteen_meal"], "coffee": ["coffee"],
    "basketball": ["basketball"], "running": ["running"],
    "study": ["study"], "shopping": ["shopping"],
    "pickup": ["package_pickup"],
}
_T5_CANDIDATE_ACTIVITIES = {
    "food_before_route": _T2_ACTIVITIES["meal"],
    "study_before_route": _T2_ACTIVITIES["study"],
    "sport_then_food": _T2_ACTIVITIES["basketball"],
}
_T3 = (
    ("ball", "我在{start}，想去打球。", "sport"),
    ("ball_steps", "从{start}出发去打球，别走台阶。", "sport"),
    ("ball_bike", "我在{start}，想骑车去打球。", "sport"),
    ("exercise", "我在{start}，想运动。", "activity"),
    ("exercise_near", "我在{start}附近，找个运动的地方。", "activity"),
    ("destination", "我从{start}出发，帮我规划步行路线。", "destination"),
    ("destination_flat", "我从{start}出发，想走平坦的路。", "destination"),
    ("destination_weather", "我在{start}，查天气后再带我过去。", "destination"),
)
_T4 = (
    ("change_mode", "改成骑行，起点和终点不要变。"),
    ("change_strategy", "改成风景优先，地点不要变。"),
    ("add_via", "途中再经过{library}。"),
    ("remove_via", "不经过{library}了，直接到终点。"),
    ("change_start", "起点改为{library}，终点还是原来的。"),
    ("change_end", "终点改为{library}，起点仍是刚才的。"),
    ("add_constraint", "保留原来地点，但不要走台阶。"),
    ("weather_after", "再告诉我天气，不要改路线。"),
)
_T5 = (
    ("route_weather", "查天气并规划从{start}到{end}的步行路线。"),
    ("route_conditions", "从{start}去{end}，顺便查路上有没有已确认管制。"),
    ("food_before_route", "从{start}先找个附近能吃饭的地方，再去{end}。"),
    ("study_before_route", "我从{start}出发，先找能自习的地方，再去{end}。"),
    ("sport_then_food", "从{start}先去打篮球，再找附近能吃饭的地方。"),
    ("two_vias_weather", "查天气，从{start}经过{library}和{other_food}到{end}。"),
    ("unknown_condition", "从{start}到{end}，如果某段路没路况记录，请明确说未知。"),
    ("no_route_then_option", "从{start}到{end}别走台阶；走不了就说明原因并给可行选择。"),
)


def _turn(*, start=None, end=None, via=None, kind="route", status="completed",
          mode="walk", strategy="shortest", constraints=None, outputs=None, slot=None):
    return {
        "status": status, "kind": kind, "feasibility": "unknown" if kind == "route" else "not_applicable",
        "start_poi_id": start, "end_poi_id": end, "via_poi_ids": via or [],
        "travel_mode": mode, "strategy": strategy, "hard_constraints": constraints or {},
        "required_outputs": outputs if outputs is not None else (["route"] if kind == "route" else []),
        "candidate_poi_ids": [], "clarification_slot": slot, "allowed_unknowns": [],
        "intent_label": None, "expected_tools": None,
    }


def build_authoring_bank(poi_path: Path, *, diverse_anchors: bool = False,
                         graph_path: Path = DEFAULT_GRAPH,
                         overrides_path: Path = DEFAULT_OVERRIDES,
                         reviews_path: Path = DEFAULT_REVIEWS) -> dict:
    poi_bytes = poi_path.read_bytes()
    poi_rows = json.loads(poi_bytes)["pois"]
    pois = {row["id"]: row for row in poi_rows}
    graph = None

    def select_nearby(start_poi_id: str, activities: set[str]) -> dict:
        nonlocal graph
        if graph is None:
            from experiments.scenario_v2 import load_frozen_graph
            graph = load_frozen_graph(graph_path, overrides_path, reviews_path)
        from experiments.candidate_reference import rank_nearby_activities

        ranked = rank_nearby_activities(graph, poi_rows, start_poi_id=start_poi_id,
                                        activities=activities)["reachable"]
        if not ranked:
            raise ValueError(f"no source-backed venue for {sorted(activities)}")
        return next((row for row in ranked[:4] if not row["access_over_30m"]), ranked[0])

    cases = []
    for group in ("T1", "T2", "T3", "T4", "T5"):
        templates = {"T1": _T1, "T2": _T2, "T3": _T3, "T4": _T4, "T5": _T5}[group]
        for family_index, family in enumerate(templates):
            family_name = family[0]
            anchors = ([(campus, starts[family_index], ends[family_index], library_id, other_id)
                        for campus, library_id, other_id, starts, ends in _DIVERSE_ANCHORS]
                       if diverse_anchors else _ANCHORS)
            for variant, (campus, start_id, end_id, library_id, other_id) in enumerate(anchors, 1):
                selected = [start_id, end_id, library_id, other_id]
                if any(poi_id not in pois for poi_id in selected):
                    raise ValueError("anchor POI disappeared; reselect source-backed tasks")
                if len({start_id, end_id, library_id, other_id}) < 3:
                    raise ValueError("anchor selection collapses distinct task roles")
                names = {"start": pois[start_id]["name"], "end": pois[end_id]["name"],
                         "library": pois[library_id]["name"],
                         "other_food": pois[other_id]["name"]}
                prompt = family[1].format(**names)
                messages = [prompt]
                turns = []
                extra_source_ids = []
                if group == "T1":
                    via = [library_id] if family_name == "one_via" else (
                        [library_id, other_id] if family_name == "two_via" else [])
                    turns = [_turn(start=start_id, end=end_id, via=via,
                                   mode="bike" if family_name == "bike" else
                                   "drive" if family_name == "drive" else "walk",
                                   strategy="scenic" if family_name == "scenic" else
                                   "flat" if family_name == "flat" else "shortest",
                                   constraints={"avoid_steps": True} if family_name == "avoid_steps" else {})]
                elif group == "T2":
                    turns = [_turn(start=start_id, kind="candidates", mode=None, strategy=None,
                                   outputs=["candidates"])]
                elif group == "T3":
                    constraints = {"avoid_steps": True} if family_name == "ball_steps" else {}
                    turns = [_turn(start=start_id, kind="clarify", status="needs_input",
                                   mode="bike" if family_name == "ball_bike" else None,
                                   strategy=None, constraints=constraints, outputs=[], slot=family[2])]
                elif group == "T4":
                    first_via = [library_id] if family_name == "remove_via" else []
                    first_prompt = (f"从{names['start']}先去{names['library']}，再到{names['end']}。"
                                    if first_via else f"从{names['start']}到{names['end']}，步行。")
                    messages = [first_prompt, prompt]
                    first = _turn(start=start_id, end=end_id, via=first_via)
                    updated = _turn(start=library_id if family_name == "change_start" else start_id,
                                    end=library_id if family_name == "change_end" else end_id,
                                    via=[library_id] if family_name == "add_via" else [],
                                    mode="bike" if family_name == "change_mode" else "walk",
                                    strategy="scenic" if family_name == "change_strategy" else "shortest",
                                    constraints={"avoid_steps": True} if family_name == "add_constraint" else {},
                                    outputs=["route", "weather"] if family_name == "weather_after" else None)
                    turns = [first, updated]
                else:
                    via = ([library_id, other_id] if family_name == "two_vias_weather" else [])
                    outputs = ["route", "weather"] if "weather" in family_name else ["route"]
                    if family_name == "route_conditions":
                        outputs = ["route", "road_conditions"]
                    if family_name in {"food_before_route", "study_before_route", "sport_then_food"}:
                        # The user has not selected a meal/study/sport venue yet.
                        # A completed route at this point would invent a destination.
                        turns = [_turn(start=start_id,
                                       end=None if family_name == "sport_then_food" else end_id,
                                       kind="candidates", status="needs_input",
                                       outputs=["candidates"])]
                        activity = set(_T5_CANDIDATE_ACTIVITIES[family_name])
                        venue_id = select_nearby(start_id, activity)["poi_id"]
                        extra_source_ids.append(venue_id)
                        if family_name == "sport_then_food":
                            food_id = select_nearby(venue_id, set(_T2_ACTIVITIES["meal"]))["poi_id"]
                            if food_id == venue_id:
                                raise ValueError("sport venue and food venue must differ")
                            extra_source_ids.append(food_id)
                            messages.append(
                                f"就去{pois[venue_id]['name']}打篮球，吃饭选{pois[food_id]['name']}；"
                                "请从刚才的起点出发，按顺序给出全程步行路线。")
                            final_end = food_id
                        else:
                            messages.append(
                                f"就去{pois[venue_id]['name']}，然后到刚才说的终点；"
                                "请从出发地按顺序给出全程步行路线。")
                            final_end = end_id
                        turns.append(_turn(start=start_id, end=final_end, via=[venue_id]))
                    else:
                        turns = [_turn(start=start_id, end=end_id, via=via,
                                       constraints={"avoid_steps": True} if family_name == "no_route_then_option" else {},
                                       outputs=outputs)]
                prefix = "v2h" if diverse_anchors else "v2"
                case = {
                    "id": f"{prefix}_{group}_{family_name}_{variant}", "category": group,
                    "template_group": f"{prefix}_{group}_{family_name}",
                    "od_group": f"{prefix}_{group}_{family_name}_{campus}",
                    "source_group": ("campus_poi_diverse_authoring_v2"
                                     if diverse_anchors else "campus_poi_authoring_v2"),
                    "source_poi_ids": list(dict.fromkeys([*selected, *extra_source_ids])),
                    "snapshot_id": "PENDING_FROZEN_SNAPSHOT",
                    "messages": [{"role": "user", "content": text} for text in messages],
                    "expected_turns": turns,
                    "adjudication": {"status": "draft", "method": None,
                                     "reviewer": None, "evidence": []},
                }
                activities = (_T2_ACTIVITIES.get(family_name) if group == "T2" else
                              _T5_CANDIDATE_ACTIVITIES.get(family_name) if group == "T5"
                              else None)
                if activities:
                    case["candidate_spec"] = {
                        "activities": activities, "coordinates": "gcj02",
                        "max_display": 4, "rank_tolerance_m": 30,
                    }
                cases.append(case)
    return {
        "protocol_version": "scenario-v2",
        "version": ("scenarios-test-v2-diverse-authoring-draft" if diverse_anchors
                    else "scenarios-test-v2-authoring-draft"),
        "split": "test", "repeats": 3,
        "authoring_status": "UNADJUDICATED CANDIDATES. No frozen graph, independent gold or blind test result.",
        "source_poi_sha256": hashlib.sha256(poi_bytes).hexdigest(),
        "source_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "source_overrides_sha256": hashlib.sha256(overrides_path.read_bytes()).hexdigest(),
        "source_reviews_sha256": hashlib.sha256(reviews_path.read_bytes()).hexdigest(),
        "poi_sha256": hashlib.sha256(poi_bytes).hexdigest(),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pois", type=Path, default=DEFAULT_POIS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diverse-anchors", action="store_true",
                        help="Use 24 distinct first-turn campus anchors per task group")
    args = parser.parse_args()
    from experiments.scenario_v2 import validate_dataset

    bank = build_authoring_bank(args.pois, diverse_anchors=args.diverse_anchors,
                                graph_path=args.graph, overrides_path=args.overrides,
                                reviews_path=args.reviews)
    validate_dataset(bank)
    args.output.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(bank["cases"]),
                      "status": "draft"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
