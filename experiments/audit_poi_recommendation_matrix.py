"""Capture a repeatable, development-only campus POI distance-ranking audit.

This exercises the real local POI master and road graph. It uses an empty road
condition snapshot and is not a production/live-conditions or paper result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.tools import _ensure_graph, _tool_search_poi_candidates  # noqa: E402
from spatial.poi import find_poi_ambiguous  # noqa: E402
from spatial.poi_taxonomy import (  # noqa: E402
    ACTIVITY_SUBCATEGORIES,
    matches_activities,
)
from spatial.coord_transform import gcj02_to_wgs84  # noqa: E402
from spatial.routing_index import get_routing_index  # noqa: E402


ORIGIN_GROUPS = {
    "信息学部": ("信息学部", "玉兰2门"),
    "工学部": ("工学部", "澄波门"),
    "文理学部": ("文理学部", "凌波门"),
}
ORIGINS = tuple(
    (campus, name)
    for campus, names in ORIGIN_GROUPS.items()
    for name in names
)
QUERY_LABELS = {
    "basketball": "篮球场", "badminton": "羽毛球场", "table_tennis": "乒乓球馆",
    "football": "足球场", "volleyball": "排球场", "tennis": "网球场",
    "running": "跑步场地", "fitness": "健身场馆", "swimming": "游泳场馆",
    "meal": "餐饮", "canteen_meal": "食堂", "restaurant_meal": "餐厅",
    "fast_food": "快餐小吃", "coffee": "咖啡店", "tea_drink": "茶饮店",
    "study": "学习场所", "shopping": "超市", "medical": "医疗服务",
    "package_pickup": "快递点", "postal_service": "邮政服务",
    "view_lake": "湖景点", "view_sakura": "赏樱点",
}
CASES = tuple({
    "activity": activity,
    "query": f"附近有哪些{QUERY_LABELS[activity]}",
    "activities": [activity],
    "subcategories": list(subcategories),
} for activity, subcategories in ACTIVITY_SUBCATEGORIES.items())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _independent_haversine_m(lon_a: float, lat_a: float,
                             lon_b: float, lat_b: float) -> float:
    radius_m = 6_371_008.8
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    d_phi = phi_b - phi_a
    d_lon = math.radians(lon_b - lon_a)
    value = (math.sin(d_phi / 2) ** 2
             + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lon / 2) ** 2)
    return 2 * radius_m * math.asin(math.sqrt(min(1.0, value)))


def _independent_nearest_node(graph, lng: float, lat: float):
    cos_lat = math.cos(math.radians(lat))
    best_node = None
    best_distance = math.inf
    for node, data in graph.nodes(data=True):
        try:
            dx = (float(data["x"]) - lng) * cos_lat
            dy = float(data["y"]) - lat
        except (KeyError, TypeError, ValueError):
            continue
        distance = dx * dx + dy * dy
        if distance < best_distance:
            best_node, best_distance = node, distance
    return best_node


def _poi_wgs84(poi: dict) -> tuple[float, float] | None:
    coordinates = poi.get("coordinates") or {}
    try:
        lng, lat = float(coordinates["lng"]), float(coordinates["lat"])
        return gcj02_to_wgs84(lng, lat)
    except (KeyError, TypeError, ValueError):
        return None


def _independent_distance_table(graph, origin: dict, pois: list[dict], *,
                                max_distance_m: float,
                                max_access_snap_m: float,
                                mode: str = "walk") -> dict[str, dict]:
    """Compute full-corpus reference distances without candidate search tools.

    The shared mode filter and activity taxonomy are the declared data contract;
    node snapping, reachability distances, cutoff, and ranking are recomputed here.
    """
    mode_graph = get_routing_index(graph).for_mode(mode).graph
    origin_wgs = _poi_wgs84(origin)
    if origin_wgs is None:
        raise ValueError(f"起点缺少有效坐标：{origin.get('id')}")
    start_node = _independent_nearest_node(mode_graph, *origin_wgs)
    if start_node is None:
        raise ValueError("独立路网扫描无法定位起点节点")
    start_data = mode_graph.nodes[start_node]
    start_access = _independent_haversine_m(
        origin_wgs[0], origin_wgs[1], float(start_data["x"]), float(start_data["y"]),
    )
    if start_access > max_access_snap_m:
        return {}
    network_distances = nx.single_source_dijkstra_path_length(
        mode_graph, start_node, cutoff=max_distance_m, weight="length",
    )
    table = {}
    for poi in pois:
        poi_id = str(poi.get("id") or "")
        poi_wgs = _poi_wgs84(poi)
        if not poi_id or poi_wgs is None:
            continue
        node = _independent_nearest_node(mode_graph, *poi_wgs)
        if node is None or node not in network_distances:
            continue
        node_data = mode_graph.nodes[node]
        access = _independent_haversine_m(
            poi_wgs[0], poi_wgs[1], float(node_data["x"]), float(node_data["y"]),
        )
        if access > max_access_snap_m:
            continue
        exact_distance = start_access + float(network_distances[node]) + access
        if exact_distance <= max_distance_m:
            table[poi_id] = {"distance_m": round(exact_distance),
                             "exact_distance_m": exact_distance}
    return table


def independent_reachable_candidate_rank(graph, origin: dict, pois: list[dict], *,
                                          max_distance_m: float,
                                          max_access_snap_m: float,
                                          limit: int,
                                          mode: str = "walk") -> dict:
    table = _independent_distance_table(
        graph, origin, pois, max_distance_m=max_distance_m,
        max_access_snap_m=max_access_snap_m, mode=mode,
    )
    names = {str(poi.get("id") or ""): str(poi.get("name") or "") for poi in pois}
    ordered = sorted(
        ({"poi_id": poi_id, "distance_m": values["distance_m"],
          "name": names.get(poi_id, "")}
         for poi_id, values in table.items()),
        key=lambda item: (item["distance_m"], item["name"]),
    )
    return {"reachable_count": len(ordered), "candidates": ordered[:limit]}


def ranked_candidates_match(expected: dict, actual: list[dict], *,
                            total_count: int | None) -> bool:
    if total_count != expected["reachable_count"]:
        return False
    expected_items = expected["candidates"]
    actual_items = []
    for item in actual:
        try:
            poi_id = str(item.get("poi_id") or item.get("id") or "")
            distance = int(item.get("network_distance_m", item.get("distance_m")))
        except (TypeError, ValueError):
            return False
        actual_items.append({"poi_id": poi_id, "distance_m": distance})
    if [item["poi_id"] for item in actual_items] != [
            item["poi_id"] for item in expected_items]:
        return False
    # The independent reference and production endpoint helper both round
    # straight-line access to integer metres. A 1 m display delta can occur at
    # the half-metre boundary; larger differences still indicate drift.
    return all(abs(actual_item["distance_m"] - expected_item["distance_m"]) <= 1
               for expected_item, actual_item in zip(expected_items, actual_items))


def candidate_distance_deltas(expected: dict, actual: list[dict]) -> list[int]:
    expected_by_id = {item["poi_id"]: item["distance_m"]
                      for item in expected["candidates"]}
    deltas = []
    for item in actual:
        try:
            poi_id = str(item.get("poi_id") or item.get("id") or "")
            distance = int(item.get("network_distance_m", item.get("distance_m")))
            if poi_id in expected_by_id:
                deltas.append(abs(distance - expected_by_id[poi_id]))
        except (TypeError, ValueError):
            continue
    return deltas


def _eligible_activity_pois(pois: list[dict], case: dict) -> list[dict]:
    subcategories = set(case["subcategories"])
    unique = {}
    for poi in pois:
        if (poi.get("subcategory") in subcategories
                and matches_activities(poi, case["activities"])):
            poi_id = str(poi.get("id") or "")
            if poi_id:
                unique.setdefault(poi_id, poi)
    return list(unique.values())


def _activity_corpus_counts(pois: list[dict], cases: tuple[dict, ...] | list[dict]) -> dict[str, int]:
    """Count only POIs satisfying both the requested activity and categories."""
    return {
        case["activities"][0]: len(_eligible_activity_pois(pois, case))
        for case in cases
    }


def collect_matrix() -> dict:
    graph = _ensure_graph()
    if graph is None:
        raise RuntimeError("本地校园路网不可用，无法执行 POI 距离矩阵")

    poi_file = ROOT / "data" / "pois.json"
    network_file = ROOT / "data" / "whu_road_network.graphml"
    poi_data = json.loads(poi_file.read_text(encoding="utf-8"))
    all_pois = poi_data.get("pois", [])
    corpus_counts = _activity_corpus_counts(all_pois, CASES)
    results = []
    for origin_campus, origin_name in ORIGINS:
        origin, alternatives = find_poi_ambiguous(origin_name)
        if origin is None or alternatives:
            raise RuntimeError(f"起点无法唯一解析：{origin_name}")
        max_distance_m = 10000
        max_access_snap_m = 80
        independent_distances = _independent_distance_table(
            graph, origin, all_pois, max_distance_m=max_distance_m,
            max_access_snap_m=max_access_snap_m,
        )
        for case in CASES:
            response, _artifact = _tool_search_poi_candidates(
                {
                    "start": {"name": origin_name, "type": "poi"},
                    "activities": list(case["activities"]),
                    "subcategories": list(case["subcategories"]),
                    "limit": 4,
                },
                {
                    "query": case["query"],
                    "_route_graph_snapshot": graph,
                    "_route_conditions_snapshot": [],
                    "_route_weather_snapshot": None,
                },
            )
            candidates = response.get("candidates") or []
            distances = [item.get("network_distance_m") for item in candidates]
            matches = [matches_activities(item, case["activities"]) for item in candidates]
            eligible = _eligible_activity_pois(all_pois, case)
            eligible_ids = {str(poi.get("id") or "") for poi in eligible}
            names = {str(poi.get("id") or ""): str(poi.get("name") or "")
                     for poi in eligible}
            ranked = sorted(
                ({"poi_id": poi_id, "distance_m": distance["distance_m"],
                  "name": names.get(poi_id, "")}
                 for poi_id, distance in independent_distances.items()
                 if poi_id in eligible_ids),
                key=lambda item: (item["distance_m"], item["name"]),
            )
            expected = {"reachable_count": len(ranked),
                        "candidates": ranked[:4]}
            reported_total = response.get("total_count", response.get("count", len(candidates)))
            full_corpus_rank_match = ranked_candidates_match(
                expected, candidates, total_count=reported_total,
            )
            distance_deltas = candidate_distance_deltas(expected, candidates)
            results.append({
                "origin": {
                    "campus_group": origin_campus,
                    "name": origin_name,
                    "poi_id": origin.get("id"),
                    "coordinates_gcj02": origin.get("coordinates"),
                },
                "query": case["query"],
                "activity": case["activities"][0],
                "activities": list(case["activities"]),
                "result": {
                    "error": response.get("error"),
                    "master_match_count": corpus_counts[case["activities"][0]],
                    "total_count": reported_total,
                    "independent_reachable_count": expected["reachable_count"],
                    "independent_top_candidates": expected["candidates"],
                    "full_corpus_rank_match": full_corpus_rank_match,
                    "max_candidate_distance_delta_m": max(distance_deltas, default=None),
                    "returned_count": len(candidates),
                    "has_more": bool(response.get("has_more", False)),
                    "distance_sorted": distances == sorted(distances),
                    "all_candidates_match_activity": all(matches),
                    "reachability_basis": response.get("reachability_basis"),
                    "message": response.get("message", ""),
                    "candidates": [
                        {
                            "poi_id": item.get("poi_id"),
                            "name": item.get("name"),
                            "subcategory": item.get("subcategory"),
                            "activities": item.get("activities", []),
                            "distance_m": item.get("network_distance_m"),
                            "access_snap_m": item.get("access_snap_m"),
                            "access_link_verified": item.get("access_link_verified", False),
                        }
                        for item in candidates
                    ],
                },
            })

    checks = []
    for case in CASES:
        group = [item for item in results if item["query"] == case["query"]]
        checks.append({
            "name": f"{case['query']}：主数据有匹配项时，六个学部起点均返回用途匹配且按距离排序的结果",
            "passed": len(group) == len(ORIGINS) and all(
                not item["result"]["error"]
                and (item["result"]["master_match_count"] == 0
                     or item["result"]["returned_count"] > 0
                     and item["result"]["distance_sorted"])
                and item["result"]["all_candidates_match_activity"]
                and item["result"]["full_corpus_rank_match"]
                for item in group
            ),
        })
        pair_checks = []
        for campus, names in ORIGIN_GROUPS.items():
            pair = [item for item in group if item["origin"]["campus_group"] == campus]
            signatures = [
                tuple((candidate["poi_id"], candidate["distance_m"])
                      for candidate in item["result"]["candidates"])
                for item in pair
            ]
            pair_checks.append(
                len(pair) == len(names)
                and (corpus_counts[case["activities"][0]] == 0
                     or all(signatures) and len(set(signatures)) > 1)
            )
        checks.append({
            "name": f"{case['query']}：三学部内更换起点会重新计算距离/候选顺序",
            "passed": all(pair_checks),
        })

    return {
        "scope": "poi_recommendation_origin_distance_audit",
        "snapshot_kind": "development_diagnostic",
        "formal_eligible": False,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "只使用本地 POI 主数据与校园路网，不代表线上 API 或生产路况。",
            "路况事件为空；距离含起点/地点到最近道路的直线接驳估算，接驳通行尚未核实。",
            "用于开发验收，不是独立标注的论文测试集或实地真值。",
        ],
        "inputs": {
            "poi_file": "data/pois.json",
            "poi_sha256": sha256_file(poi_file),
            "network_file": "data/whu_road_network.graphml",
            "network_sha256": sha256_file(network_file),
            "poi_count": len(all_pois),
            "graph_node_count": graph.number_of_nodes(),
            "graph_directed_edge_count": graph.number_of_edges(),
            "active_road_conditions": 0,
            "activities_covered": len(CASES),
            "origin_count": len(ORIGINS),
            "origin_groups": {campus: list(names) for campus, names in ORIGIN_GROUPS.items()},
        },
        "checks": checks,
        "all_checks_passed": all(item["passed"] for item in checks),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="结果 JSON 路径；默认写入唯一时间戳文件")
    args = parser.parse_args()
    default_name = f"poi_recommendation_matrix_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output = args.output or ROOT / "output" / "experiments" / default_name
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        parser.error(f"拒绝覆盖已有结果文件：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = collect_matrix()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "all_checks_passed": report["all_checks_passed"],
        "poi_count": report["inputs"]["poi_count"],
        "graph_nodes": report["inputs"]["graph_node_count"],
        "graph_directed_edges": report["inputs"]["graph_directed_edge_count"],
        "checks": report["checks"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
