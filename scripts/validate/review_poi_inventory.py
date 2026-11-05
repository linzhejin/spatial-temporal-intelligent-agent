"""Create a row-level, read-only evidence review for the campus POI inventory.

This report distinguishes source-backed identity from field verification. A map
snap only measures proximity to the routed graph; it does not establish an
entrance or a safe pedestrian connection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _read(path: Path, default=None):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized(value):
    value = str(value or "").casefold()
    value = re.sub(r"[\s\u3000\-—_()（）·,，.。]", "", value)
    return value.translate(str.maketrans("零一二两三四五六七八九", "01223456789"))


def _distance_m(a, b):
    mean_lat = math.radians((a[1] + b[1]) / 2)
    return math.hypot((a[0] - b[0]) * 111320 * math.cos(mean_lat),
                      (a[1] - b[1]) * 110540)


def build_report():
    data_path = ROOT / "data" / "pois.json"
    review_path = ROOT / "data" / "campus_review_decisions.json"
    candidates_path = ROOT / "data" / "pois_osm_candidates.json"
    graph_path = ROOT / "data" / "whu_road_network.graphml"
    data = _read(data_path, {})
    pois = data.get("pois", [])
    review = _read(review_path, {})
    decisions = {(item.get("source"), item.get("source_id")): item
                 for item in review.get("poi_decisions", [])}
    candidate_data = _read(candidates_path, {}) or {}
    osm_by_poi = defaultdict(list)
    for candidate in candidate_data.get("candidates", []):
        stable_id = candidate.get("matched_poi_id") or candidate.get("poi_id")
        if stable_id:
            osm_by_poi[stable_id].append(candidate)
    for item in decisions.values():
        if item.get("poi_id"):
            osm_by_poi[item["poi_id"]].append(item)

    valid_types = {"study", "dining", "landmark", "scenery", "gate", "service",
                   "sports", "medical", "building", "area", "transport"}
    coords_by_id = {}
    normalized_names = defaultdict(list)
    for poi in pois:
        coord = poi.get("coordinates") or {}
        try:
            point = (float(coord["lng"]), float(coord["lat"]))
            if not (-180 <= point[0] <= 180 and -90 <= point[1] <= 90):
                point = None
        except (KeyError, TypeError, ValueError):
            point = None
        coords_by_id[poi.get("id")] = point
        normalized_names[_normalized(poi.get("name"))].append(poi)

    duplicates = defaultdict(list)
    for name, rows in normalized_names.items():
        if not name or len(rows) < 2:
            continue
        for row in rows:
            for other in rows:
                if other is row:
                    continue
                first, second = coords_by_id.get(row.get("id")), coords_by_id.get(other.get("id"))
                if first is None or second is None or _distance_m(first, second) <= 20:
                    duplicates[row.get("id")].append(other.get("id"))

    snaps = {}
    graph_status = "unavailable"
    if graph_path.exists():
        try:
            import networkx as nx
            from spatial.coord_transform import gcj02_to_wgs84
            from spatial.network import get_nearest_node
            graph = nx.read_graphml(graph_path, node_type=int)
            graph_status = "available_proximity_only"
            for poi in pois:
                point = coords_by_id.get(poi.get("id"))
                if point is None:
                    continue
                lng, lat = gcj02_to_wgs84(*point)
                node = get_nearest_node(graph, lng, lat)
                node_lng, node_lat = float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])
                snaps[poi.get("id")] = {"nearest_node": int(node),
                                         "snap_distance_m": round(_distance_m((lng, lat), (node_lng, node_lat)), 1)}
        except Exception as exc:  # report the limitation without aborting the inventory.
            graph_status = "calculation_failed:" + type(exc).__name__

    rows = []
    counts = Counter()
    for poi in pois:
        poi_id = poi.get("id")
        coord = coords_by_id.get(poi_id)
        evidence = []
        direct_refs = poi.get("source_refs") or poi.get("sources") or []
        for ref in direct_refs:
            if isinstance(ref, str):
                evidence.append({"source": ref})
            elif isinstance(ref, dict):
                evidence.append({key: ref.get(key) for key in ("source", "id", "url", "license")
                                 if ref.get(key) is not None})
        reviewed_osm = []
        for item in osm_by_poi.get(poi_id, []):
            source_id = item.get("source_id") or item.get("osm_id")
            decision = item.get("decision") or item.get("review_decision")
            record = decisions.get((item.get("source", "OpenStreetMap"), source_id), {})
            if not decision:
                decision = record.get("decision")
            reviewed_osm.append({"source": item.get("source", "OpenStreetMap"),
                                 "source_id": source_id,
                                 "decision": decision or "unreviewed",
                                 "evidence": record.get("evidence") or item.get("evidence")})
            if decision in {"existing_name_match", "included", "alias_or_source_enrichment", "campus_corrected"}:
                evidence.append({"source": "OpenStreetMap", "id": source_id,
                                 "decision": decision,
                                 "evidence": record.get("evidence") or item.get("evidence")})
        status = poi.get("verification_status") or data.get("provenance_defaults", {}).get(
            "verification_status", "legacy_unverified")
        if poi.get("recommendation_status") == "reported_absent_pending_review":
            disposition = "exclude_pending_user_report"
            rationale = "用户反馈该具体门店/位置不存在；在线查询同名同位置结果也须暂时排除，不能据此推断其他分店。"
        elif poi.get("type") == "dining" and poi.get("subcategory") in {
                "restaurant", "fastfood", "coffee", "tea_drink"}:
            disposition = "online_lookup_only"
            rationale = "历史静态商户记录不主动推荐；规划前按商户名称实时查询并核对位置。静态记录保留用于别名与反馈审计。"
        elif evidence:
            disposition = "source_backed_provisional"
            rationale = "存在逐条来源或明确接受的开放数据审查记录；这不证明当前营业、入口或现场可通行。"
        else:
            disposition = "hold_unverified"
            rationale = "没有可定位到该条记录的逐条来源；不进入主动推荐，明确搜索时需说明目录证据不足。"
        coord_valid = coord is not None
        issue_codes = []
        if not coord_valid:
            issue_codes.append("missing_or_invalid_coordinates")
        if poi.get("type") not in valid_types:
            issue_codes.append("type_outside_known_vocabulary")
        if not poi.get("campus"):
            issue_codes.append("campus_unassigned")
        if duplicates.get(poi_id):
            issue_codes.append("exact_name_duplicate_needs_review")
        if poi_id not in snaps:
            issue_codes.append("network_snap_unavailable_or_invalid")
        row = {
            "poi_id": poi_id, "name": poi.get("name"),
            "aliases": list(poi.get("aliases") or []),
            "type": poi.get("type"), "subcategory": poi.get("subcategory"),
            "campus": poi.get("campus"),
            "coordinate_system": poi.get("coordinate_system") or "GCJ-02 (catalog convention)",
            "coordinates": poi.get("coordinates"),
            "coordinate_valid": coord_valid,
            "verification_status": status,
            "source_evidence": evidence,
            "osm_review_records": reviewed_osm,
            "near_exact_name_ids_within_20m": sorted(set(duplicates.get(poi_id, []))),
            "road_network_proximity": snaps.get(poi_id),
            "recommendation_disposition": disposition,
            "classification_rationale": rationale,
            "issue_codes": issue_codes,
            "limitations": ["source presence does not prove current existence or exact location",
                            "road snap is proximity only; entrance and access are unverified"],
        }
        rows.append(row)
        counts[disposition] += 1

    return {
        "schema_version": 1,
        "created_local_date": "2026-11-05",
        "scope": "read-only row-by-row evidence classification for all POIs",
        "source_fingerprints": {str(path.relative_to(ROOT)): _sha256(path)
                                for path in (data_path, review_path, candidates_path, graph_path)
                                if path.exists()},
        "declared_count": data.get("count"),
        "actual_count": len(pois),
        "row_count_matches_declared": len(pois) == data.get("count"),
        "disposition_counts": dict(counts),
        "road_graph_status": graph_status,
        "interpretation": "该清单分类的是现有记录可支持的推荐处置，不等于逐店现场核实。来源、位置、校区和路网接入仍须按 issue_codes 逐项补证。",
        "records": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "local_acceptance" / "20261105_candidate_v18" / "poi_data_review.json")
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"清单已生成：{args.output}")
    print(f"逐条记录：{report['actual_count']} / 声明 {report['declared_count']}；处置分类：{report['disposition_counts']}")


if __name__ == "__main__":
    main()
