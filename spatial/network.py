"""
OSMnx 路网加载与缓存模块

负责下载/加载武大校园路网（步行道/小路），支持 GraphML 缓存。
首次启动时从 OSM 下载，后续直接加载缓存文件（< 1 秒）。
"""

import json
import logging
import os
import ast
from typing import Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

_G: Optional[nx.MultiDiGraph] = None
_annotation_coverage_rate: Optional[float] = None


def _cache_path() -> str:
    try:
        from config import ROAD_NETWORK_CACHE
        return ROAD_NETWORK_CACHE
    except ImportError:
        return os.path.join(os.path.dirname(__file__), "..", "data", "whu_road_network.graphml")


def _bbox() -> dict:
    try:
        from config import WHU_BBOX
        return WHU_BBOX
    except ImportError:
        return {"north": 30.5480, "south": 30.5280, "east": 114.3750, "west": 114.3500}


def get_network() -> Optional[nx.MultiDiGraph]:
    """
    获取已加载的路网图。

    Returns:
        networkx MultiDiGraph 或 None（未加载）
    """
    return _G


def load_or_download_network(bbox: Optional[dict] = None) -> nx.MultiDiGraph:
    """
    加载缓存路网或从 OSM 下载。

    优先加载 GraphML 缓存；缓存不存在时下载并保存。

    Args:
        bbox: 边界框 dict(north, south, east, west)，默认从 config 读取

    Returns:
        networkx MultiDiGraph（有向图，含步行道/小路）

    Raises:
        RuntimeError: 下载失败且无缓存时抛出
    """
    global _G

    if _G is not None:
        return _G

    global _annotation_coverage_rate

    cache = _cache_path()
    bbox = bbox or _bbox()

    if os.path.exists(cache):
        try:
            logger.info("从缓存加载路网: %s", cache)
            _G = _load_graphml(cache)
        except RuntimeError:
            logger.warning("路网缓存损坏，删除后重新下载: %s", cache)
            try:
                os.remove(cache)
            except OSError:
                pass
            _G = None

    if _G is None:
        logger.info("从 OSM 下载路网...")
        _G = _download_network(bbox)
        try:
            _save_graphml(_G, cache)
            logger.info("路网已缓存: %s", cache)
        except Exception as e:
            logger.warning("路网缓存失败（不影响运行）: %s", e)

    ann_path = _annotations_path()
    expected_network_version = None
    if os.path.exists(ann_path):
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                snapshot_version = json.load(f).get("network_version")
            if snapshot_version:
                from scripts.network.build_edge_attribute_master import graph_fingerprint
                expected_network_version = graph_fingerprint(_G)
        except Exception:
            # _merge_annotations reports malformed snapshots through the normal path below.
            expected_network_version = None

    _mark_osm_provenance(_G)

    if os.path.exists(ann_path):
        try:
            _annotation_coverage_rate = _merge_annotations(
                _G, ann_path, expected_network_version=expected_network_version
            )
            logger.info(
                "路段标注 merge 完成: coverage_rate=%.3f (%s)",
                _annotation_coverage_rate, ann_path
            )
        except Exception as e:
            logger.warning("路段标注 merge 失败（不影响运行）: %s", e)
            _annotation_coverage_rate = 0.0
    else:
        _annotation_coverage_rate = 0.0
        logger.info("未找到路段标注文件: %s (覆盖率=0.0)", ann_path)

    # 人工路网覆盖（穿楼边封禁/软惩罚等），在标注之后叠加
    ov_path = _overrides_path()
    if os.path.exists(ov_path):
        try:
            n = _merge_overrides(_G, ov_path)
            logger.info("路网人工覆盖 merge 完成: %d 条边 (%s)", n, ov_path)
        except Exception as e:
            logger.warning("路网人工覆盖 merge 失败（不影响运行）: %s", e)

    review_path = os.path.join(os.path.dirname(__file__), "..", "data",
                               "campus_review_decisions.json")
    if os.path.exists(review_path):
        try:
            n = _merge_verified_road_reviews(_G, review_path)
            logger.info("已应用现场/校方核实路段: %d 条", n)
        except Exception as e:
            logger.warning("核实路段记录加载失败（不影响运行）: %s", e)

    return _G


def _mark_osm_provenance(G: nx.MultiDiGraph) -> None:
    """合并 OSM 底稿来源；不得覆盖已附加的其他数据来源。"""
    for _, _, _, edge in G.edges(keys=True, data=True):
        raw = edge.get("osmid")
        if isinstance(raw, str) and raw.startswith("["):
            try:
                raw = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                pass
        ids = raw if isinstance(raw, (list, tuple)) else [raw]
        osm_refs = [
            {"source": "OpenStreetMap", "id": f"way/{osm_id}", "license": "ODbL-1.0"}
            for osm_id in ids if osm_id is not None
        ]
        existing_refs = edge.get("source_refs") or []
        if isinstance(existing_refs, str):
            try:
                existing_refs = ast.literal_eval(existing_refs)
            except (ValueError, SyntaxError):
                try:
                    existing_refs = json.loads(existing_refs)
                except (json.JSONDecodeError, TypeError):
                    existing_refs = []
        if isinstance(existing_refs, dict):
            existing_refs = [existing_refs]
        merged_refs = list(existing_refs) if isinstance(existing_refs, (list, tuple)) else []
        for source_ref in osm_refs:
            if source_ref not in merged_refs:
                merged_refs.append(source_ref)
        if merged_refs:
            edge["source_refs"] = merged_refs
        edge.setdefault("verification_status", "source_only")


def _merge_verified_road_reviews(G: nx.MultiDiGraph, review_path: str) -> int:
    """仅应用有证据和日期、且精确匹配边 ID 的通行决定。"""
    with open(review_path, encoding="utf-8") as handle:
        decisions = json.load(handle).get("road_decisions", [])
    applied = 0
    valid_modes = {"walk", "bike", "drive"}
    for review in decisions:
        if (G.graph.get("annotation_policy") == "source_scoped_only"
                and review.get("graph_source_sha256") != G.graph.get("source_sha256")):
            continue
        if review.get("verification_status") not in {"field_verified", "institution_verified"}:
            continue
        if not review.get("evidence") or not review.get("verified_at"):
            continue
        raw_id = review.get("edge_id")
        if not isinstance(raw_id, list) or len(raw_id) != 3:
            continue
        u, v, key = raw_id
        if not G.has_edge(u, v, key):
            continue
        blocked = set(review.get("blocked_modes") or [])
        passable = set(review.get("passable_modes") or [])
        if not blocked <= valid_modes or not passable <= valid_modes or blocked & passable:
            continue
        edge = G[u][v][key]
        old = edge.get("blocked_modes") or []
        if isinstance(old, str):
            try:
                old = ast.literal_eval(old)
            except (ValueError, SyntaxError):
                old = [old]
        modes = set(old)
        if "all" in modes:
            modes = (modes - {"all"}) | valid_modes
        modes -= passable
        edge["blocked_modes"] = sorted(modes | blocked)
        if review.get("clear_walk_penalty") and "walk" in passable:
            edge.pop("walk_penalty", None)
        edge["verification_status"] = review["verification_status"]
        edge["verified_at"] = review["verified_at"]
        edge["review_flag"] = "resolved_by_verified_evidence"
        applied += 1
    return applied


def _load_graphml(path: str) -> nx.MultiDiGraph:
    try:
        G = nx.read_graphml(path, node_type=int)
        logger.info("路网加载成功: %d 节点, %d 边", G.number_of_nodes(), G.number_of_edges())
        return G
    except Exception as e:
        raise RuntimeError(f"路网缓存加载失败: {e}")


def _save_graphml(G: nx.MultiDiGraph, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 清洗非原始类型属性值（osmnx 2.x 可能产生 list/dict/geometry，GraphML 不支持）
    G_clean = G.copy()
    for key, value in list(G_clean.graph.items()):
        if key in {"node_default", "edge_default"}:
            if value is None:
                G_clean.graph[key] = {}
            elif not isinstance(value, dict):
                del G_clean.graph[key]
            continue
        if value is None:
            del G_clean.graph[key]
        else:
            G_clean.graph[key] = _graphml_safe_value(value)
    for _, _, data in G_clean.edges(data=True):
        for key in list(data.keys()):
            if data[key] is None:
                del data[key]
            else:
                data[key] = _graphml_safe_value(data[key])
    for _, data in G_clean.nodes(data=True):
        for key in list(data.keys()):
            if data[key] is None:
                del data[key]
            else:
                data[key] = _graphml_safe_value(data[key])
    nx.write_graphml(G_clean, path)


def _graphml_safe_value(val):
    """Convert a graph attribute value into a GraphML-serializable form."""
    if isinstance(val, (list, dict, set, tuple)):
        return str(val)
    if hasattr(val, "wkt"):
        return val.wkt
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    return str(val)


def _download_network(bbox: dict) -> nx.MultiDiGraph:
    try:
        import osmnx as ox
    except ImportError:
        raise RuntimeError("osmnx 未安装，无法下载路网。请执行: pip install osmnx")

    north, south = bbox["north"], bbox["south"]
    east, west = bbox["east"], bbox["west"]

    try:
        # osmnx >= 2.0: bbox 元组参数 (west, south, east, north)
        G = ox.graph_from_bbox(
            bbox=(west, south, east, north),
            network_type="walk",
            simplify=True,
            retain_all=False,
            truncate_by_edge=True,
        )
    except TypeError:
        # osmnx < 2.0: 独立关键字参数 (fallback)
        G = ox.graph_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            network_type="walk",
            simplify=True,
            retain_all=False,
            truncate_by_edge=True,
        )
    except Exception as e:
        raise RuntimeError(f"OSM 路网下载失败: {e}")

    if G.number_of_nodes() == 0:
        raise RuntimeError("OSM 路网为空，请检查 bbox 范围或网络连接")

    logger.info("路网下载成功: %d 节点, %d 边", G.number_of_nodes(), G.number_of_edges())
    return G


def reload_network(bbox: Optional[dict] = None) -> nx.MultiDiGraph:
    """
    强制重新下载路网（忽略缓存）。

    Args:
        bbox: 边界框 dict

    Returns:
        重新下载的 MultiDiGraph
    """
    global _G
    _G = None
    from spatial.routing_index import invalidate_routing_index
    invalidate_routing_index()

    cache = _cache_path()
    if os.path.exists(cache):
        os.remove(cache)
        logger.info("已删除旧缓存: %s", cache)

    return load_or_download_network(bbox)


def clear_cache() -> None:
    """删除路网缓存文件。"""
    from spatial.routing_index import invalidate_routing_index
    invalidate_routing_index()
    cache = _cache_path()
    if os.path.exists(cache):
        os.remove(cache)
        logger.info("路网缓存已清除: %s", cache)


def get_nearest_node(G: nx.MultiDiGraph, lng: float, lat: float) -> int:
    """查找距离 (lng, lat) 最近的路网节点。

    纯 networkx 实现，不依赖 osmnx（减少运行时内存占用）。
    校园尺度下用等距圆柱近似即可，精度足够（误差 < 1 米）。
    """
    best_node = None
    best_dist = float("inf")
    # 纬度修正因子：经度 1 度对应的实际距离随纬度缩短
    cos_lat = __import__("math").cos(__import__("math").radians(lat))

    for node_id, data in G.nodes(data=True):
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
        # 等距圆柱近似：经度差 × cos(lat) 修正
        dx = (x - lng) * cos_lat
        dy = y - lat
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_node = node_id

    if best_node is None:
        raise RuntimeError("路网为空，无法查找最近节点")

    return int(best_node)


def get_node_coords(G: nx.MultiDiGraph, node_id: int) -> tuple:
    node_data = G.nodes[node_id]
    lng = float(node_data.get("x", 0))
    lat = float(node_data.get("y", 0))
    return (lng, lat)


def _annotations_path() -> str:
    override = os.environ.get("ROAD_ANNOTATIONS_PATH")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    try:
        from config import ROAD_ANNOTATIONS_PATH
        return ROAD_ANNOTATIONS_PATH
    except ImportError:
        return os.path.join(os.path.dirname(__file__), "..", "data", "road_annotations.json")


def _parse_edge_id(edge_id_raw) -> Optional[Tuple]:
    if edge_id_raw is None:
        return None
    if isinstance(edge_id_raw, list) or isinstance(edge_id_raw, tuple):
        if len(edge_id_raw) >= 3:
            return (str(edge_id_raw[0]), str(edge_id_raw[1]), int(edge_id_raw[2]))
        elif len(edge_id_raw) == 2:
            return (str(edge_id_raw[0]), str(edge_id_raw[1]), 0)
    if isinstance(edge_id_raw, str):
        try:
            parts = edge_id_raw.strip("[]()").split(",")
            parts = [p.strip() for p in parts]
            if len(parts) >= 3:
                return (parts[0], parts[1], int(parts[2]))
            elif len(parts) == 2:
                return (parts[0], parts[1], 0)
        except (ValueError, IndexError):
            pass
    return None


def _merge_annotations(
    G: nx.MultiDiGraph,
    annotations_path: str,
    expected_network_version: Optional[str] = None,
) -> float:
    with open(annotations_path, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    snapshot_version = ann_data.get("network_version")
    if snapshot_version:
        if expected_network_version is None:
            from scripts.network.build_edge_attribute_master import graph_fingerprint
            expected_network_version = graph_fingerprint(G)
        if snapshot_version != expected_network_version:
            raise ValueError("annotation network_version does not match the current graph")

    if (G.graph.get("annotation_policy") == "source_scoped_only"
            and ann_data.get("graph_source_sha256") != G.graph.get("source_sha256")):
        logger.info("旧边号标注未绑定当前路网来源版本，跳过合并")
        return 0.0

    edges_ann = ann_data.get("edges", []) or []
    total_edges = G.number_of_edges()

    if total_edges == 0:
        return 0.0

    graph_keys = set()
    graph_uv_to_keys = {}
    key_to_orig = {}  # (str_u, str_v, int_k) → (orig_u, orig_v, orig_k)  O(1) 查找
    for u, v, k in G.edges(keys=True):
        u_s, v_s = str(u), str(v)
        k_int = int(k)
        graph_keys.add((u_s, v_s, k_int))
        graph_uv_to_keys.setdefault((u_s, v_s), []).append(k_int)
        key_to_orig[(u_s, v_s, k_int)] = (u, v, k)

    annotated_set = set()

    for ann in edges_ann:
        edge_id = _parse_edge_id(ann.get("edge_id"))
        u_raw = ann.get("u")
        v_raw = ann.get("v")

        target_key = None
        if edge_id is not None:
            if edge_id in graph_keys:
                target_key = edge_id
        if target_key is None and u_raw is not None and v_raw is not None:
            u_s, v_s = str(u_raw), str(v_raw)
            candidates = graph_uv_to_keys.get((u_s, v_s), [])
            if candidates:
                target_key = (u_s, v_s, candidates[0])

        if target_key is None:
            continue

        orig = key_to_orig.get(target_key)
        if orig is None:
            continue
        orig_u, orig_v, orig_k = orig

        edge_data = G[orig_u][orig_v][orig_k]
        course_geometry_replaced = str(edge_data.get("course_geometry_replaced", "")).lower() == "true"
        course_name_authoritative = edge_data.get("course_name_source") == "course"
        current_graph_oge_candidate = bool(snapshot_version and ann.get("terrain_source_id"))
        if ((not course_geometry_replaced or current_graph_oge_candidate) and "slope_level" in ann
                and ann["slope_level"] is not None):
            try:
                edge_data["slope_level"] = int(ann["slope_level"])
            except (ValueError, TypeError):
                pass
        for attr in (
            "terrain_confidence", "terrain_verification_status", "grade_abs_pct",
            "grade_p90_pct", "terrain_source_id", "terrain_method_version",
            "terrain_source_refs",
        ):
            if attr in ann:
                edge_data[attr] = ann[attr]
        if (not course_geometry_replaced and "scenery_level" in ann
                and ann["scenery_level"] is not None):
            try:
                edge_data["scenery_level"] = int(ann["scenery_level"])
            except (ValueError, TypeError):
                pass
        if "name" in ann and ann["name"] and not course_name_authoritative:
            edge_data["name"] = str(ann["name"])
        # 人工标注的步行成本倍数：用于处理 OSM 把建筑内台阶/连廊画成穿楼捷径等问题
        # （只软惩罚不封死：起终点就在该边时仍可通行）
        if "walk_penalty" in ann and ann["walk_penalty"] is not None:
            try:
                wp = float(ann["walk_penalty"])
                if wp > 0:
                    edge_data["walk_penalty"] = wp
            except (ValueError, TypeError):
                pass

        annotated_set.add((orig_u, orig_v, orig_k))

    return len(annotated_set) / total_edges if total_edges > 0 else 0.0


def get_annotation_coverage_rate() -> float:
    global _annotation_coverage_rate
    if _annotation_coverage_rate is None:
        return 0.0
    return float(_annotation_coverage_rate)


def _overrides_path() -> str:
    try:
        from config import EDGE_OVERRIDES_PATH
        return EDGE_OVERRIDES_PATH
    except ImportError:
        return os.path.join(os.path.dirname(__file__), "..", "data", "edge_overrides.json")


def _merge_overrides(G: nx.MultiDiGraph, overrides_path: str) -> int:
    """
    合并人工路网覆盖到边属性（不写 GraphML，每次加载叠加）：
      - blocked_modes: ["bike","drive"] / ["all"]，filter_graph_for_mode 读取后硬删
      - walk_penalty: 步行软惩罚倍数（与 road_annotations 同机制，可叠加相乘）
    匹配规则与 _merge_annotations 一致：edge_id 优先，否则 u/v + 最小 key。
    """
    with open(overrides_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if (G.graph.get("annotation_policy") == "source_scoped_only"
            and data.get("graph_source_sha256") != G.graph.get("source_sha256")):
        logger.info("旧路网覆盖未绑定当前路网来源版本，跳过合并")
        return 0

    edges_ov = data.get("edges", []) or []
    graph_uv_to_keys = {}
    key_to_orig = {}
    for u, v, k in G.edges(keys=True):
        graph_uv_to_keys.setdefault((str(u), str(v)), []).append(int(k))
        key_to_orig[(str(u), str(v), int(k))] = (u, v, k)

    applied = 0
    for ov in edges_ov:
        edge_id = _parse_edge_id(ov.get("edge_id"))
        target = None
        if edge_id is not None and edge_id in key_to_orig:
            target = edge_id
        if target is None and ov.get("u") is not None and ov.get("v") is not None:
            cands = graph_uv_to_keys.get((str(ov["u"]), str(ov["v"])), [])
            want_k = int(ov.get("k", 0))
            if want_k in cands:
                target = (str(ov["u"]), str(ov["v"]), want_k)
            elif cands:
                target = (str(ov["u"]), str(ov["v"]), cands[0])
        if target is None:
            continue

        ou, ov_, ok = key_to_orig[target]
        edge_data = G[ou][ov_][ok]

        if ov.get("reason") == "crosses_building":
            edge_data["review_flag"] = "suspected_building_crossing"
        if ov.get("verification_status"):
            edge_data["verification_status"] = str(ov["verification_status"])

        modes = ov.get("blocked_modes")
        if modes:
            if isinstance(modes, str):
                modes = [modes]
            modes = [str(m) for m in modes]
            if modes:
                edge_data["blocked_modes"] = modes

        wp = ov.get("walk_penalty")
        if wp is not None:
            try:
                wp_f = float(wp)
                if wp_f > 0:
                    edge_data["walk_penalty"] = wp_f
            except (TypeError, ValueError):
                pass

        applied += 1

    return applied
