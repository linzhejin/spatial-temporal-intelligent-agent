"""
多因素路径计算模块

基于硬约束过滤 + 软成本优化的两步路径计算算法（DEC-011）。

核心算法：
  1. 硬约束过滤：根据 constraints 过滤不可通行路段
  2. 软成本优化：Cost = w_d × D + w_s × S + w_v × (1 − V)
  3. 路径长度上限：recommended ≤ shortest × 3 或 ≤ 2000m
  4. 不可达策略：保留硬约束，明确报告无可行路线

数据源：
  - G: OSMnx 路网（含 length 属性）
  - 路段坡度/景观标注：通过 edge 属性 slope_level / scenery_level
"""

import ast
import heapq
import itertools
import logging
import math
import os
import time
from typing import Optional

import networkx as nx

from spatial.route_constraints import positive_budget

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"distance": 0.90, "slope": 0.05, "scenery": 0.05}
WEIGHT_BOUNDS = {"min": 0.05, "max": 0.90}

_PATH_LENGTH_CAP_MULTIPLIER = 3.0
_PATH_LENGTH_CAP_MAX = 2000.0
_MAX_CONSTRAINED_LABELS = 50000


class BudgetSearchIncomplete(ValueError):
    """The bounded search could not establish whether a budgeted path exists."""

_EDGE_ATTR_DEFAULTS = {
    "slope_level": 3,
    "scenery_level": 3,
}

# ====== 校园边界多边形（精确剔除"真校外"路段） ======
# 仅两端都落在校园多边形外的边才算"真校外"并删除；
# 一端在内一端在外的跨边界边（如东湖南路沿湖段、珞瑜路校门段）保留以维持学部连通。
_campus_polygon = None
_outside_edge_cache = {}  # key: id(G) → set of (u,v,k)

def get_campus_polygon():
    """懒加载校园边界多边形（WGS-84，不加缓冲）。"""
    global _campus_polygon
    if _campus_polygon is None:
        try:
            from config import CAMPUS_POLYS_GCJ
            from spatial.coord_transform import gcj02_to_wgs84
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
            polys = [Polygon([gcj02_to_wgs84(lng, lat) for lng, lat in poly])
                     for poly in CAMPUS_POLYS_GCJ.values()]
            _campus_polygon = unary_union(polys)
        except Exception as e:
            logger.warning("校园多边形加载失败，校外边检测降级为路名匹配: %s", e)
            _campus_polygon = False
    return _campus_polygon


def _get_outside_edges(G: nx.MultiDiGraph) -> set:
    """
    返回"两端均在校外"的边 key 集合。
    用校园多边形精确判定（非路名），避免误删穿越校园的市政路段。
    结果按 id(G) 缓存，路网不变时只算一次。
    """
    poly = get_campus_polygon()
    if G.graph.get("routing_boundary_wkt"):
        from shapely.wkt import loads as _load_boundary
        poly = _load_boundary(G.graph["routing_boundary_wkt"])
    if not poly:
        # 多边形不可用 → 退路名匹配（保守）
        return set()
    cache_key = id(G)
    cached = _outside_edge_cache.get(cache_key)
    if cached is not None:
        return cached
    from shapely.geometry import Point
    outside = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        ud = G.nodes[u]
        vd = G.nodes[v]
        pu = Point(float(ud.get("x", 0)), float(ud.get("y", 0)))
        pv = Point(float(vd.get("x", 0)), float(vd.get("y", 0)))
        if not poly.covers(pu) and not poly.covers(pv):
            from shapely.geometry import LineString
            from shapely.wkt import loads as _load_geometry
            raw = d.get("geometry")
            try:
                line = _load_geometry(raw) if isinstance(raw, str) else raw
                if line is None:
                    line = LineString([pu, pv])
                if poly.intersects(line):
                    continue
            except (ValueError, TypeError):
                # Failed geometry parsing is not evidence a crossing is absent.
                if poly.intersects(LineString([pu, pv])):
                    continue
            outside.add((u, v, k))
    _outside_edge_cache[cache_key] = outside
    return outside


# 校外市政道路（武大校园周边的马路）：推荐路径应尽量避开，
# 只在起终点本身就在这些路边时（凌波门/牌坊/珞瑜门等）才必要地经过。
# 这些路的 scenery 标注可能很高（如东湖南路沿湖），若不惩罚会导致推荐路线绕出校园。
# 与 scripts/clip_to_campus.py 的 OUTSIDE_ROAD_NAMES 保持同步。
_OUTSIDE_ROAD_NAMES = {
    "八一路", "东湖南路", "卓刀泉北路", "卓刀泉南路", "卓刀泉路",
    "广八路", "茶港路", "广卓路", "珞狮路", "珞狮北路", "珞狮路辅路",
    "珞瑜路", "珞喻路", "珞喻路辅路", "珞瑜路辅路",
    "武珞路", "武珞路辅路", "群光南路", "洪福巷",
    "武工路", "机电路", "科技小路", "明志路", "汇志大道", "神龙园路",
}
_OUTSIDE_ROAD_PENALTY = 10.0  # 校外道路的距离成本放大倍数（从3.0调至10.0，强避免穿城）


def _road_name_list(raw) -> list:
    """路名可能是 str / list / 被str()序列化的list（"['珞喻路辅路', ...]"），统一拆成字符串列表。"""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    s = str(raw)
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    return [s]


def _is_outside_road(raw_name) -> bool:
    for n in _road_name_list(raw_name):
        if any(bad in n for bad in _OUTSIDE_ROAD_NAMES):
            return True
    return False


# ---------------------------------------------------------------------------
# 出行方式（walk 步行 / bike 骑行 / drive 驾车）
# ---------------------------------------------------------------------------

TRAVEL_MODES = ("walk", "bike", "drive")

MODE_SPEEDS_KMH = {"walk": 4.5, "bike": 14.0, "drive": 25.0}

# 各模式默认多因素权重。设计原则：distance 为主导，slope/scenery 是弱偏好——
# 用户没显式表达偏好时，推荐路线应接近最短路径；只有显式说"风景好/平坦"时才大幅偏离。
# 实测：校园短边（中位 28m）上坡度按边计费，slope 权重 0.1 就会导致 20%+ 绕行，
# 故步行默认 slope 压到 0.05（OD 实测绕行 <3%）。
MODE_DEFAULT_WEIGHTS = {
    # 三种出行方式的通勤默认统一为 0.90/0.05/0.05。
    # 用户明确表达"看风景/避坡"偏好时 LLM 才调高非距离权重，否则不主动偏航。
    "walk": dict(DEFAULT_WEIGHTS),
    "bike": dict(DEFAULT_WEIGHTS),
    "drive": dict(DEFAULT_WEIGHTS),
}

# 校外市政道路惩罚倍数：步行强避免（10×，与 _OUTSIDE_ROAD_PENALTY 保持一致）；
# 骑行/驾车本身就常走市政路，惩罚逐档降低。
MODE_OUTSIDE_ROAD_PENALTY = {"walk": _OUTSIDE_ROAD_PENALTY, "bike": 3.0, "drive": 1.2}

# 步行台阶成本倍数：台阶步行速度约为平地一半，且无法推行。
# 另一作用：防止 OSM 把教学楼/食堂的入口台阶、连廊画成穿越建筑的"捷径"
# （如信息学部学生四食堂，长边全是 steps，125 对起终点曾穿楼而过）。
_WALK_STEPS_PENALTY = 2.5

# 非机动车设施"一票否决"标签：边只要含其中任意一个，骑行/驾车一律剔除。
# 背景：OSMnx simplify 会把平行的车行道与台阶/走廊合并成同一条多标签边
# （实测 ['service','steps'] 54 条、['footway','steps'] 360 条、
# ['residential','steps'] 8 条等，共 462 条）。旧逻辑下 drive"白名单任一命中
# 即保留"、bike"全部标签都不可通行才删"，导致 80 条台阶边在驾车图中漏网、
# 几乎全部台阶边在骑行图中漏网（线上"车上台阶/车穿楼"问题根因）。
_NON_VEHICULAR_HIGHWAY = {"steps", "elevator", "escalator", "corridor"}

# 驾车白名单：边的 highway 标签中任意一个命中即视为车行道
_DRIVE_ALLOWED_HIGHWAY = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "service", "living_street", "road",
}


def normalize_mode(mode) -> str:
    """归一化出行方式：None / 非法值统一回退 "walk"，合法值原样返回。"""
    if isinstance(mode, str) and mode in TRAVEL_MODES:
        return mode
    return "walk"


def nearest_in_mode_node(G: nx.MultiDiGraph, G_mode: nx.MultiDiGraph,
                          lng: float, lat: float, max_dist_m: float = 50.0):
    """
    在 G_mode 中找距离 (lng, lat) 最近的节点，限半径 max_dist_m。

    用于 bike/drive 起终点吸附到 walk-only POI 时的兜底：原 POI 节点已被
    filter_graph_for_mode 删除，需要找一个该模式下仍可通行的最近节点
    作为"换乘锚点"，前后拼接 walk 段。

    Returns:
        (node_id, dist_m) 或 (None, inf)。dist_m 为 haversine 米距。
    """
    import math
    if G_mode.number_of_nodes() == 0:
        return None, float("inf")
    cos_lat = math.cos(math.radians(lat))
    best_node = None
    best_dist_m = float("inf")
    for node_id, data in G_mode.nodes(data=True):
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
        # 等距圆柱近似：经度差 × cos(lat)
        dlat = (y - lat)
        dlng = (x - lng) * cos_lat
        # 度 → 米：1° 纬度 ≈ 111000m
        dist_m = math.sqrt((dlat * 111000.0) ** 2 + (dlng * 111000.0) ** 2)
        if dist_m < best_dist_m:
            best_dist_m = dist_m
            best_node = node_id
    if best_node is None or best_dist_m > max_dist_m:
        return None, float("inf")
    return best_node, best_dist_m


def _edge_highway_tags(edge_data: dict) -> list:
    """解析边的 highway 属性（str / list / 被 str() 序列化的 list），返回标签字符串列表。"""
    if not edge_data:
        return []
    return _road_name_list(edge_data.get("highway"))


def _edge_blocked_modes(edge_data: dict) -> set:
    """读取人工覆盖的硬封禁模式集合（edge_overrides.json 合并到边属性）。"""
    if not edge_data:
        return set()
    raw = edge_data.get("blocked_modes")
    if not raw:
        return set()
    if isinstance(raw, str):
        raw = _road_name_list(raw)
    try:
        return {str(x) for x in raw}
    except TypeError:
        return set()


def _merge_penalty_maps(*maps: dict) -> dict:
    """合并多个 penalty_map：同一 (u, v, k) 上的惩罚倍数相乘。"""
    merged = {}
    for m in maps:
        if not m:
            continue
        for key, multiplier in m.items():
            if key in merged:
                merged[key] *= multiplier
            else:
                merged[key] = multiplier
    return merged


def filter_graph_for_mode(G: nx.MultiDiGraph, mode) -> tuple:
    """
    按出行方式过滤路网。

    第一步（所有模式）：剔除"真校外"边——两端均在校园多边形外的边。
        跨边界边（一端在内）保留以维持学部连通。
    第二步（按模式）：
    - walk:  仅剔校外边后返回
    - bike:  再移除纯台阶/垂直交通边；陡坡软惩罚
    - drive: 仅保留车行道边

    过滤后图为空时回退，status 追加 "_degraded"。

    Returns:
        (G_filtered, status_str, penalty_map)
    """
    mode = normalize_mode(mode)

    from spatial.osm_access import edge_access, node_access

    # —— 第一步：所有模式都剔除真校外边 ——
    outside_edges = _get_outside_edges(G)
    if outside_edges:
        G_mode = G.copy()
        G_mode.remove_edges_from(list(outside_edges))
        isolated = [n for n, deg in G_mode.degree() if deg == 0]
        if isolated:
            G_mode.remove_nodes_from(isolated)
    else:
        G_mode = G
    outside_status = "no_outside" if outside_edges else "no_filter"

    # Preserve OSM mode-specific access and direction instead of deriving motor
    # permissions from a bidirectional walking graph. Unknown tags remain open
    # for review; explicit no/private and physical barriers are respected.
    denied_nodes = {n for n, d in G_mode.nodes(data=True) if not node_access(d, mode)[0]}
    denied_edges = []
    for u, v, k, data in G_mode.edges(keys=True, data=True):
        declared = data.get("allowed_modes")
        allowed = _road_name_list(declared) if declared is not None else None
        if (u in denied_nodes or v in denied_nodes or not edge_access(data, mode)[0]
                or (allowed is not None and mode not in allowed)):
            denied_edges.append((u, v, k))
    if denied_edges:
        G_mode = G_mode.copy()
        G_mode.remove_edges_from(denied_edges)
        G_mode.remove_nodes_from([n for n, degree in G_mode.degree() if degree == 0])

    if mode == "walk":
        # 台阶软惩罚：不封死（台阶本身可走，且可能是唯一通道），但不让它成为穿楼捷径。
        # 人工标注 walk_penalty（建筑内连廊等）与台阶惩罚叠加。
        # Indoor/corridor tags describe infrastructure, not a walk prohibition.
        # Explicit foot/access and barrier rules are applied above.
        steps_penalty = {}
        hard_blocked = []
        for u, v, k, data in G_mode.edges(keys=True, data=True):
            tags = _edge_highway_tags(data)
            mult = 1.0
            if "steps" in tags:
                mult *= _WALK_STEPS_PENALTY
            wp = data.get("walk_penalty")
            if wp:
                try:
                    mult *= float(wp)
                except (TypeError, ValueError):
                    pass
            if mult > 1.0:
                steps_penalty[(u, v, k)] = mult
            bm = _edge_blocked_modes(data)
            if "all" in bm or "walk" in bm:
                hard_blocked.append((u, v, k))
        if hard_blocked:
            # 有人工封禁（"all"）边时才复制图，避免常态下整图拷贝
            G_mode = G_mode.copy()
            G_mode.remove_edges_from(hard_blocked)
            isolated = [n for n, deg in G_mode.degree() if deg == 0]
            if isolated:
                G_mode.remove_nodes_from(isolated)
        return G_mode, outside_status, steps_penalty

    # bike/drive：在已剔校外边的图副本上继续删边（绝不能改原图）
    G_mode = G_mode.copy()
    edges_to_remove = []

    if mode == "bike":
        for u, v, k, data in list(G_mode.edges(keys=True, data=True)):
            tags = _edge_highway_tags(data)
            bm = _edge_blocked_modes(data)
            # 一票否决：只要含台阶/电梯/走廊等非机动车设施标签即剔除
            # （多标签边 ['service','steps'] 同样剔除，杜绝"车上台阶"）；
            # 人工覆盖 blocked_modes 命中本模式同样剔除
            if mode in bm or "all" in bm:
                edges_to_remove.append((u, v, k))
        base_status = "mode_bike"
    else:  # drive
        for u, v, k, data in list(G_mode.edges(keys=True, data=True)):
            tags = _edge_highway_tags(data)
            bm = _edge_blocked_modes(data)
            # 双重条件：含非机动车设施标签一票否决；否则要求至少一个车行道标签；
            # 人工覆盖（如穿楼车行道）命中同样剔除
            if mode in bm or "all" in bm:
                edges_to_remove.append((u, v, k))
        base_status = "mode_drive"

    G_mode.remove_edges_from(edges_to_remove)

    if G_mode.number_of_edges() == 0:
        # Empty legal-edge graph is unreachable; never restore prohibited roads.
        status = f"{base_status}_empty"
    else:
        isolated = [n for n, deg in G_mode.degree() if deg == 0]
        if isolated:
            G_mode.remove_nodes_from(isolated)
            logger.info("出行方式 %s：移除 %d 条不可通行边、%d 个孤立节点",
                        mode, len(edges_to_remove), len(isolated))
        status = base_status

    penalty_map = {}
    if mode == "bike":
        for u, v, k, data in G_mode.edges(keys=True, data=True):
            lvl = data.get("slope_level")
            if lvl == 5:
                penalty_map[(u, v, k)] = 3.0
            elif lvl == 4:
                penalty_map[(u, v, k)] = 1.5

    return G_mode, status, penalty_map


def estimate_duration_min(length_m, mode, G=None, route_nodes=None,
                          penalty_map=None, route_edges=None) -> float:
    """
    按模式平均速度估算通行时长（分钟），round 到 1 位小数。

    两种调用形式（重载）：
      1. estimate_duration_min(length_m: number, mode) — 旧签名，仅按总长×平均速度，
         不考虑坡度/台阶/路况。保留兼容（TestEstimateDuration 旧用例）。
      2. estimate_duration_min(length_m=None, mode, G, route_nodes, penalty_map) —
         新签名，按路径每条边累加时长，speed 由 slope_level / steps 标签 /
         penalty_map 路况惩罚倒推（penalty F → 速度 ÷ F），更贴近实际。

    length 为 0 / None 且无 route_nodes 时返回 0.0。
    """
    if route_nodes is None or G is None:
        if not length_m:
            return 0.0
        mode = normalize_mode(mode)
        meters_per_min = MODE_SPEEDS_KMH[mode] * 1000.0 / 60.0
        return round(length_m / meters_per_min, 1)
    return _estimate_route_duration_min(G, route_nodes, mode, penalty_map, route_edges)


def _estimate_route_duration_min(G, route_nodes, mode, penalty_map=None,
                                 route_edges=None) -> float:
    """按路径每条边累加时长，速度受 slope/steps/路况惩罚影响。

    - slope_factor：walk slope_level 5→0.5、4→0.7；bike 5→0.4、4→0.6；drive 全 1.0
    - steps_factor：walk 命中 steps 标签→0.4（与 _WALK_STEPS_PENALTY 2.5× 减速语义对齐）
    - road_factor：penalty_map[(u,v,k)] 数值 F → 速度 ÷ F（block 边已删不入 route）
    """
    mode = normalize_mode(mode)
    if not route_nodes or len(route_nodes) < 2:
        return 0.0
    penalty_map = penalty_map or {}
    total_min = 0.0
    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        edge_data = G.get_edge_data(u, v)
        if not edge_data:
            continue
        if route_edges is not None:
            edge_key = route_edges[i][2]
            d = edge_data[edge_key]
        else:
            edge_key = min(edge_data, key=lambda k: edge_data[k].get("length", float("inf")))
            d = edge_data[edge_key]
        total_min += _edge_duration_min(d, mode, penalty_map.get((u, v, edge_key)))

    return round(total_min, 1)


def _edge_duration_min(data, mode, road_factor=None) -> float:
    """Unrounded contribution of one exact edge to the public duration estimate."""
    try:
        length_m = float(data.get("length", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    if length_m <= 0:
        return 0.0
    factor = 1.0
    level = data.get("slope_level")
    if mode == "walk":
        if "steps" in _edge_highway_tags(data):
            factor *= 0.4
        if level == 5:
            factor *= 0.5
        elif level == 4:
            factor *= 0.7
    elif mode == "bike":
        if level == 5:
            factor *= 0.4
        elif level == 4:
            factor *= 0.6
    if road_factor and road_factor > 1.0:
        factor /= road_factor
    meters_per_min = MODE_SPEEDS_KMH[mode] * 1000.0 / 60.0
    return length_m / (meters_per_min * max(factor, 0.05))


# ---------------------------------------------------------------------------
# 转向指令（turn-by-turn）
# ---------------------------------------------------------------------------


def _bearing(p1, p2) -> float:
    """线段方位角（度）：0=正北，顺时针到 360。p=(lng,lat)。"""
    import math
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _classify_turn(in_bearing: float, out_bearing: float) -> tuple:
    """根据进入/离开方位角返回 (action, 中文口令)。"""
    import math
    # 带符号夹角：[-180, 180)，正=右转
    delta = (out_bearing - in_bearing + 540.0) % 360.0 - 180.0
    a = abs(delta)
    if a < 10:
        return "straight", "直行"
    if a < 25:
        return ("slight_right", "靠右") if delta > 0 else ("slight_left", "靠左")
    if a < 120:
        return ("right", "右转") if delta > 0 else ("left", "左转")
    if a < 170:
        return ("sharp_right", "向右急转") if delta > 0 else ("sharp_left", "向左急转")
    return "uturn", "掉头"


def _edge_points(G, u, v, data) -> list:
    """取边的几何点序列 [(lng,lat),...]；无 geometry 时用节点连线。"""
    raw = data.get("geometry") if data else None
    if raw and isinstance(raw, str):
        try:
            from shapely.wkt import loads as wkt_loads
            g = wkt_loads(raw)
            points = list(g.coords)
            u_coords = (float(G.nodes[u].get("x", 0)), float(G.nodes[u].get("y", 0)))
            if points and (points[0][0] - u_coords[0]) ** 2 + (points[0][1] - u_coords[1]) ** 2 > (points[-1][0] - u_coords[0]) ** 2 + (points[-1][1] - u_coords[1]) ** 2:
                points.reverse()
            return points
        except Exception:
            pass
    xu = float(G.nodes[u].get("x", 0))
    yu = float(G.nodes[u].get("y", 0))
    xv = float(G.nodes[v].get("x", 0))
    yv = float(G.nodes[v].get("y", 0))
    return [(xu, yu), (xv, yv)]


def _edge_road_name(data) -> str:
    """取路名；无名（含 OSM 占位"未命名路 u-v"）返回空串。"""
    nm = str(data.get("name", "") or "").strip()
    if not nm or "未命名" in nm:
        return ""
    return nm


def _human_distance(m: float) -> str:
    """距离口语化：≥100m 按 10m 取整，其余按 5m 取整，<15m 用"马上"。"""
    if m < 15:
        return "马上"
    step = 10 if m >= 100 else 5
    d = int(round(m / step) * step)
    return f"约{d}米"


def build_turn_by_turn(G, route_nodes: list, mode: str = "walk",
                       end_name: str = "", route_edges=None) -> list:
    """
    由路径节点序列生成转向指令（坐标 WGS-84，与路网一致）。

    连续同名道路的边先合并成"路段"，再在每个路段边界（拐点/路名变化点）
    生成一条指令。指令 point 为动作发生点（拐点坐标）。

    Returns:
        list[dict]：
          {seq, type(depart|turn|arrive), action, distance_m, cumulative_m,
           road_name, next_road_name, text, point:{lng,lat}}
    """
    if not route_nodes:
        return []

    # ---------- 1. 逐边取几何/路名/长度 ----------
    raw_edges = []
    for i, (a, b) in enumerate(zip(route_nodes, route_nodes[1:])):
        data = G.get_edge_data(a, b)
        if not data:
            # 回退：尝试反向边（理论上路径有向不会发生）
            data = G.get_edge_data(b, a) or {}
        k0 = route_edges[i][2] if route_edges is not None else (sorted(data.keys())[0] if data else 0)
        d = data.get(k0, {})
        pts = _edge_points(G, a, b, d)
        if len(pts) < 2:
            continue
        try:
            length = float(d.get("length", 0) or 0)
        except (TypeError, ValueError):
            length = 0.0
        raw_edges.append({
            "name": _edge_road_name(d), "points": pts, "length": length,
        })

    if not raw_edges:
        return []

    # ---------- 2. 同名连续边合并成路段 ----------
    segments = []
    for e in raw_edges:
        if segments and segments[-1]["name"] == e["name"]:
            seg = segments[-1]
            # 拼接几何：跳过重复首点
            seg["points"].extend(e["points"][1:])
            seg["length"] += e["length"]
        else:
            segments.append({
                "name": e["name"],
                "points": list(e["points"]),
                "length": e["length"],
            })

    def seg_bearing(seg, at_start: bool) -> float:
        pts = seg["points"]
        return _bearing(pts[0], pts[1]) if at_start else _bearing(pts[-2], pts[-1])

    # ---------- 3. 生成指令 ----------
    steps = []
    cum = 0.0
    first_name = segments[0]["name"] or "小路"

    # 出发指令：动作点在第一个拐点（只有一段时直接是终点）
    first_point = segments[0]["points"][-1]
    depart_text = f"出发，沿{first_name}前行"
    steps.append({
        "seq": 0, "type": "depart", "action": "depart",
        "distance_m": round(segments[0]["length"], 1),
        "cumulative_m": round(cum + segments[0]["length"], 1),
        "road_name": first_name, "next_road_name": "",
        "text": depart_text,
        "point": {"lng": round(first_point[0], 6), "lat": round(first_point[1], 6)},
    })

    # 拐点指令：segments[i] → segments[i+1]
    for i in range(len(segments) - 1):
        cur, nxt = segments[i], segments[i + 1]
        cum += cur["length"]
        in_b = seg_bearing(cur, at_start=False)
        out_b = seg_bearing(nxt, at_start=True)
        action, zh = _classify_turn(in_b, out_b)
        d_text = _human_distance(cur["length"])
        nxt_name = nxt["name"] or "小路"
        if action == "straight":
            text = f"继续直行{('，进入' + nxt_name) if nxt['name'] else ''}".rstrip("，")
        else:
            text = f"{d_text}{zh}"
            if nxt["name"]:
                text += f"，进入{nxt_name}"
        point = cur["points"][-1]
        steps.append({
            "seq": len(steps), "type": "turn", "action": action,
            "distance_m": round(cur["length"], 1),
            "cumulative_m": round(cum, 1),
            "road_name": cur["name"] or "小路",
            "next_road_name": nxt_name,
            "text": text,
            "point": {"lng": round(point[0], 6), "lat": round(point[1], 6)},
        })

    # 到达指令：动作点在路径终点，距离取最后一段长度
    last_seg = segments[-1]
    cum_total = cum + last_seg["length"]
    end_p = last_seg["points"][-1]
    arrive_text = "到达终点"
    if end_name:
        arrive_text += f"（{end_name}）"
    arrive_text += "，导航结束"
    steps.append({
        "seq": len(steps), "type": "arrive", "action": "arrive",
        "distance_m": round(last_seg["length"], 1),
        "cumulative_m": round(cum_total, 1),
        "road_name": last_seg["name"] or "小路", "next_road_name": "",
        "text": arrive_text,
        "point": {"lng": round(end_p[0], 6), "lat": round(end_p[1], 6)},
    })

    return steps


def resolve_weights(llm_weights: Optional[dict]) -> dict:
    """
    解析 LLM 输出的 weights，校验后返回最终权重。

    - 入参为 None 时返回 DEFAULT_WEIGHTS
    - 每个权重强制限制在 [0.05, 0.90]
    - 归一化使总和 = 1

    Args:
        llm_weights: LLM 输出的 weights dict，可为 None

    Returns:
        校验并归一化后的权重 dict {"distance": float, "slope": float, "scenery": float}
    """
    if llm_weights is None:
        return dict(DEFAULT_WEIGHTS)

    bounds_min = WEIGHT_BOUNDS["min"]
    bounds_max = WEIGHT_BOUNDS["max"]

    w = {}
    for key in ("distance", "slope", "scenery"):
        raw = float(llm_weights.get(key, DEFAULT_WEIGHTS[key]))
        w[key] = max(bounds_min, min(bounds_max, raw))

    total = sum(w.values())
    if total == 0:
        return dict(DEFAULT_WEIGHTS)

    return {k: v / total for k, v in w.items()}


def _normalize_lengths(G: nx.MultiDiGraph) -> tuple:
    """
    计算所有边的归一化长度。

    Returns:
        (max_len, norm_map) 元组：
        - max_len: 全图最长边长度（米）
        - norm_map: {(u, v): norm_length} 归一化长度映射
    """
    max_len = 0.0
    for u, v, data in G.edges(data=True):
        length = data.get("length", 0)
        if length > max_len:
            max_len = length

    if max_len == 0:
        return 0.0, {}

    norm_map = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        norm_map[(u, v, k)] = data.get("length", 0) / max_len
    return max_len, norm_map


def _get_edge_attr(data: dict, attr: str) -> tuple:
    """
    获取边属性值，返回 (值, 是否为降级默认值)。

    Args:
        data: edge data dict
        attr: 属性名

    Returns:
        (value, is_default) 元组
    """
    if attr in data and data[attr] is not None:
        return float(data[attr]), False
    return float(_EDGE_ATTR_DEFAULTS.get(attr, 3)), True


def _compute_edge_cost(
    data: dict,
    norm_length: float,
    weights: dict,
) -> tuple:
    """
    计算单条边的多因素成本。

    Cost = w_d × D + w_s × S + w_v × (1 − V)

    Args:
        data: edge data dict
        norm_length: 归一化距离（0-1）
        weights: 权重 dict

    Returns:
        (cost, is_degraded) 元组
    """
    slope, slope_degraded = _get_edge_attr(data, "slope_level")
    scenery, scenery_degraded = _get_edge_attr(data, "scenery_level")

    is_degraded = slope_degraded or scenery_degraded

    d_component = weights["distance"] * norm_length
    s_component = weights["slope"] * (slope / 5.0)
    v_component = weights["scenery"] * (1.0 - scenery / 5.0)

    return d_component + s_component + v_component, is_degraded


def _filter_by_constraints(G: nx.MultiDiGraph, constraints: dict) -> tuple:
    """
    基于硬约束过滤不可通行路段。

    DEC-011 规则：
      - slope=avoid: 过滤 slope_level=5 的路段
                     slope_level=4 的路段加 2× 距离惩罚
                     无可行路径时保留硬约束，由调用方报告不可达
      - 其他约束等级（normal/any）：不过滤

    Args:
        G: 路网图
        constraints: 约束 dict，如 {"slope": "avoid"}

    Returns:
        (G_filtered, filter_status, penalty_map) 元组
        - G_filtered: 过滤后的图（可能是原图副本）
        - filter_status: 状态标记 ("filtered" | "no_filter")
        - penalty_map: {(u, v, k): penalty_multiplier} 用于成本调整
    """
    from spatial.route_constraints import excluded_edge_ids

    excluded = excluded_edge_ids(constraints)
    slope_constraint = constraints.get("slope", "normal")
    avoid_steps = bool(constraints.get("avoid_steps", False))

    if slope_constraint != "avoid" and not avoid_steps and not excluded:
        return G, "no_filter", {}

    edges_to_remove = list(excluded)
    penalty_map = {}

    for u, v, k, data in G.edges(keys=True, data=True):
        slope_level = data.get("slope_level")
        has_steps = "steps" in _edge_highway_tags(data)

        if avoid_steps and has_steps:
            edges_to_remove.append((u, v, k))
        if slope_constraint == "avoid" and slope_level == 5:
            edges_to_remove.append((u, v, k))
        elif slope_constraint == "avoid" and slope_level == 4:
            penalty_map[(u, v, k)] = 2.0

    if not edges_to_remove and not penalty_map and not avoid_steps:
        return G, "no_filter", {}

    G_filtered = G.copy()
    G_filtered.remove_edges_from(set(edges_to_remove))

    status_parts = []
    if slope_constraint == "avoid" and (edges_to_remove or penalty_map):
        status_parts.append("filtered")
    if avoid_steps:
        status_parts.append("steps_avoid")
        status_parts.append("steps_unverified")
    if excluded:
        status_parts.append("excluded_edges")

    return G_filtered, "+".join(status_parts) or "no_filter", penalty_map


def _should_degrade_annotations() -> Optional[str]:
    try:
        from spatial.network import get_annotation_coverage_rate
        rate = get_annotation_coverage_rate()
    except Exception:
        rate = 0.0

    if rate <= 0.0:
        return "no_annotations"
    if rate < 0.8:
        return "degraded_annotations"
    return None


def _edge_cost_factory(
    G: nx.MultiDiGraph,
    norm_lengths: dict,
    weights: dict,
    penalty_map: dict,
    annotation_degraded_tag: Optional[str] = None,
    outside_road_penalty: float = _OUTSIDE_ROAD_PENALTY,
):
    """
    创建边权函数（用于 Dijkstra 路径计算）。

    Args:
        outside_road_penalty: 校外市政道路成本放大倍数（按出行模式取，
            walk=10.0 与历史行为一致）。

    返回的函数签名: edge_weight(u, v, data) -> float
    """
    max_len_for_fallback = max(
        (float(attrs.get("length", 0) or 0) for _, _, attrs in G.edges(data=True)),
        default=1.0,
    )

    def single_edge_cost(u, v, edge_k, edge_data):
        key = (u, v, edge_k)
        raw_length = edge_data.get("length", 0)
        fallback_norm = raw_length / max_len_for_fallback if max_len_for_fallback > 0 else 0.0
        norm = norm_lengths.get(key, fallback_norm)

        if annotation_degraded_tag is not None:
            cost = weights["distance"] * norm
        else:
            cost, _ = _compute_edge_cost(edge_data, norm, weights)

        # 校外道路惩罚：尽量避免推荐路线绕出校园（东湖南路/八一路等市政路）
        if _is_outside_road(edge_data.get("name")):
            cost *= outside_road_penalty

        penalty = penalty_map.get(key, 1.0)
        cost *= penalty

        return cost

    def edge_weight(u, v, data):
        # MultiDiGraph 的 weight 回调一次收到全部平行边；必须按实际成本选边。
        if data and all(isinstance(value, dict) for value in data.values()):
            return min(
                single_edge_cost(u, v, key, attrs)
                for key, attrs in data.items()
            )
        return single_edge_cost(u, v, 0, data or {})

    return edge_weight


def _select_path_edges(G: nx.MultiDiGraph, path: list, weight) -> list:
    """将节点路径还原成 Dijkstra 实际计费的 (u, v, key) 序列。"""
    selected = []
    for u, v in zip(path, path[1:]):
        parallel = G.get_edge_data(u, v)
        if not parallel:
            raise ValueError(f"路径边不存在: {u}->{v}")
        key = min(
            parallel,
            key=lambda k: (weight(u, v, {k: parallel[k]}), str(k)),
        )
        selected.append((u, v, key))
    return selected


def _length_weight(u, v, data):
    """Return physical edge length for an exact shortest-path search."""
    if data and all(isinstance(value, dict) for value in data.values()):
        return min(float(attrs.get("length", 0) or 0) for attrs in data.values())
    return float((data or {}).get("length", 0) or 0)


def _run_path(G: nx.MultiDiGraph, start_node: int, end_node: int, weight):
    """Run one Dijkstra search and retain the exact parallel edges it selected."""
    nodes = nx.dijkstra_path(G, start_node, end_node, weight=weight)
    edges = _select_path_edges(G, nodes, weight)
    return nodes, edges


def _budgeted_path(G, start_node, end_node, *, mode, penalty_map, edge_weight,
                   max_distance_m, max_duration_s, duration_penalty_map=None,
                   return_frontier=False, max_labels=None):
    """Find minimum soft-cost path under additive distance and time budgets.

    Pareto labels retain the exact directed edge key. A bounded search returns
    an explicit unknown state on resource exhaustion, never a false no-route.
    """
    labels = [{"node": start_node, "cost": 0.0, "distance": 0.0,
               "seconds": 0.0, "previous": None, "edge": None, "active": True}]
    at_node = {start_node: [0]}
    sequence = itertools.count()
    queue = [(0.0, next(sequence), 0)]
    duration_penalty_map = (penalty_map if duration_penalty_map is None
                            else duration_penalty_map)
    label_limit = max(1, int(max_labels or _MAX_CONSTRAINED_LABELS))

    def reconstruct(label_index):
        edges = []
        nodes = [labels[label_index]["node"]]
        while label_index:
            label = labels[label_index]
            edges.append(label["edge"])
            label_index = label["previous"]
            nodes.append(labels[label_index]["node"])
        return list(reversed(nodes)), list(reversed(edges))

    while queue:
        _, _, label_index = heapq.heappop(queue)
        current = labels[label_index]
        if not current["active"]:
            continue
        u = current["node"]
        if u == end_node:
            if max_duration_s is None or round(current["seconds"] / 60.0, 1) * 60 <= max_duration_s + 3:
                if not return_frontier:
                    return reconstruct(label_index)
            continue

        for _, v, key, data in G.out_edges(u, keys=True, data=True):
            try:
                length = float(data.get("length", 0) or 0)
                cost = float(edge_weight(u, v, {key: data}))
            except (TypeError, ValueError, OverflowError) as exc:
                raise BudgetSearchIncomplete("路网含有无法计算的边，预算路线尚不能确认。") from exc
            if not math.isfinite(length) or length < 0 or not math.isfinite(cost) or cost < 0:
                raise BudgetSearchIncomplete("路网含有无效距离或代价，预算路线尚不能确认。")
            edge = (u, v, key)
            seconds = _edge_duration_min(data, mode, duration_penalty_map.get(edge)) * 60.0
            if not math.isfinite(seconds) or seconds < 0:
                raise BudgetSearchIncomplete("路网含有无效通行时间，预算路线尚不能确认。")
            distance = current["distance"] + length
            elapsed = current["seconds"] + seconds
            if max_distance_m is not None and distance > max_distance_m + 0.15:
                continue
            # Public ETA rounds the whole route to 0.1 min (6 s). A raw
            # duration slightly above max + 3 can still round down into the
            # accepted displayed duration, so use a safe bound here and check
            # the exact public rule when the destination label is popped.
            if max_duration_s is not None and elapsed > max_duration_s + 6:
                continue
            next_cost = current["cost"] + cost
            others = at_node.get(v, [])
            if any(labels[i]["cost"] <= next_cost
                   and labels[i]["distance"] <= distance
                   and labels[i]["seconds"] <= elapsed for i in others):
                continue
            remaining = []
            for i in others:
                old = labels[i]
                if next_cost <= old["cost"] and distance <= old["distance"] and elapsed <= old["seconds"]:
                    old["active"] = False
                else:
                    remaining.append(i)
            if len(labels) >= label_limit:
                raise BudgetSearchIncomplete("预算路线搜索超过计算上限，暂时无法确认是否有可行路线。")
            next_index = len(labels)
            labels.append({"node": v, "cost": next_cost, "distance": distance,
                           "seconds": elapsed, "previous": label_index,
                           "edge": edge, "active": True})
            remaining.append(next_index)
            at_node[v] = remaining
            heapq.heappush(queue, (next_cost, next(sequence), next_index))
    if return_frontier:
        frontier = []
        for label_index in at_node.get(end_node, []):
            label = labels[label_index]
            if not label["active"]:
                continue
            duration_min = round(label["seconds"] / 60.0, 1)
            if (max_duration_s is not None
                    and duration_min * 60.0 > max_duration_s + 3.0):
                continue
            nodes, edges = reconstruct(label_index)
            frontier.append({
                "nodes": nodes, "edges": edges, "cost": label["cost"],
                "distance_m": round(label["distance"], 1),
                "duration_min": duration_min,
            })
        return frontier
    return None


def _budgeted_ordered_path(G, waypoints, *, mode, edge_weight,
                           duration_penalty_map=None, max_distance_m=None,
                           max_duration_s=None):
    """Find a minimum-cost path through ordered waypoints under whole-trip budgets.

    Labels retain the next waypoint index, so distance and time budgets are
    cumulative across the entire itinerary rather than independently reused
    for each leg. ``G`` must already have mode, closure, and hard-constraint
    filtering applied. Exact directed edge keys are returned for each leg.
    """
    if not isinstance(waypoints, (list, tuple)) or len(waypoints) < 2:
        raise ValueError("ordered route needs at least two waypoints")
    waypoints = list(waypoints)
    if any(left == right for left, right in zip(waypoints, waypoints[1:])):
        raise ValueError("adjacent ordered waypoints must differ")
    if any(node not in G for node in waypoints):
        return None
    mode = normalize_mode(mode)
    duration_penalty_map = duration_penalty_map or {}

    # A label's stage is the index of the next waypoint to reach. Multiple
    # labels at the same (node, stage) are kept only when they are Pareto
    # incomparable in cost, distance, and elapsed time.
    labels = [{
        "node": waypoints[0], "stage": 1, "cost": 0.0, "distance": 0.0,
        "seconds": 0.0, "previous": None, "edge": None,
        "segment_index": None, "active": True,
    }]
    at_state = {(waypoints[0], 1): [0]}
    sequence = itertools.count()
    queue = [(0.0, next(sequence), 0)]

    def build_result(label_index):
        chain = []
        while label_index:
            label = labels[label_index]
            chain.append(label)
            label_index = label["previous"]
        chain.reverse()
        nodes = [waypoints[0]] + [label["node"] for label in chain]
        leg_count = len(waypoints) - 1
        leg_nodes = [[waypoints[index]] for index in range(leg_count)]
        leg_edges = [[] for _ in range(leg_count)]
        leg_seconds = [0.0] * leg_count
        leg_distances = [0.0] * leg_count
        for edge_index, label in enumerate(chain):
            segment = label["segment_index"]
            edge = label["edge"]
            if segment is None or not 0 <= segment < leg_count:
                raise BudgetSearchIncomplete("有序预算路线的阶段记录无效。")
            leg_edges[segment].append(edge)
            leg_nodes[segment].append(nodes[edge_index + 1])
            u, v, key = edge
            attrs = G[u][v][key]
            leg_distances[segment] += float(attrs.get("length", 0) or 0)
            leg_seconds[segment] += _edge_duration_min(
                attrs, mode, duration_penalty_map.get(edge),
            ) * 60.0
        legs = [
            {
                "nodes": leg_nodes[index],
                "edges": leg_edges[index],
                "distance_m": round(leg_distances[index], 1),
                "duration_min": round(leg_seconds[index] / 60.0, 1),
            }
            for index in range(leg_count)
        ]
        return {
            "legs": legs,
            "distance_m": round(sum(leg_distances), 1),
            # Match the composite payload, which sums independently rounded
            # per-leg estimates before exposing the total.
            "duration_min": round(sum(leg["duration_min"] for leg in legs), 1),
        }

    while queue:
        _, _, label_index = heapq.heappop(queue)
        current = labels[label_index]
        if not current["active"]:
            continue
        if current["stage"] == len(waypoints):
            result = build_result(label_index)
            if (max_distance_m is None or result["distance_m"] <= max_distance_m + 0.15) and (
                max_duration_s is None
                or result["duration_min"] * 60.0 <= max_duration_s + 3.0
            ):
                return result
            continue

        u, stage = current["node"], current["stage"]
        for _, v, key, attrs in G.out_edges(u, keys=True, data=True):
            edge = (u, v, key)
            try:
                length = float(attrs.get("length", 0) or 0)
                cost = float(edge_weight(u, v, {key: attrs}))
                seconds = _edge_duration_min(
                    attrs, mode, duration_penalty_map.get(edge),
                ) * 60.0
            except (TypeError, ValueError, OverflowError) as exc:
                raise BudgetSearchIncomplete("路网含有无法计算的边，有序预算路线暂不能确认。") from exc
            if (not math.isfinite(length) or length < 0
                    or not math.isfinite(cost) or cost < 0
                    or not math.isfinite(seconds) or seconds < 0):
                raise BudgetSearchIncomplete("路网含有无效距离或代价，有序预算路线暂不能确认。")

            next_stage = stage + 1 if v == waypoints[stage] else stage
            distance = current["distance"] + length
            elapsed = current["seconds"] + seconds
            if max_distance_m is not None and distance > max_distance_m + 0.15:
                continue
            # Leave enough room for the same public one-decimal-minute rounding
            # used by the single-route budget solver; reconstructed leg totals
            # are checked exactly when the final label is popped.
            if max_duration_s is not None and elapsed > max_duration_s + 6.0:
                continue
            next_cost = current["cost"] + cost
            state = (v, next_stage)
            others = at_state.get(state, [])
            if any(labels[index]["cost"] <= next_cost
                   and labels[index]["distance"] <= distance
                   and labels[index]["seconds"] <= elapsed for index in others):
                continue
            remaining = []
            for index in others:
                old = labels[index]
                if (next_cost <= old["cost"] and distance <= old["distance"]
                        and elapsed <= old["seconds"]):
                    old["active"] = False
                else:
                    remaining.append(index)
            if len(labels) >= _MAX_CONSTRAINED_LABELS:
                raise BudgetSearchIncomplete("有序预算路线搜索超过计算上限，暂时无法确认是否有可行路线。")
            next_index = len(labels)
            labels.append({
                "node": v, "stage": next_stage, "cost": next_cost,
                "distance": distance, "seconds": elapsed,
                "previous": label_index, "edge": edge,
                "segment_index": stage - 1, "active": True,
            })
            remaining.append(next_index)
            at_state[state] = remaining
            heapq.heappush(queue, (next_cost, next(sequence), next_index))
    return None


def compute_budgeted_ordered_route(
    G, waypoints, *, constraints=None, weights=None, mode="walk",
    road_conditions=None, weather_info=None, strategy_name="recommended",
):
    """Find per-leg paths that jointly satisfy a hard whole-trip budget.

    Returns an exact edge sequence for each ordered leg, or ``None`` when no
    feasible complete chain exists. The fallback minimizes the configured
    route cost subject to *cumulative* distance and duration limits. It uses
    the same mode, event, hard-constraint, weather, and ETA rules as
    :func:`compute_route`.
    """
    constraints = constraints or {}
    max_distance_m = positive_budget(constraints, "max_distance_m")
    max_duration_s = positive_budget(constraints, "max_duration_s")
    if max_distance_m is None and max_duration_s is None:
        return None
    mode = normalize_mode(mode)

    from spatial.routing_index import get_routing_index
    prepared = get_routing_index(G).for_mode(mode)
    G_mode = prepared.graph
    if road_conditions is None:
        from spatial.road_conditions import list_conditions
        road_conditions = list_conditions(strict=True)
    road_penalty = {}
    if road_conditions:
        from spatial.road_conditions import apply_conditions_to_graph
        G_mode, road_penalty, _, _ = apply_conditions_to_graph(
            G_mode, road_conditions, mode, strict=True,
        )
    G_filtered, _, constraint_penalty = _filter_by_constraints(G_mode, constraints)

    weather_penalty = {}
    if weather_info:
        try:
            from spatial.weather import weather_slope_penalty
            slope_penalties = weather_slope_penalty(weather_info)
            if slope_penalties:
                for u, v, key, attrs in G_filtered.edges(keys=True, data=True):
                    multiplier = slope_penalties.get(attrs.get("slope_level", 3))
                    if multiplier:
                        weather_penalty[(u, v, key)] = multiplier
        except Exception as exc:
            logger.warning("有序预算路线的天气惩罚无法计算: %s", exc)

    if strategy_name == "shortest":
        resolved_weights = {"distance": 1.0, "slope": 0.0, "scenery": 0.0}
    elif weights is None:
        resolved_weights = dict(MODE_DEFAULT_WEIGHTS[mode])
    else:
        resolved_weights = resolve_weights(weights)
    penalty_map = _merge_penalty_maps(
        prepared.mode_penalty, road_penalty, weather_penalty, constraint_penalty,
    )
    annotation_degraded = _should_degrade_annotations()
    _, norm_lengths = _normalize_lengths(G_filtered)
    edge_weight = _edge_cost_factory(
        G_filtered, norm_lengths, resolved_weights, penalty_map,
        annotation_degraded,
        outside_road_penalty=MODE_OUTSIDE_ROAD_PENALTY[mode],
    )
    duration_penalty_map = _merge_penalty_maps(road_penalty, weather_penalty)
    started = time.perf_counter()
    result = _budgeted_ordered_path(
        G_filtered, waypoints, mode=mode, edge_weight=edge_weight,
        duration_penalty_map=duration_penalty_map,
        max_distance_m=max_distance_m, max_duration_s=max_duration_s,
    )
    if result is None:
        return None

    for leg in result["legs"]:
        nodes, edges = leg["nodes"], leg["edges"]
        leg["recommended_length_m"] = leg.pop("distance_m")
        leg["duration_min"] = estimate_duration_min(
            None, mode, G=G, route_nodes=nodes, route_edges=edges,
            penalty_map=duration_penalty_map,
        )
        leg["slope_avg_recommended"] = round(
            _weighted_avg_attr(G, nodes, "slope_level", edges), 3,
        )
        leg["scenery_avg_recommended"] = round(
            _weighted_avg_attr(G, nodes, "scenery_level", edges), 3,
        )
        leg["degraded"] = any(
            _get_edge_attr(G[u][v][key], "slope_level")[1]
            or _get_edge_attr(G[u][v][key], "scenery_level")[1]
            for u, v, key in edges
        )
    result["duration_min"] = round(sum(leg["duration_min"] for leg in result["legs"]), 1)
    result["search_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return result


def compute_budgeted_multimodal_route(
    G, segments, *, constraints=None, weights=None, road_conditions=None,
    weather_info=None, strategy_name="recommended",
):
    """Find exact per-mode segment paths that jointly satisfy trip budgets.

    Each segment is searched on its own mode-filtered graph. The solver keeps
    each segment's Pareto frontier (soft cost, distance, ETA), then combines
    those frontiers under the public whole-trip rounding rules. A bounded
    search raises :class:`BudgetSearchIncomplete` rather than claiming that no
    compliant multimodal route exists.
    """
    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError("multimodal budget search needs at least one segment")
    constraints = constraints or {}
    max_distance_m = positive_budget(constraints, "max_distance_m")
    max_duration_s = positive_budget(constraints, "max_duration_s")
    if max_distance_m is None and max_duration_s is None:
        return None
    if road_conditions is None:
        from spatial.road_conditions import list_conditions
        road_conditions = list_conditions(strict=True)

    from spatial.routing_index import get_routing_index

    started = time.perf_counter()
    # Each atomic leg gets the same bounded label allowance as a direct
    # constrained search. Segment searches run serially, so raising the per-leg
    # allowance does not multiply peak label memory; the combined frontier has
    # its own global cap below.
    per_segment_limit = _MAX_CONSTRAINED_LABELS
    frontiers = []
    prepared_segments = []
    _, global_norm_lengths = _normalize_lengths(G)
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("multimodal budget segment must be a mapping")
        mode = normalize_mode(segment.get("mode"))
        start_node, end_node = segment.get("start_node"), segment.get("end_node")
        if start_node not in G or end_node not in G:
            return None
        prepared = get_routing_index(G).for_mode(mode)
        G_mode = prepared.graph
        road_penalty = {}
        if road_conditions:
            from spatial.road_conditions import apply_conditions_to_graph
            G_mode, road_penalty, _, _ = apply_conditions_to_graph(
                G_mode, road_conditions, mode, strict=True,
            )
        G_filtered, _, constraint_penalty = _filter_by_constraints(G_mode, constraints)
        if start_node not in G_filtered or end_node not in G_filtered:
            return None

        weather_penalty = {}
        if weather_info:
            try:
                from spatial.weather import weather_slope_penalty
                slope_penalties = weather_slope_penalty(weather_info)
                for u, v, key, attrs in G_filtered.edges(keys=True, data=True):
                    multiplier = slope_penalties.get(attrs.get("slope_level", 3))
                    if multiplier:
                        weather_penalty[(u, v, key)] = multiplier
            except Exception as exc:
                logger.warning("多方式预算路线的天气惩罚无法计算: %s", exc)

        if strategy_name == "shortest":
            edge_weight = _length_weight
        else:
            if weights is None:
                resolved_weights = dict(MODE_DEFAULT_WEIGHTS[mode])
            else:
                resolved_weights = resolve_weights(weights)
            penalty_map = _merge_penalty_maps(
                prepared.mode_penalty, road_penalty, weather_penalty,
                constraint_penalty,
            )
            annotation_degraded = _should_degrade_annotations()
            edge_weight = _edge_cost_factory(
                G_filtered, global_norm_lengths, resolved_weights, penalty_map,
                annotation_degraded,
                outside_road_penalty=MODE_OUTSIDE_ROAD_PENALTY[mode],
            )

        duration_penalty_map = _merge_penalty_maps(road_penalty, weather_penalty)
        frontier = _budgeted_path(
            G_filtered, start_node, end_node, mode=mode,
            penalty_map=duration_penalty_map, edge_weight=edge_weight,
            max_distance_m=max_distance_m, max_duration_s=max_duration_s,
            duration_penalty_map=duration_penalty_map, return_frontier=True,
            max_labels=per_segment_limit,
        )
        if not frontier:
            return None
        for candidate in frontier:
            candidate["duration_min"] = estimate_duration_min(
                None, mode, G=G, route_nodes=candidate["nodes"],
                route_edges=candidate["edges"], penalty_map=duration_penalty_map,
            )
            candidate["slope_avg_recommended"] = round(
                _weighted_avg_attr(G, candidate["nodes"], "slope_level",
                                   candidate["edges"]), 3,
            )
            candidate["scenery_avg_recommended"] = round(
                _weighted_avg_attr(G, candidate["nodes"], "scenery_level",
                                   candidate["edges"]), 3,
            )
            candidate["degraded"] = any(
                _get_edge_attr(G[u][v][key], "slope_level")[1]
                or _get_edge_attr(G[u][v][key], "scenery_level")[1]
                for u, v, key in candidate["edges"]
            )
            candidate["mode"] = mode
        frontiers.append(frontier)
        prepared_segments.append({"mode": mode, "start_node": start_node,
                                  "end_node": end_node})

    partials = [{"cost": 0.0, "distance_m": 0.0, "duration_s": 0.0,
                 "legs": []}]
    for frontier in frontiers:
        expanded = []
        for partial in partials:
            for candidate in frontier:
                distance_m = round(partial["distance_m"] + candidate["distance_m"], 1)
                duration_s = partial["duration_s"] + candidate["duration_min"] * 60.0
                if (max_distance_m is not None
                        and distance_m > max_distance_m + 0.15):
                    continue
                if (max_duration_s is not None
                        and round(duration_s / 60.0, 1) * 60.0 > max_duration_s + 3.0):
                    continue
                expanded.append({
                    "cost": partial["cost"] + candidate["cost"],
                    "distance_m": distance_m,
                    "duration_s": duration_s,
                    "legs": partial["legs"] + [candidate],
                })
                if len(expanded) > _MAX_CONSTRAINED_LABELS:
                    raise BudgetSearchIncomplete(
                        "多方式整程预算组合超过计算上限，暂时无法确认可行路线。",
                    )

        nondominated = []
        for candidate in sorted(
            expanded,
            key=lambda item: (item["cost"], item["distance_m"], item["duration_s"]),
        ):
            if any(
                old["cost"] <= candidate["cost"]
                and old["distance_m"] <= candidate["distance_m"]
                and old["duration_s"] <= candidate["duration_s"]
                for old in nondominated
            ):
                continue
            nondominated = [
                old for old in nondominated
                if not (
                    candidate["cost"] <= old["cost"]
                    and candidate["distance_m"] <= old["distance_m"]
                    and candidate["duration_s"] <= old["duration_s"]
                )
            ]
            nondominated.append(candidate)
        partials = nondominated
        if not partials:
            return None

    best = min(
        partials,
        key=lambda item: (item["cost"], item["distance_m"], item["duration_s"]),
    )
    return {
        "legs": best["legs"],
        "distance_m": best["distance_m"],
        "duration_min": round(best["duration_s"] / 60.0, 1),
        "search_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "segments": prepared_segments,
    }


def _interpolate_to_distance(weights: dict, factor: float) -> dict:
    """Move a soft preference toward pure distance by a deterministic factor."""
    return {
        "distance": factor * weights["distance"] + (1.0 - factor),
        "slope": factor * weights["slope"],
        "scenery": factor * weights["scenery"],
    }


def _compute_overlap(route_a: list, route_b: list) -> float:
    """
    计算两条路径的元素重叠率；主流程传入精确边 ID。

    overlap_rate = 共同边数 / max(len(route_a), len(route_b))

    Args:
        route_a: 节点路径 [node1, node2, ...]
        route_b: 节点路径 [node1, node2, ...]

    Returns:
        重叠率 0.0-1.0
    """
    if not route_a or not route_b:
        return 0.0

    set_a = set(route_a)
    set_b = set(route_b)
    overlap = len(set_a & set_b)
    max_len = max(len(route_a), len(route_b))

    return overlap / max_len if max_len > 0 else 0.0


def _path_length(G: nx.MultiDiGraph, path: list, route_edges=None) -> float:
    """
    计算路径总长度（米）。

    Args:
        G: 路网图
        path: 节点路径

    Returns:
        路径总长度（米）
    """
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            if route_edges is not None:
                total += float(edge_data[route_edges[i][2]].get("length", 0))
            else:
                total += min(float(d.get("length", 0)) for d in edge_data.values())
    return total


def _weighted_avg_attr(G: nx.MultiDiGraph, path: list, attr: str,
                       route_edges=None) -> float:
    """计算路径的长度加权平均属性值（如坡度/景观等级）。

    用于实验评估：Σ(attr(e) × L(e)) / Σ L(e)
    """
    total_weighted = 0.0
    total_length = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G.get_edge_data(u, v)
        if not edge_data:
            continue
        if route_edges is not None:
            data = edge_data[route_edges[i][2]]
        else:
            data = min(edge_data.values(), key=lambda d: d.get("length", float("inf")))
        length = data.get("length", 0)
        attr_val = data.get(attr, 3)  # 默认值 3（中等）
        total_weighted += float(attr_val) * length
        total_length += length
    return total_weighted / total_length if total_length > 0 else 0.0


def _raise_no_path(G, start_node, end_node, filter_status, G_filtered, mode="walk"):
    """不可达诊断：记录日志并抛出带友好提示的 ValueError。"""
    mode = normalize_mode(mode)
    from spatial.network import get_node_coords
    try:
        sn_coords = get_node_coords(G, start_node)
        en_coords = get_node_coords(G, end_node)
    except Exception:
        sn_coords, en_coords = None, None
    logger.warning(
        "路径不可达: start=%s (%s) end=%s (%s) mode=%s filter=%s edges=%d/%d",
        start_node, sn_coords, end_node, en_coords, mode,
        filter_status, G_filtered.number_of_edges(), G.number_of_edges(),
    )
    # 路况导致不可达时，给出针对性提示
    is_road_blocked = "road_closure" in filter_status or "road_blocked" in filter_status
    if is_road_blocked:
        if mode == "drive":
            raise ValueError(
                "驾车路线因道路封闭/施工无法通行，建议切换骑行或步行，或选择其他路线～"
            )
        if mode == "bike":
            raise ValueError(
                "骑行路线因道路封闭/施工无法通行，建议切换步行，或选择其他路线～"
            )
        raise ValueError(
            "该路线因道路封闭/施工暂时无法通行，建议选择附近的其他地点作为起终点，或稍后再试～"
        )
    if "slope_avoid" in filter_status:
        raise ValueError("暂未找到满足避坡要求的路线，已保留避开陡坡的约束，请更换起终点。")
    if mode == "drive":
        raise ValueError(
            "驾车无法到达该地点（附近可能只有步行道/台阶），建议切换骑行或步行～"
        )
    if mode == "bike":
        raise ValueError(
            "骑行无法到达该地点（附近可能只有台阶/陡坡），建议切换步行～"
        )
    raise ValueError(
        f"该两点之间没有完全在校内的步行路线，"
        f"可能需要经过校外市政道路。"
        f"可试试选择附近的其他校门或地点作为起终点。"
    )


def compute_route(
    G: nx.MultiDiGraph,
    start_node: int,
    end_node: int,
    constraints: Optional[dict] = None,
    weights: Optional[dict] = None,
    mode: str = "walk",
    road_conditions: Optional[list] = None,
    weather_info: Optional[dict] = None,
    disable_hard_filter: bool = False,
    strategy_name: str = "recommended",
    detour_cap: float = 1.25,
    prepared=None,
    reuse_exact_shortest_edges=None,
) -> dict:
    """
    多因素路径计算主函数。

    流程：
      1. 解析出行方式与权重（默认 → 校验 → 归一化）
      2. 出行方式过滤（walk 不过滤；bike 删纯台阶；drive 仅留车行道）
      3. 特殊路况处理（封闭边移除，施工/积水等施加惩罚）
      4. 硬约束过滤（坡度等）
      5. 软成本优化（Dijkstra 计算推荐路线）
      6. 计算最短路径基线
      7. 路径长度上限裁剪
      8. 计算重叠率

    Args:
        G: OSMnx MultiDiGraph 路网
        start_node: 起点节点 ID
        end_node: 终点节点 ID
        constraints: 约束 dict，如 {"slope": "avoid"}
        weights: 权重 dict，如 {"distance": 0.2, "slope": 0.5, "scenery": 0.3}；
                 None 时取 MODE_DEFAULT_WEIGHTS[mode]
        mode: 出行方式 "walk" / "bike" / "drive"（默认 "walk"）
        road_conditions: 路况事件列表，None 时自动加载 active 事件
        disable_hard_filter: 设为 True 时跳过硬约束过滤（消融实验用）

    Returns:
        {
            "recommended": [node_id, ...], 推荐路径
            "shortest": [node_id, ...], 最短路径
            "filter_status": str, 过滤状态
            "overlap_rate": float, 重叠率
            "degraded": bool, 是否降级
            "degraded_count": int, 降级边数
            "recommended_length_m": float, 推荐路径长度
            "shortest_length_m": float, 最短路径长度
            "applied_weights": dict, 实际使用的权重
            "length_capped": bool, 是否触发了长度上限
            "max_len": float, 全图最长边长度
            "_annotation_degraded": bool, 是否处于标注降级
            "mode": str, 出行方式
            "duration_min": float, 推荐路径预估时长（分钟）
            "shortest_duration_min": float, 最短路径预估时长（分钟）
            "speed_kmh": float, 模式平均速度
            "road_conditions_applied": int, 实际生效的路况事件数
        }

    Raises:
        ValueError: 起终点不可达时
    """
    timing_started = time.perf_counter()
    constraints = constraints or {}
    max_distance_m = positive_budget(constraints, "max_distance_m")
    max_duration_s = positive_budget(constraints, "max_duration_s")
    mode = normalize_mode(mode)

    if strategy_name == "shortest":
        resolved_weights = {"distance": 1.0, "slope": 0.0, "scenery": 0.0}
        detour_cap = 1.0
    elif weights is None:
        resolved_weights = dict(MODE_DEFAULT_WEIGHTS[mode])
    else:
        resolved_weights = resolve_weights(weights)

    # 天气只提供状态与路段安全惩罚，不修改默认或用户指定的偏好权重。
    weather_penalty = {}
    weather_applied = False
    if weather_info:
        try:
            from spatial.weather import classify_weather
            weather_applied = bool(classify_weather(weather_info).get("label"))
        except Exception as e:
            logger.warning("天气分类失败，跳过: %s", e)

    annotation_degraded = _should_degrade_annotations()

    # 1) 出行方式过滤（所有模式先剔真校外边；bike/drive 再按方式删边+删孤立节点）
    if prepared is None:
        from spatial.routing_index import get_routing_index
        prepared = get_routing_index(G).for_mode(mode)
    G_mode = prepared.graph
    mode_status = prepared.mode_status
    mode_penalty = dict(prepared.mode_penalty)
    max_len = prepared.max_len
    norm_lengths = prepared.norm_lengths

    # 起终点在过滤后被作为孤立节点移除（如驾车时起终点只连台阶/步行道，
    # 或步行时起终点只连校外路段）→ 该方式不可达，给友好提示
    if start_node not in G_mode or end_node not in G_mode:
        _raise_no_path(G, start_node, end_node, mode_status, G_mode, mode=mode)

    # 1.5) 特殊路况处理：按 CONDITION_EFFECTS 分模式施加硬封/软惩罚
    # （道路封闭、深积水等对步行同样不可通行，不再豁免步行模式）
    road_penalty = {}
    road_conditions_applied = 0
    closed_edges = set()
    if road_conditions is None:
        from spatial.road_conditions import list_conditions
        road_conditions = list_conditions(strict=True)
    if road_conditions:
        from spatial.road_conditions import apply_conditions_to_graph
        G_mode, road_penalty, closed_edges, road_conditions_applied = (
            apply_conditions_to_graph(G_mode, road_conditions, mode, strict=True)
        )
        if closed_edges:
            mode_status = f"{mode_status}+road_closure"
        if road_penalty:
            mode_status = f"{mode_status}+road_penalty"
        # 路况封闭边可能导致起终点变成孤立节点
        if start_node not in G_mode or end_node not in G_mode:
            _raise_no_path(G, start_node, end_node, f"{mode_status}+road_blocked", G_mode, mode=mode)

    # 2) 硬约束过滤（坡度等），在方式过滤图上进行
    if disable_hard_filter:
        G_filtered, filter_status, constraint_penalty = G_mode, "no_filter", {}
    else:
        G_filtered, filter_status, constraint_penalty = _filter_by_constraints(G_mode, constraints)

    # 天气湿滑惩罚：雨雪天对陡坡/台阶边（slope_level 4/5）施加额外成本，智能避坡
    if weather_info:
        try:
            from spatial.weather import weather_slope_penalty
            slope_pen = weather_slope_penalty(weather_info)
            if slope_pen:
                for u, v, k, data in G_filtered.edges(keys=True, data=True):
                    lvl = data.get("slope_level", 3)
                    mult = slope_pen.get(lvl)
                    if mult:
                        weather_penalty[(u, v, k)] = mult
        except Exception as e:
            logger.warning("天气湿滑惩罚应用失败，跳过: %s", e)

    # 方式惩罚 + 路况惩罚 + 天气惩罚 + 约束惩罚合并（同一 key 相乘）
    penalty_map = _merge_penalty_maps(mode_penalty, road_penalty)
    penalty_map = _merge_penalty_maps(penalty_map, weather_penalty)
    penalty_map = _merge_penalty_maps(penalty_map, constraint_penalty)
    # Mode/constraint penalties guide path choice, while ETA separately applies
    # physical slope/step factors. Only real slowdown conditions affect time,
    # otherwise steps and steep-bike penalties would be counted twice.
    duration_penalty_map = _merge_penalty_maps(road_penalty, weather_penalty)

    if mode == "walk":
        status_parts = [mode_status] if mode_status != "no_filter" else []
        if filter_status != "no_filter":
            status_parts.append(filter_status)
        if annotation_degraded is not None:
            status_parts.append(annotation_degraded)
        filter_status = "+".join(status_parts) if status_parts else "no_filter"
    else:
        status_parts = [mode_status]
        if filter_status != "no_filter":
            status_parts.append(filter_status)
        if annotation_degraded is not None:
            status_parts.append(annotation_degraded)
        filter_status = "+".join(status_parts)

    outside_penalty = MODE_OUTSIDE_ROAD_PENALTY[mode]

    edge_weight = _edge_cost_factory(
        G_filtered, norm_lengths, resolved_weights, penalty_map,
        annotation_degraded, outside_road_penalty=outside_penalty,
    )

    graph_prepare_ms = (time.perf_counter() - timing_started) * 1000
    path_search_started = time.perf_counter()

    # 最短基线与推荐路线使用同一张已经完成方式、管制和硬约束过滤的图。
    # 普通通勤直接复用这一次搜索，避免再运行一遍软成本 Dijkstra。
    reusable_edges = None
    if (strategy_name == "shortest" and max_distance_m is None and max_duration_s is None
            and isinstance(reuse_exact_shortest_edges, (list, tuple))):
        candidate_edges = [tuple(edge) for edge in reuse_exact_shortest_edges
                           if isinstance(edge, (list, tuple)) and len(edge) == 3]
        previous_node = start_node
        if len(candidate_edges) == len(reuse_exact_shortest_edges):
            valid_chain = True
            for u, v, key in candidate_edges:
                if u != previous_node or not G_filtered.has_edge(u, v, key):
                    valid_chain = False
                    break
                previous_node = v
            if valid_chain and previous_node == end_node:
                reusable_edges = candidate_edges

    if reusable_edges is not None:
        shortest_edges = reusable_edges
        shortest = [start_node, *(edge[1] for edge in shortest_edges)]
        dijkstra_runs = 0
    else:
        try:
            shortest, shortest_edges = _run_path(
                G_filtered, start_node, end_node, _length_weight
            )
        except nx.NetworkXNoPath:
            if constraints.get("slope") == "avoid" and not disable_hard_filter:
                filter_status = f"{filter_status}+slope_avoid"
            _raise_no_path(G, start_node, end_node, filter_status, G_filtered, mode=mode)
        dijkstra_runs = 1

    if strategy_name == "shortest":
        recommended, recommended_edges = shortest, shortest_edges
    else:
        try:
            recommended, recommended_edges = _run_path(
                G_filtered, start_node, end_node, edge_weight
            )
            dijkstra_runs += 1
        except nx.NetworkXNoPath:
            if constraints.get("slope") == "avoid" and not disable_hard_filter:
                filter_status = f"{filter_status}+slope_avoid"
            _raise_no_path(G, start_node, end_node, filter_status, G_filtered, mode=mode)

    recommended_len = _path_length(G, recommended, recommended_edges)
    shortest_len = _path_length(G, shortest, shortest_edges)

    length_capped = False
    length_limit = shortest_len * max(1.0, float(detour_cap))
    if strategy_name != "shortest" and shortest_len > 0 and recommended_len > length_limit:
        length_capped = True
        logger.info(
            "推荐路径 %.0fm 超过最短路径 %.0fm 的 %.2f 倍上限，开始收敛偏好",
            recommended_len, shortest_len, detour_cap,
        )
        for factor in (0.75, 0.50, 0.25):
            candidate_weights = _interpolate_to_distance(resolved_weights, factor)
            candidate_weight = _edge_cost_factory(
                G_filtered,
                norm_lengths,
                candidate_weights,
                penalty_map,
                annotation_degraded,
                outside_road_penalty=outside_penalty,
            )
            candidate, candidate_edges = _run_path(
                G_filtered, start_node, end_node, candidate_weight
            )
            dijkstra_runs += 1
            candidate_len = _path_length(G, candidate, candidate_edges)
            if candidate_len <= length_limit:
                recommended, recommended_edges = candidate, candidate_edges
                recommended_len = candidate_len
                break
        else:
            recommended, recommended_edges = shortest, shortest_edges
            recommended_len = shortest_len

    recommended_duration = estimate_duration_min(
        None, mode, G=G, route_nodes=recommended,
        route_edges=recommended_edges, penalty_map=duration_penalty_map,
    )
    shortest_duration = estimate_duration_min(
        None, mode, G=G, route_nodes=shortest,
        route_edges=shortest_edges, penalty_map=duration_penalty_map,
    )

    def within_budget(length_m, duration_min):
        return ((max_distance_m is None or length_m <= max_distance_m + 0.15)
                and (max_duration_s is None or duration_min * 60 <= max_duration_s + 3))

    hard_budget_fallback = False
    if not within_budget(recommended_len, recommended_duration):
        feasible = _budgeted_path(
            G_filtered, start_node, end_node, mode=mode,
            penalty_map=penalty_map, edge_weight=edge_weight,
            max_distance_m=max_distance_m, max_duration_s=max_duration_s,
            duration_penalty_map=duration_penalty_map,
        )
        if feasible is None:
            limit_name = ("距离和时间" if max_distance_m is not None and max_duration_s is not None
                          else "距离" if max_distance_m is not None else "时间")
            raise ValueError(f"当前条件下没有满足{limit_name}上限的可通行路线。")
        recommended, recommended_edges = feasible
        recommended_len = _path_length(G, recommended, recommended_edges)
        recommended_duration = estimate_duration_min(
            None, mode, G=G, route_nodes=recommended,
            route_edges=recommended_edges, penalty_map=duration_penalty_map,
        )
        if not within_budget(recommended_len, recommended_duration):
            raise BudgetSearchIncomplete("预算路线与公开估时结果不一致，暂时无法确认可行路线。")
        hard_budget_fallback = True
    shortest_baseline_feasible = within_budget(shortest_len, shortest_duration)

    path_search_ms = (time.perf_counter() - path_search_started) * 1000
    response_build_started = time.perf_counter()

    overlap_rate = _compute_overlap(recommended_edges, shortest_edges)

    # 从实际结果路径计算 degraded_count（避免闭包中 Dijkstra 重复评估边导致计数虚高）
    degraded_count = 0
    for u, v, key in recommended_edges:
        data = G[u][v][key]
        _, is_d = _compute_edge_cost(
            data, norm_lengths.get((u, v, key), 0.001), resolved_weights,
        )
        if is_d:
            degraded_count += 1
    is_degraded = degraded_count > 0

    # 计算路径加权平均坡度和景观（用于实验评估）
    slope_avg_rec = _weighted_avg_attr(G, recommended, "slope_level", recommended_edges)
    scenery_avg_rec = _weighted_avg_attr(G, recommended, "scenery_level", recommended_edges)
    slope_avg_short = _weighted_avg_attr(G, shortest, "slope_level", shortest_edges)
    scenery_avg_short = _weighted_avg_attr(G, shortest, "scenery_level", shortest_edges)

    result = {
        "recommended": recommended,
        "shortest": shortest,
        "recommended_edges": recommended_edges,
        "shortest_edges": shortest_edges,
        "filter_status": filter_status,
        "overlap_rate": round(overlap_rate, 4),
        "degraded": is_degraded,
        "degraded_count": degraded_count,
        "recommended_length_m": round(recommended_len, 1),
        "shortest_length_m": round(shortest_len, 1),
        "applied_weights": resolved_weights,
        "length_capped": length_capped,
        "hard_budget_fallback": hard_budget_fallback,
        "shortest_baseline_feasible": shortest_baseline_feasible,
        "strategy": strategy_name,
        "detour_cap": float(detour_cap),
        "dijkstra_runs": dijkstra_runs,
        "max_len": max_len,
        "_annotation_degraded": annotation_degraded is not None,
        "mode": mode,
        "duration_min": recommended_duration,
        "shortest_duration_min": shortest_duration,
        "speed_kmh": MODE_SPEEDS_KMH[mode],
        "road_conditions_applied": road_conditions_applied,
        "weather_applied": weather_applied,
        "slope_avg_recommended": round(slope_avg_rec, 3),
        "scenery_avg_recommended": round(scenery_avg_rec, 3),
        "slope_avg_shortest": round(slope_avg_short, 3),
        "scenery_avg_shortest": round(scenery_avg_short, 3),
    }
    result["timings_ms"] = {
        "agent": 0.0,
        "poi_resolution": 0.0,
        "graph_prepare": round(graph_prepare_ms, 3),
        "path_search": round(path_search_ms, 3),
        "response_build": round(
            (time.perf_counter() - response_build_started) * 1000, 3
        ),
    }
    return result


def compute_route_with_annotations(
    G: nx.MultiDiGraph,
    start_node: int,
    end_node: int,
    constraints: Optional[dict] = None,
    weights: Optional[dict] = None,
    annotations: Optional[list] = None,
    mode: str = "walk",
    road_conditions: Optional[list] = None,
    weather_info: Optional[dict] = None,
    strategy_name: str = "recommended",
    detour_cap: float = 1.25,
) -> dict:
    """
    带路段标注的路径计算（slope_level / scenery_level 注入路网边属性）。

    Args:
        G: OSMnx MultiDiGraph 路网
        start_node: 起点节点 ID
        end_node: 终点节点 ID
        constraints: 约束 dict
        weights: 权重 dict
        annotations: 标注列表，每项含 edge_id 列表和属性值
        mode: 出行方式 "walk" / "bike" / "drive"（默认 "walk"）
        road_conditions: 路况事件列表

    Returns:
        同 compute_route 返回结构
    """
    G_annotated = G.copy()

    if annotations:
        for ann in annotations:
            edge_ids = ann.get("edge_id", [])
            slope = ann.get("slope_level")
            scenery = ann.get("scenery_level")
            name = ann.get("name", "")

            # edge_ids = [u_node, v_node, key] — use as triple, don't iterate
            if len(edge_ids) >= 3:
                u_target, v_target, k_target = int(edge_ids[0]), int(edge_ids[1]), int(edge_ids[2])
                if G_annotated.has_edge(u_target, v_target, k_target):
                    data = G_annotated[u_target][v_target][k_target]
                    if slope is not None:
                        data["slope_level"] = int(slope)
                    if scenery is not None:
                        data["scenery_level"] = int(scenery)
                    if name:
                        data["name"] = str(name)

    return compute_route(
        G_annotated, start_node, end_node, constraints, weights, mode=mode,
        road_conditions=road_conditions, weather_info=weather_info,
        strategy_name=strategy_name, detour_cap=detour_cap,
    )


# ============================================================================
# 途经点路径规划（via）
# ============================================================================

# 顺路性阈值：绕行比 = (起点→途经点 + 途经点→终点) / 起点→终点（最短路径口径）
# 超过该比值的途经点判定为"不顺路"，不推荐。可在 config 中覆盖。
VIA_MAX_DETOUR_RATIO = 1.5


def _shortest_distance(G_mode: nx.MultiDiGraph, u: int, v: int) -> float:
    """两点最短路径长度（米），不可达返回 inf。"""
    try:
        return nx.shortest_path_length(G_mode, u, v, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def rank_via_candidates(
    G: nx.MultiDiGraph,
    start_node: int,
    end_node: int,
    candidates: list,
    max_detour_ratio: float = VIA_MAX_DETOUR_RATIO,
    top_k: int = 3,
    mode: str = "walk",
) -> tuple:
    """对途经点候选按顺路性排序。

    Args:
        candidates: [(poi_dict, node_id), ...]，poi_dict 至少含 name；
                    node_id 由调用方用 get_nearest_node 解析（WGS-84）。
        max_detour_ratio: 绕行比上限，超过即判不顺路。
        top_k: 返回的顺路候选数量。

    Returns:
        (on_the_way, off_the_way)：
          on_the_way = [(poi_dict, node_id, detour_ratio), ...] 按绕行比升序
          off_the_way = 同上结构，全部超阈值（供"都不顺路"时说明）
    """
    from spatial.routing_index import get_routing_index
    G_mode = get_routing_index(G).for_mode(mode).graph
    direct = _shortest_distance(G_mode, start_node, end_node)
    if direct <= 0 or direct == float("inf"):
        return [], []

    on_the_way, off_the_way = [], []
    for poi, node in candidates:
        if node in (start_node, end_node):
            ratio = 1.0  # 途经点就是起终点本身，零绕行
        else:
            d1 = _shortest_distance(G_mode, start_node, node)
            d2 = _shortest_distance(G_mode, node, end_node)
            if d1 == float("inf") or d2 == float("inf"):
                continue  # 不可达候选直接丢弃
            ratio = (d1 + d2) / direct
        entry = (poi, node, round(ratio, 3))
        (on_the_way if ratio <= max_detour_ratio else off_the_way).append(entry)

    on_the_way.sort(key=lambda x: x[2])
    off_the_way.sort(key=lambda x: x[2])
    return on_the_way[:top_k], off_the_way


def compute_via_route(
    G: nx.MultiDiGraph,
    start_node: int,
    via_node: int,
    end_node: int,
    constraints: Optional[dict] = None,
    weights: Optional[dict] = None,
    mode: str = "walk",
    road_conditions: Optional[list] = None,
    weather_info: Optional[dict] = None,
    strategy_name: str = "recommended",
    detour_cap: float = 1.25,
) -> dict:
    """起点 → 途经点 → 终点两段路径拼接。

    Returns:
        {
            "leg1": compute_route 结果（start→via）,
            "leg2": compute_route 结果（via→end）,
            "direct": compute_route 结果（start→end 直达基线）,
            "total_length_m": float,
            "detour_ratio": float,   # 两段合计 / 直达（推荐路径口径）
            "via_node": int,
        }
    """
    from spatial.routing_index import get_routing_index
    prepared = get_routing_index(G).for_mode(mode)
    kwargs = dict(
        mode=mode,
        road_conditions=road_conditions,
        weather_info=weather_info,
        prepared=prepared,
        strategy_name=strategy_name,
        detour_cap=detour_cap,
    )
    leg1 = compute_route(G, start_node, via_node, constraints, weights, **kwargs)
    leg2 = compute_route(G, via_node, end_node, constraints, weights, **kwargs)
    direct = compute_route(G, start_node, end_node, constraints, weights, **kwargs)

    total = leg1["recommended_length_m"] + leg2["recommended_length_m"]
    base = direct["recommended_length_m"] or direct["shortest_length_m"]
    detour_ratio = total / base if base > 0 else float("inf")

    return {
        "leg1": leg1,
        "leg2": leg2,
        "direct": direct,
        "total_length_m": round(total, 1),
        "detour_ratio": round(detour_ratio, 3),
        "via_node": via_node,
    }


# ============================================================================
# 多 POI 游览环线（tour）
# ============================================================================

TOUR_MAX_POIS = 8                 # 游览点数量上限（全对距离矩阵 28 次 Dijkstra 以内）
TOUR_MAX_TOTAL_M = 6000.0         # 环线总长上限（游客步行为参照）
_2OPT_MAX_ROUNDS = 200            # 2-opt 最大迭代轮次


def _tour_distance_matrix(G_mode: nx.MultiDiGraph, nodes: list) -> dict:
    """全对最短距离矩阵：{ (i, j): 米 }，不可达为 inf。"""
    matrix = {}
    for i, u in enumerate(nodes):
        try:
            lengths = nx.single_source_dijkstra_path_length(G_mode, u, weight="length")
        except nx.NodeNotFound:
            lengths = {}
        for j, v in enumerate(nodes):
            if i != j:
                matrix[(i, j)] = lengths.get(v, float("inf"))
    return matrix


def _greedy_nn_order(matrix: dict, n: int, start_idx: int = 0) -> list:
    """贪心最近邻初解。"""
    unvisited = set(range(n)) - {start_idx}
    order = [start_idx]
    while unvisited:
        cur = order[-1]
        nxt = min(unvisited, key=lambda j: matrix.get((cur, j), float("inf")))
        if matrix.get((cur, nxt), float("inf")) == float("inf"):
            break  # 剩余点不可达，留给调用方剔除
        order.append(nxt)
        unvisited.remove(nxt)
    return order


def _order_length(matrix: dict, order: list, loop: bool) -> float:
    total = 0.0
    for i in range(len(order) - 1):
        d = matrix.get((order[i], order[i + 1]), float("inf"))
        if d == float("inf"):
            return float("inf")
        total += d
    if loop and len(order) > 1:
        d = matrix.get((order[-1], order[0]), float("inf"))
        if d == float("inf"):
            return float("inf")
        total += d
    return total


def _two_opt(matrix: dict, order: list, loop: bool) -> list:
    """2-opt 改进游览顺序（固定起点在 order[0]）。"""
    best = order[:]
    best_len = _order_length(matrix, best, loop)
    improved = True
    rounds = 0
    while improved and rounds < _2OPT_MAX_ROUNDS:
        improved = False
        rounds += 1
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cand_len = _order_length(matrix, cand, loop)
                if cand_len < best_len - 1e-6:
                    best, best_len = cand, cand_len
                    improved = True
    return best


def compute_tour_route(
    G: nx.MultiDiGraph,
    poi_nodes: list,
    start_node: Optional[int] = None,
    loop: bool = True,
    constraints: Optional[dict] = None,
    weights: Optional[dict] = None,
    mode: str = "walk",
    max_total_m: float = TOUR_MAX_TOTAL_M,
    road_conditions: Optional[list] = None,
    weather_info: Optional[dict] = None,
    strategy_name: str = "recommended",
    detour_cap: float = 1.25,
) -> dict:
    """多 POI 游览环线：贪心最近邻 + 2-opt 排序，逐段生成真实加权路径。

    Args:
        poi_nodes: [(poi_dict, node_id), ...]，poi_dict 至少含 name/importance 信息；
                   数量 ≤ TOUR_MAX_POIS，超出部分按列表顺序截断前的由调用方筛选。
        start_node: 游览起点（如"我的位置"/校门）；None 时以第一个 POI 为起点。
        loop: True 回到起点（环线），False 开放路径。
        max_total_m: 总长上限，超出时按 poi_dict["importance"] 从低到高剔点。

    Returns:
        {
            "ordered_pois": [poi_dict, ...] 游览顺序,
            "ordered_nodes": [node_id, ...],
            "legs": [compute_route 结果, ...] 每段路径,
            "total_length_m": float,
            "dropped": [poi_dict, ...] 被剔除的点（不可达或超长）,
            "loop": bool,
        }
    """
    if not poi_nodes:
        return {"ordered_pois": [], "ordered_nodes": [], "legs": [],
                "total_length_m": 0.0, "dropped": [], "loop": loop}

    poi_nodes = poi_nodes[:TOUR_MAX_POIS]
    from spatial.routing_index import get_routing_index
    prepared = get_routing_index(G).for_mode(mode)
    G_mode = prepared.graph

    # 起点固定为顺序首位：有 start_node 时把它作为 0 号节点加入矩阵
    nodes = [n for _, n in poi_nodes]
    anchor_first = False
    if start_node is not None:
        nodes = [start_node] + nodes
        anchor_first = True

    matrix = _tour_distance_matrix(G_mode, nodes)

    # 剔除不可达点（到任何其他点都不可达的孤立点）
    keep = list(range(len(nodes)))
    if anchor_first:
        keep.remove(0)
    connected = []
    for i in keep:
        if any(matrix.get((i, j), float("inf")) < float("inf") for j in range(len(nodes)) if j != i):
            connected.append(i)
    dropped = [poi_nodes[i - 1][0] for i in keep if i not in connected] if anchor_first \
        else [poi_nodes[i][0] for i in keep if i not in connected]

    if anchor_first:
        sub = [0] + connected
    else:
        sub = connected
    if len(sub) < (2 if anchor_first else 1):
        return {"ordered_pois": [], "ordered_nodes": [], "legs": [],
                "total_length_m": 0.0, "dropped": dropped, "loop": loop}

    order = _greedy_nn_order(matrix, len(nodes), start_idx=0)
    order = [i for i in order if i in sub]
    order = _two_opt(matrix, order, loop and not anchor_first)

    # 总长超限：按重要度从低到高剔点（不动起点锚）
    def _imp(idx):
        if anchor_first and idx == 0:
            return float("inf")
        poi = poi_nodes[idx - 1][0] if anchor_first else poi_nodes[idx][0]
        return float(poi.get("importance", 0))

    while len(order) > 2 and _order_length(matrix, order, loop) > max_total_m:
        victim = min(order[1:], key=_imp)
        order.remove(victim)
        dropped.append(poi_nodes[victim - 1][0] if anchor_first else poi_nodes[victim][0])

    # 逐段生成满足约束的路径；不可达时明确失败，不回退到未过滤的最短路径。
    ordered_nodes = [nodes[i] for i in order]
    ordered_pois = [poi_nodes[i - 1][0] if anchor_first else poi_nodes[i][0]
                    for i in order if not (anchor_first and i == 0)]
    seq = ordered_nodes + ([ordered_nodes[0]] if loop and len(ordered_nodes) > 1 else [])
    legs = []
    total = 0.0
    for a, b in zip(seq, seq[1:]):
        if a == b:
            continue
        leg = compute_route(G, a, b, constraints, weights, mode=mode,
                            road_conditions=road_conditions, weather_info=weather_info,
                            prepared=prepared, strategy_name=strategy_name,
                            detour_cap=detour_cap)
        legs.append(leg)
        total += leg["recommended_length_m"]

    return {
        "ordered_pois": ordered_pois,
        "ordered_nodes": ordered_nodes,
        "legs": legs,
        "total_length_m": round(total, 1),
        "dropped": dropped,
        "loop": loop,
    }
