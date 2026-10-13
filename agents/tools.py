"""珞珈智行 — Agent 工具箱（LLM function calling 的工具定义与执行器）。

设计原则（v2 全 Agent 架构）：
- LLM 只做决策，不做计算：坐标转换、路网吸附、Dijkstra、顺路比、游览排序全部在此
- 所有工具返回 dict（可 JSON 序列化），失败返回 {"error": ..., "message": ...}
  由 planner 回注给 LLM 自我修正，不抛异常中断循环
- 涉及校内地点的信息一律来自这里（resolve_poi / search_poi_candidates），
  配合系统提示约束 LLM 不编造校内设施
"""

import json
import logging
import math
import networkx as nx
import re
import time

import config
from agents import route_state as route_state_module
from spatial.poi import (
    find_poi_ambiguous, search_by_category, search_pois, list_all_pois,
    load_pois, importance_score, _flatten_poi,
)
from spatial.poi_taxonomy import (
    ACTIVITY_LABELS, ACTIVITY_SUBCATEGORIES, activities_for_poi,
    matches_activities, resolve_activity_query, resolve_activity_sequence,
)
from spatial.network import get_network, load_or_download_network, get_nearest_node, get_node_coords
from spatial.routing import (
    compute_route, compute_via_route, compute_tour_route,
    rank_via_candidates, resolve_weights,
    estimate_duration_min, MODE_SPEEDS_KMH, build_turn_by_turn,
)
from spatial.routing_index import get_routing_index
from spatial.coord_transform import gcj02_to_wgs84, wgs84_to_gcj02
from spatial.amap_poi import navigation_wgs
from spatial.road_conditions import (
    list_conditions, CONDITION_LABELS, apply_conditions_to_graph,
    RoadConditionsUnavailableError,
)
from spatial import weather as weather_mod
from agents.routing_policy import select_route_strategy
from agents.preferences import detect_strategy_hint
from agents.route_state import (
    build_route_state,
    current_data_version,
    current_road_condition_version,
)
from agents.timings import add_timings, normalize_timings

logger = logging.getLogger(__name__)

# 沿途 POI 标注阈值（与 routes.py 保持一致的体验）
_ALONG_ROUTE_THRESHOLD_M = 100.0
_ALONG_ROUTE_LIMIT = 8
_ALONG_ROUTE_MIN_IMPORTANCE = 8.0

# 游览默认选点数量与总长上限
_TOUR_DEFAULT_MAX_POIS = 6
_TOUR_MAX_TOTAL_M = 6000.0
_MAX_VIA_POINTS = 10
_REACHABILITY_MAX_ACCESS_SNAP_M = 60.0
_MAX_ENDPOINT_ACCESS_SNAP_M = 80.0
_COMPARISON_STRATEGIES = (
    ("shortest", "最短路径"),
    ("scenery", "风景优先"),
    ("flat", "平坦优先"),
)


def _data_version_for_graph(graph) -> str:
    """Expose a real graph version when available; keep metadata failure non-fatal."""
    try:
        return current_data_version(graph)
    except (AttributeError, TypeError, ValueError):
        logger.warning("无法从当前工具上下文计算路网版本标识")
        return "unavailable"


# ===========================================================================
# 工具入参 Schema（OpenAI function calling 格式，DeepSeek 兼容）
# ===========================================================================

_ENDPOINT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "地点名称（POI 名/别名）；coord 类型时为显示名"},
        "poi_id": {"type": "string", "description": "已检索候选地点的稳定 ID；优先于名称重新匹配"},
        "type": {"type": "string", "enum": ["poi", "coord"],
                 "description": "poi=校内地点名；coord=WGS-84 坐标（如用户 GPS 定位）"},
        "lng": {"type": "number", "description": "经度（仅 type=coord 时填，WGS-84）"},
        "lat": {"type": "number", "description": "纬度（仅 type=coord 时填，WGS-84）"},
    },
    "required": ["name"],
}

_PREFERENCE_SCHEMA = {
    "constraints": {
            "type": "object",
            "description": "硬约束。仅在用户明确说'避开/不要'时设置",
            "properties": {
                "distance": {"type": "string", "enum": ["short", "medium", "long"]},
                "slope": {"type": "string", "enum": ["avoid", "normal", "prefer"]},
                "avoid_steps": {"type": "boolean", "description": "用户明确要求不走台阶时设为 true；仅过滤已标注台阶的路段，未知通行属性不能视为已核实"},
                "scenery": {"type": "string", "enum": ["high", "normal"]},
            },
    },
    "weights": {
        "type": "object",
        "description": "仅当用户明确给出数值比例时填写。普通通勤不填，由策略层执行纯最短路径 1.00/0.00/0.00",
        "properties": {
            "distance": {"type": "number"},
            "slope": {"type": "number"},
            "scenery": {"type": "number"},
        },
    },
    "mode": {"type": "string", "enum": ["walk", "bike", "drive"],
             "description": "出行方式，默认 walk；用户明确说骑车/开车时才改"},
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "resolve_poi",
            "description": "按名称解析校内地点。用户提到的地点名不明确或你想确认它是否存在时调用。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "地点名称或别名"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_place_details",
            "description": "返回校园主数据中地点的名称、别名、类别、位置、来源与核实状态。开放时间、入口和无障碍信息没有登记时明确返回 unknown；查不到只表示当前主数据未匹配，不代表现实中不存在。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "地点正式名称或别名"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_poi_candidates",
            "description": "按类别/关键词检索校内地点候选。推荐附近地点时必须传用户本轮起点；工具会先检索完整类别，再按当前可用路网距离排序，估算距离包含起点和地点到路网节点的直线接驳，两端都未经现场核实。没有起点时应先询问；列举目录时可以不传起点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {**_ENDPOINT_SCHEMA, "description": "附近推荐的本轮起点，可为坐标或校内地点；不可用旧路线终点代替"},
                    "subcategory": {
                        "type": "string",
                        "description": "细分类别：canteen 食堂 / restaurant 餐厅 / fastfood 快餐小吃 / "
                                       "coffee 咖啡 / tea_drink 奶茶饮品 / supermarket 超市 / "
                                       "library 图书馆 / classroom 教学楼 / gym 体育馆 / field 运动场 / "
                                       "pool 游泳池 / sakura 赏樱 / lake 湖泊 / hill 山景 / park 公园广场 / "
                                       "landmark 地标 / bank 银行 / hospital 医院 / post 邮政 等",
                    },
                    "subcategories": {
                        "type": "array", "items": {"type": "string"},
                        "description": "一个模糊需求覆盖的多个细分类；例如‘美食’传 canteen、restaurant、fastfood",
                    },
                    "activities": {
                        "type": "array", "items": {"type": "string", "enum": sorted(ACTIVITY_LABELS)},
                        "description": "按地点支持的用途精确筛选；例如 basketball、badminton、running、meal",
                    },
                    "poi_type": {"type": "string", "enum": ["dining", "study", "sports", "dorm", "gate",
                                                            "scenery", "service", "area"]},
                    "keyword": {"type": "string", "description": "名称关键词模糊匹配"},
                    "season": {"type": "string", "enum": ["spring", "summer", "autumn", "winter"]},
                    "near_poi": {"type": "string", "description": "兼容旧调用的起点 POI 名；推荐优先传 start"},
                    "mode": {"type": "string", "enum": ["walk", "bike", "drive"], "default": "walk"},
                    "include_minor": {"type": "boolean",
                                      "description": "是否包含小店铺（连锁奶茶/咖啡档口），默认 true"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20,
                               "description": "附近推荐默认 4 个，目录检索默认 10 个"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_reachable_places",
            "description": "按已标记可通行路网搜索附近地点候选；地点到路网的最后一段是直线接驳估算、未经通行核实。用户问‘步行十分钟内有什么食堂/附近能去哪’时调用，不能把候选宣称为入口已确认可达。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": _ENDPOINT_SCHEMA,
                    "subcategory": {"type": "string", "description": "地点细分类，如 canteen、coffee、gym、library"},
                    "activities": {
                        "type": "array", "items": {"type": "string", "enum": sorted(ACTIVITY_LABELS)},
                        "description": "按地点支持的用途精确筛选，如 basketball、running、meal",
                    },
                    "poi_type": {"type": "string", "enum": ["dining", "study", "sports", "dorm", "gate", "scenery", "service", "area"]},
                    "keyword": {"type": "string", "description": "地点名称关键词"},
                    "mode": {"type": "string", "enum": ["walk", "bike", "drive"], "default": "walk"},
                    "max_distance_m": {"type": "number", "description": "最大路网距离（米）；与 max_time_min 二选一"},
                    "max_time_min": {"type": "number", "description": "最大估算路网时间（分钟）；与 max_distance_m 二选一，默认 10 分钟"},
                    "include_minor": {"type": "boolean", "default": True},
                    "constraints": {"type": "object", "properties": {
                        "slope": {"type": "string", "enum": ["avoid", "normal"]},
                        "avoid_steps": {"type": "boolean", "description": "用户明确要求不走台阶时设为 true；只过滤已标记台阶的边"},
                    }},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
                },
                "required": ["start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_routes",
            "description": "在相同起终点、出行方式、硬约束和路况快照下，对比最短、风景优先、平坦优先三种路线。仅在用户明确要求比较路线或选择策略时调用；返回可核对的距离、估算时间、坡度与景观指标，不替用户切换当前路线。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": _ENDPOINT_SCHEMA,
                    "end": _ENDPOINT_SCHEMA,
                    "mode": {"type": "string", "enum": ["walk", "bike", "drive"], "default": "walk"},
                    "constraints": {
                        "type": "object",
                        "properties": {
                            "slope": {"type": "string", "enum": ["avoid", "normal", "prefer"]},
                            "avoid_steps": {"type": "boolean", "description": "用户明确要求不走台阶时设为 true；只过滤已标记台阶的边"},
                        },
                    },
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_route",
            "description": "规划起点到终点的路径（多因素优化：距离/坡度/景观）。起点终点都明确时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": _ENDPOINT_SCHEMA,
                    "end": _ENDPOINT_SCHEMA,
                    **_PREFERENCE_SCHEMA,
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_via_route",
            "description": "规划带途经点的路径：如'去上课路上顺便买个笔记本'。via 可以指名具体地点，"
                         "也可以只给类别（自动从该类别里挑最顺路的），也可以用坐标设途经点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": _ENDPOINT_SCHEMA,
                    "end": _ENDPOINT_SCHEMA,
                    "via_name": {"type": "string", "description": "途经地点名（与 via_subcategory/via_coord 三选一）"},
                    "via_subcategory": {"type": "string",
                                        "description": "途经类别（如 supermarket/coffee/canteen），"
                                                       "自动选顺路的，与 via_name 二选一"},
                    "via_activity": {"type": "string", "enum": sorted(ACTIVITY_LABELS),
                                     "description": "按用途选一个顺路地点，例如 basketball、coffee、meal"},
                    "via_coord": {"type": "object",
                                  "description": "途经点坐标（WGS-84），与 via_name/via_subcategory 三选一",
                                  "properties": {
                                      "lng": {"type": "number", "description": "经度 WGS-84"},
                                      "lat": {"type": "number", "description": "纬度 WGS-84"},
                                  },
                                  "required": ["lng", "lat"]},
                    "via_points": {
                        "type": "array",
                        "description": "按用户给出的先后顺序必须经过的多个途经点；每项为地点名或 WGS-84 坐标。最多 10 个。与 via_name/via_subcategory/via_coord 单点参数互斥。",
                        "minItems": 1,
                        "maxItems": _MAX_VIA_POINTS,
                        "items": _ENDPOINT_SCHEMA,
                    },
                    **_PREFERENCE_SCHEMA,
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_itinerary",
            "description": "按用户给出的地点顺序规划有时间预算的行程。计算路网步行时间估算与每站停留时间；超预算时只返回差额和可行性，不把路线作为已满足要求的结果提交。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": _ENDPOINT_SCHEMA,
                    "stops": {"type": "array", "minItems": 1, "maxItems": _MAX_VIA_POINTS,
                              "items": _ENDPOINT_SCHEMA,
                              "description": "按用户指定顺序访问的地点"},
                    "end": {**_ENDPOINT_SCHEMA, "description": "结束地点；不填时默认回到起点"},
                    "time_budget_min": {"type": "number", "exclusiveMinimum": 0,
                                        "maximum": 1440, "description": "可用总时间，分钟"},
                    "stop_duration_min": {"type": "number", "minimum": 0, "maximum": 240,
                                          "default": 15, "description": "每个地点的停留估算分钟数，默认 15"},
                    "mode": {"type": "string", "enum": ["walk", "bike", "drive"], "default": "walk"},
                    "constraints": {"type": "object", "properties": {
                        "slope": {"type": "string", "enum": ["avoid", "normal", "prefer"]},
                        "avoid_steps": {"type": "boolean"},
                    }},
                    "weights": _PREFERENCE_SCHEMA["weights"],
                },
                "required": ["start", "stops", "time_budget_min"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_tour",
            "description": "规划多点游览路线：如'游客想逛遍全校''推荐一条赏樱路线'。"
                         "未指定地点时自动选点；用户点名地点时将其作为必经站，生成环线或开放路线。",
            "parameters": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string",
                              "description": "游览主题：sakura 赏樱 / scenery 全校风景（默认）/ "
                                             "lake 湖畔 / landmark 地标建筑"},
                    "season": {"type": "string", "enum": ["spring", "summer", "autumn", "winter"],
                               "description": "按季节标签筛点，如春天赏樱"},
                    "start": _ENDPOINT_SCHEMA,
                    "loop": {"type": "boolean", "description": "true=回到起点环线（默认），false=开放路线"},
                    "max_pois": {"type": "integer", "description": "游览点数量上限 2~8，默认 6"},
                    "poi_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 8,
                        "description": "用户明确指定的必经游览地点，全部纳入路线；不得用其他景点替代。",
                    },
                    **_PREFERENCE_SCHEMA,
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_multimodal_route",
            "description": "规划分段换乘路径：如'先骑车到樱花大道再步行到珞珈山'"
                         "'骑车到校门再走进去'这类一次出行含多种出行方式的需求。"
                         "每段单独给 mode 和终点；第 N 段终点自动作为第 N+1 段起点。"
                         "不用于普通单方式路径（用 plan_route 即可）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": _ENDPOINT_SCHEMA,
                    "legs": {
                        "type": "array",
                        "description": "换乘段列表（2~4 段）。每段独立规划，按顺序拼接",
                        "items": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string",
                                         "enum": ["walk", "bike", "drive"],
                                         "description": "本段出行方式"},
                                "end": _ENDPOINT_SCHEMA,
                                "via_name": {"type": "string",
                                             "description": "本段途经点（可选）；"
                                                            "不填则直接走终点"},
                            },
                            "required": ["mode", "end"],
                        },
                        "minItems": 2,
                        "maxItems": 4,
                    },
                    **{k: v for k, v in _PREFERENCE_SCHEMA.items()
                       if k != "mode"},
                },
                "required": ["start", "legs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询武汉当前实时天气。用户问天气/要不要带伞/适不适合出门时必须调用，不允许凭印象回答。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_road_conditions",
            "description": "查询当前生效的封路/施工/积水/事故/活动管制等路况事件（绑定具体路段，步行也会绕封路）。用户问'哪里在修路''哪段封了''能不能走'时必须调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "信息不足或存在歧义时向用户提问澄清（如起点缺失、地点有多个候选）。调用后本轮对话结束。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "给用户的问题，口语化、简短"},
                    "options": {"type": "array", "items": {"type": "string"},
                                "description": "可选回答，最多 4 个"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_followup",
            "description": "向用户推荐 2~3 个后续可能的问法。在给出最终答复前调用，让用户能一键继续。",
            "parameters": {
                "type": "object",
                "properties": {
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "object",
                                  "properties": {
                                      "label": {"type": "string", "description": "按钮显示文字（≤10 字）"},
                                      "query": {"type": "string", "description": "点击后发送的自然语言 query"},
                                  },
                                  "required": ["label", "query"]},
                        "description": "2~3 个跟进建议",
                    },
                },
                "required": ["suggestions"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}

# 终止型工具：调用即结束本轮循环
TERMINAL_TOOLS = {"ask_user"}


# ===========================================================================
# 内部辅助（确定性计算）
# ===========================================================================

def _ensure_graph():
    G = get_network()
    if G is not None:
        return G
    return load_or_download_network()


def _parse_linestring(wkt):
    if not wkt or not isinstance(wkt, str):
        return None
    m = re.match(r"LINESTRING\s*\((.*)\)", wkt.strip(), re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    pts = []
    for pair in m.group(1).split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pts if pts else None


def _path_to_coords(G, route_nodes, route_edges=None):
    """路径节点序列 → 密集坐标序列（WGS-84，含边 geometry 中间点）。"""
    if not route_nodes:
        return []
    coords = []
    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        edge_data = G.get_edge_data(u, v)
        if not edge_data:
            continue
        data = (edge_data[route_edges[i][2]] if route_edges is not None
                else min(edge_data.values(), key=lambda d: d.get("length", float("inf"))))
        if not coords:
            ulng, ulat = get_node_coords(G, u)
            coords.append({"lng": round(ulng, 6), "lat": round(ulat, 6)})
        pts = _parse_linestring(data.get("geometry")) if data.get("geometry") else None
        if pts and len(pts) >= 2:
            ulng, ulat = get_node_coords(G, u)
            if (pts[0][0] - ulng) ** 2 + (pts[0][1] - ulat) ** 2 > (pts[-1][0] - ulng) ** 2 + (pts[-1][1] - ulat) ** 2:
                pts.reverse()
            for lng, lat in pts[1:]:
                coords.append({"lng": round(lng, 6), "lat": round(lat, 6)})
        else:
            vlng, vlat = get_node_coords(G, v)
            coords.append({"lng": round(vlng, 6), "lat": round(vlat, 6)})
    return coords


def _coords_wgs_to_gcj(coords_list):
    if not coords_list:
        return coords_list
    return [{"lng": round(gcj_lng, 6), "lat": round(gcj_lat, 6)}
            for c in coords_list
            for gcj_lng, gcj_lat in [wgs84_to_gcj02(c["lng"], c["lat"])]]


def _build_steps_gcj(G, route_nodes, mode, end_name="", route_edges=None):
    """逐步转向指令（动作点转 GCJ-02）。失败返回 []，不阻断路径规划。"""
    if not route_nodes or len(route_nodes) < 2:
        return []
    try:
        steps = build_turn_by_turn(
            G, route_nodes, mode=mode, end_name=end_name or "", route_edges=route_edges,
        )
    except Exception:
        logger.warning("转向指令生成失败 mode=%s", mode, exc_info=True)
        return []
    for s in steps:
        p = s.get("point")
        if p:
            g_lng, g_lat = wgs84_to_gcj02(p["lng"], p["lat"])
            s["point"] = {"lng": round(g_lng, 6), "lat": round(g_lat, 6)}
    return steps


def _merge_leg_steps(legs):
    """合并多段（途经/环线）指令：中间到达点转为"途经点"提示，去掉后续段的出发指令，
    累计距离整体平移并重排 seq。"""
    merged = []
    cum_offset = 0.0
    valid = [leg.get("steps") or [] for leg in legs]
    valid = [(i, st) for i, st in enumerate(valid) if st]
    for idx, (leg_i, steps) in enumerate(valid):
        is_last = idx == len(valid) - 1
        for s in steps:
            t = s.get("type")
            if idx > 0 and t == "depart":
                continue
            ns = dict(s)
            if t == "arrive" and not is_last:
                ns["type"] = "via"
                ns["action"] = "via"
                via_name = legs[leg_i].get("end_name") or "途经点"
                ns["text"] = f"到达途经点（{via_name}），继续前行"
            ns["cumulative_m"] = round(cum_offset + float(s.get("cumulative_m", 0) or 0), 1)
            merged.append(ns)
        # 下一段的累计距离偏移 = 本段最后一条累计值（本段总长）
        cum_offset += float(steps[-1].get("cumulative_m", 0) or 0)
    for i, s in enumerate(merged):
        s["seq"] = i
    return merged


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _poi_public(poi: dict, with_coords: bool = True, with_provenance: bool = False) -> dict:
    """给 LLM/前端看的精简 POI 结构（坐标保持 GCJ-02，与前端一致）。"""
    subcategory = poi.get("subcategory", "")
    supported_activities = sorted(activities_for_poi(poi))
    out = {
        "name": poi.get("name", ""),
        "type": poi.get("type", ""),
        "subcategory": subcategory,
        "subcategory_label": _SUBCATEGORY_LABELS.get(subcategory, subcategory),
        "campus": poi.get("campus") or "",
        "activities": supported_activities,
        "activity_labels": [ACTIVITY_LABELS[value] for value in supported_activities],
        "description": (poi.get("description") or "")[:120],
        "scenery_score": poi.get("scenery_score"),
    }
    if with_coords:
        out["lat"] = poi.get("lat")
        out["lng"] = poi.get("lon") or poi.get("lng")
    if with_provenance:
        out.update({
            "poi_id": poi.get("id"),
            "source_refs": list(poi.get("source_refs") or [])[:5],
            "verification_status": poi.get("verification_status") or "legacy_unverified",
            "coordinate_verification_status": poi.get("coordinate_verification_status") or "unknown",
            "opening_hours": poi.get("opening_hours") or "unknown",
        })
    return out


_SUBCATEGORY_LABELS = {
    "canteen": "食堂", "restaurant": "餐厅", "fastfood": "快餐/小吃",
    "coffee": "咖啡", "tea_drink": "茶饮", "supermarket": "超市",
    "library": "图书馆", "classroom": "教学楼", "college": "院系楼",
    "laboratory": "实验室", "building": "建筑",
    "study_other": "学习场所", "field": "运动场", "sports_field": "运动场地",
    "gym": "体育馆", "court": "球场", "pool": "游泳场馆",
    "sports_centre": "综合体育设施", "sports_other": "其他运动设施",
    "sakura": "赏樱点", "lake": "湖景", "hill": "山景", "park": "公园",
    "square": "广场", "landmark": "地标", "gate": "校门", "dormitory": "宿舍",
    "hospital": "医院", "bank": "银行", "post": "邮政", "activity_center": "活动中心",
    "area": "区域", "museum": "展馆", "culture": "文化场所", "service": "服务点",
    "pavilion": "亭阁",
}


def infer_recommendation_subcategories(query: str) -> list[str] | None:
    """把模糊地点需求映射到校园主数据已有类别；无法可靠判断时返回 None。"""
    controlled = resolve_activity_query(query)
    if controlled:
        if controlled.get("status") != "resolved":
            return None
        return list(controlled.get("subcategories") or []) or None
    text = re.sub(r"\s+", "", str(query or "")).lower()
    if not text:
        return None

    broad_food = r"美食|吃饭|吃点|吃什么|吃东西|吃的|觅食|饿了|早点|午饭|午餐|晚饭|晚餐|早餐|早饭|夜宵|餐饮"
    if "食堂" in text and not re.search(broad_food, text):
        return ["canteen"]
    if "餐厅" in text and not re.search(broad_food, text):
        return ["restaurant"]
    if "快餐" in text and not re.search(broad_food, text):
        return ["fastfood"]
    if re.search(broad_food + r"|好吃|味道好", text):
        return ["canteen", "restaurant", "fastfood"]
    if any(term in text for term in ("快餐", "小吃", "炸鸡", "汉堡")):
        return ["fastfood"]
    if any(term in text for term in ("餐厅", "饭店", "聚餐")):
        return ["restaurant"]

    asks_coffee = any(term in text for term in ("咖啡", "coffee", "拿铁", "美式"))
    asks_tea = any(term in text for term in ("奶茶", "茶饮", "果茶", "果汁", "柠檬茶", "喝饮料"))
    if asks_coffee and asks_tea:
        return ["coffee", "tea_drink"]
    if asks_coffee:
        return ["coffee"]
    if asks_tea:
        return ["tea_drink"]
    if any(term in text for term in ("喝点东西", "喝点啥", "喝的", "饮品", "饮品店", "饮料", "饮料店", "来杯")):
        return ["coffee", "tea_drink"]

    if any(term in text for term in ("游泳", "游个泳")):
        return ["pool"]
    if any(term in text for term in ("打球", "球场")):
        return ["court"]
    if any(term in text for term in ("跑步", "操场")):
        return ["field", "sports_field"]
    if any(term in text for term in ("运动", "锻炼", "健身")):
        return ["field", "sports_field", "court", "gym", "pool", "sports_centre", "sports_other"]

    if "图书馆" in text:
        return ["library"]
    if any(term in text for term in (
            "自习", "学习的地方", "找地方学习", "学习场所", "想学习",
            "看书", "复习", "备考", "写作业", "安静的地方")):
        return ["library", "study_other"]
    if any(term in text for term in (
            "超市", "便利店", "买东西", "零食", "文具", "购物")):
        return ["supermarket"]
    if any(term in text for term in ("赏樱", "看樱花", "樱花")):
        return ["sakura"]
    if any(term in text for term in ("看湖", "湖边", "东湖边", "看湖景")):
        return ["lake"]
    return None


def is_poi_catalog_query(query: str) -> bool:
    """区分无位置要求的目录问题与需要按当前位置排序的推荐。"""
    text = str(query or "")
    if re.search(r"附近|周边|最近|离我|从.{1,20}(?:找|推荐|去哪)|按距离|推薦|推荐|帮我挑", text):
        return False
    return bool(re.search(r"有哪些|都有哪些|列出|列一下|名单|清单|全部|所有|一共有多少|有多少", text))


def recommendation_origin_from_text(query: str) -> str | None:
    """提取用户本轮明确说出的推荐起点，不从上一条路线推断。"""
    text = str(query or "").strip()
    patterns = (
        r"我(?:现在)?在\s*([^，,。；;]+)",
        r"(?:从|由)\s*([^，,。；;]+)",
        r"起点(?:设为|是|为|在|[:：])\s*([^，,。；;]+)",
    )
    stop_words = r"(?:附近|周边|旁边|周围|一带|这边|出发|开始|想|准备|打算|帮我|请|推荐|找|吃|喝|运动|锻炼|学习|逛|游玩|去|到).*$"
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        name = re.sub(stop_words, "", match.group(1)).strip()
        if name and name not in {"我这", "我这里", "我的位置", "当前位置", "这里", "这儿"}:
            return name
    return None


def _query_has_explicit_poi_destination(query: str) -> bool:
    text = re.sub(r"\s+", "", str(query or ""))
    if not text:
        return False
    target = re.search(r"(?:导航到|带我(?:去|到)|前往|去|到|找|定位到)(.+)$", text)
    if not target:
        return False
    tail = target.group(1)
    generic = {"食堂", "餐厅", "饭店", "咖啡", "咖啡店", "奶茶", "茶饮", "超市",
               "图书馆", "自习室", "操场", "体育馆", "球场", "篮球场", "篮球馆",
               "羽毛球场", "乒乓球馆", "足球场", "排球场", "网球场", "游泳馆", "学习的地方"}
    for poi in load_pois():
        name = re.sub(r"\s+", "", str(poi.get("name") or ""))
        if len(name) >= 3 and name not in generic and name in tail:
            return True
        for alias in poi.get("aliases") or []:
            alias = re.sub(r"\s+", "", str(alias or ""))
            if len(alias) >= 3 and alias not in generic and alias in tail:
                return True
    return False


def query_requests_route(query: str) -> bool:
    """True when the user explicitly wants directions rather than a place shortlist."""
    text = str(query or "")
    return bool(re.search(
        r"路线|导航|怎么走|帶我(?:去|到)|带我(?:去|到)|送我(?:去|到)|"
        r"从.{1,30}(?:到|前往)|由.{1,30}(?:到|前往)|规划.{0,8}(?:路线|怎么走)", text
    ))


def _search_poi_categories(subcategories: list[str] | None, poi_type: str | None,
                           keyword: str, season: str | None,
                           include_minor: bool, activities: list[str] | None = None) -> list[dict]:
    """先完整检索所有目标类别，再做跨类别去重；不受热度截断影响。"""
    categories = list(dict.fromkeys(str(item) for item in (subcategories or []) if item))
    if categories:
        records = []
        for category in categories:
            found = (search_pois(keyword, poi_type=poi_type, season=season,
                                 subcategory=category, include_minor=include_minor, limit=2000)
                     if keyword else
                     search_by_category(subcategory=category, poi_type=poi_type,
                                        season=season, include_minor=include_minor, limit=2000))
            records.extend(found)
    elif keyword:
        records = search_pois(keyword, poi_type=poi_type, season=season,
                              include_minor=include_minor, limit=2000)
    else:
        records = search_by_category(poi_type=poi_type, season=season,
                                     include_minor=include_minor, limit=2000)

    if activities:
        records = [poi for poi in records if matches_activities(poi, activities)]

    unique = []
    seen_ids = set()
    for poi in records:
        poi_id = str(poi.get("id") or "")
        normalized_name = re.sub(r"[\s\u3000\-—_()（）·]+", "", str(poi.get("name") or "")).casefold()
        if poi_id and poi_id in seen_ids:
            continue
        if poi_id:
            seen_ids.add(poi_id)
        coords = poi.get("coordinates") or {}
        try:
            lng = float(coords.get("lng", poi.get("lon", poi.get("lng"))))
            lat = float(coords.get("lat", poi.get("lat")))
            has_coords = math.isfinite(lng) and math.isfinite(lat)
        except (TypeError, ValueError):
            has_coords = False
            lng = lat = 0.0
        tokens = {
            re.sub(r"[\s\u3000\-—_()（）·]+", "", str(value or "")).casefold()
            for value in [poi.get("name"), *(poi.get("aliases") or [])]
        }
        duplicate = False
        for existing in unique:
            if (poi.get("type") != existing.get("type")
                    or poi.get("subcategory") != existing.get("subcategory")):
                continue
            old_coords = existing.get("coordinates") or {}
            try:
                old_lng = float(old_coords.get("lng", existing.get("lon", existing.get("lng"))))
                old_lat = float(old_coords.get("lat", existing.get("lat")))
                old_has_coords = math.isfinite(old_lng) and math.isfinite(old_lat)
            except (TypeError, ValueError):
                old_has_coords = False
                old_lng = old_lat = 0.0
            old_tokens = {
                re.sub(r"[\s\u3000\-—_()（）·]+", "", str(value or "")).casefold()
                for value in [existing.get("name"), *(existing.get("aliases") or [])]
            }
            common_tokens = {token for token in tokens & old_tokens if len(token) >= 3}
            if normalized_name and normalized_name == re.sub(
                    r"[\s\u3000\-—_()（）·]+", "", str(existing.get("name") or "")).casefold():
                duplicate = not (has_coords and old_has_coords) or _haversine(lat, lng, old_lat, old_lng) <= 15
            elif common_tokens and has_coords and old_has_coords:
                duplicate = _haversine(lat, lng, old_lat, old_lng) <= 15
            if duplicate:
                break
        if duplicate:
            continue
        unique.append(poi)
    return unique


def _endpoint_target_wgs(ref: dict, mode: str):
    """解析地点/坐标的 WGS-84 目标坐标，不在此处执行路网吸附。"""
    if not isinstance(ref, dict):
        return None, None, None, {"error": "invalid_endpoint", "message": "起终点格式不正确"}
    name = (ref.get("name") or "").strip()
    if ref.get("type") == "coord":
        coord = ref.get("coordinates") or {}
        lng = coord.get("lng") if coord else None
        lat = coord.get("lat") if coord else None
        # 兼容 LLM 直接在顶层放 lng/lat 的情况
        if lng is None:
            lng = ref.get("lng")
        if lat is None:
            lat = ref.get("lat")
        if lng is not None and lat is not None:
            try:
                lng, lat = float(lng), float(lat)
            except (TypeError, ValueError):
                return None, None, None, {"error": "invalid_coord", "message": "坐标无法识别"}
            if not (math.isfinite(lng) and math.isfinite(lat)
                    and -180 <= lng <= 180 and -90 <= lat <= 90):
                return None, None, None, {"error": "invalid_coord", "message": "坐标超出有效范围"}
            return lng, lat, name or "我的位置", None
        return None, None, None, {"error": "invalid_coord", "message": "缺少有效坐标"}
    selected_id = str(ref.get("poi_id") or "").strip()
    if selected_id:
        poi = next((item for item in load_pois() if item.get("id") == selected_id), None)
        if poi is None:
            return None, None, None, {"error": "poi_not_found", "message": "候选地点已不在当前主数据中"}
        lng_wgs, lat_wgs = navigation_wgs(poi, mode)
        return lng_wgs, lat_wgs, poi.get("name") or name or selected_id, None
    if not name:
        return None, None, None, {"error": "missing_name", "message": "地点名为空"}
    poi, alts = find_poi_ambiguous(name)
    if poi is None:
        return None, None, None, {
            "error": "poi_not_found",
            "message": f"校内没找到「{name}」",
            "candidates": [_poi_public(a, with_coords=False) for a in (alts or [])][:5],
        }
    lng_wgs, lat_wgs = navigation_wgs(poi, mode)
    return lng_wgs, lat_wgs, poi["name"], None


def _endpoint_access_evidence(ref: dict, node: int, G_mode, mode: str) -> dict:
    """Report the unverified straight connector from the requested point to graph node."""
    lng_wgs, lat_wgs, display_name, error = _endpoint_target_wgs(ref, mode)
    if error:
        return {"status": "unavailable", "message": error.get("message", "端点无法核验")}
    return _snap_evidence_for_wgs(lng_wgs, lat_wgs, display_name, node, G_mode)


def _snap_evidence_for_wgs(lng_wgs, lat_wgs, display_name, node, G_mode) -> dict:
    node_lng, node_lat = get_node_coords(G_mode, node)
    distance = _haversine(lat_wgs, lng_wgs, node_lat, node_lng)
    return {
        "name": display_name,
        "snap_distance_m": round(distance, 1),
        "status": "unverified_long_connector" if distance > _REACHABILITY_MAX_ACCESS_SNAP_M
                 else "unverified_nearby_network_node",
        "access_link_verified": False,
        "access_link_basis": "straight_line_to_nearest_routable_network_node",
        "note": "起终点到路网节点之间按直线距离吸附，未核实末端是否存在实际通行连接。",
    }


def _snap_wgs_to_network(lng_wgs, lat_wgs, G_mode, display_name):
    """Snap an already-resolved WGS-84 coordinate with the same cap/evidence as endpoints."""
    try:
        lng_wgs, lat_wgs = float(lng_wgs), float(lat_wgs)
        if not (math.isfinite(lng_wgs) and math.isfinite(lat_wgs)
                and -180 <= lng_wgs <= 180 and -90 <= lat_wgs <= 90):
            raise ValueError("坐标超出有效范围")
        node = get_nearest_node(G_mode, lng_wgs, lat_wgs)
        evidence = _snap_evidence_for_wgs(lng_wgs, lat_wgs, display_name, node, G_mode)
    except (RuntimeError, KeyError, TypeError, ValueError) as exc:
        return None, None, {"error": "nearest_node_failed", "message": f"最近路网节点查找失败: {exc}"}
    if evidence["snap_distance_m"] > _MAX_ENDPOINT_ACCESS_SNAP_M:
        return None, evidence, {
            "error": "endpoint_too_far_from_network",
            "message": f"「{display_name}」距可规划道路约 {round(evidence['snap_distance_m'])} 米，末端接驳无法可靠确认。请在地图上选择附近道路或校门。",
            "snap_distance_m": evidence["snap_distance_m"],
            "max_access_snap_m": _MAX_ENDPOINT_ACCESS_SNAP_M,
        }
    return node, evidence, None


def _resolve_endpoint(ref: dict, G_mode):
    """把 {name} 或坐标解析到路网节点，并限制未核实的末端接驳长度。

    Returns: (node_id, display_name, error_dict_or_None)
    """
    mode = G_mode.graph.get("travel_mode", "walk")
    lng_wgs, lat_wgs, display_name, error = _endpoint_target_wgs(ref, mode)
    if error:
        return None, None, error
    node, _, error = _snap_wgs_to_network(lng_wgs, lat_wgs, G_mode, display_name)
    if error:
        return None, None, error
    return node, display_name, None


def _pois_along_route(G, route_nodes):
    """沿途重要 POI（返回 GCJ-02，与前端一致）。"""
    route_coords = []
    for nid in route_nodes:
        try:
            lng, lat = get_node_coords(G, nid)
            route_coords.append((lng, lat))
        except Exception:
            continue
    if not route_coords:
        return []
    along = []
    for poi in load_pois():
        coords = poi.get("coordinates", {})
        poi_lng_wgs, poi_lat_wgs = gcj02_to_wgs84(coords.get("lng", 0), coords.get("lat", 0))
        min_dist = min((_haversine(poi_lat_wgs, poi_lng_wgs, rlat, rlng)
                        for rlng, rlat in route_coords), default=float("inf"))
        if min_dist <= _ALONG_ROUTE_THRESHOLD_M:
            imp = importance_score(poi)
            if imp < _ALONG_ROUTE_MIN_IMPORTANCE:
                continue
            flat = _flatten_poi(poi)
            flat["distance_to_route_m"] = round(min_dist, 1)
            flat["importance"] = round(imp, 2)
            along.append(flat)
    along.sort(key=lambda p: (-p["importance"], p["distance_to_route_m"]))
    return along[:_ALONG_ROUTE_LIMIT]


def _weather_snapshot():
    try:
        live = weather_mod.fetch_weather_live()
        if not live:
            return None
        return {"live": live, "impact": weather_mod.classify_weather(live)}
    except Exception as e:
        logger.warning("天气快照获取失败: %s", e)
        return None


def _weather_public(snap):
    if not snap:
        return None
    live, impact = snap["live"], snap["impact"]
    return {
        "weather": live.get("weather", ""),
        "temperature": live.get("temperature"),
        "windpower": live.get("windpower", ""),
        "slippery": impact["slippery"],
        "hot": impact["hot"],
        "low_visibility": impact["low_visibility"],
        "label": impact["label"],
        "advice": impact["advice"],
    }


def _route_payload(G, route_result, start_name, end_name, mode):
    """compute_route 结果 → 前端/LLM 可用的路径包（坐标 GCJ-02）。"""
    response_started = time.perf_counter()
    recommended_nodes = route_result["recommended"]
    shortest_nodes = route_result["shortest"]
    recommended_edges = route_result["recommended_edges"]
    shortest_edges = route_result["shortest_edges"]
    payload = {
        "start_name": start_name,
        "end_name": end_name,
        "recommended": _coords_wgs_to_gcj(
            _path_to_coords(G, recommended_nodes, recommended_edges)),
        "shortest": _coords_wgs_to_gcj(
            _path_to_coords(G, shortest_nodes, shortest_edges)),
        "recommended_edge_ids": recommended_edges,
        "shortest_edge_ids": shortest_edges,
        "steps": _build_steps_gcj(
            G, recommended_nodes, mode, end_name=end_name, route_edges=recommended_edges),
        "pois": _pois_along_route(G, recommended_nodes),
        "filter_status": route_result["filter_status"],
        "overlap_rate": route_result["overlap_rate"],
        "recommended_length_m": route_result["recommended_length_m"],
        "shortest_length_m": route_result["shortest_length_m"],
        "distance_m": route_result["recommended_length_m"],
        "duration_min": route_result["duration_min"],
        "shortest_duration_min": route_result["shortest_duration_min"],
        "applied_weights": route_result.get("applied_weights"),
        "slope_avg_recommended": route_result.get("slope_avg_recommended"),
        "scenery_avg_recommended": route_result.get("scenery_avg_recommended"),
        "degraded": route_result["degraded"],
        "length_capped": route_result["length_capped"],
        "mode": route_result["mode"],
        "speed_kmh": route_result.get("speed_kmh", MODE_SPEEDS_KMH.get(mode, 4.5)),
    }
    timings = normalize_timings(route_result.get("timings_ms"))
    timings["response_build"] = round(
        timings["response_build"] + (time.perf_counter() - response_started) * 1000,
        3,
    )
    payload["timings_ms"] = normalize_timings(timings)
    return payload


# ===========================================================================
# 工具执行器
# ===========================================================================

def _tool_resolve_poi(args, ctx):
    poi, alts = find_poi_ambiguous(args.get("name", ""))
    if poi:
        return {"found": True, "poi": _poi_public(poi)}, None
    return {
        "found": False,
        "message": f"校内没找到「{args.get('name', '')}」",
        "candidates": [_poi_public(a, with_coords=False) for a in (alts or [])][:5],
    }, None


def _tool_search_poi_candidates(args, ctx):
    context = dict(ctx) if isinstance(ctx, dict) else {}
    query = str(context.get("query") or "")
    is_catalog = is_poi_catalog_query(query)
    default_limit = 4 if args.get("start") or args.get("near_poi") else 20 if is_catalog else 10
    limit = max(1, min(int(args.get("limit", default_limit) or default_limit), 20))
    include_minor = bool(args.get("include_minor", True))
    raw_subcategories = args.get("subcategories")
    subcategories = ([str(item) for item in raw_subcategories if item]
                     if isinstance(raw_subcategories, list) else [])
    subcategory = args.get("subcategory")
    if not subcategories and subcategory:
        subcategories = [subcategory]
    activity_request = resolve_activity_query(query)
    activities = [str(value) for value in args.get("activities", []) if value] \
        if isinstance(args.get("activities"), list) else []
    if activity_request and activity_request.get("status") == "needs_clarification" and not activities:
        question = activity_request["question"]
        return {"question": question, "options": activity_request["options"]}, {
            "clarify": {"question": question, "options": activity_request["options"]}
        }
    if activity_request and activity_request.get("status") == "resolved" and not is_catalog:
        subcategories = list(activity_request.get("subcategories") or subcategories)
        activities = list(activity_request.get("activities") or activities)
    elif activities and not subcategories:
        subcategories = list(dict.fromkeys(
            category for activity in activities
            for category in ACTIVITY_SUBCATEGORIES.get(activity, ())
        ))
    inferred = infer_recommendation_subcategories(query)
    if inferred and not is_catalog:
        subcategories = inferred
    poi_type = args.get("poi_type")
    keyword = (args.get("keyword") or "").strip()
    season = args.get("season")

    start_ref = args.get("start")
    near_name = (args.get("near_poi") or "").strip()
    if near_name and not start_ref:
        near_poi, alternatives = find_poi_ambiguous(near_name)
        if near_poi is None:
            return {"error": "poi_not_found", "message": f"无法确认附近起点「{near_name}」",
                    "alternatives": [_poi_public(item, with_coords=False) for item in alternatives]}, None
        start_ref = {"name": near_poi["name"], "type": "poi"}

    needs_origin = bool(subcategories or activities or inferred) and not is_catalog
    if (needs_origin and not start_ref and not near_name
            and not _query_has_explicit_poi_destination(query)):
        question = "你从哪里出发？可以在地图上点起点，或告诉我所在的地点，我再按路网距离推荐。"
        return {"question": question, "options": ["地图上选起点", "告诉我所在地点"]}, {
            "clarify": {"question": question, "options": ["地图上选起点", "告诉我所在地点"]}
        }

    if start_ref:
        reachable_args = {
            "start": start_ref,
            "subcategories": subcategories or None,
            "activities": activities or None,
            "subcategory": subcategory if not subcategories else None,
            "poi_type": poi_type,
            "keyword": keyword,
            "season": season,
            "include_minor": include_minor,
            "mode": args.get("mode") or "walk",
            "max_distance_m": args.get("max_distance_m") or 10000,
            "limit": limit,
        }
        result, artifact = _tool_find_reachable_places(reachable_args, context)
        if result.get("error"):
            return result, None
        candidates = result.get("candidates") or []
        for candidate in candidates:
            candidate["distance_m"] = candidate.get("network_distance_m")
            candidate["distance_basis"] = result.get("reachability_basis")
        result["candidate_count"] = len(candidates)
        result["count"] = len(candidates)
        result["candidates"] = candidates
        result["message"] = (
            "已按当前路网距离排序；起点和地点到道路的末端接驳均为直线估算，尚未核实通行。"
            if candidates else result.get("message", "校内没有找到符合条件的附近地点。")
        )
        return result, {"candidates": candidates}

    results = _search_poi_categories(
        subcategories or None, poi_type, keyword, season, include_minor, activities=activities,
    )
    results.sort(key=importance_score, reverse=True)
    results = results[:limit]
    if not results:
        return {"candidates": [], "message": "校内暂无匹配的地点，可换个类别或关键词试试"}, None

    candidates = [_poi_public(p, with_provenance=True) for p in results]
    return {"candidates": candidates, "count": len(candidates)}, {"candidates": candidates}


def _tool_find_reachable_places(args, ctx):
    """按已标注路网搜索候选地点，将起点与地点的接驳估算计入距离。"""
    has_distance = args.get("max_distance_m") is not None
    has_time = args.get("max_time_min") is not None
    if has_distance and has_time:
        return {"error": "invalid_reachability_budget",
                "message": "最大距离和最大时间只能选一个"}, None

    mode = args.get("mode") or "walk"
    if mode not in ("walk", "bike", "drive"):
        mode = "walk"
    max_time = 10.0 if not has_distance and not has_time else None
    try:
        if has_distance:
            max_distance = float(args["max_distance_m"])
            if not math.isfinite(max_distance) or not 1 <= max_distance <= 10000:
                raise ValueError
        else:
            max_time = float(args["max_time_min"] if has_time else max_time)
            if not math.isfinite(max_time) or not 1 <= max_time <= 180:
                raise ValueError
            max_distance = max_time * MODE_SPEEDS_KMH[mode] * 1000.0 / 60.0
    except (TypeError, ValueError):
        return {"error": "invalid_reachability_budget",
                "message": "距离需为 1–10000 米，时间需为 1–180 分钟"}, None

    start_ref = args.get("start")
    if not isinstance(start_ref, dict):
        return {"error": "invalid_endpoint", "message": "起点格式不正确"}, None
    context = dict(ctx) if isinstance(ctx, dict) else {}
    graph = context.get("_route_graph_snapshot")
    if graph is None:
        graph = _ensure_graph()
    if graph is None:
        return {"error": "network_unavailable", "message": "校园路网暂时不可用"}, None
    mode_graph = get_routing_index(graph).for_mode(mode).graph
    start_node, start_name, error = _resolve_endpoint(start_ref, mode_graph)
    if error:
        return error, None

    try:
        active_conditions = (context["_route_conditions_snapshot"]
                             if "_route_conditions_snapshot" in context
                             else list_conditions(strict=True))
    except Exception:
        logger.exception("读取路况失败，拒绝按错误的可达范围推荐地点")
        return {"error": "road_conditions_unavailable", "message": "当前路况暂时无法确认，附近地点候选暂不可核实。"}, None
    try:
        mode_graph, _, _, conditions_applied = apply_conditions_to_graph(
            mode_graph, active_conditions, mode, strict=True,
        )
    except RoadConditionsUnavailableError:
        logger.exception("路况事件无法匹配当前路网，拒绝生成可达地点结果")
        return {"error": "road_conditions_unavailable",
                "message": "当前管制信息无法与路网对应，附近地点暂不可核实。"}, None
    from spatial.routing import _filter_by_constraints
    constraints = args.get("constraints") or {}
    if not isinstance(constraints, dict):
        return {"error": "invalid_constraints", "message": "通行约束格式不正确"}, None
    if ("avoid_steps" in constraints
            and not isinstance(constraints["avoid_steps"], bool)):
        return {"error": "invalid_constraints", "message": "avoid_steps 必须是布尔值"}, None
    filtered_graph, filter_status, _ = _filter_by_constraints(mode_graph, constraints)
    tagged_steps_before = {
        (u, v, k) for u, v, k, data in mode_graph.edges(keys=True, data=True)
        if "steps" in str(data.get("highway", "")).lower().split(";")
        or (isinstance(data.get("highway"), (list, tuple)) and "steps" in data.get("highway"))
    }
    tagged_steps_after = {
        (u, v, k) for u, v, k in filtered_graph.edges(keys=True)
    }
    known_tagged_steps_filtered = len(tagged_steps_before - tagged_steps_after)
    avoid_steps_requested = bool(constraints.get("avoid_steps", False))
    if start_node not in filtered_graph:
        return {"error": "start_unreachable",
                "message": f"起点「{start_name}」在当前出行方式或管制条件下无法接入路网"}, None

    start_access = _endpoint_access_evidence(start_ref, start_node, filtered_graph, mode)
    try:
        start_access_distance = float(start_access["snap_distance_m"])
    except (KeyError, TypeError, ValueError):
        return {"error": "start_access_unavailable",
                "message": "起点到可规划道路的接驳距离暂时无法核实，附近地点候选暂不可用。"}, None
    if not math.isfinite(start_access_distance):
        return {"error": "start_access_unavailable",
                "message": "起点到可规划道路的接驳距离暂时无法核实，附近地点候选暂不可用。"}, None

    try:
        distances = nx.single_source_dijkstra_path_length(
            filtered_graph, start_node, cutoff=max_distance, weight="length",
        )
    except (nx.NodeNotFound, nx.NetworkXError) as exc:
        logger.warning("可达地点搜索失败: %s", exc, exc_info=True)
        return {"error": "reachability_failed", "message": "附近地点暂时查不到，请稍后重试"}, None

    subcategory = args.get("subcategory")
    raw_subcategories = args.get("subcategories")
    subcategories = ([str(item) for item in raw_subcategories if item]
                     if isinstance(raw_subcategories, list) else [])
    if not subcategories and subcategory:
        subcategories = [subcategory]
    if not subcategories:
        subcategories = infer_recommendation_subcategories(
            (ctx or {}).get("query", "") if isinstance(ctx, dict) else ""
        ) or []
    query = (ctx or {}).get("query", "") if isinstance(ctx, dict) else ""
    activity_request = resolve_activity_query(query)
    activities = ([str(value) for value in args.get("activities", []) if value]
                  if isinstance(args.get("activities"), list) else [])
    if activity_request and activity_request.get("status") == "needs_clarification" and not activities:
        question = activity_request["question"]
        return {"question": question, "options": activity_request["options"]}, {
            "clarify": {"question": question, "options": activity_request["options"]}
        }
    if activity_request and activity_request.get("status") == "resolved" and not is_poi_catalog_query(query):
        activities = list(activity_request.get("activities") or activities)
        subcategories = list(activity_request.get("subcategories") or subcategories)
    elif activities and not subcategories:
        subcategories = list(dict.fromkeys(
            category for activity in activities
            for category in ACTIVITY_SUBCATEGORIES.get(activity, ())
        ))
    poi_type = args.get("poi_type")
    keyword = (args.get("keyword") or "").strip()
    include_minor = bool(args.get("include_minor", True))
    season = args.get("season")
    places = _search_poi_categories(
        subcategories or None, poi_type, keyword, season, include_minor, activities=activities,
    )

    reachable = []
    excluded_long_access = 0
    for poi in places:
        coords = poi.get("coordinates") or {}
        try:
            poi_lng_gcj = float(coords.get("lng", poi.get("lon", poi.get("lng"))))
            poi_lat_gcj = float(coords.get("lat", poi.get("lat")))
            poi_lng_wgs, poi_lat_wgs = gcj02_to_wgs84(poi_lng_gcj, poi_lat_gcj)
            node = get_nearest_node(filtered_graph, poi_lng_wgs, poi_lat_wgs)
            if node not in distances:
                continue
            node_lng, node_lat = get_node_coords(filtered_graph, node)
            access_distance = _haversine(poi_lat_wgs, poi_lng_wgs, node_lat, node_lng)
            if access_distance > _REACHABILITY_MAX_ACCESS_SNAP_M:
                excluded_long_access += 1
                continue
            network_path_distance = float(distances[node])
            network_distance = start_access_distance + network_path_distance + access_distance
        except (TypeError, ValueError, KeyError, RuntimeError):
            continue
        if network_distance > max_distance:
            continue
        public = _poi_public(poi, with_provenance=True)
        public["network_path_distance_m"] = round(network_path_distance)
        public["network_distance_m"] = round(network_distance)
        public["distance_m"] = round(network_distance)
        public["distance_basis"] = "network_path_plus_unverified_straight_line_access_estimate"
        public["start_access_m"] = round(start_access_distance, 1)
        public["access_snap_m"] = round(access_distance, 1)
        public["access_link_verified"] = False
        public["access_link_basis"] = "straight_line_to_nearest_network_node"
        public["start_access_link_verified"] = False
        if avoid_steps_requested:
            public["step_access_status"] = "known_tagged_steps_filtered_unknown_edges_unverified"
        public["estimated_duration_min"] = estimate_duration_min(network_distance, mode)
        reachable.append(public)

    reachable.sort(key=lambda place: (place["network_distance_m"], place["name"]))
    limit = max(1, min(int(args.get("limit", 8) or 8), 20))
    reachable = reachable[:limit]
    result = {
        "start_name": start_name,
        "mode": mode,
        "count": len(reachable),
        "candidates": reachable,
        "reachability_basis": "network_path_plus_unverified_straight_line_access_estimate",
        "start_access": start_access,
        "max_distance_m": round(max_distance),
        "max_time_min": round(max_time, 1) if max_time is not None else None,
        "max_access_snap_m": _REACHABILITY_MAX_ACCESS_SNAP_M,
        "excluded_long_access_count": excluded_long_access,
        "avoid_steps_requested": avoid_steps_requested,
        "known_tagged_steps_filtered": known_tagged_steps_filtered if avoid_steps_requested else 0,
        "step_access_status": (
            "known_tagged_steps_filtered_unknown_edges_unverified"
            if avoid_steps_requested else "not_requested"
        ),
        "remaining_step_access_verified": False if avoid_steps_requested else None,
        "road_conditions_applied": conditions_applied,
        "data_version": _data_version_for_graph(graph),
        "road_condition_version": current_road_condition_version(active_conditions),
        "data_source": "校园 POI 主数据与当前交通方式可用的已标记路网",
        "geometry_crs": "GCJ-02",
        "access_link_note": "起点和地点到最近路网节点的末端接驳只按直线距离估算，未核实是否存在实际步行通道；地点接驳距离超过 60 米的不列为候选。候选地点应再用路线工具核查。",
        "estimate_note": "距离由起点接驳、路网路径和地点接驳组成；两端接驳均为未核实的直线估算。时间按该出行方式平均速度估算，不等同导航实时 ETA。",
    }
    if not reachable:
        result["message"] = "当前路网估算范围内没有找到候选地点；地点入口与末端通道尚未逐一核实，可扩大范围或换个类别。"
    return result, {"candidates": reachable}


def _tool_compare_routes(args, ctx):
    """用同一数据、天气与路况快照比较三种策略，不替换当前路线。"""
    context, snapshot_error = _prepare_route_context(ctx)
    if snapshot_error:
        return snapshot_error, None

    alternatives = []
    unavailable = []
    start_name = end_name = None
    for strategy, label in _COMPARISON_STRATEGIES:
        route_args = dict(args)
        route_args["strategy"] = strategy
        route, _artifact = _tool_plan_route(route_args, context)
        if not isinstance(route, dict) or route.get("error"):
            reason = (route.get("message", "该策略暂时无法生成路线")
                      if isinstance(route, dict) else "路线结果格式无效")
            unavailable.append({"strategy": strategy, "label": label, "reason": reason})
            continue
        try:
            distance = float(route.get("recommended_length_m", route.get("distance_m")))
            if not math.isfinite(distance) or distance < 0:
                raise ValueError
        except (TypeError, ValueError):
            unavailable.append({"strategy": strategy, "label": label, "reason": "路线缺少有效距离"})
            continue
        if start_name is None:
            start_name, end_name = route.get("start_name"), route.get("end_name")
        duration = route.get("duration_min")
        try:
            duration = float(duration) if duration is not None else None
            if duration is not None and (not math.isfinite(duration) or duration < 0):
                duration = None
        except (TypeError, ValueError):
            duration = None
        alternatives.append({
            "strategy": strategy,
            "label": label,
            "distance_m": round(distance),
            "duration_min": round(duration, 1) if duration is not None else None,
            "slope_level_avg": route.get("slope_avg_recommended"),
            "scenery_level_avg": route.get("scenery_avg_recommended"),
            "weights": route.get("applied_weights"),
        })

    if not alternatives:
        return {
            "error": "route_comparison_unavailable",
            "message": "三种路线都无法生成，请检查起终点、出行方式和硬约束。",
            "unavailable": unavailable,
        }, None

    shortest = next((item["distance_m"] for item in alternatives
                     if item["strategy"] == "shortest"), None)
    for item in alternatives:
        delta = item["distance_m"] - shortest if shortest is not None else None
        item["distance_delta_from_shortest_m"] = delta
        item["detour_ratio"] = round(item["distance_m"] / shortest, 3) if shortest else None

    return {
        "status": "complete" if not unavailable else "partial",
        "start_name": start_name,
        "end_name": end_name,
        "mode": args.get("mode") or "walk",
        "alternatives": alternatives,
        "unavailable": unavailable,
        "evidence": {
            "data_version": _data_version_for_graph(context["_route_graph_snapshot"]),
            "road_condition_version": route_state_module.current_road_condition_version(
                context["_route_conditions_snapshot"]
            ),
            "geometry_crs": "GCJ-02",
            "distance_basis": "所选路线边的路网长度",
            "duration_basis": "速度模型估算，不含实时拥堵 ETA",
        },
        "comparison_basis": "相同起终点、出行方式、硬约束、路网/路况与天气快照；距离按所选路网边计算，时间为估算值。",
        "selection_note": "此结果仅供比较，没有改变当前地图路线；用户选定策略后再单独规划。",
    }, None


def _tool_get_place_details(args, ctx):
    requested = str(args.get("name") or "").strip()
    poi, alternatives = find_poi_ambiguous(requested)
    if not poi:
        return {
            "found": False,
            "requested_name": requested,
            "source_status": "not_found_in_current_campus_master",
            "message": "当前校园地点主数据没有匹配到该名称；这不等于现实中该地点不存在。可核对别名或补充地点资料。",
            "candidates": [_poi_public(item, with_coords=False) for item in (alternatives or [])[:5]],
            "opening_hours": "unknown",
            "accessibility": "unknown",
            "data_version": _data_version_for_graph(None),
        }, None

    place = _poi_public(poi, with_provenance=True)
    coordinates = poi.get("coordinates") or {}
    place.update({
        "poi_id": poi.get("id"),
        "aliases": list(poi.get("aliases") or []),
        "campus": poi.get("campus"),
        "source_refs": list(poi.get("source_refs") or [])[:10],
        "verification_status": poi.get("verification_status") or "legacy_unverified",
        "coordinate_verification_status": poi.get("coordinate_verification_status") or "unknown",
        "opening_hours": {"status": "known", "value": poi.get("opening_hours")}
        if poi.get("opening_hours") else {"status": "unknown", "value": None},
        "accessibility": poi.get("accessibility") or "unknown",
        "location_crs": poi.get("coordinate_crs") or "GCJ-02",
        "data_source": "校园 POI 主数据；source_refs 标注其来源",
        "data_version": _data_version_for_graph(None),
    })
    if place.get("lng") is None:
        place["lng"] = poi.get("lon", coordinates.get("lng"))
    if place.get("lat") is None:
        place["lat"] = poi.get("lat", coordinates.get("lat"))
    return {"found": True, "place": place}, None


def _plan_common(args, ctx):
    """公共准备：路网、方式过滤、天气。返回 (G, G_mode, mode, weather_info)。"""
    context = ctx if isinstance(ctx, dict) else {}
    G = context.get("_route_graph_snapshot") or _ensure_graph()
    mode = args.get("mode") or "walk"
    if mode not in ("walk", "bike", "drive"):
        mode = "walk"
    G_mode = get_routing_index(G).for_mode(mode).graph
    snap = (context.get("_route_weather_snapshot")
            if "_route_weather_snapshot" in context else _weather_snapshot())
    return G, G_mode, mode, (snap["live"] if snap else None)


def _prepare_route_context(ctx):
    """Freeze graph, active events and weather once for a complete spatial run."""
    context = dict(ctx) if isinstance(ctx, dict) else {}
    graph = context.get("_route_graph_snapshot")
    if graph is None:
        graph = _ensure_graph()
    if graph is None:
        return None, {"error": "network_unavailable", "message": "校园路网暂时不可用。"}
    context["_route_graph_snapshot"] = graph
    if "_route_conditions_snapshot" not in context:
        try:
            context["_route_conditions_snapshot"] = list_conditions(strict=True)
        except Exception:
            logger.exception("读取路况快照失败，拒绝使用可能过期或漏封路的路线数据")
            return None, {"error": "road_conditions_unavailable", "message": "当前路况暂时无法确认，无法可靠规划路线。"}
    if "_route_weather_snapshot" not in context:
        try:
            context["_route_weather_snapshot"] = _weather_snapshot()
        except Exception:
            logger.exception("读取天气快照失败；本次路线不应用天气成本")
            context["_route_weather_snapshot"] = None
    return context, None


def _strategy_for_args(args, ctx=None, end_poi=None):
    explicit = args.get("strategy") or args.get("strategy_name")
    raw_weights = args.get("weights")
    context = ctx if isinstance(ctx, dict) else {}
    if explicit is None and context.get("query"):
        explicit = detect_strategy_hint(context["query"])
    elif explicit is None and raw_weights is not None:
        explicit = "custom"
    return select_route_strategy(
        query=context.get("query", ""),
        end_poi=end_poi,
        explicit_strategy=explicit,
        strategy_source=args.get("strategy_source", "explicit_nl"),
        profile=context.get("profile"),
        custom_weights=raw_weights,
    )


def _attach_route_state(
    payload, route_kind, args, ctx, graph, decision, start, end,
    *, via=None, tour=None, legs=None, travel_mode="walk",
):
    data_version = _data_version_for_graph(graph)
    condition_snapshot = ((ctx or {}).get("_route_conditions_snapshot")
                          if isinstance(ctx, dict) else None)
    condition_version = route_state_module.current_road_condition_version(condition_snapshot)
    payload["route_state"] = build_route_state(
        route_kind=route_kind,
        original_query=(ctx or {}).get("query", "") if isinstance(ctx, dict) else "",
        start=start,
        end=end,
        via=via,
        tour=tour,
        legs=legs or [],
        travel_mode=travel_mode,
        hard_constraints=args.get("constraints") or {},
        strategy=decision.as_dict(),
        data_version=data_version,
        road_condition_version=condition_version,
    )
    payload["evidence"] = {
        "network_source": "武汉大学校园主路网（OSM 与校方数据融合）",
        "poi_source": "项目校园 POI 主数据",
        "data_version": data_version,
        "road_condition_version": condition_version,
        "geometry_crs": "GCJ-02",
        "distance_basis": "所选路线边的路网长度",
        "duration_basis": "路网长度与出行方式速度模型估算；不是实时导航 ETA",
        "attribute_caveat": "坡度、景观和无障碍字段可能缺测或为派生值；degraded 与通行约束状态须一并查看。",
    }
    return payload


def _tool_plan_route(args, ctx):
    graph_started = time.perf_counter()
    context, snapshot_error = _prepare_route_context(ctx)
    if snapshot_error:
        return snapshot_error, None
    G, G_mode, mode, weather_info = _plan_common(args, context)
    common_graph_ms = (time.perf_counter() - graph_started) * 1000
    poi_started = time.perf_counter()
    start_node, start_name, err = _resolve_endpoint(args.get("start"), G_mode)
    if err:
        return err, None
    end_node, end_name, err = _resolve_endpoint(args.get("end"), G_mode)
    if err:
        return err, None
    if start_node == end_node:
        return {"error": "same_poi", "message": "起点和终点相同（或距离太近），换个目的地试试"}, None

    end_poi, _ = find_poi_ambiguous((args.get("end") or {}).get("name", ""))
    poi_resolution_ms = (time.perf_counter() - poi_started) * 1000
    decision = _strategy_for_args(args, ctx, end_poi=end_poi)

    try:
        result = compute_route(
            G=G, start_node=start_node, end_node=end_node,
            constraints=args.get("constraints") or {},
            weights=decision.weights, mode=mode, weather_info=weather_info,
            strategy_name=decision.name, detour_cap=decision.detour_cap,
            road_conditions=context["_route_conditions_snapshot"],
        )
    except ValueError as e:
        return {"error": "route_not_found", "message": str(e)}, None

    payload = _route_payload(G, result, start_name, end_name, mode)
    payload["timings_ms"]["graph_prepare"] = round(
        payload["timings_ms"]["graph_prepare"] + common_graph_ms, 3
    )
    payload["timings_ms"]["poi_resolution"] = round(poi_resolution_ms, 3)
    payload["timings_ms"] = normalize_timings(payload["timings_ms"])
    payload["strategy"] = decision.as_dict()
    payload["endpoint_access"] = {
        "start": _endpoint_access_evidence(args.get("start"), start_node, G_mode, mode),
        "end": _endpoint_access_evidence(args.get("end"), end_node, G_mode, mode),
        "max_snap_m": _MAX_ENDPOINT_ACCESS_SNAP_M,
    }
    _attach_route_state(
        payload, "direct", args, context, G, decision,
        args["start"], args["end"], travel_mode=mode,
    )
    return payload, {"route": payload}


def _tool_plan_via_route(args, ctx):
    context, snapshot_error = _prepare_route_context(ctx)
    if snapshot_error:
        return snapshot_error, None
    G, G_mode, mode, weather_info = _plan_common(args, context)
    start_node, start_name, err = _resolve_endpoint(args.get("start"), G_mode)
    if err:
        return err, None
    end_node, end_name, err = _resolve_endpoint(args.get("end"), G_mode)
    if err:
        return err, None

    requested_points = args.get("via_points")
    via_name = (args.get("via_name") or "").strip()
    via_sub = (args.get("via_subcategory") or "").strip()
    via_activity = (args.get("via_activity") or "").strip()
    if not via_activity and via_sub in ACTIVITY_LABELS:
        via_activity, via_sub = via_sub, ""
    via_coord = args.get("via_coord")
    has_single_via = bool(via_name or via_sub or via_activity or via_coord)
    if requested_points is not None and has_single_via:
        return {"error": "conflicting_via_arguments",
                "message": "多个途经点 via_points 不能与单个途经点参数同时使用"}, None
    if requested_points is not None:
        if (not isinstance(requested_points, list) or not requested_points
                or len(requested_points) > _MAX_VIA_POINTS):
            return {"error": "invalid_via_points",
                    "message": f"途经点需按顺序提供 1 到 {_MAX_VIA_POINTS} 个地点或坐标"}, None
    elif not has_single_via:
        return {"error": "missing_via", "message": "缺少途经点：请用 via_name、via_subcategory、via_activity 或 via_coord 指定"}, None

    via_entries = []
    detour_ratio = None
    if requested_points is not None:
        for idx, ref in enumerate(requested_points, start=1):
            if not isinstance(ref, dict):
                return {"error": "invalid_via_points",
                        "message": f"第 {idx} 个途经点格式无效，请提供地点名或坐标"}, None
            if ref.get("type") == "coord":
                coords = ref.get("coordinates") or ref
                try:
                    lng, lat = float(coords.get("lng")), float(coords.get("lat"))
                except (AttributeError, TypeError, ValueError):
                    return {"error": "via_coord_invalid",
                            "message": f"第 {idx} 个途经点缺少有效的 WGS-84 坐标"}, None
                if not (math.isfinite(lng) and math.isfinite(lat)
                        and -180 <= lng <= 180 and -90 <= lat <= 90):
                    return {"error": "via_coord_invalid",
                            "message": f"第 {idx} 个途经点坐标超出有效范围"}, None
                ref = {**ref, "coordinates": {"lng": lng, "lat": lat}}
            node, display, err = _resolve_endpoint(ref, G_mode)
            if err:
                return {**err, "message": f"第 {idx} 个途经点无法解析：{err.get('message', '地点无效')}"}, None
            via_info = {"name": display}
            normalized_ref = {"name": display, "type": "poi"}
            if isinstance(ref, dict) and ref.get("type") == "coord":
                coords = ref.get("coordinates") or ref
                normalized_coords = {"lng": float(coords["lng"]), "lat": float(coords["lat"])}
                via_info["coordinates"] = normalized_coords
                normalized_ref = {"name": display, "type": "coord",
                                  "coordinates": normalized_coords}
            else:
                poi, _ = find_poi_ambiguous(display)
                if poi:
                    via_info = _poi_public(poi)
            access_evidence = _endpoint_access_evidence(ref, node, G_mode, mode)
            via_entries.append({"node": node, "name": display, "info": via_info,
                                "ref": normalized_ref, "access": access_evidence})
    elif via_coord:
        # 坐标途经点：直接吸附到最近路网节点
        try:
            via_lng = float(via_coord.get("lng"))
            via_lat = float(via_coord.get("lat"))
            via_display = "地图途经点"
            via_node, via_access, error = _snap_wgs_to_network(
                via_lng, via_lat, G_mode, via_display,
            )
            if error:
                return error, None
            via_info = {"name": via_display}
            via_entries = [{"node": via_node, "name": via_display, "info": via_info,
                            "ref": {"name": via_display, "type": "coord",
                                    "coordinates": {"lng": via_lng, "lat": via_lat}},
                            "access": via_access}]
            detour_ratio = None
        except (TypeError, ValueError, RuntimeError) as e:
            return {"error": "via_coord_invalid", "message": f"途经点坐标无效: {e}"}, None
    elif via_name:
        via_node, via_display, err = _resolve_endpoint({"name": via_name}, G_mode)
        if err:
            return err, None
        via_poi, _ = find_poi_ambiguous(via_name)
        via_info = _poi_public(via_poi) if via_poi else {"name": via_display}
        via_access = _endpoint_access_evidence(
            {"name": via_display, "type": "poi"}, via_node, G_mode, mode,
        )
        via_entries = [{"node": via_node, "name": via_display, "info": via_info,
                        "ref": {"name": via_display, "type": "poi"}, "access": via_access}]
        detour_ratio = None
    else:
        # 类别途经：检索候选 → 顺路性排序 → 取最顺路的
        via_activities = [via_activity] if via_activity else None
        via_categories = (list(ACTIVITY_SUBCATEGORIES.get(via_activity, ()))
                          if via_activity else [via_sub])
        via_type = "sports" if via_activity in {
            "basketball", "badminton", "table_tennis", "football", "volleyball",
            "tennis", "running", "fitness", "swimming",
        } else None
        cands = _search_poi_categories(
            via_categories, via_type, "", None, True, activities=via_activities,
        )
        if not cands:
            return {"error": "no_via_candidates",
                    "message": f"校园主数据中没有找到「{ACTIVITY_LABELS.get(via_activity, via_sub)}」途经地点"}, None
        poi_nodes = []
        for p in cands:
            try:
                lng_wgs, lat_wgs = gcj02_to_wgs84(p["lon"], p["lat"])
                node, access, snap_error = _snap_wgs_to_network(
                    lng_wgs, lat_wgs, G_mode, p.get("name", "途经地点"),
                )
                if not snap_error:
                    enriched = dict(p)
                    enriched["_endpoint_access"] = access
                    poi_nodes.append((enriched, node))
            except Exception:
                continue
        on_the_way, off_the_way = rank_via_candidates(G, start_node, end_node, poi_nodes, mode=mode)
        if not on_the_way:
            return {
                "error": "no_on_the_way_via",
                "message": "顺路范围内没有合适的途经点（绕行比都超过 1.5），"
                           "可以放弃途经直接规划，或让用户在以下相对近的候选中选择",
                "off_the_way": [{"poi": _poi_public(p), "detour_ratio": r,
                                 "endpoint_access": p.get("_endpoint_access")}
                                for p, _, r in off_the_way[:3]],
            }, None
        via_poi, via_node, detour_ratio = on_the_way[0]
        via_info = _poi_public(via_poi)
        via_display = via_poi["name"]
        via_access = via_poi.get("_endpoint_access") or _snap_evidence_for_wgs(
            *gcj02_to_wgs84(via_poi["lon"], via_poi["lat"]), via_display, via_node, G_mode,
        )
        via_entries = [{"node": via_node, "name": via_display, "info": via_info,
                        "ref": {"name": via_display, "type": "poi"}, "access": via_access}]

    ordered_nodes = [start_node] + [entry["node"] for entry in via_entries] + [end_node]
    for idx, (first, second) in enumerate(zip(ordered_nodes, ordered_nodes[1:]), start=1):
        if first == second:
            return {"error": "duplicate_via_point",
                    "message": f"第 {idx} 段的途经点与相邻地点落在同一路网位置，请调整途经点"}, None

    decision = _strategy_for_args(args, ctx)
    try:
        route_kwargs = {
            "constraints": args.get("constraints") or {},
            "weights": decision.weights,
            "mode": mode,
            "weather_info": weather_info,
            "road_conditions": context["_route_conditions_snapshot"],
            "strategy_name": decision.name,
            "detour_cap": decision.detour_cap,
        }
        leg_results = [
            compute_route(G, ordered_nodes[i], ordered_nodes[i + 1], **route_kwargs)
            for i in range(len(ordered_nodes) - 1)
        ]
        direct = compute_route(G, start_node, end_node, **route_kwargs)
    except ValueError as e:
        return {"error": "route_not_found", "message": str(e)}, None

    point_names = [entry["name"] for entry in via_entries]
    waypoint_names = [start_name] + point_names + [end_name]
    legs = [
        _route_payload(G, result, waypoint_names[i], waypoint_names[i + 1], mode)
        for i, result in enumerate(leg_results)
    ]
    direct_length = direct["recommended_length_m"] or direct["shortest_length_m"]
    total_length = sum(result["recommended_length_m"] for result in leg_results)
    total_shortest = sum(result["shortest_length_m"] for result in leg_results)
    detour_ratio = total_length / direct_length if direct_length > 0 else float("inf")

    def join_path(field):
        merged = []
        for leg in legs:
            part = leg.get(field) or []
            if merged and part and merged[-1] == part[0]:
                part = part[1:]
            merged.extend(part)
        return merged

    points_public = [entry["info"] for entry in via_entries]
    via_public = (points_public[0] if len(points_public) == 1 else {
        "name": "、".join(point_names), "type": "multi", "points": points_public,
    })
    direct_shortest = direct["shortest_length_m"]
    shortest_detour = total_shortest / direct_shortest if direct_shortest > 0 else float("inf")
    payload = {
        "via": via_public,
        "legs": legs,
        "start_name": start_name,
        "end_name": end_name,
        "recommended": join_path("recommended"),
        "shortest": join_path("shortest"),
        "recommended_edge_ids": [edge for leg in legs for edge in leg.get("recommended_edge_ids", [])],
        "shortest_edge_ids": [edge for leg in legs for edge in leg.get("shortest_edge_ids", [])],
        "steps": _merge_leg_steps(legs),
        "recommended_length_m": round(total_length, 1),
        "shortest_length_m": round(total_shortest, 1),
        "distance_m": round(total_length, 1),
        "detour_ratio": round(detour_ratio, 3),
        "detour_ratio_shortest": round(shortest_detour, 3),
        "duration_min": round(estimate_duration_min(total_length, mode), 1),
        "shortest_duration_min": round(estimate_duration_min(total_shortest, mode), 1),
        "mode": mode,
        "strategy": decision.as_dict(),
        "pois": [poi for leg in legs for poi in leg.get("pois", [])],
        "filter_status": ";".join(dict.fromkeys(
            status for leg in legs for status in (leg.get("filter_status") or "").split(";")
            if status
        )),
        "overlap_rate": round(
            sum((leg.get("overlap_rate") or 0) * (leg.get("recommended_length_m") or 0)
                for leg in legs) / total_length, 4
        ) if total_length > 0 else 1.0,
        "applied_weights": decision.weights,
        "degraded": any(bool(leg.get("degraded")) for leg in legs),
        "length_capped": any(bool(leg.get("length_capped")) for leg in legs),
    }
    payload["endpoint_access"] = {
        "start": _endpoint_access_evidence(args.get("start"), start_node, G_mode, mode),
        "via": [
            entry.get("access") or _endpoint_access_evidence(entry.get("ref"), entry["node"], G_mode, mode)
            for entry in via_entries
        ],
        "end": _endpoint_access_evidence(args.get("end"), end_node, G_mode, mode),
        "max_snap_m": _MAX_ENDPOINT_ACCESS_SNAP_M,
    }
    via_ref = (via_entries[0]["ref"] if len(via_entries) == 1 else {
        "name": "、".join(point_names), "type": "multi",
        "points": [entry["ref"] for entry in via_entries],
    })
    _attach_route_state(
        payload, "via", args, context, G, decision,
        args["start"], args["end"], via=via_ref, travel_mode=mode,
    )
    return payload, {"route": payload, "route_kind": "via"}


def _tool_plan_itinerary(args, ctx):
    """Plan an ordered stop list and enforce the stated total-time budget."""
    stops = args.get("stops")
    if not isinstance(stops, list) or not 1 <= len(stops) <= _MAX_VIA_POINTS:
        return {"error": "invalid_itinerary", "message": "行程至少需要 1 个、最多 10 个途经地点。"}, None
    raw_budget = args.get("time_budget_min")
    raw_dwell = args.get("stop_duration_min", 15)
    if isinstance(raw_budget, bool) or isinstance(raw_dwell, bool):
        return {"error": "invalid_itinerary", "message": "请提供有效的总时间和每站停留时间。"}, None
    try:
        budget = float(raw_budget)
        dwell = float(raw_dwell)
    except (TypeError, ValueError):
        return {"error": "invalid_itinerary", "message": "请提供有效的总时间和每站停留时间。"}, None
    if (not math.isfinite(budget) or budget <= 0 or budget > 1440
            or not math.isfinite(dwell) or dwell < 0 or dwell > 240):
        return {"error": "invalid_itinerary", "message": "总时间需在 1 到 1440 分钟内，每站停留时间需在 0 到 240 分钟内。"}, None
    start = args.get("start")
    if not isinstance(start, dict) or not start.get("name"):
        return {"error": "invalid_itinerary", "message": "行程起点无效。"}, None

    end = args.get("end") or start
    route_args = {
        "start": start,
        "end": end,
        "via_points": stops,
        "mode": args.get("mode", "walk"),
        "constraints": args.get("constraints") or {},
        "weights": args.get("weights"),
        "strategy": args.get("strategy"),
        "strategy_source": args.get("strategy_source", "explicit_nl"),
    }
    route, artifact = _tool_plan_via_route(route_args, ctx)
    if not isinstance(route, dict) or route.get("error"):
        return route if isinstance(route, dict) else {
            "error": "itinerary_route_invalid", "message": "路线计算未返回可用结果。"
        }, None
    try:
        travel = float(route.get("duration_min"))
        distance = float(route.get("recommended_length_m", route.get("distance_m")))
        if (not math.isfinite(travel) or travel < 0
                or not math.isfinite(distance) or distance < 0):
            raise ValueError
    except (TypeError, ValueError):
        return {"error": "itinerary_estimate_missing", "message": "路线缺少有效的路网距离或时间估算，暂时不能判断是否满足预算。"}, None

    dwell_total = dwell * len(stops)
    total = travel + dwell_total
    over_by = max(0.0, total - budget)
    summary = {
        "status": "feasible" if over_by <= 0.05 else "over_budget",
        "start_name": route.get("start_name"),
        "end_name": route.get("end_name"),
        "stop_names": [str(stop.get("name", "未命名地点")) for stop in stops if isinstance(stop, dict)],
        "stop_count": len(stops),
        "travel_duration_min": round(travel, 1),
        "stop_duration_min": round(dwell_total, 1),
        "per_stop_duration_min": round(dwell, 1),
        "total_duration_min": round(total, 1),
        "distance_m": round(distance),
        "time_budget_min": round(budget, 1),
        "over_by_min": round(over_by, 1),
        "opening_hours": "unknown",
        "estimate_note": "路上时间按当前路网和出行方式的速度模型估算；未计入排队、实际停留差异和地点开放状态。",
    }
    if summary["status"] != "feasible":
        summary["next_step"] = "说明超出预算的分钟数，并询问用户是否增加时间或删减途经点；不要把这条路线作为满足预算的路线展示。"
        return summary, None

    route["itinerary"] = summary
    route["route_kind"] = "itinerary"
    route["travel_duration_min"] = round(travel, 1)
    route["stop_duration_min"] = round(dwell_total, 1)
    route["itinerary_total_duration_min"] = round(total, 1)
    route["time_budget_min"] = round(budget, 1)
    if isinstance(route.get("route_state"), dict):
        route["route_state"]["route_kind"] = "itinerary"
        route["route_state"]["itinerary"] = {
            "time_budget_min": round(budget, 1),
            "stop_duration_min": round(dwell, 1),
        }
    route_artifact = dict(artifact) if isinstance(artifact, dict) else {}
    route_artifact["route"] = route
    route_artifact["route_kind"] = "itinerary"
    return summary, route_artifact


def _tool_plan_multimodal_route(args, ctx):
    """分段换乘路径：用户显式给每段 mode + end，按顺序拼接。

    schema: start + legs: array[{mode, end, via_name?}]。
    第 N 段终点自动作为第 N+1 段起点（节点重新吸附到对应 mode 的 G_mode）。
    每段独立 compute_route，最后合并 legs/recommended/steps/duration_min。
    """
    legs_in = args.get("legs") or []
    if not isinstance(legs_in, list) or len(legs_in) < 2:
        return {"error": "invalid_legs",
                "message": "至少需要 2 段换乘信息（每段含 mode 和 end）"}, None
    if len(legs_in) > 4:
        return {"error": "too_many_legs",
                "message": "换乘段数过多（最多 4 段），可拆成多次规划"}, None

    context, snapshot_error = _prepare_route_context(ctx)
    if snapshot_error:
        return snapshot_error, None
    G = context["_route_graph_snapshot"]
    snap = context.get("_route_weather_snapshot")
    weather_info = snap.get("live") if snap else None
    conditions = context["_route_conditions_snapshot"]

    # 起点（按第一段 mode 吸附）
    first_mode = legs_in[0].get("mode") or "walk"
    if first_mode not in ("walk", "bike", "drive"):
        first_mode = "walk"
    G_first = get_routing_index(G).for_mode(first_mode).graph
    start_node, start_name, err = _resolve_endpoint(args.get("start"), G_first)
    if err:
        return err, None
    start_access = _endpoint_access_evidence(args.get("start"), start_node, G_first, first_mode)

    decision = _strategy_for_args(args, ctx)
    legs_payload = []
    prev_node = start_node
    prev_name = start_name
    cumulative_len = 0.0
    cumulative_dur = 0.0
    recommended_all = []
    pois_all = []

    for idx, leg in enumerate(legs_in):
        mode = leg.get("mode") or "walk"
        if mode not in ("walk", "bike", "drive"):
            mode = "walk"
        end_ref = leg.get("end")
        if not isinstance(end_ref, dict):
            return {"error": "invalid_leg_end",
                    "message": f"第 {idx + 1} 段终点格式不正确"}, None

        # 每段独立过滤路网（mode 不同 → G_mode 不同）
        G_mode = get_routing_index(G).for_mode(mode).graph
        end_node, end_name, err = _resolve_endpoint(end_ref, G_mode)
        if err:
            return err, None
        end_access = _endpoint_access_evidence(end_ref, end_node, G_mode, mode)
        leg_start_access = start_access if idx == 0 else None
        # 起点也要重新吸附到本段 G_mode（前一段终点是 walk-only 时可能不在本段图里）
        if prev_node not in G_mode:
            # 用前一终点坐标重新吸附，并限制/披露模式转换接驳距离。
            try:
                prev_lng, prev_lat = get_node_coords(G, prev_node)
                prev_node, leg_start_access, transfer_error = _snap_wgs_to_network(
                    prev_lng, prev_lat, G_mode, f"第 {idx + 1} 段换乘起点",
                )
                if transfer_error:
                    return {**transfer_error,
                            "message": f"第 {idx + 1} 段换乘点无法可靠接入当前出行方式的道路：{transfer_error['message']}"}, None
            except Exception:
                return {"error": "leg_start_unreachable",
                        "message": f"第 {idx + 1} 段起点在 {mode} 模式下无可达节点附近，"
                                   f"可调整换乘点位置"}, None
        elif idx > 0:
            prev_lng, prev_lat = get_node_coords(G_mode, prev_node)
            leg_start_access = _snap_evidence_for_wgs(
                prev_lng, prev_lat, f"第 {idx + 1} 段换乘起点", prev_node, G_mode,
            )

        if prev_node == end_node:
            return {"error": "same_poi",
                    "message": f"第 {idx + 1} 段起终点相同，可省略该段"}, None

        via_name = (leg.get("via_name") or "").strip()
        via_access = None
        try:
            if via_name:
                via_node, via_display, err = _resolve_endpoint({"name": via_name}, G_mode)
                if err:
                    return err, None
                via_access = _endpoint_access_evidence(
                    {"name": via_display, "type": "poi"}, via_node, G_mode, mode,
                )
                result = compute_via_route(
                    G, prev_node, via_node, end_node,
                    constraints=args.get("constraints") or {},
                    weights=decision.weights, mode=mode,
                    road_conditions=conditions,
                    weather_info=weather_info,
                    strategy_name=decision.name, detour_cap=decision.detour_cap,
                )
                leg1 = _route_payload(G, result["leg1"], prev_name, via_display, mode)
                leg2 = _route_payload(G, result["leg2"], via_display, end_name, mode)
                leg_payload = {
                    "via": {"name": via_display},
                    "legs": [leg1, leg2],
                    "recommended": leg1["recommended"] + leg2["recommended"],
                    "recommended_edge_ids": (
                        leg1["recommended_edge_ids"] + leg2["recommended_edge_ids"]
                    ),
                    "steps": _merge_leg_steps([leg1, leg2]),
                    "recommended_length_m": result["total_length_m"],
                    "distance_m": result["total_length_m"],
                    "duration_min": round(estimate_duration_min(result["total_length_m"], mode), 1),
                    "mode": mode,
                    "pois": leg1["pois"] + leg2["pois"],
                }
            else:
                result = compute_route(
                    G=G, start_node=prev_node, end_node=end_node,
                    constraints=args.get("constraints") or {},
                    weights=decision.weights, mode=mode,
                    road_conditions=conditions,
                    weather_info=weather_info,
                    strategy_name=decision.name, detour_cap=decision.detour_cap,
                )
                leg_payload = _route_payload(G, result, prev_name, end_name, mode)
        except ValueError as e:
            return {"error": "route_not_found",
                    "message": f"第 {idx + 1} 段（{mode}：{prev_name}→{end_name}）不可达：{e}"}, None

        leg_payload["endpoint_access"] = {
            "start": leg_start_access,
            "end": end_access,
            "via": via_access,
            "max_snap_m": _MAX_ENDPOINT_ACCESS_SNAP_M,
        }

        legs_payload.append(leg_payload)
        cumulative_len += float(leg_payload["recommended_length_m"] or 0)
        cumulative_dur += float(leg_payload["duration_min"] or 0)
        recommended_all += leg_payload["recommended"]
        pois_all += leg_payload.get("pois") or []
        prev_node = end_node
        prev_name = end_name

    # 合并多段 steps（_merge_leg_steps 已处理跨段 arrive→via 转换）
    merged_steps = _merge_leg_steps(legs_payload)

    payload = {
        "legs": legs_payload,
        "start_name": start_name,
        "end_name": prev_name,
        "recommended": recommended_all,
        "recommended_edge_ids": [
            edge for leg in legs_payload for edge in leg["recommended_edge_ids"]
        ],
        "steps": merged_steps,
        "recommended_length_m": round(cumulative_len, 1),
        "distance_m": round(cumulative_len, 1),
        "duration_min": round(cumulative_dur, 1),
        # 整体 mode 用第一段（前端默认渲染色），每段 mode 在 legs[].mode 里
        "mode": legs_payload[0]["mode"],
        "strategy": decision.as_dict(),
        "pois": pois_all,
        "endpoint_access": {
            "start": start_access,
            "legs": [leg.get("endpoint_access") for leg in legs_payload],
            "max_snap_m": _MAX_ENDPOINT_ACCESS_SNAP_M,
        },
    }
    state_legs = [
        {
            "end": leg.get("end"),
            "travel_mode": leg.get("mode") or "walk",
            **({"via": {"name": leg["via_name"], "type": "poi"}}
               if leg.get("via_name") else {}),
        }
        for leg in legs_in
    ]
    _attach_route_state(
        payload, "multimodal", args, context, G, decision,
        args["start"], legs_in[-1]["end"], legs=state_legs,
        travel_mode=legs_payload[0]["mode"],
    )
    return payload, {"route": payload, "route_kind": "multimodal"}


def _tour_clause_is_negated(text, position):
    prefix = str(text or "")[:position]
    clause = re.split(r"[，,。；;！？!\n]|但是|不过|然而|但", prefix)[-1]
    return bool(re.search(r"(?:不想|不愿(?:意)?|不希望|不要|不去|(?:先)?别|避免|避开|排除)\s*$", clause))


def _infer_excluded_tour_poi_names(query):
    """Resolve places inside explicit negative tour clauses for auto selection."""
    text = str(query or "").strip()
    pattern = (
        r"(?:不想|不愿(?:意)?|不希望|不要|不去|(?:先)?别|避免|避开|排除)\s*"
        r"(?:(?:再)?(?:去|游览|游玩|游逛|逛|经过|途经|途径)\s*)?"
        r"(.*?)(?=(?:但是|不过|然而|但)|[，,。；;！？!\n]|$)"
    )
    excluded = set()
    for match in re.finditer(pattern, text):
        for fragment in re.split(r"(?:以及|和|与|及|、|到|至)", match.group(1)):
            candidate = re.sub(r"^\s*(?:去|到|逛|游览|游玩|经过|途经|途径)+", "", fragment).strip()
            if not candidate:
                continue
            poi, _ = find_poi_ambiguous(candidate)
            if poi:
                excluded.add(poi.get("name", candidate))
    return excluded


def _infer_explicit_tour_poi_names(query):
    """Recover named tour stops from an explicit multi-place tour request.

    This is a guard against a model call that selects plan_tour but omits the
    poi_names argument. Generic campus-tour requests remain category-driven.
    """
    text = str(query or "").strip()
    excluded = _infer_excluded_tour_poi_names(text)
    target_patterns = (
        # Destination-first wording: “经过 A 和 B 的游览路线”.
        r"(?:经过|途经|途径)\s*(.+?)(?:的)?(?:游览|游玩|游逛|逛一逛|逛逛|路线|环线)",
        # Place names before a route request: “规划 A 和 B 的游览路线”.
        r"(?:规划|安排|推荐|设计|生成)(?:一条|一段|一圈)?\s*(.+?)(?:的)?(?:游览路线|游玩路线|游逛路线|路线|环线)",
        # Compact title-style request: “A 和 B 游览路线”.
        r"^\s*(?:(?:我想要|我要|我想|帮我|请|带我|给我)\s*)?"
        r"(.+?)(?:的)?(?:游览路线|游玩路线|游逛路线|路线|环线)\s*$",
        # Place names before the tour verb: “去 A 和 B 逛逛”.
        r"(?:去|到)\s*(.+?)(?:游览|游玩|游逛|逛一逛|逛逛|逛遍|转一转|转转|走走)",
        # Tour verb before place names: “游览 A 和 B”.
        r"(?:游览|游玩|游逛|逛一逛|逛逛|逛遍|逛)([^，,。；;！？!\n]*)",
    )
    target_text = None
    for pattern in target_patterns:
        for match in re.finditer(pattern, text):
            if _tour_clause_is_negated(text, match.start()):
                continue
            target_text = match.group(1)
            break
        if target_text is not None:
            break
    if target_text is None:
        return [], []

    target_text = re.split(
        r"[，,。；;！？!\n]|路线|怎么走|多久|大概|预计",
        target_text, maxsplit=1,
    )[0]
    fragments = re.split(r"(?:以及|和|与|及|、)", target_text)
    if len(fragments) < 2:
        return [], []

    names, missing = [], []
    seen = set()
    for fragment in fragments:
        candidate = re.sub(
            r"^\s*(?:(?:一下|一趟|一圈|一条|一段|我要|我想要|我想|帮我|请|带我|给我|"
            r"规划|安排|推荐|设计|生成|经过|途经|途径|去|到)\s*)+",
            "", fragment,
        ).strip()
        candidate = re.sub(r"(?:附近|周边|一带|沿线|景区|景点|看看|游览|游玩|等)+\s*$", "", candidate).strip()
        candidate = re.sub(r"^(?:武汉大学|武大)(?:的)?", "", candidate).strip()
        if not candidate:
            continue
        poi, _ = find_poi_ambiguous(candidate)
        if not poi:
            missing.append(candidate)
            continue
        if poi.get("name") in excluded:
            continue
        identity = poi.get("id") or poi.get("name")
        if identity not in seen:
            names.append(poi.get("name", candidate))
            seen.add(identity)
    return names, missing


def _tool_plan_tour(args, ctx):
    context, snapshot_error = _prepare_route_context(ctx)
    if snapshot_error:
        return snapshot_error, None
    G, G_mode, mode, weather_info = _plan_common(args, context)

    theme = (args.get("theme") or "scenery").strip()
    theme_map = {
        "sakura": {"subcategory": "sakura"},
        "lake": {"subcategory": "lake"},
        "landmark": {"subcategory": "landmark"},
        "scenery": {"poi_type": "scenery"},
    }
    sel = theme_map.get(theme, {"poi_type": "scenery"})
    max_pois = max(2, min(int(args.get("max_pois", _TOUR_DEFAULT_MAX_POIS) or _TOUR_DEFAULT_MAX_POIS), 8))
    query = context.get("query")
    excluded_names = _infer_excluded_tour_poi_names(query)
    inferred_names, inferred_missing = _infer_explicit_tour_poi_names(query)
    requested_names = [str(name).strip() for name in (args.get("poi_names") or []) if str(name).strip()]
    if inferred_names or inferred_missing:
        requested_names = inferred_names
        if inferred_missing:
            return {
                "error": "tour_poi_not_found",
                "message": "找不到这些指定游览地点：" + "、".join(inferred_missing)
                           + "。我不会改成普通校园景点环线。",
                "missing_pois": inferred_missing,
            }, None
    requested_names = [
        name for name in requested_names
        if (find_poi_ambiguous(name)[0] or {}).get("name", name) not in excluded_names
    ]
    if len(requested_names) > 8:
        return {
            "error": "too_many_tour_stops",
            "message": "一次游览最多安排 8 个指定地点，请删减后再规划。",
        }, None
    if requested_names:
        pois = []
        missing_names = []
        alternatives = {}
        seen_pois = set()
        for name in requested_names[:8]:
            poi, matches = find_poi_ambiguous(name)
            if poi:
                identity = poi.get("id") or poi.get("name")
                if identity not in seen_pois:
                    pois.append(poi)
                    seen_pois.add(identity)
            else:
                missing_names.append(name)
                if matches:
                    alternatives[name] = [match.get("name", "") for match in matches[:5]]
        if missing_names:
            return {
                "error": "tour_poi_not_found",
                "message": "找不到这些指定地点：" + "、".join(missing_names)
                           + "。我不会用其他景点替代，请确认地点名称。",
                "missing_pois": missing_names,
                "alternatives": alternatives,
            }, None
    else:
        pois = search_by_category(
            season=args.get("season"), include_minor=False,
            limit=min(max_pois + len(excluded_names), 24), **sel,
        )
        pois = [poi for poi in pois if poi.get("name") not in excluded_names][:max_pois]
    if not requested_names and theme != "scenery" and len(pois) < 2:
        # 主题点太少时并入全校风景点
        additional = search_by_category(
            poi_type="scenery", season=args.get("season"), include_minor=False,
            limit=min(max_pois + len(excluded_names), 24),
        )
        pois = [poi for poi in pois + additional if poi.get("name") not in excluded_names][:max_pois]
    if len(pois) < 2:
        if requested_names:
            return {
                "error": "not_enough_tour_stops",
                "message": "指定地点少于两个不同地点，暂时无法生成游览路线。",
                "requested_pois": requested_names,
            }, None
        return {"error": "no_tour_pois", "message": "可游览的校内景点不足，换个主题试试"}, None

    # 起点：可选；默认以重要度最高的点为锚
    start_node, start_name = None, None
    start_access = None
    start_ref = args.get("start")
    if start_ref:
        start_node, start_name, err = _resolve_endpoint(start_ref, G_mode)
        if err:
            return err, None
        start_access = _endpoint_access_evidence(start_ref, start_node, G_mode, mode)

    poi_nodes = []
    for p in pois:
        try:
            lng_wgs, lat_wgs = gcj02_to_wgs84(p["lon"], p["lat"])
            node, access, snap_error = _snap_wgs_to_network(
                lng_wgs, lat_wgs, G_mode, p.get("name", "游览地点"),
            )
            if snap_error:
                continue
        except Exception:
            continue
        p2 = dict(p)
        p2["importance"] = importance_score(p)
        p2["_endpoint_access"] = access
        poi_nodes.append((p2, node))

    loop = bool(args.get("loop", True)) if start_node is None else bool(args.get("loop", True))
    decision = _strategy_for_args(args, ctx)
    result = compute_tour_route(
        G, poi_nodes, start_node=start_node, loop=loop,
        constraints=args.get("constraints") or {}, weights=decision.weights,
        mode=mode, max_total_m=_TOUR_MAX_TOTAL_M, weather_info=weather_info,
        road_conditions=context["_route_conditions_snapshot"],
        strategy_name=decision.name, detour_cap=decision.detour_cap,
    )
    if requested_names:
        required_names = {poi.get("name") for poi in pois}
        routed_names = {poi.get("name") for poi in result.get("ordered_pois", [])}
        omitted_names = sorted(name for name in required_names if name not in routed_names)
        if omitted_names:
            return {
                "error": "tour_stops_unroutable",
                "message": "指定地点无法全部纳入同一条可行路线：" + "、".join(omitted_names)
                           + "。我不会静默删掉这些地点。",
                "missing_pois": omitted_names,
            }, None
    if not result["legs"]:
        return {"error": "tour_failed", "message": "这些景点之间暂时无法连通，换一批试试"}, None

    legs_payload = []
    seq_names = [start_name] if start_name else []
    seq_names += [p["name"] for p in result["ordered_pois"]]
    for i, leg in enumerate(result["legs"]):
        a = seq_names[i] if i < len(seq_names) else f"第{i + 1}站"
        b = seq_names[i + 1] if i + 1 < len(seq_names) else seq_names[0]
        legs_payload.append(_route_payload(G, leg, a, b, mode))

    ordered_public = [_poi_public(p) for p in result["ordered_pois"]]
    for public_poi, source_poi in zip(ordered_public, result["ordered_pois"]):
        public_poi["endpoint_access"] = source_poi.get("_endpoint_access")
    payload = {
        "tour": {
            "theme": theme,
            "ordered_pois": ordered_public,
            "loop": result["loop"],
            "legs": legs_payload,
            "total_length_m": result["total_length_m"],
            "duration_min": round(estimate_duration_min(result["total_length_m"], mode), 1),
            "dropped": [p.get("name", "") for p in result["dropped"]],
            "mode": mode,
        },
        # 前端兼容：整条环线作为 recommended 返回
        "start_name": seq_names[0] if seq_names else "",
        "recommended": [c for leg in legs_payload for c in leg["recommended"]],
        "recommended_edge_ids": [
            edge for leg in legs_payload for edge in leg["recommended_edge_ids"]
        ],
        "steps": _merge_leg_steps(legs_payload),
        "recommended_length_m": result["total_length_m"],
        "distance_m": result["total_length_m"],
        "strategy": decision.as_dict(),
        "endpoint_access": {
            "start": start_access,
            "places": [p.get("_endpoint_access") for p in result["ordered_pois"]],
            "max_snap_m": _MAX_ENDPOINT_ACCESS_SNAP_M,
        },
    }
    state_pois = [{"name": p["name"], "type": "poi"} for p in ordered_public]
    state_start = args.get("start") or state_pois[0]
    state_end = state_start if result["loop"] else state_pois[-1]
    _attach_route_state(
        payload, "tour", args, context, G, decision,
        state_start, state_end,
        tour={"theme": theme, "pois": state_pois, "loop": result["loop"]},
        travel_mode=mode,
    )
    return payload, {"route": payload, "route_kind": "tour"}


def _tool_get_weather(args, ctx):
    pub = _weather_public(_weather_snapshot())
    if not pub:
        return {"error": "weather_unavailable", "message": "天气暂时获取不到"}, None
    return pub, None


def _tool_list_road_conditions(args, ctx):
    try:
        conds = list_conditions(strict=True)
    except RoadConditionsUnavailableError:
        return {
            "error": "road_conditions_unavailable",
            "message": "当前路况数据暂不可用，不能确认是否存在管制信息；请稍后再试。",
        }, None
    out = []
    for c in conds:
        road = (c.get("edge") or {}).get("road_name", "")
        action = c.get("action")
        source = c.get("source") or {}
        audit = c.get("audit") or []
        observation_time = source.get("captured_at") or c.get("created_at")
        out.append({
            "id": c.get("id"),
            "name": c.get("name", ""),
            "type_label": CONDITION_LABELS.get(c.get("type"), c.get("type", "")),
            "action": action or "legacy_rule",
            "action_label": {"notice": "提示", "slowdown": "缓行", "closure": "禁止通行"}.get(
                action, "按原有事件规则"
            ),
            "affected_modes": c.get("affected_modes") or [],
            "cost_multiplier": c.get("cost_multiplier"),
            "road_name": road,
            "location": (road + "路段") if road else "",
            "description": (c.get("description") or "")[:100],
            "observation_time": observation_time,
            "confirmed_time": c.get("created_at") or next((
                item.get("at") for item in reversed(audit)
                if item.get("action") in {"created", "confirmed"}
            ), None),
            "start_time": c.get("start_time"),
            "end_time": c.get("end_time") or None,
            "status": "confirmed_active",
            "source_kind": (c.get("source") or {}).get("kind", "manager_record"),
            "confirmation_note": source.get("field_confirmation") or "",
            "route_effect": {
                "action": action or "legacy_rule",
                "affected_modes": c.get("affected_modes") or [],
                "cost_multiplier": c.get("cost_multiplier"),
            },
        })
    return {"conditions": out, "count": len(out),
            "message": "当前没有生效的路况信息" if not out else ""}, None


def _tool_ask_user(args, ctx):
    question = (args.get("question") or "").strip() or "能再说得具体一点吗？"
    options = [str(o)[:30] for o in (args.get("options") or [])][:4]
    return {"question": question, "options": options}, {"clarify": {"question": question, "options": options}}


def _tool_suggest_followup(args, ctx):
    raw = args.get("suggestions") or []
    suggestions = []
    for s in raw[:3]:
        if not isinstance(s, dict):
            continue
        label = str(s.get("label", ""))[:10]
        query = str(s.get("query", ""))[:200]
        if label and query:
            suggestions.append({"label": label, "query": query})
    if not suggestions:
        return {"suggestions": []}, None
    return {"suggestions": suggestions}, {"suggestions": suggestions}


_EXECUTORS = {
    "resolve_poi": _tool_resolve_poi,
    "get_place_details": _tool_get_place_details,
    "search_poi_candidates": _tool_search_poi_candidates,
    "find_reachable_places": _tool_find_reachable_places,
    "compare_routes": _tool_compare_routes,
    "plan_route": _tool_plan_route,
    "plan_via_route": _tool_plan_via_route,
    "plan_itinerary": _tool_plan_itinerary,
    "plan_multimodal_route": _tool_plan_multimodal_route,
    "plan_tour": _tool_plan_tour,
    "get_weather": _tool_get_weather,
    "list_road_conditions": _tool_list_road_conditions,
    "ask_user": _tool_ask_user,
    "suggest_followup": _tool_suggest_followup,
}


def execute_tool(name: str, args: dict, ctx: dict = None) -> tuple:
    """执行工具。返回 (result_for_llm, artifact_or_None)。

    artifact 用于 planner 组装最终响应（route / candidates / clarify）。
    任何内部异常都转为 error dict 回注，不抛出。
    """
    executor = _EXECUTORS.get(name)
    if executor is None:
        return {"error": "unknown_tool", "message": f"工具 {name} 不存在"}, None
    if not isinstance(args, dict):
        args = {}
    started = time.perf_counter()
    try:
        result, artifact = executor(args, ctx)
        route = (artifact or {}).get("route") if isinstance(artifact, dict) else None
        if isinstance(route, dict):
            timings = route.get("timings_ms")
            if not isinstance(timings, dict):
                leg_timings = []
                for leg in route.get("legs") or []:
                    if isinstance(leg, dict):
                        leg_timings.append(leg.get("timings_ms"))
                tour = route.get("tour")
                if isinstance(tour, dict):
                    for leg in tour.get("legs") or []:
                        if isinstance(leg, dict):
                            leg_timings.append(leg.get("timings_ms"))
                timings = add_timings(*leg_timings)
            timings = normalize_timings(timings)
            measured = sum(timings[key] for key in timings if key != "agent")
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings["response_build"] = round(
                timings["response_build"] + max(0.0, elapsed_ms - measured), 3
            )
            route["timings_ms"] = normalize_timings(timings)
            if isinstance(result, dict):
                result["timings_ms"] = route["timings_ms"]
        return result, artifact
    except RoadConditionsUnavailableError:
        logger.exception("工具 %s 无法将当前路况安全应用到路网", name)
        return {
            "error": "road_conditions_unavailable",
            "message": "当前管制信息无法与路网对应，暂时不能可靠规划路线。",
        }, None
    except Exception as e:
        logger.exception("工具 %s 执行异常", name)
        return {"error": "tool_exception", "message": f"{type(e).__name__}: {e}"}, None


def result_to_json(result: dict, max_len: int = 4000) -> str:
    """工具结果压缩为 LLM 可读的 JSON 文本（截断路径坐标等大体量字段）。

    路径坐标对 LLM 决策无用且极占 token，只保留统计信息；
    完整坐标由 planner 从 artifact 里取，直接进 API 响应。
    """
    compact = {}
    for k, v in result.items():
        if k in ("recommended", "shortest"):
            compact[k] = f"[{len(v)} 个坐标点]" if isinstance(v, list) else v
        elif k == "steps":
            compact[k] = f"[{len(v)} 条转向指令]" if isinstance(v, list) else v
        elif k == "legs":
            compact[k] = [
                {kk: (f"[{len(vv)} 个坐标点]" if kk in ("recommended", "shortest") and isinstance(vv, list)
                      else (f"[{len(vv)} 条转向指令]" if kk == "steps" and isinstance(vv, list)
                            else ("[略]" if kk == "pois" else vv)))
                 for kk, vv in leg.items()}
                for leg in v
            ] if isinstance(v, list) else v
        elif k == "pois" and isinstance(v, list):
            compact[k] = [p.get("name", "") for p in v][:10]
        else:
            compact[k] = v
    text = json.dumps(compact, ensure_ascii=False)
    return text[:max_len]
