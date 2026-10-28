"""
特殊路况管理模块（边绑定模型）

路况事件绑定到具体路网边，而不是"一个点 + 影响半径"的圆：
  - 管理员在地图上点出事发位置，snap_to_edge() 服务端吸附到最近路段
  - 事件持久化边标识（OSM 节点 u/v/key）+ 吸附点 + 道路名 + 边几何快照
  - 规划时按 (u, v, key) 精确命中（有向图双向同时生效）
  - 路网重新下载导致节点 id 失效时，用吸附点几何回退（12m 容差）
  - 更早的 radius_m 圆模型数据仍可读取，按旧半径几何回退

所有路况类型默认对步行、骑行、驾车硬封路。管理者可为现场明确确认仍可通行的方式
取消禁行；路线规划只消费 blocked_modes，不根据事件类型猜测现场通行情况。

数据默认在 data/road_conditions.json；生产由 ROAD_CONDITIONS_FILE 指向运行目录。
坐标全部 GCJ-02（与 POI/前端一致）。
"""

import json
import logging
import math
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

_CONDITIONS_FILE = os.path.abspath(os.getenv("ROAD_CONDITIONS_FILE") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "road_conditions.json",
))

# "block" = 硬禁止；浮点数 = 边成本惩罚倍数
# 设计依据：边链扩展后惩罚范围为"两个路口之间整段道路"(300-600m)，
# 倍数代表对人的真实感知影响——×1.5 ≈ 速度减半（不是×4那种"绕行"级惩罚）。
CONDITION_EFFECTS = {
    # A new manager restriction is fail-closed for all travel modes by default.
    # Managers may explicitly leave a confirmed-passable mode out of blocked_modes.
    # Legacy rows without blocked_modes also fail closed instead of relying on a
    # type label that cannot describe what the manager actually observed.
    "closure":      {"walk": "block", "bike": "block", "drive": "block"},
    "construction": {"walk": "block", "bike": "block", "drive": "block"},
    "flooding":     {"walk": "block", "bike": "block", "drive": "block"},
    "accident":     {"walk": "block", "bike": "block", "drive": "block"},
    "event":        {"walk": "block", "bike": "block", "drive": "block"},
}

TRAVEL_MODE_KEYS = ("walk", "bike", "drive")

CONDITION_LABELS = {
    "closure": "道路封闭",
    "construction": "施工",
    "event": "活动",
    "flooding": "积水",
    "accident": "事故",
}
CONDITION_ACTIONS = {"notice", "slowdown", "closure"}
TRAVEL_MODES = {"walk", "bike", "drive"}

# 管理员点击点离最近路段超过该距离（米）则拒绝吸附
SNAP_MAX_DIST_M = 30.0
# 边绑定数据在节点 id 失效时的几何回退容差
_EDGE_FALLBACK_DIST_M = 12.0
# When multiple source ways represent the same physical corridor, closing only
# the selected MultiDiGraph key can leave a duplicate route through that road.
# A production closure was bypassed by a parallel footway 11.7 m from its
# centerline, so the managed corridor includes aligned ways within 15 m. The
# longitudinal-overlap checks below continue to exclude perpendicular crossings.
_PARALLEL_CORRIDOR_DIST_M = 15.0
# 旧圆模型数据缺省半径
_LEGACY_DEFAULT_RADIUS_M = 30.0

# 边链扩展：一次管制影响的是"两个路口之间的整段道路"，而非单个 OSM edge。
# 从吸附边沿拓扑向两端延伸，直连中间节点（无向度=2）直接穿过，直到路口/断头。
_CHAIN_MAX_EDGES = 40        # 单条链最多合并的边数
_CHAIN_MAX_LENGTH_M = 600.0  # 单条链最长（米），校园一个街区足够
_CHAIN_MAX_TURN_DEG = 100.0  # degree=2 处近乎折返（>100°）视为不同道路，停止延伸

_lock = threading.Lock()
_write_lock = threading.Lock()
_cache = None  # list of condition dicts
_cache_mtime = 0.0
_cache_error = None


@contextmanager
def _condition_write_lock():
    """Serialize read-modify-write across the production web processes."""
    if (os.getenv("AGENT_REPLAY_EVENT_EPOCH") is not None
            or os.getenv("AGENT_REPLAY_EVENT_SHA256") is not None):
        raise RoadConditionsUnavailableError("模拟路况重放期间不能修改路况文件")
    directory = os.path.dirname(_CONDITIONS_FILE)
    os.makedirs(directory, exist_ok=True)
    with _write_lock, open(_CONDITIONS_FILE + ".lock", "a+b") as lock_file:
        if os.name == "nt":
            import msvcrt
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class RoadConditionsUnavailableError(RuntimeError):
    """Raised when route planning cannot safely read the road-condition store."""


class RoadConditionBindingError(RoadConditionsUnavailableError):
    """Raised when an active condition cannot be safely bound to the current graph."""


class VisionConditionAlreadyPublishedError(RuntimeError):
    """Raised when a reviewed image candidate already has a road-condition event."""

    def __init__(self, condition: dict):
        self.condition = deepcopy(condition)
        super().__init__("影像候选已发布为路况事件")


def _finite_number(value) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _valid_lon_lat(lng, lat) -> bool:
    return (_finite_number(lng) and _finite_number(lat)
            and -180 <= float(lng) <= 180 and -90 <= float(lat) <= 90)


def _valid_edge_binding(edge_info: dict) -> bool:
    if not isinstance(edge_info, dict):
        return False
    chain = edge_info.get("edges")
    if isinstance(chain, list) and chain:
        chain_valid = True
        for triple in chain:
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                chain_valid = False
                break
            try:
                int(triple[0]); int(triple[1]); int(triple[2])
            except (TypeError, ValueError):
                chain_valid = False
                break
        if chain_valid:
            return True
    if edge_info.get("u") is not None and edge_info.get("v") is not None:
        try:
            int(edge_info["u"]); int(edge_info["v"]); int(edge_info.get("key", 0))
            return True
        except (TypeError, ValueError):
            pass
    geometry = edge_info.get("geometry_gcj")
    if isinstance(geometry, list) and len(geometry) >= 2:
        if all(isinstance(point, (list, tuple)) and len(point) >= 2
               and _valid_lon_lat(point[0], point[1]) for point in geometry):
            return True
    snap = edge_info.get("snap") or {}
    return isinstance(snap, dict) and _valid_lon_lat(snap.get("lng"), snap.get("lat"))


def _valid_blocked_modes(value) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(mode, str) and mode in TRAVEL_MODE_KEYS for mode in value)
            and len(value) == len(set(value)))


def _validate_condition_record(condition: dict) -> None:
    """Validate persistent event shape before it can influence routing."""
    if condition.get("type") not in CONDITION_EFFECTS:
        raise ValueError("路况事件类型无效")
    if "blocked_modes" in condition:
        blocked_modes = condition.get("blocked_modes")
        if not _valid_blocked_modes(blocked_modes):
            raise ValueError("路况事件必须明确至少一种禁行方式")
    if "action" in condition:
        if condition.get("action") not in CONDITION_ACTIONS:
            raise ValueError("路况影响动作无效")
        modes = condition.get("affected_modes")
        if (not isinstance(modes, list) or not modes
                or len(modes) != len(set(modes))
                or any(mode not in TRAVEL_MODES for mode in modes)):
            raise ValueError("受影响的出行方式无效")
        if condition["action"] == "slowdown":
            factor = condition.get("cost_multiplier")
            if not _finite_number(factor) or not 1.0 <= float(factor) <= 5.0:
                raise ValueError("缓行成本倍数必须在 1 到 5 之间")
    start = condition.get("start_time", 0) or 0
    end = condition.get("end_time", 0) or 0
    if not _finite_number(start) or not _finite_number(end):
        raise ValueError("路况事件时间无效")
    start, end = float(start), float(end)
    if start < 0 or end < 0 or (end and start and end <= start):
        raise ValueError("路况事件时间范围无效")

    edge_ok = _valid_edge_binding(condition.get("edge"))
    coords = condition.get("coordinates")
    coords_ok = isinstance(coords, dict) and _valid_lon_lat(coords.get("lng"), coords.get("lat"))
    if coords_ok:
        radius = condition.get("radius_m", _LEGACY_DEFAULT_RADIUS_M)
        coords_ok = _finite_number(radius) and 0 < float(radius) <= 1000
    if not edge_ok and not coords_ok:
        raise ValueError("路况事件缺少有效道路绑定")


# ===================== 持久化 =====================

def _load_conditions(*, strict: bool = False) -> list:
    """从 JSON 文件加载路况事件，带 mtime 缓存。"""
    global _cache, _cache_mtime, _cache_error
    with _lock:
        try:
            mtime = os.stat(_CONDITIONS_FILE).st_mtime_ns
        except FileNotFoundError as exc:
            mtime = 0.0
            if strict and os.getenv("ROAD_CONDITIONS_FILE"):
                raise RoadConditionsUnavailableError("已配置的路况文件不存在，无法确认当前封路信息") from exc
            if strict and _cache is not None and _cache_mtime > 0:
                raise RoadConditionsUnavailableError("路况文件丢失，无法确认当前封路信息") from exc
            if strict:
                _cache = []
                _cache_mtime = 0.0
                _cache_error = None
                return []
        except OSError as exc:
            if strict:
                raise RoadConditionsUnavailableError("无法读取路况文件状态") from exc
            mtime = 0.0
        if _cache is not None and mtime == _cache_mtime and not (strict and _cache_error):
            if strict:
                try:
                    for item in _cache:
                        if not isinstance(item, dict):
                            raise ValueError("路况文件必须是对象数组")
                        _validate_condition_record(item)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise RoadConditionsUnavailableError("路况文件无效，无法确认当前封路信息") from exc
            return deepcopy(_cache)
        try:
            with open(_CONDITIONS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list) or any(not isinstance(item, dict) for item in loaded):
                raise ValueError("路况文件必须是对象数组")
            if strict:
                for item in loaded:
                    _validate_condition_record(item)
            _cache = loaded
            _cache_error = None
        except (OSError, json.JSONDecodeError, ValueError) as e:
            if strict:
                raise RoadConditionsUnavailableError("路况文件无效，无法确认当前封路信息") from e
            logger.warning("路况文件加载失败，使用空列表: %s", e)
            _cache = []
            _cache_error = "路况文件无效，无法确认当前封路信息"
        _cache_mtime = mtime
        return deepcopy(_cache)


def _save_conditions(conditions: list) -> None:
    """Replace a complete JSON snapshot; a failed write preserves the old file."""
    global _cache, _cache_mtime, _cache_error
    for condition in conditions:
        _validate_condition_record(condition)
    directory = os.path.dirname(_CONDITIONS_FILE)
    os.makedirs(directory, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         prefix=".road-conditions-", suffix=".json",
                                         delete=False) as stream:
            temp_path = stream.name
            json.dump(conditions, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers can temporarily hold the destination without delete
        # sharing. Keep the atomic replacement and retry briefly; persistent
        # permission errors still propagate without modifying the old snapshot.
        for attempt in range(6):
            try:
                os.replace(temp_path, _CONDITIONS_FILE)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 5:
                    raise
                time.sleep(0.05 * (attempt + 1))
        temp_path = None
        with _lock:
            _cache = deepcopy(conditions)
            _cache_mtime = os.stat(_CONDITIONS_FILE).st_mtime_ns
            _cache_error = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def list_conditions(include_inactive: bool = False, *, strict: bool = False) -> list:
    """返回路况事件。

    生效判定：start_time <= now < end_time（end_time 为 0 表示长期有效）。
    include_inactive=True 时返回全部事件（含未开始/已过期），供管理端展示。
    """
    from experiments.event_replay_guard import certified_event_snapshot

    try:
        replay_snapshot = certified_event_snapshot(_CONDITIONS_FILE)
    except (OSError, ValueError) as exc:
        raise RoadConditionsUnavailableError(
            "模拟路况重放配置无效，无法确认当前路况"
        ) from exc
    conditions = (replay_snapshot[1] if replay_snapshot is not None
                  else _load_conditions(strict=strict))
    if include_inactive:
        return list(conditions)
    now = replay_snapshot[0] if replay_snapshot is not None else time.time()
    active = []
    for c in conditions:
        if c.get("revoked_at"):
            continue
        start = c.get("start_time", 0) or 0
        end = c.get("end_time", 0) or 0
        if start and now < start:
            continue  # 尚未开始（如预录的樱花节管制）
        if end and now >= end:
            continue  # 已过期
        active.append(c)
    return active


def purge_revoked_conditions() -> int:
    """Permanently remove old soft-deleted records left by earlier releases."""
    with _condition_write_lock():
        conditions = _load_conditions(strict=True)
        retained = [condition for condition in conditions if not condition.get("revoked_at")]
        removed_count = len(conditions) - len(retained)
        if removed_count:
            _save_conditions(retained)
    return removed_count


def add_condition(
    cond_type: str,
    name: str,
    edge: dict,
    click_point: Optional[dict] = None,
    description: str = "",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    created_by: str = "web",
    source: Optional[dict] = None,
    blocked_modes: Optional[list] = None,
    action: Optional[str] = None,
    affected_modes: Optional[list[str]] = None,
    cost_multiplier: Optional[float] = None,
) -> dict:
    """
    添加一个绑定到具体路段的路况事件。

    Args:
        cond_type: 事件类型（closure/construction/event/flooding/accident）
        name: 事件名称
        edge: snap_to_edge() 的返回结果（含 u/v/key/road_name/snap/geometry）
        click_point: 管理员原始点击坐标 {"lng","lat"}（GCJ-02），审计用
        description: 描述
        start_time/end_time: 生效时间戳（秒），None = 立即生效/长期有效
        created_by: 录入来源（web / token:保卫部）

    Returns:
        新创建的事件 dict
    """
    if cond_type not in CONDITION_EFFECTS:
        raise ValueError(f"未知路况类型: {cond_type}")
    if action is not None:
        if action not in CONDITION_ACTIONS:
            raise ValueError("路况影响动作无效")
        if (not isinstance(affected_modes, list) or not affected_modes
                or len(affected_modes) != len(set(affected_modes))
                or any(mode not in TRAVEL_MODE_KEYS for mode in affected_modes)):
            raise ValueError("受影响的出行方式无效")
        if action == "slowdown" and (
                not _finite_number(cost_multiplier)
                or not 1.0 <= float(cost_multiplier) <= 5.0):
            raise ValueError("缓行成本倍数必须在 1 到 5 之间")
    else:
        if blocked_modes is None:
            blocked_modes = list(TRAVEL_MODE_KEYS)
        if not _valid_blocked_modes(blocked_modes):
            raise ValueError("路况事件必须明确至少一种禁行方式")
    if not edge or edge.get("u") is None or edge.get("v") is None:
        raise ValueError("缺少绑定路段信息 edge（u/v）")

    now = time.time()
    condition = {
        "id": str(uuid.uuid4())[:8],
        "type": cond_type,
        **({
            "action": action,
            "affected_modes": list(affected_modes),
            "cost_multiplier": float(cost_multiplier) if action == "slowdown" else None,
        } if action else {"blocked_modes": list(blocked_modes)}),
        "name": name,
        "description": description or "",
        "edge": {
            "u": int(edge["u"]),
            "v": int(edge["v"]),
            "key": int(edge.get("key", 0)),
            # 完整边链（路口到路口）；旧数据无此字段时由 u/v 单条边兜底
            "edges": edge.get("edges") or [
                [int(edge["u"]), int(edge["v"]), int(edge.get("key", 0))]
            ],
            "chain_length_m": edge.get("chain_length_m", 0.0),
            "road_name": edge.get("road_name") or "",
            "snap": {
                "lng": float(edge["snap_lng_gcj"]),
                "lat": float(edge["snap_lat_gcj"]),
            },
            "geometry_gcj": edge.get("geometry_gcj") or [],
            "snap_dist_m": round(float(edge.get("dist_m", 0.0)), 1),
        },
        "click_point": click_point or {},
        "start_time": start_time if start_time is not None else now,
        "end_time": end_time if end_time is not None else 0,  # 0 = 长期有效
        "created_by": created_by,
        "created_at": now,
        "updated_at": now,
        "audit": [{"action": "created", "actor": created_by, "at": now}],
    }
    if source:
        condition["source"] = deepcopy(source)
    if action is not None:
        condition["action"] = action
        condition["affected_modes"] = list(affected_modes or [])
        if cost_multiplier is not None:
            condition["cost_multiplier"] = float(cost_multiplier)
    with _condition_write_lock():
        conditions = _load_conditions(strict=True)
        source_kind = (source or {}).get("kind")
        source_job_id = (source or {}).get("job_id")
        candidate_index = (source or {}).get("candidate_index")
        if source_kind == "vision_job" and candidate_index is not None:
            existing = next((item for item in conditions
                             if (item.get("source") or {}).get("kind") == "vision_job"
                             and (item.get("source") or {}).get("job_id") == source_job_id
                             and (item.get("source") or {}).get("candidate_index") == candidate_index),
                            None)
            if existing is not None:
                raise VisionConditionAlreadyPublishedError(existing)
        conditions.append(condition)
        _save_conditions(conditions)
    logger.info(
        "新增路况: %s (%s) edge=(%s,%s,k%s) by=%s",
        name, cond_type, condition["edge"]["u"], condition["edge"]["v"],
        condition["edge"]["key"], created_by,
    )
    return condition


def remove_condition(cond_id: str, *, actor: str = "web") -> bool:
    """Permanently remove an event from the active condition store."""
    with _condition_write_lock():
        conditions = _load_conditions(strict=True)
        retained = [item for item in conditions if item.get("id") != cond_id]
        if len(retained) == len(conditions):
            return False
        _save_conditions(retained)
    logger.info("永久删除路况事件: %s by=%s", cond_id, actor)
    return True


def update_condition(cond_id: str, changes: dict, *, actor: str = "web",
                     action: str = "updated") -> Optional[dict]:
    """更新事件字段（name/description/start_time/end_time），返回更新后的事件。

    特殊用法：changes={"end_time": 0} 由调用方把"立即结束"翻译成当前时间戳传入。
    """
    with _condition_write_lock():
        conditions = _load_conditions(strict=True)
        target = next((item for item in conditions if item.get("id") == cond_id), None)
        if target is None or target.get("revoked_at"):
            return None
        applied = {}
        for field in (
            "name", "description", "start_time", "end_time", "action",
            "affected_modes", "cost_multiplier",
        ):
            if field in changes and changes[field] is not None:
                target[field] = changes[field]
                applied[field] = changes[field]
        now = time.time()
        target["updated_at"] = now
        target.setdefault("audit", []).append({
            "action": action, "actor": actor, "at": now, "changes": applied,
        })
        _save_conditions(conditions)
    logger.info("更新路况事件 %s: %s", cond_id, list(changes.keys()))
    return target


def rebind_vision_condition_edge(cond_id: str, edge: dict, *,
                                 field_confirmation: str,
                                 actor: str = "web") -> Optional[dict]:
    """Correct the road association of a published vision-derived event, with audit history."""
    if not isinstance(edge, dict) or edge.get("u") is None or edge.get("v") is None:
        raise ValueError("缺少新的路段绑定信息")

    def integer_id(value, label):
        if not _finite_number(value) or not float(value).is_integer():
            raise ValueError(f"新的道路{label}无效")
        return int(value)

    u, v = integer_id(edge.get("u"), "起点"), integer_id(edge.get("v"), "终点")
    key = integer_id(edge.get("key", 0), "边标识")
    snap_lng, snap_lat, snap_dist = (
        edge.get("snap_lng_gcj"), edge.get("snap_lat_gcj"), edge.get("dist_m"),
    )
    if (not _valid_lon_lat(snap_lng, snap_lat)
            or not 70 <= float(snap_lng) <= 140 or not 0 <= float(snap_lat) <= 60
            or not _finite_number(snap_dist)
            or not 0 <= float(snap_dist) <= SNAP_MAX_DIST_M):
        raise ValueError("新的道路吸附结果无效")
    chain_length = edge.get("chain_length_m", 0)
    if not _finite_number(chain_length) or float(chain_length) < 0:
        raise ValueError("新的道路长度无效")
    raw_chain = edge.get("edges")
    if raw_chain is None:
        raw_chain = [[u, v, key]]
    if not isinstance(raw_chain, list) or not raw_chain:
        raise ValueError("新的路段边链无效")
    chain = []
    for triple in raw_chain:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            raise ValueError("新的路段边链无效")
        chain.append([integer_id(triple[0], "边链起点"),
                      integer_id(triple[1], "边链终点"),
                      integer_id(triple[2], "边链标识")])
    geometry = edge.get("geometry_gcj")
    if (not isinstance(geometry, list) or len(geometry) < 2
            or not all(isinstance(point, (list, tuple)) and len(point) >= 2
                       and _valid_lon_lat(point[0], point[1])
                       for point in geometry)):
        raise ValueError("新的道路几何无效")
    reason = str(field_confirmation or "").strip()
    if not 8 <= len(reason) <= 300:
        raise ValueError("道路更正依据须为 8 到 300 字")

    new_binding = {
        "u": u,
        "v": v,
        "key": key,
        "edges": chain,
        "chain_length_m": float(chain_length),
        "road_name": str(edge.get("road_name") or ""),
        "snap": {"lng": float(snap_lng), "lat": float(snap_lat)},
        "geometry_gcj": [[float(point[0]), float(point[1])] for point in geometry],
        "snap_dist_m": round(float(snap_dist), 1),
    }
    associated_edge = {
        "u": new_binding["u"], "v": new_binding["v"], "key": new_binding["key"],
        "road_name": new_binding["road_name"], "snap_dist_m": new_binding["snap_dist_m"],
    }

    with _condition_write_lock():
        conditions = _load_conditions(strict=True)
        target = next((item for item in conditions if item.get("id") == cond_id), None)
        if target is None or target.get("revoked_at"):
            return None
        source = target.get("source") or {}
        if source.get("kind") != "vision_job":
            raise ValueError("只有影像识别来源的事件可以更正绑定道路")
        previous_edge = target.get("edge") or {}
        previous_association = {
            "road_name": str(previous_edge.get("road_name") or ""),
            "u": previous_edge.get("u"), "v": previous_edge.get("v"),
            "key": previous_edge.get("key", 0),
        }
        now = time.time()
        history = source.get("association_history")
        history = list(history) if isinstance(history, list) else []
        history.append({
            "associated_edge": previous_association,
            "field_confirmation": source.get("field_confirmation", ""),
            "at": now,
        })
        source = deepcopy(source)
        source["association_history"] = history[-10:]
        source["associated_edge"] = associated_edge
        source["field_confirmation"] = reason
        source["road_association"] = "manager_corrected_edge"
        target["source"] = source
        target["edge"] = new_binding
        target["click_point"] = {
            "lng": float(edge["snap_lng_gcj"]), "lat": float(edge["snap_lat_gcj"]),
        }
        target["updated_at"] = now
        target.setdefault("audit", []).append({
            "action": "vision_road_rebound", "actor": actor, "at": now,
            "changes": {
                "previous_edge": previous_association,
                "new_edge": associated_edge,
                "field_confirmation": reason,
            },
        })
        _save_conditions(conditions)
    logger.info("更正影像路况事件 %s 的道路绑定: %s -> %s by=%s",
                cond_id, previous_association["road_name"], associated_edge["road_name"], actor)
    return target


# ===================== 路网吸附 =====================

def _edge_line(G, u, v, data):
    """取边的 shapely LineString（WGS-84）；geometry 可能是 LineString 或 WKT 字符串。"""
    from shapely.geometry import LineString
    from shapely import wkt
    geom = data.get("geometry")
    if geom is None:
        ud, vd = G.nodes[u], G.nodes[v]
        return LineString([
            (float(ud.get("x", 0)), float(ud.get("y", 0))),
            (float(vd.get("x", 0)), float(vd.get("y", 0))),
        ])
    if hasattr(geom, "coords"):
        return geom
    if isinstance(geom, str) and geom.strip():
        try:
            return wkt.loads(geom)
        except Exception:
            pass
    ud, vd = G.nodes[u], G.nodes[v]
    return LineString([
        (float(ud.get("x", 0)), float(ud.get("y", 0))),
        (float(vd.get("x", 0)), float(vd.get("y", 0))),
    ])


def _line_coords(line) -> list:
    """LineString → [[lng, lat], ...]（WGS-84）。"""
    try:
        return [[float(x), float(y)] for x, y in line.coords]
    except Exception:
        return []


def _road_name_of(edge_data: dict) -> str:
    """从边属性取规范道路名（无名路返回 ''）。"""
    raw = edge_data.get("name")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, list) and raw:
        return str(raw[0]).strip()
    return ""


def _undirected_degree(G, node) -> int:
    """节点的无向连接度：双向平行边只算一个邻居。==2 表示道路直连点，可穿过。"""
    return len(set(G.successors(node)) | set(G.predecessors(node)))


def _bearing_deg(p1, p2) -> float:
    import math
    dx = (p2[0] - p1[0]) * math.cos(math.radians((p1[1] + p2[1]) / 2.0))
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _edge_end_bearing(coords: list, forward: bool) -> float:
    """边几何在起点(forward=True)/终点(forward=False)端的切线方位角。"""
    if len(coords) < 2:
        return 0.0
    if forward:
        return _bearing_deg(coords[0], coords[1])
    return _bearing_deg(coords[-2], coords[-1])


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _walk_one_direction(G, cur, prev, heading, road_name, budget_edges, budget_m):
    """沿 cur 端单向延伸（道路为双向 MultiDiGraph，用出边即可）。

    停止条件：到达路口/断头（无向度≠2）、边数/长度预算耗尽、
    或直连点处道路名变化 / 近乎折返（防止穿到匝道、另一条路上）。

    Returns:
        keys: [(a,b,k), ...] 沿延伸方向的有向边
        pts:  [[lng,lat], ...] 从 cur 节点开始（含 cur）的拼接坐标
        length_m: 新增边总长度
    """
    keys, pts, total = [], [], 0.0
    while budget_edges > 0 and budget_m > 0:
        if _undirected_degree(G, cur) != 2:
            break  # 路口、丁字、断头：管制段到此为止
        candidates = []
        for _, nxt, k, d in G.out_edges(cur, keys=True, data=True):
            if nxt == prev:
                continue
            cs = _line_coords(_edge_line(G, cur, nxt, d))
            if len(cs) < 2:
                continue
            turn = _angle_diff(heading, _edge_end_bearing(cs, True))
            name = _road_name_of(d)
            length = float(d.get("length", 0) or 0)
            candidates.append((turn, nxt, k, cs, name, length))
        if not candidates:
            break
        candidates.sort(key=lambda c: c[0])
        turn, nxt, k, cs, name, length = candidates[0]
        if turn > _CHAIN_MAX_TURN_DEG:
            break
        if road_name and name and name != road_name:
            break
        if length <= 0 or length > budget_m:
            break
        keys.append((cur, nxt, k))
        if not pts:
            pts.append(cs[0])
        pts.extend(cs[1:])
        total += length
        budget_edges -= 1
        budget_m -= length
        prev, cur = cur, nxt
        heading = _edge_end_bearing(cs, False)
        if name:
            road_name = name
    return keys, pts, total


def _expand_edge_chain(G, u, v, k):
    """把吸附边 (u,v,k) 扩展为"路口到路口"的完整道路链。

    Returns:
        ordered_keys: [(a,b,k), ...] 沿几何方向（道路一端→另一端）的有向边
        coords_wgs:   [[lng,lat], ...] 合并去重后的整链几何（WGS-84）
        length_m:     整链长度
        edge_count:   合并边数
    """
    main_data = G.get_edge_data(u, v, k) or {}
    main_coords = _line_coords(_edge_line(G, u, v, main_data))
    main_len = float(main_data.get("length", 0) or 0)
    main_name = _road_name_of(main_data)
    budget_e = _CHAIN_MAX_EDGES - 1
    budget_m = _CHAIN_MAX_LENGTH_M - main_len

    # 前向：v 端继续；后向：u 端逆推（沿出边走向 v 的反方向）
    fwd_keys, fwd_pts, fwd_len = _walk_one_direction(
        G, v, u, _edge_end_bearing(main_coords, False), main_name,
        budget_e // 2, budget_m / 2,
    )
    bwd_keys, bwd_pts, bwd_len = _walk_one_direction(
        G, u, v, (_edge_end_bearing(main_coords, True) + 180.0) % 360.0, main_name,
        budget_e - budget_e // 2, budget_m - budget_m / 2,
    )

    # bwd_pts 方向为 u→道路远端，反转为 远端→…→u，再拼主边与前向
    coords = list(reversed(bwd_pts)) + main_coords
    if fwd_pts:
        coords.extend(fwd_pts[1:])
    # 相邻重复点去重
    deduped = []
    for p in coords:
        if not deduped or abs(deduped[-1][0] - p[0]) > 1e-12 or abs(deduped[-1][1] - p[1]) > 1e-12:
            deduped.append(p)

    ordered_keys = [(b, a, kk) for a, b, kk in reversed(bwd_keys)] + [(u, v, k)] + fwd_keys
    return ordered_keys, deduped, main_len + fwd_len + bwd_len, len(ordered_keys)


def snap_to_edge(
    G: nx.MultiDiGraph,
    lng_gcj: float,
    lat_gcj: float,
    max_dist_m: float = SNAP_MAX_DIST_M,
) -> Optional[dict]:
    """
    把管理员点击的 GCJ-02 坐标吸附到最近的路网边。

    Returns:
        {
          "u","v","key": 吸附主边标识,
          "edges": [[u,v,k],...] 主边沿道路扩展到两端路口后的完整边链,
          "chain_length_m": 整段道路长度,
          "road_name": 路名（可能为空）,
          "snap_lng_gcj","snap_lat_gcj": 吸附点（GCJ-02）,
          "snap_lng_wgs","snap_lat_wgs": 吸附点（WGS-84）,
          "dist_m": 点击点到边的距离（米）,
          "geometry_gcj": [[lng,lat],...] 整段道路几何（GCJ-02，前端直接画线）
        }
        最近边超过 max_dist_m 时返回 None。
    """
    from spatial.coord_transform import gcj02_to_wgs84, wgs84_to_gcj02
    from shapely.geometry import Point

    lng_wgs, lat_wgs = gcj02_to_wgs84(lng_gcj, lat_gcj)
    click = Point(lng_wgs, lat_wgs)
    cos_lat = __import__("math").cos(__import__("math").radians(lat_wgs))

    best = None  # (dist_deg_sq, u, v, k, line, proj_point)
    for u, v, k, data in G.edges(keys=True, data=True):
        line = _edge_line(G, u, v, data)
        proj = line.interpolate(line.project(click))
        dx = (proj.x - lng_wgs) * cos_lat
        dy = proj.y - lat_wgs
        dist_sq = dx * dx + dy * dy
        if best is None or dist_sq < best[0]:
            best = (dist_sq, u, v, k, line, proj)

    if best is None:
        return None

    dist_deg = best[0] ** 0.5
    dist_m = dist_deg * 111320.0
    if dist_m > max_dist_m:
        return None

    _, u, v, k, line, proj = best
    snap_lng_gcj, snap_lat_gcj = wgs84_to_gcj02(proj.x, proj.y)

    # 沿道路扩展到两端路口：管制影响的是一整段道路，不是单个 OSM edge
    try:
        chain_keys, chain_coords_wgs, chain_len_m, _ = _expand_edge_chain(G, u, v, k)
    except Exception:
        logger.warning("边链扩展失败，回退为单条边 (%s,%s,%s)", u, v, k, exc_info=True)
        chain_keys, chain_coords_wgs, chain_len_m = (
            [(u, v, k)], _line_coords(line), float((G.get_edge_data(u, v, k) or {}).get("length", 0) or 0),
        )
    geometry_gcj = [list(wgs84_to_gcj02(x, y)) for x, y in chain_coords_wgs]
    road_name = _road_name_of(G.get_edge_data(u, v, k) or {})

    return {
        "u": int(u),
        "v": int(v),
        "key": int(k),
        "edges": [[int(a), int(b), int(kk)] for a, b, kk in chain_keys],
        "chain_length_m": round(float(chain_len_m), 1),
        "road_name": road_name,
        "snap_lng_gcj": float(snap_lng_gcj),
        "snap_lat_gcj": float(snap_lat_gcj),
        "snap_lng_wgs": float(proj.x),
        "snap_lat_wgs": float(proj.y),
        "dist_m": round(float(dist_m), 1),
        "geometry_gcj": geometry_gcj,
    }


# ===================== 规划期应用 =====================

def _resolve_edge_keys(G: nx.MultiDiGraph, cond: dict, *, strict: bool = False) -> set:
    """
    解析事件影响的有向边 key 集合（道路双向通行，双向同时生效）。

    优先级：
      1. edge.edges 边链（路口到路口整段）/ 旧数据 edge.u/v 精确命中（当前路网）
      2. 节点 id 失效（路网重建）→ 整链几何 12m 回退，吸附点回退兜底
      3. 旧版 radius_m 圆模型 → 按原半径几何回退
    """
    edge_info = cond.get("edge")
    if isinstance(edge_info, dict) and edge_info:
        keys = set()
        chain = edge_info.get("edges")
        if isinstance(chain, list) and chain:
            chain_complete = True
            for triple in chain:
                try:
                    a, b, kk = int(triple[0]), int(triple[1]), int(triple[2])
                except (TypeError, ValueError, IndexError):
                    chain_complete = False
                    continue
                segment_found = False
                if G.has_edge(a, b, kk):
                    keys.add((a, b, kk))
                    segment_found = True
                if G.has_edge(b, a, kk):
                    keys.add((b, a, kk))
                    segment_found = True
                if not segment_found:
                    chain_complete = False
            if chain_complete and keys:
                return keys
            # Never apply only the subset of a stored closure chain that still has matching IDs.
            keys.clear()
        elif edge_info.get("u") is not None and edge_info.get("v") is not None:
            try:
                a, b = int(edge_info["u"]), int(edge_info["v"])
                kk = int(edge_info.get("key", 0))
                if G.has_edge(a, b, kk):
                    keys.add((a, b, kk))
                if G.has_edge(b, a, kk):
                    keys.add((b, a, kk))
            except (TypeError, ValueError):
                pass
        if keys:
            return keys
        # IDs失效或部分失效 → 必须通过整段几何回退，不应用残缺的边链。
        geom = edge_info.get("geometry_gcj") or []
        if len(geom) >= 2:
            chain_keys = _edges_near_geometry(G, geom, _EDGE_FALLBACK_DIST_M)
            if chain_keys and (not strict or _geometry_covered_by_edges(
                    G, geom, chain_keys, _EDGE_FALLBACK_DIST_M)):
                return chain_keys
        snap = edge_info.get("snap") or {}
        if (not strict and isinstance(snap, dict)
                and snap.get("lng") is not None and snap.get("lat") is not None):
            return _edges_near_point(
                G, float(snap["lng"]), float(snap["lat"]),
                gcj=True, radius_m=_EDGE_FALLBACK_DIST_M,
            )

    # 旧圆模型：coordinates + radius_m
    coords = cond.get("coordinates")
    if coords and coords.get("lng") is not None:
        return _edges_near_point(
            G, float(coords["lng"]), float(coords["lat"]),
            gcj=True, radius_m=float(cond.get("radius_m", _LEGACY_DEFAULT_RADIUS_M)),
        )
    return set()


def _expand_coincident_corridor_edges(G: nx.MultiDiGraph, keys: set, edge_info: dict) -> set:
    """Close duplicate/parallel graph edges that follow the selected road corridor.

    Node ids alone are insufficient in a MultiDiGraph: two OSM ways can encode the
    same physical road with different keys, or a rebuilt source can split it at
    different nodes. Only edges within the configured 15 m corridor and aligned
    along it are included. A perpendicular crossing that only touches the corridor is excluded.
    """
    from shapely.geometry import LineString, Point
    from spatial.coord_transform import gcj02_to_wgs84

    corridor_lines = []
    geometry_gcj = edge_info.get("geometry_gcj") if isinstance(edge_info, dict) else None
    if isinstance(geometry_gcj, list) and len(geometry_gcj) >= 2:
        try:
            wgs_coords = [gcj02_to_wgs84(float(point[0]), float(point[1]))
                          for point in geometry_gcj if isinstance(point, (list, tuple)) and len(point) >= 2]
            if len(wgs_coords) >= 2:
                corridor_lines = [wgs_coords]
        except (TypeError, ValueError, IndexError):
            corridor_lines = []

    if not corridor_lines:
        seen = set()
        for u, v, key in keys:
            identity = (frozenset((u, v)), key)
            if identity in seen or not G.has_edge(u, v, key):
                continue
            seen.add(identity)
            coords = _line_coords(_edge_line(G, u, v, G.get_edge_data(u, v, key) or {}))
            if len(coords) >= 2:
                corridor_lines.append(coords)

    if not corridor_lines:
        return set(keys)

    all_points = [point for line in corridor_lines for point in line]
    origin_lon = sum(point[0] for point in all_points) / len(all_points)
    origin_lat = sum(point[1] for point in all_points) / len(all_points)
    lon_scale = 111320.0 * math.cos(math.radians(origin_lat))

    def metric_line(coords):
        return LineString([
            ((float(lon) - origin_lon) * lon_scale,
             (float(lat) - origin_lat) * 111320.0)
            for lon, lat in coords
        ])

    try:
        metric_corridors = [metric_line(coords) for coords in corridor_lines]
        metric_corridors = [line for line in metric_corridors if not line.is_empty and line.length > 0]
    except (TypeError, ValueError):
        return set(keys)
    if not metric_corridors:
        return set(keys)

    corridor_buffers = [line.buffer(_PARALLEL_CORRIDOR_DIST_M) for line in metric_corridors]
    matched = set(keys)
    for u, v, key, data in G.edges(keys=True, data=True):
        candidate_coords = _line_coords(_edge_line(G, u, v, data))
        if len(candidate_coords) < 2:
            continue
        try:
            candidate = metric_line(candidate_coords)
        except (TypeError, ValueError):
            continue
        if candidate.is_empty or candidate.length <= 0:
            continue
        start = Point(candidate.coords[0])
        end = Point(candidate.coords[-1])
        for corridor, buffered in zip(metric_corridors, corridor_buffers):
            if candidate.distance(corridor) > _PARALLEL_CORRIDOR_DIST_M:
                continue
            overlap = candidate.intersection(buffered).length
            # The edge must run along the managed road for a meaningful length;
            # mere contact at a junction is not enough to bind the crossing road.
            if overlap < min(8.0, candidate.length * 0.7):
                continue
            # A short stretch inside the buffer is not evidence that a long
            # diagonal branch is the same road. Compare against the shorter
            # geometry so a long duplicate edge can still cover a short closure.
            comparable_length = min(candidate.length, corridor.length)
            if overlap < comparable_length * 0.7:
                continue
            along_span = abs(corridor.project(start) - corridor.project(end))
            if along_span < min(6.0, candidate.length * 0.7):
                continue
            # Reject oblique branches that remain near the junction but quickly
            # diverge; projected travel along the corridor must dominate.
            if along_span < comparable_length * math.cos(math.radians(30)):
                continue
            matched.add((u, v, key))
            break
    return matched


def _edges_near_point(G, lng, lat, gcj, radius_m) -> set:
    """几何回退：找到点(GCJ-02)半径内的边，双向返回。"""
    from spatial.coord_transform import gcj02_to_wgs84
    from shapely.geometry import Point
    lng_wgs, lat_wgs = gcj02_to_wgs84(lng, lat)
    # 经纬度 → 米的保守换算（高纬方向 1°≈111.3km，经度乘 cos(lat)）
    radius_deg = radius_m / 111320.0
    keys = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        line = _edge_line(G, u, v, data)
        if line.distance(Point(lng_wgs, lat_wgs)) <= radius_deg:
            keys.add((u, v, k))
            if G.has_edge(v, u, k):
                keys.add((v, u, k))
    return keys


def _edges_near_geometry(G, coords_gcj, radius_m=_EDGE_FALLBACK_DIST_M) -> set:
    """整链几何回退：与事件边链折线(GCJ-02)相距 radius_m 内的所有边，双向返回。

    路网重建后节点 id 全部变化时使用——一条链折线与候选边各做一次距离计算。
    """
    from spatial.coord_transform import gcj02_to_wgs84
    from shapely.geometry import LineString
    wgs = [gcj02_to_wgs84(float(p[0]), float(p[1])) for p in coords_gcj if p and len(p) >= 2]
    if len(wgs) < 2:
        return set()
    try:
        chain_line = LineString(wgs)
    except Exception:
        return set()
    radius_deg = radius_m / 111320.0
    keys = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        if _edge_line(G, u, v, data).distance(chain_line) <= radius_deg:
            keys.add((u, v, k))
            if G.has_edge(v, u, k):
                keys.add((v, u, k))
    return keys


def _geometry_covered_by_edges(G, coords_gcj, keys, tolerance_m) -> bool:
    """Require fallback edges to cover the entire stored road geometry within tolerance."""
    from math import ceil
    from shapely.geometry import LineString
    from spatial.coord_transform import gcj02_to_wgs84

    try:
        wgs = [gcj02_to_wgs84(float(point[0]), float(point[1]))
               for point in coords_gcj if isinstance(point, (list, tuple)) and len(point) >= 2]
        event_line = LineString(wgs)
        candidate_lines = [
            _edge_line(G, u, v, G.get_edge_data(u, v, k) or {})
            for u, v, k in keys if G.has_edge(u, v, k)
        ]
        if len(wgs) < 2 or not candidate_lines or event_line.length <= 0:
            return False
        interval_deg = max(tolerance_m / 111320.0 / 2.0, 1e-7)
        sample_count = max(2, int(ceil(event_line.length / interval_deg)))
        max_distance_deg = tolerance_m / 111320.0
        for index in range(sample_count + 1):
            point = event_line.interpolate(index / sample_count, normalized=True)
            if min(line.distance(point) for line in candidate_lines) > max_distance_deg:
                return False
        return True
    except (TypeError, ValueError, IndexError, KeyError):
        return False


def resolve_condition_effects(
    G: nx.MultiDiGraph,
    conditions: Optional[list] = None,
    mode: str = "walk",
    *,
    strict: bool = False,
    include_event_details: bool = False,
) -> tuple:
    """Resolve event effects without copying or mutating the supplied graph."""
    if conditions is None:
        conditions = list_conditions(strict=strict)
    if not conditions:
        return ({}, set(), 0, []) if include_event_details else ({}, set(), 0)

    if strict and mode not in {"walk", "bike", "drive"}:
        raise RoadConditionBindingError("未知出行方式，无法校验路况影响")

    penalties = {}
    closed = set()
    applied = 0
    details = []
    for cond in conditions:
        try:
            if strict:
                if not isinstance(cond, dict):
                    raise ValueError("路况事件格式无效")
                _validate_condition_record(cond)
            if "action" in cond:
                action = cond["action"]
                affected_modes = cond["affected_modes"]
                if mode not in affected_modes:
                    continue
                effect = (
                    "block" if action == "closure"
                    else float(cond["cost_multiplier"]) if action == "slowdown"
                    else None
                )
            else:
                action = None
                blocked_modes = cond.get("blocked_modes")
                if blocked_modes is None or not _valid_blocked_modes(blocked_modes):
                    if strict and blocked_modes is not None:
                        raise ValueError("路况事件禁行方式无效")
                    blocked_modes = TRAVEL_MODE_KEYS
                if mode not in blocked_modes:
                    continue
                effect = "block"
        except (AttributeError, TypeError, ValueError) as exc:
            if strict:
                raise RoadConditionBindingError("路况事件格式无效，无法安全应用") from exc
            continue
        if effect is None and action != "notice":
            if strict and action is None:
                raise RoadConditionBindingError("路况事件没有当前出行方式的通行规则")
            continue
        try:
            keys = _resolve_edge_keys(G, cond, strict=strict)
            edge_info = cond.get("edge") if isinstance(cond, dict) else {}
            # Corridor expansion is only valid after the stored binding (or its
            # complete-geometry fallback) has resolved. Expanding from geometry
            # alone here could silently revive a partially missing road chain.
            if keys:
                keys = _expand_coincident_corridor_edges(G, keys, edge_info or {})
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as exc:
            if strict:
                raise RoadConditionBindingError("路况事件道路绑定无法解析") from exc
            continue
        if not keys:
            if strict:
                raise RoadConditionBindingError("生效中的路况事件无法匹配当前路网，已停止路线规划")
            continue
        applied += 1
        if include_event_details:
            details.append((cond, keys))
        if effect is None:  # notice: report it without changing route costs
            continue
        if effect == "block":
            closed.update(keys)
        else:
            factor = float(effect)
            for key in keys:
                if factor > penalties.get(key, 1.0):
                    penalties[key] = factor

    return (penalties, closed, applied, details) if include_event_details else (penalties, closed, applied)


def _condition_overview(details: list) -> list:
    """Describe resolved events without implying they affect the selected route."""
    overview = []
    for condition, _keys in details:
        edge = condition.get("edge") if isinstance(condition.get("edge"), dict) else {}
        name = condition.get("name")
        road_name = edge.get("road_name")
        overview.append({
            "id": condition.get("id"),
            "name": " ".join(name.split())[:60] if isinstance(name, str) and name.strip() else "未命名路况",
            "road_name": " ".join(road_name.split())[:60] if isinstance(road_name, str) else "",
            "action": condition.get("action") or "closure",
        })
    return overview


def active_condition_overview(G, conditions: list, mode: str) -> tuple:
    """Describe events applied to this mode in the supplied active snapshot."""
    _, _, applied, details = resolve_condition_effects(G, conditions, mode, strict=True, include_event_details=True)
    return _condition_overview(details), applied


def route_condition_notices(G, conditions: list, mode: str, route_edges: list, *, include_overview: bool = False) -> tuple:
    """Return active notice/slowdown events overlapping actual selected edges.

    ``conditions`` must be the same already-filtered active snapshot used to
    plan and verify the route. The returned count is global to that snapshot;
    the events are limited to the selected edges and disclose no media paths.
    """
    if not isinstance(route_edges, list):
        raise RoadConditionBindingError("路线缺少已核对的道路边序列")
    try:
        order = {tuple(int(value) for value in edge): index
                 for index, edge in reversed(list(enumerate(route_edges)))}
        if any(len(edge) != 3 for edge in order):
            raise ValueError("invalid edge")
    except (TypeError, ValueError):
        raise RoadConditionBindingError("路线道路边序列无效") from None
    _, _, applied, details = resolve_condition_effects(
        G, conditions, mode, strict=True, include_event_details=True,
    )
    matches = []
    for condition, keys in details:
        if condition.get("action") not in {"notice", "slowdown"}:
            continue
        indices = [order[key] for key in keys if key in order]
        if not indices:
            continue
        edge = condition.get("edge") if isinstance(condition.get("edge"), dict) else {}
        source = condition.get("source") if isinstance(condition.get("source"), dict) else {}
        description = condition.get("description")
        public = {
            "id": condition.get("id"),
            "name": condition.get("name") or "路况提醒",
            "action": condition["action"],
            "road_name": edge.get("road_name") or "",
            "description": description[:100] if isinstance(description, str) else "",
            "start_time": condition.get("start_time"),
            "end_time": condition.get("end_time"),
            "observation_time": source.get("captured_at") or condition.get("created_at"),
            "cost_multiplier": condition.get("cost_multiplier") if condition["action"] == "slowdown" else None,
        }
        matches.append((min(indices), public))
    matches.sort(key=lambda item: item[0])
    unique, seen = [], set()
    for _, item in matches:
        identifier = item["id"]
        if not isinstance(identifier, (str, int)) or isinstance(identifier, bool):
            identifier = None
            item["id"] = None
        if identifier is not None and identifier in seen:
            continue
        if identifier is not None:
            seen.add(identifier)
        unique.append(item)
    return (unique, applied, _condition_overview(details)) if include_overview else (unique, applied)


def apply_conditions_to_graph(
    G: nx.MultiDiGraph,
    conditions: Optional[list] = None,
    mode: str = "walk",
    *,
    strict: bool = False,
) -> tuple:
    """Apply active event effects, materializing a graph copy only for closures.

    Returns ``(G_modified, penalty_map, closed_edges, applied_count)``. Callers
    that only need to compare snapshots should use ``resolve_condition_effects``
    and avoid copying a campus-sized graph.
    """
    penalties, closed, applied = resolve_condition_effects(
        G, conditions, mode, strict=strict,
    )
    G_modified = G
    if closed:
        G_modified = G.copy()
        G_modified.remove_edges_from(list(closed))
        isolated = [n for n, deg in G_modified.degree() if deg == 0]
        if isolated:
            G_modified.remove_nodes_from(isolated)
    return G_modified, penalties, closed, applied
