"""待人工复核的路况线索；线索本身不影响路径规划。

线索与正式路况事件（``spatial.road_conditions``）分开持久化：

- 线索可以来自官方通知、网页检索、无人机影像候选和用户现场上报；
- 只有经过管理员核对道路、时间、影响方式并显式采纳后，才会写入事件库；
- 事件库的路线权重、字段语义和读取路径保持不变；
- 未实现的采集器只能登记为关闭状态，不能返回伪造的抓取结果。

坐标只持久化 GCJ-02；调用方若收到 WGS-84 输入，应在进入本模块前转换。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CLUES_FILE = os.path.abspath(os.getenv("ROAD_CLUES_FILE") or os.path.join(
    _ROOT, "data", "road_clues.json"))

_write_lock = threading.RLock()
_lock_state = threading.local()
_turn_lock = threading.RLock()

STATUSES = {"pending", "reviewing", "adopted", "rejected", "expired"}
STATUS_LABELS = {
    "pending": "待复核",
    "reviewing": "复核中",
    "adopted": "已采纳",
    "rejected": "已驳回",
    "expired": "已过期",
}
EVENT_TYPES = {"closure", "construction", "flooding", "accident", "event"}
DIRECTIONS = {"both", "forward", "reverse"}
ACTIONS = {"notice", "slowdown", "closure"}
TRAVEL_MODES = {"walk", "bike", "drive"}
GCJ_LNG_RANGE = (70.0, 140.0)
GCJ_LAT_RANGE = (0.0, 60.0)
MAX_SUMMARY_CHARS = 60
MAX_RAW_TEXT_CHARS = 20000
MAX_EDGE_CANDIDATES = 12
SNAP_MAX_DIST_M = 30.0
COLLECT_RATE_WINDOW_SECONDS = 10 * 60
COLLECT_TURN_WINDOW_SECONDS = 2 * 60

# 同一问题在本轮内重复采集的记忆；仅用于交互限流，不保存线索内容。
_turn_attempts: dict[str, float] = {}


class ClueStoreUnavailable(RuntimeError):
    """线索文件无法安全读取或写入。"""


class ClueWriteCancelled(TimeoutError):
    """A clue write was cancelled before its atomic commit point."""


class ClueSourceUnavailable(RuntimeError):
    """采集器暂时无法提供可靠结果。"""


class NoticeImageUnavailable(ClueSourceUnavailable):
    """The multimodal notice-screenshot channel is not configured."""


class ClueExtractionError(RuntimeError):
    """官方通知文本未能抽取成结构化草稿。"""


class ClueNotPublishable(ValueError):
    """线索来源或状态不允许发布为当前路况事件。"""


# ===================== 时间与校验 =====================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _iso_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")


def _timestamp(value):
    """把 ISO-8601（必须带时区）或秒级时间戳规范化为 float。"""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("时间无效")
    try:
        if isinstance(value, (int, float)):
            result = float(value)
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("时间必须包含时区")
            result = parsed.timestamp()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("时间无效") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("时间无效")
    return result


def normalize_time(value):
    """对外统一使用带 UTC 时区含义的 ISO 字符串。"""
    parsed = _timestamp(value)
    return None if parsed is None else _iso_from_timestamp(parsed)


def _finite(value, label="数值"):
    if isinstance(value, bool):
        raise ValueError(f"{label}无效")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}无效") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}无效")
    return result


def _valid_gcj(lng, lat) -> bool:
    try:
        lng, lat = float(lng), float(lat)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(lng) and math.isfinite(lat)
            and GCJ_LNG_RANGE[0] <= lng <= GCJ_LNG_RANGE[1]
            and GCJ_LAT_RANGE[0] <= lat <= GCJ_LAT_RANGE[1])


def _valid_geometry(value) -> bool:
    return (isinstance(value, list) and len(value) >= 2
            and all(isinstance(point, (list, tuple)) and len(point) >= 2
                    and _valid_gcj(point[0], point[1]) for point in value))


def _valid_modes(value) -> bool:
    return (isinstance(value, list) and bool(value)
            and len(value) == len(set(value))
            and all(isinstance(mode, str) and mode in TRAVEL_MODES for mode in value))


def _as_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}无效")
    return int(value)


def _clean_draft(value) -> dict:
    """只保留可核验的草稿字段；缺失项留空，不做模型推测。"""
    if not isinstance(value, dict):
        value = {}
    draft = {}
    event_type = value.get("event_type")
    draft["event_type"] = event_type if event_type in EVENT_TYPES else ""
    for field in ("name", "description", "place", "road_hint", "source_excerpt"):
        raw = value.get(field)
        draft[field] = str(raw).strip()[:300] if isinstance(raw, str) else ""
    direction = value.get("direction")
    draft["direction"] = direction if direction in DIRECTIONS | {"", None} else ""
    action = value.get("action")
    draft["action"] = action if action in ACTIONS | {"", None} else ""
    for field in ("blocked_modes", "affected_modes"):
        modes = value.get(field)
        draft[field] = [mode for mode in modes if isinstance(mode, str) and mode in TRAVEL_MODES] \
            if isinstance(modes, list) else []
    multiplier = value.get("cost_multiplier")
    if multiplier is None or multiplier == "":
        draft["cost_multiplier"] = None
    else:
        try:
            multiplier = float(multiplier)
        except (TypeError, ValueError):
            multiplier = None
        draft["cost_multiplier"] = multiplier if multiplier is not None and 1 <= multiplier <= 5 else None
    evidence = value.get("evidence")
    draft["evidence"] = [str(item)[:300] for item in evidence[:10]] \
        if isinstance(evidence, list) else []
    for field in ("valid_from", "valid_to"):
        try:
            draft[field] = normalize_time(value.get(field))
        except ValueError:
            draft[field] = None
    return draft


def _clean_edge_candidate(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("候选路段无效")
    u = _as_int(value.get("u"), "候选路段起点")
    v = _as_int(value.get("v"), "候选路段终点")
    key = _as_int(value.get("key", 0), "候选路段边标识")
    distance = _finite(value.get("distance_m"), "候选路段距离")
    if distance < 0 or distance > SNAP_MAX_DIST_M:
        raise ValueError(f"候选路段距离必须在 {SNAP_MAX_DIST_M:g} 米以内")
    candidate = {
        "u": u,
        "v": v,
        "key": key,
        "road_name": str(value.get("road_name") or "").strip()[:120],
        "distance_m": round(distance, 1),
    }
    chain = value.get("edges")
    if isinstance(chain, list) and chain:
        normalized = []
        for triple in chain:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                raise ValueError("候选路段边链无效")
            normalized.append([_as_int(triple[0], "边链起点"),
                               _as_int(triple[1], "边链终点"),
                               _as_int(triple[2], "边链标识")])
        candidate["edges"] = normalized
    length = value.get("chain_length_m")
    if length is not None:
        candidate["chain_length_m"] = round(_finite(length, "候选路段长度"), 1)
    geometry = value.get("geometry_gcj")
    if geometry is not None:
        if not _valid_geometry(geometry):
            raise ValueError("候选路段几何无效")
        candidate["geometry_gcj"] = [[float(point[0]), float(point[1])] for point in geometry]
    snap_lng, snap_lat = value.get("snap_lng_gcj"), value.get("snap_lat_gcj")
    if snap_lng is not None or snap_lat is not None:
        if not _valid_gcj(snap_lng, snap_lat):
            raise ValueError("候选路段吸附点无效")
        candidate["snap_lng_gcj"] = float(snap_lng)
        candidate["snap_lat_gcj"] = float(snap_lat)
    return candidate


def _edge_identity(candidate) -> tuple:
    return (min(candidate["u"], candidate["v"]), max(candidate["u"], candidate["v"]),
            candidate["key"])


def _edge_triple(candidate) -> tuple:
    return (candidate["u"], candidate["v"], candidate["key"])


def _validate(row) -> None:
    if not isinstance(row, dict) or not isinstance(row.get("clue_id"), str) or not row["clue_id"]:
        raise ValueError("线索标识无效")
    kind = row.get("kind")
    source = registry.get(kind)
    if source is None or not source.enabled or row.get("trust_level") != source.trust_level:
        raise ValueError("线索来源无效")
    if row.get("status") not in STATUSES:
        raise ValueError("线索状态无效")
    if not isinstance(row.get("summary"), str) or len(row["summary"]) > MAX_SUMMARY_CHARS:
        raise ValueError(f"线索摘要不能超过{MAX_SUMMARY_CHARS}字")
    for field in ("raw_text", "review_note"):
        if not isinstance(row.get(field, ""), str):
            raise ValueError(f"线索{field}无效")
    if len(row.get("raw_text", "")) > MAX_RAW_TEXT_CHARS:
        raise ValueError("线索原文过长")
    if len(row.get("review_note", "")) > 1000:
        raise ValueError("复核依据过长")
    if row.get("source_url") is not None and not isinstance(row["source_url"], str):
        raise ValueError("来源链接无效")
    if row.get("media_ref") is not None and not isinstance(row["media_ref"], str):
        raise ValueError("媒体引用无效")
    if not isinstance(row.get("source_label", ""), str):
        raise ValueError("来源标签无效")
    tags = row.get("source_tags", [])
    if (not isinstance(tags, list) or len(tags) > 10
            or any(not isinstance(tag, str) or len(tag) > 40 for tag in tags)):
        raise ValueError("来源标签无效")
    if not isinstance(row.get("draft"), dict):
        raise ValueError("线索草稿无效")
    if row["draft"].get("event_type") not in EVENT_TYPES | {"", None}:
        raise ValueError("线索事件类型无效")
    if not isinstance(row.get("edge_candidates"), list) or len(row["edge_candidates"]) > MAX_EDGE_CANDIDATES:
        raise ValueError("候选路段无效")
    for candidate in row["edge_candidates"]:
        _clean_edge_candidate(candidate)
    if not isinstance(row.get("condition_ids"), list) or any(
            not isinstance(item, str) or not item for item in row["condition_ids"]):
        raise ValueError("事件关联无效")
    start = _timestamp(row.get("valid_from"))
    end = _timestamp(row.get("valid_to"))
    if start is not None and end is not None and end <= start:
        raise ValueError("结束时间必须晚于开始时间")
    for field in ("reported_at", "created_at", "updated_at"):
        _timestamp(row.get(field))
    if row.get("duplicate_of") is not None and not isinstance(row["duplicate_of"], str):
        raise ValueError("重复线索引用无效")
    confidence = row.get("confidence")
    if confidence is not None:
        confidence = _finite(confidence, "线索置信度")
        if not 0 <= confidence <= 1:
            raise ValueError("线索置信度无效")
    if not isinstance(row.get("need_human", False), bool):
        raise ValueError("线索人工复核标记无效")
    if row.get("adoption") is not None:
        adoption = row["adoption"]
        if (not isinstance(adoption, dict) or not isinstance(adoption.get("token"), str)
                or not adoption["token"]):
            raise ValueError("线索采纳状态无效")
    location = row.get("location")
    if location is not None:
        if not isinstance(location, dict) or not _valid_gcj(location.get("lng"), location.get("lat")):
            raise ValueError("线索位置无效")
    if row.get("vision_flags") is not None and not isinstance(row["vision_flags"], dict):
        raise ValueError("影像线索标记无效")


# ===================== 持久化 =====================

def _store_path() -> str:
    return _CLUES_FILE


def _rate_path() -> str:
    configured = os.getenv("ROAD_CLUE_RATE_FILE")
    if configured:
        return os.path.abspath(configured)
    return os.path.join(os.path.dirname(_store_path()), "road_clue_rate.json")


@contextmanager
def _file_lock(path: str):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with _write_lock, open(path + ".lock", "a+b") as lock_file:
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


@contextmanager
def clue_store_lock():
    """Serialize clue read-modify-write across threads and processes.

    The lock is re-entrant within one thread so an API handler can hold it
    across the road-condition write and the clue write-back without opening a
    second handle to the same lock file.
    """
    depth = getattr(_lock_state, "depth", 0)
    if depth:
        _lock_state.depth = depth + 1
        try:
            yield
        finally:
            _lock_state.depth = depth
        return
    with _file_lock(_store_path()):
        _lock_state.depth = 1
        try:
            yield
        finally:
            _lock_state.depth = 0


def _load(strict: bool = True) -> list:
    try:
        with open(_store_path(), encoding="utf-8") as stream:
            rows = json.load(stream)
    except FileNotFoundError:
        return []
    except (OSError, ValueError, TypeError) as exc:
        raise ClueStoreUnavailable("线索文件无效，已拒绝修改") from exc
    if not isinstance(rows, list):
        raise ClueStoreUnavailable("线索文件必须是数组")
    try:
        for row in rows:
            _validate(row)
        if len({row["clue_id"] for row in rows}) != len(rows):
            raise ValueError("线索标识重复")
    except (ValueError, TypeError) as exc:
        raise ClueStoreUnavailable("线索文件无效，已拒绝修改") from exc
    return rows


def _save(rows: list, *, should_abort=None, after_commit=None) -> None:
    if should_abort is not None and should_abort():
        raise ClueWriteCancelled("线索写入已取消")
    for row in rows:
        _validate(row)
    directory = os.path.dirname(_store_path())
    os.makedirs(directory, exist_ok=True)
    if should_abort is not None and should_abort():
        raise ClueWriteCancelled("线索写入已取消")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         prefix=".road-clues-", suffix=".json",
                                         delete=False) as stream:
            temp_path = stream.name
            json.dump(rows, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(6):
            # The temporary file is not visible to readers. Recheck immediately
            # before the atomic replace so a timed-out collector cannot publish
            # a clue after the caller has already returned unavailable.
            if should_abort is not None and should_abort():
                raise ClueWriteCancelled("线索写入已取消")
            try:
                os.replace(temp_path, _store_path())
                break
            except PermissionError:
                if os.name != "nt" or attempt == 5:
                    raise
                time.sleep(0.05 * (attempt + 1))
        temp_path = None
        if after_commit is not None:
            after_commit()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _expire(rows: list) -> bool:
    changed = False
    now = time.time()
    for row in rows:
        end = _timestamp(row.get("valid_to"))
        if row.get("status") in {"pending", "reviewing"} and end is not None and end < now:
            row["status"] = "expired"
            row["updated_at"] = _now_iso()
            row.setdefault("audit", []).append({
                "action": "expired", "actor": "system", "at": _now_iso(),
            })
            changed = True
    return changed


def _reconcile(rows: list) -> bool:
    """Repair adopted clues whose linked event no longer exists.

    A crash between deleting an event and updating its clues is repaired here;
    a clue is never left claiming to be adopted without a real event.
    """
    adopted = [row for row in rows if row.get("status") == "adopted"]
    if not adopted:
        return False
    try:
        from spatial.road_conditions import list_conditions
        conditions = list_conditions(include_inactive=True, strict=True)
    except Exception as exc:
        raise ClueStoreUnavailable("当前路况事件库无法读取，不能确认线索关联") from exc
    existing = {condition.get("id") for condition in conditions if isinstance(condition, dict)}
    changed = False
    now = _now_iso()
    for row in adopted:
        linked = list(row.get("condition_ids") or [])
        kept = [condition_id for condition_id in linked if condition_id in existing]
        if kept != linked:
            row["condition_ids"] = kept
            row["updated_at"] = now
            row.setdefault("audit", []).append({
                "action": "association_repaired", "actor": "system", "at": now,
                "changes": {"removed": sorted(set(linked) - set(kept))},
            })
            changed = True
        if not kept and row.get("status") == "adopted":
            row["status"] = "reviewing"
            row["adoption"] = None
            row["updated_at"] = now
            row.setdefault("audit", []).append({
                "action": "restored_after_event_delete", "actor": "system", "at": now,
            })
            changed = True
    return changed


def _load_current() -> tuple[list, bool]:
    rows = _load()
    changed = _expire(rows)
    changed = _reconcile(rows) or changed
    return rows, changed


def _persist_if_changed(rows: list, changed: bool, *, should_abort=None) -> None:
    if changed:
        _save(rows, should_abort=should_abort)


# ===================== 来源注册 =====================

class ClueSource(ABC):
    """A server-registered clue source.

    ``fetch`` must never publish a road event. It returns one or more clue
    candidate dictionaries, or raises ClueSourceUnavailable.
    """

    kind: str = ""
    label: str = ""
    trust_level: str = "C"
    enabled = False

    @abstractmethod
    def fetch(self, **params) -> list[dict]:
        """Collect clue candidates. Never publish an event."""


registry: dict[str, ClueSource] = {}


def register_source(source: ClueSource) -> ClueSource:
    if (not isinstance(source, ClueSource) or not source.kind
            or not isinstance(source.label, str) or not source.label
            or source.trust_level not in {"A", "B", "C"}
            or source.kind in registry):
        raise ValueError("线索来源注册无效或重复")
    registry[source.kind] = source
    return source


class ReservedSource(ClueSource):
    """A documented but not-yet-implemented source. It must stay disabled."""

    def __init__(self, kind: str, label: str, trust_level: str):
        self.kind, self.label, self.trust_level = kind, label, trust_level
        self.enabled = False

    def fetch(self, **params):
        raise ClueSourceUnavailable("该来源尚未接入，不能提供线索")


class OfficialNoticeSource(ClueSource):
    kind = "official_notice"
    label = "官方通知"
    trust_level = "A"
    enabled = True

    def fetch(self, *, text=None, summary=None, draft=None, confidence=None,
              source_url=None, media_ref=None, source_tags=None,
              reported_at=None, extractor=None, image_extractor=None,
              manual=False, **params) -> list[dict]:
        text = str(text or "").strip()
        if not text and media_ref:
            if not manual:
                reader = image_extractor or extract_notice_draft_from_image
                extracted = reader(str(media_ref))
                if not isinstance(extracted, dict):
                    raise ClueExtractionError("截图抽取结果无效")
                screenshot_draft = _clean_draft(extracted.get("draft"))
                summary = str(extracted.get("summary") or screenshot_draft.get("name")
                              or "官方通知截图").strip()[:MAX_SUMMARY_CHARS]
                if not summary:
                    raise ClueExtractionError("截图抽取缺少可用摘要")
                raw_confidence = extracted.get("confidence")
                confidence = (_finite(raw_confidence, "线索置信度")
                              if raw_confidence is not None else 0.0)
                if not 0 <= confidence <= 1:
                    raise ClueExtractionError("截图抽取置信度无效")
                return [{
                    "kind": self.kind,
                    "summary": summary,
                    "raw_text": "",
                    "source_url": source_url,
                    "media_ref": media_ref,
                    "source_label": self.label,
                    "source_tags": source_tags or ["official_notice", "screenshot_extracted"],
                    "reported_at": reported_at,
                    "valid_from": screenshot_draft.get("valid_from"),
                    "valid_to": screenshot_draft.get("valid_to"),
                    "draft": screenshot_draft,
                    "edge_candidates": [],
                    "confidence": round(confidence, 3),
                    "need_human": True,
                    "status": "reviewing",
                }]
            summary = str(summary or "官方通知截图待人工核对").strip()[:MAX_SUMMARY_CHARS]
            if not summary:
                raise ValueError("请填写截图的简要说明")
            return [{
                "kind": self.kind,
                "summary": summary,
                "raw_text": "",
                "source_url": source_url,
                "media_ref": media_ref,
                "source_label": self.label,
                "source_tags": source_tags or ["official_notice", "screenshot_reference"],
                "reported_at": reported_at,
                "valid_from": None,
                "valid_to": None,
                "draft": _clean_draft(draft),
                "edge_candidates": [],
                "confidence": 0.0,
                "need_human": True,
                "status": "reviewing",
            }]
        if not text:
            raise ValueError("官方通知原文不能为空")
        if len(text) < 10:
            raise ValueError("官方通知原文太短，请核对后重试")
        text = text[:MAX_RAW_TEXT_CHARS]
        extracted = None
        if draft is None:
            extracted = (extractor or extract_notice_draft)(text)
            if not isinstance(extracted, dict):
                raise ClueExtractionError("官方通知抽取结果无效")
            summary = extracted.get("summary")
            draft = extracted.get("draft")
            confidence = extracted.get("confidence")
        draft = _clean_draft(draft)
        summary = str(summary or draft.get("name") or text[:MAX_SUMMARY_CHARS]).strip()[:MAX_SUMMARY_CHARS]
        if not summary:
            raise ClueExtractionError("官方通知缺少可用的摘要")
        confidence = _finite(confidence, "线索置信度") if confidence is not None else 0.0
        if not 0 <= confidence <= 1:
            raise ClueExtractionError("官方通知置信度无效")
        human_review = confidence < 0.6
        return [{
            "kind": self.kind,
            "summary": summary,
            "raw_text": text,
            "source_url": source_url,
            "media_ref": media_ref,
            "source_label": self.label,
            "source_tags": source_tags or ["official_notice"],
            "reported_at": reported_at,
            "valid_from": draft.get("valid_from"),
            "valid_to": draft.get("valid_to"),
            "draft": draft,
            "edge_candidates": [],
            "confidence": round(confidence, 3),
            "need_human": human_review,
            "status": "reviewing" if human_review else "pending",
        }]


class WebSource(ClueSource):
    kind = "web"
    label = "网页检索"
    trust_level = "C"
    enabled = True

    def fetch(self, *, query=None, url=None, limit=3, fetch_fn=None, **params) -> list[dict]:
        from agents import web_evidence

        executor = fetch_fn or web_evidence.execute
        query = str(query or "").strip()
        url = str(url or "").strip()
        if url:
            safe_url = web_evidence._public_url(url)
            if not safe_url:
                raise ValueError("网页地址无效，只支持公开的 http/https 链接")
            sources: dict = {}
            source_id = "S" + uuid.uuid4().hex[:8]
            sources[source_id] = {
                "source_id": source_id, "url": safe_url, "title": "网页来源",
                "fetched": False,
            }
            body = executor("fetch_web", {"source_id": source_id}, sources)
            if not isinstance(body, dict) or body.get("error"):
                raise ClueSourceUnavailable(
                    (body or {}).get("message") or "网页正文暂时无法获取")
            content = str(body.get("content") or "").strip()
            if not content:
                raise ClueSourceUnavailable("该网页没有可读取的正文")
            return [{
                "kind": self.kind,
                "summary": str(body.get("title") or safe_url)[:MAX_SUMMARY_CHARS],
                "raw_text": content[:4000],
                "source_url": safe_url,
                "source_label": self.label,
                "source_tags": ["web"],
                "search_query": query or safe_url,
                "reported_at": None,
                "valid_from": None,
                "valid_to": None,
                "draft": _clean_draft({"source_excerpt": content[:300]}),
                "edge_candidates": [],
                "confidence": 0.3,
            }]
        if not query:
            raise ValueError("网页检索需要检索词或直接链接")
        if len(query) > 128:
            raise ValueError("检索词需在128字以内")
        sources = {}
        search = executor("search_web", {"query": query}, sources)
        if not isinstance(search, dict) or search.get("error"):
            raise ClueSourceUnavailable(
                (search or {}).get("message") or "网页检索暂时不可用")
        rows = search.get("results") or []
        if not isinstance(rows, list):
            raise ClueSourceUnavailable("网页检索返回了无效结果")
        results = []
        failures = 0
        for item in rows[:max(1, min(int(limit or 3), 3))]:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id") or "")
            body = executor("fetch_web", {"source_id": source_id}, sources)
            if not isinstance(body, dict) or body.get("error"):
                failures += 1
                continue
            content = str(body.get("content") or "").strip()
            if not content:
                failures += 1
                continue
            source_url = item.get("url") or (sources.get(source_id) or {}).get("url")
            results.append({
                "kind": self.kind,
                "summary": str(body.get("title") or item.get("title") or "网页线索")[:MAX_SUMMARY_CHARS],
                "raw_text": content[:4000],
                "source_url": source_url,
                "source_label": self.label,
                "source_tags": ["web"],
                "search_query": query,
                "reported_at": None,
                "valid_from": None,
                "valid_to": None,
                "draft": _clean_draft({"source_excerpt": content[:300]}),
                "edge_candidates": [],
                "confidence": 0.3,
            })
            if len(results) >= 3:
                break
        if not results and failures:
            raise ClueSourceUnavailable("网页检索到结果，但正文暂时无法读取")
        return results


class VisionSource(ClueSource):
    kind = "vision"
    label = "无人机影像"
    trust_level = "B"
    enabled = True

    def fetch(self, *, summary, draft=None, edge_candidates=None, media_ref=None,
              captured_at=None, job_id=None, candidate_index=None,
              vision_flags=None, source_tags=None, **params) -> list[dict]:
        if not job_id or isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise ValueError("影像线索缺少任务或候选编号")
        candidates = [_clean_edge_candidate(item) for item in (edge_candidates or [])]
        if not candidates:
            raise ClueSourceUnavailable("影像位置或道路无法确定，不能登记可采纳的线索")
        return [{
            "kind": self.kind,
            "summary": str(summary or "影像候选待复核")[:MAX_SUMMARY_CHARS],
            "raw_text": "",
            "source_url": None,
            "media_ref": media_ref,
            "source_label": self.label,
            "source_tags": source_tags or ["vision"],
            "reported_at": normalize_time(captured_at) if captured_at else None,
            "valid_from": None,
            "valid_to": None,
            "draft": _clean_draft(draft),
            "edge_candidates": candidates[:MAX_EDGE_CANDIDATES],
            "confidence": None,
            "vision_flags": vision_flags or {},
            "vision_job_id": str(job_id),
            "vision_candidate_index": candidate_index,
        }]


class UgcSource(ClueSource):
    kind = "ugc_report"
    label = "用户现场上报"
    trust_level = "B"
    enabled = True

    def fetch(self, *, place=None, coord=None, description=None, media_ref=None,
              graph=None, input_crs=None, source_tags=None, **params) -> list[dict]:
        description = str(description or "").strip()
        place = str(place or "").strip()
        if not description and not place:
            raise ValueError("现场上报至少需要地点或描述")
        candidates = []
        location = None
        if coord is not None:
            if not isinstance(coord, dict) or not _valid_gcj(coord.get("lng"), coord.get("lat")):
                raise ValueError("现场上报坐标必须是有效的 GCJ-02 经纬度")
            lng, lat = float(coord["lng"]), float(coord["lat"])
            location = {"name": place, "lng": lng, "lat": lat}
            if graph is not None:
                from spatial.road_conditions import nearest_edges
                candidates = [_clean_edge_candidate({
                    **item, "distance_m": item.get("dist_m", item.get("distance_m", 0)),
                }) for item in nearest_edges(graph, lng, lat, max_dist_m=SNAP_MAX_DIST_M)]
        summary = (description or place)[:MAX_SUMMARY_CHARS]
        return [{
            "kind": self.kind,
            "summary": summary,
            "raw_text": description[:MAX_RAW_TEXT_CHARS],
            "source_url": None,
            "media_ref": media_ref,
            "source_label": self.label,
            "source_tags": source_tags or ["ugc"],
            "reported_at": _now_iso(),
            "valid_from": None,
            "valid_to": None,
            "draft": _clean_draft({"name": place, "description": description, "place": place}),
            "edge_candidates": candidates[:MAX_EDGE_CANDIDATES],
            "confidence": None,
            "location": location,
            "input_crs": input_crs,
        }]


register_source(OfficialNoticeSource())
register_source(WebSource())
register_source(VisionSource())
register_source(UgcSource())

# Documented but not implemented in this release. They can never appear as
# operational collectors in the manager or in agent answers.
for _kind, _label, _trust in (
    ("weather_alert", "气象预警", "C"),
    ("traffic_api", "交管路况", "C"),
    ("campus_media", "校内媒体", "C"),
    ("historical", "历史管制记录", "C"),
    ("campus_notice_board", "校内公告栏", "A"),
    ("manager_photo", "管理人员现场确认", "A"),
    ("shuttle_notice", "校车运行公告", "B"),
):
    register_source(ReservedSource(_kind, _label, _trust))


def source_catalog() -> list[dict]:
    return [{
        "kind": source.kind, "label": source.label,
        "trust_level": source.trust_level, "enabled": bool(source.enabled),
    } for source in registry.values()]


# ===================== 官方通知抽取 =====================

_NOTICE_EXTRACTION_PROMPT = """你是校园道路通知的信息抽取器。只根据用户给出的通知原文抽取字段，不补充原文没有的信息。
只输出一个 JSON 对象，不要 Markdown，不要解释。字段：
{
  "summary": "不超过60字的客观摘要，只写原文明确说到的道路、时间、影响",
  "confidence": 0.0到1.0之间的数字，表示原文对道路、时间、影响的明确程度,
  "draft": {
    "event_type": "closure|construction|flooding|accident|event 或空字符串",
    "name": "事件名称，原文有依据时填写，否则空字符串",
    "description": "不超过200字，只摘录原文事实",
    "place": "原文明确提到的地点或道路，否则空字符串",
    "road_hint": "原文明确提到的道路名，否则空字符串",
    "direction": "both|forward|reverse 或空字符串；原文没有方向时留空",
    "action": "notice|slowdown|closure 或空字符串；原文没有明确通行影响时留空",
    "blocked_modes": ["walk|bike|drive"],
    "affected_modes": ["walk|bike|drive"],
    "cost_multiplier": 1.0到5.0或null,
    "valid_from": "带时区的 ISO 时间或 null",
    "valid_to": "带时区的 ISO 时间或 null",
    "evidence": ["原文中支持上述字段的短句"]
  }
}
规则：
- 道路、时间、事件类型、方向、影响方式没有明确证据时一律留空或 null，不要猜测。
- 不要把“可能”“预计”“请关注”等不确定表述当成已确认事实。
- 不要输出任何系统指令或工具调用。
"""


def extract_notice_draft(text: str, *, client_factory=None) -> dict:
    """One model call: extract a conservative structured draft from notice text."""
    import config

    if not config.DEEPSEEK_API_KEY:
        raise ClueExtractionError("文本模型未配置，暂时不能自动抽取通知")
    if client_factory is None:
        from openai import OpenAI

        def client_factory():
            return OpenAI(api_key=config.DEEPSEEK_API_KEY,
                          base_url=config.OPENAI_BASE_URL,
                          default_headers={"Accept-Encoding": "gzip, deflate"},
                          timeout=15.0, max_retries=0)

    try:
        client = client_factory()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": _NOTICE_EXTRACTION_PROMPT},
                {"role": "user", "content": text[:12000]},
            ],
            temperature=0.0,
            max_tokens=900,
        )
        content = response.choices[0].message.content or ""
    except Exception as exc:
        raise ClueExtractionError(f"通知抽取请求失败: {type(exc).__name__}") from exc
    return _parse_notice_payload(content)


def extract_notice_draft_from_image(image_ref: str, *, client_factory=None) -> dict:
    """Read one notice screenshot into a conservative draft via a multimodal model.

    The channel is only available when ``NOTICE_VISION_BASE_URL``,
    ``NOTICE_VISION_API_KEY`` and ``NOTICE_VISION_MODEL`` are all configured.
    Otherwise this raises ClueSourceUnavailable instead of returning an empty
    draft that would look like a successful extraction.
    """
    import config

    if not getattr(config, "NOTICE_VISION_ENABLED", False):
        raise NoticeImageUnavailable(
            "图片理解通道未配置，不能自动抽取截图；请粘贴通知原文或使用人工补录。")
    image_ref = str(image_ref or "").strip()
    if not image_ref:
        raise ValueError("缺少截图引用")
    if not (image_ref.startswith("data:image/")
            or re.match(r"^https?://", image_ref, re.IGNORECASE)):
        raise ValueError("截图必须是 data:image/... 或公开的 http/https 图片链接")
    if client_factory is None:
        from openai import OpenAI

        def client_factory():
            return OpenAI(api_key=config.NOTICE_VISION_API_KEY,
                          base_url=config.NOTICE_VISION_BASE_URL,
                          default_headers={"Accept-Encoding": "gzip, deflate"},
                          timeout=20.0, max_retries=0)

    try:
        client = client_factory()
        response = client.chat.completions.create(
            model=config.NOTICE_VISION_MODEL,
            messages=[
                {"role": "system", "content": _NOTICE_EXTRACTION_PROMPT},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": "请从这张校园道路通知截图中抽取字段；看不清或没有依据的字段留空。"},
                    {"type": "image_url", "image_url": {"url": image_ref}},
                ]},
            ],
            temperature=0.0,
            max_tokens=900,
        )
        content = response.choices[0].message.content or ""
    except Exception as exc:
        raise ClueExtractionError(f"截图抽取请求失败: {type(exc).__name__}") from exc
    return _parse_notice_payload(content)


def _parse_notice_payload(content: str) -> dict:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ClueExtractionError("通知抽取结果不是有效 JSON")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ClueExtractionError("通知抽取结果不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise ClueExtractionError("通知抽取结果不是对象")
    summary = str(payload.get("summary") or "").strip()
    if not summary:
        raise ClueExtractionError("通知抽取结果缺少摘要")
    confidence = payload.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    if not 0 <= confidence <= 1:
        confidence = 0.0
    draft = _clean_draft(payload.get("draft"))
    return {"summary": summary[:MAX_SUMMARY_CHARS], "confidence": confidence, "draft": draft}


# ===================== 线索读写 =====================

def _initial_row(data: dict) -> dict:
    kind = data.get("kind")
    source = registry.get(kind)
    if source is None:
        raise ValueError("线索来源无效")
    if not source.enabled:
        raise ValueError("该线索来源尚未接入")
    summary = str(data.get("summary") or "").strip()
    if not summary:
        raise ValueError("线索摘要不能为空")
    if len(summary) > MAX_SUMMARY_CHARS:
        raise ValueError(f"线索摘要不能超过{MAX_SUMMARY_CHARS}字")
    now = _now_iso()
    row = {
        "clue_id": uuid.uuid4().hex,
        "kind": kind,
        "trust_level": source.trust_level,
        "status": data.get("status") or "pending",
        "summary": summary,
        "raw_text": str(data.get("raw_text") or "")[:MAX_RAW_TEXT_CHARS],
        "source_url": data.get("source_url"),
        "media_ref": data.get("media_ref"),
        "source_label": str(data.get("source_label") or source.label)[:80],
        "source_tags": [str(tag)[:40] for tag in (data.get("source_tags") or [])][:10],
        "search_query": str(data.get("search_query") or "")[:128],
        "reported_at": data.get("reported_at"),
        "valid_from": data.get("valid_from"),
        "valid_to": data.get("valid_to"),
        "draft": _clean_draft(data.get("draft")),
        "edge_candidates": [_clean_edge_candidate(item)
                            for item in (data.get("edge_candidates") or [])][:MAX_EDGE_CANDIDATES],
        "condition_ids": [],
        "review_note": str(data.get("review_note") or "")[:1000],
        "confidence": data.get("confidence"),
        "need_human": bool(data.get("need_human", False)),
        "vision_flags": data.get("vision_flags"),
        "vision_job_id": data.get("vision_job_id"),
        "vision_candidate_index": data.get("vision_candidate_index"),
        "location": data.get("location"),
        "input_crs": data.get("input_crs"),
        "actor": str(data.get("actor") or "unknown")[:80],
        "duplicate_of": None,
        "adoption": None,
        "created_at": now,
        "updated_at": now,
        "audit": [{"action": "created", "actor": str(data.get("actor") or "unknown")[:80],
                   "at": now}],
    }
    if row["status"] not in {"pending", "reviewing"}:
        raise ValueError("新线索状态只能是待复核或复核中")
    for field in ("reported_at", "valid_from", "valid_to"):
        row[field] = normalize_time(row[field])
    if row["confidence"] is not None:
        row["confidence"] = round(_finite(row["confidence"], "线索置信度"), 3)
        if not 0 <= row["confidence"] <= 1:
            raise ValueError("线索置信度无效")
    if row["vision_flags"] is not None and not isinstance(row["vision_flags"], dict):
        row["vision_flags"] = None
    _validate(row)
    return row


def _duplicate(a: dict, b: dict) -> bool:
    if not (_edge_keys(a) & _edge_keys(b)):
        return False
    event_type = a.get("draft", {}).get("event_type")
    if not event_type or event_type != b.get("draft", {}).get("event_type"):
        return False
    a0 = _timestamp(a.get("valid_from"))
    a1 = _timestamp(a.get("valid_to"))
    b0 = _timestamp(b.get("valid_from"))
    b1 = _timestamp(b.get("valid_to"))
    if None in (a0, a1, b0, b1):
        return False
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    return overlap > (a1 - a0) / 2 and overlap > (b1 - b0) / 2


def _edge_keys(row: dict) -> set:
    keys = set()
    for edge in row.get("edge_candidates", []):
        if isinstance(edge, dict) and all(edge.get(name) is not None
                                          for name in ("u", "v", "key")):
            keys.add(_edge_identity(edge))
    return keys


def create_clue(data: dict, *, actor=None, cancel_event=None,
                deadline_monotonic=None, on_created=None) -> dict:
    if not isinstance(data, dict):
        raise ValueError("线索数据无效")

    def should_abort():
        return bool(
            (cancel_event is not None and cancel_event.is_set())
            or (deadline_monotonic is not None
                and time.monotonic() >= deadline_monotonic)
        )

    if should_abort():
        raise ClueWriteCancelled("线索写入已取消")
    payload = dict(data)
    if actor is not None:
        payload["actor"] = actor
    row = _initial_row(payload)
    with clue_store_lock():
        if should_abort():
            raise ClueWriteCancelled("线索写入已取消")
        rows, changed = _load_current()
        _persist_if_changed(rows, changed, should_abort=should_abort)
        row["duplicate_of"] = next((
            other["clue_id"] for other in rows
            if other.get("status") not in {"rejected", "expired"} and _duplicate(row, other)
        ), None)
        if should_abort():
            raise ClueWriteCancelled("线索写入已取消")
        rows.append(row)
        after_commit = (
            (lambda: on_created(deepcopy(row))) if on_created is not None else None
        )
        _save(rows, should_abort=should_abort, after_commit=after_commit)
    return deepcopy(row)


def list_clues(*, status=None, statuses=None, kind=None, keyword=None,
               clue_id=None, limit=None) -> list:
    with clue_store_lock():
        rows, changed = _load_current()
        _persist_if_changed(rows, changed)
        selected = list(rows)
    if clue_id is not None:
        selected = [row for row in selected if row["clue_id"] == clue_id]
    if statuses is not None:
        allowed = set(statuses)
        selected = [row for row in selected if row["status"] in allowed]
    if status is not None:
        selected = [row for row in selected if row["status"] == status]
    if kind is not None:
        selected = [row for row in selected if row["kind"] == kind]
    if keyword:
        needle = str(keyword).strip().casefold()
        def matches(row):
            searchable = [
                row.get("clue_id", ""), row.get("summary", ""),
                row.get("source_label", ""), row.get("raw_text", ""),
                row.get("draft", {}).get("place", ""),
                row.get("draft", {}).get("road_hint", ""),
            ]
            searchable.extend(candidate.get("road_name", "")
                              for candidate in row.get("edge_candidates", []))
            return any(needle in str(value).casefold() for value in searchable if value)
        selected = [row for row in selected if matches(row)]
    selected.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
    if limit is not None:
        selected = selected[:max(0, int(limit))]
    return deepcopy(selected)


def get_clue(clue_id: str):
    rows = list_clues(clue_id=clue_id)
    return rows[0] if rows else None


def _allowed_patch_fields() -> set:
    return {"summary", "draft", "valid_from", "valid_to", "edge_candidates",
            "review_note", "status", "location"}


def update_clue(clue_id: str, changes: dict, *, actor="unknown") -> dict:
    if not isinstance(changes, dict) or set(changes) - _allowed_patch_fields():
        raise ValueError("包含不可修改的线索字段")
    with clue_store_lock():
        rows, changed = _load_current()
        _persist_if_changed(rows, changed)
        row = next((item for item in rows if item["clue_id"] == clue_id), None)
        if row is None:
            raise KeyError(clue_id)
        if row["status"] in {"rejected", "expired", "adopted"}:
            raise ValueError("已结束的线索不能修改")
        status = changes.get("status", row["status"])
        if status == "pending":
            if row["status"] != "pending":
                raise ValueError("不允许的线索状态转换")
        elif status == "reviewing":
            if row["status"] not in {"pending", "reviewing"}:
                raise ValueError("不允许的线索状态转换")
        elif status == "rejected":
            note = str(changes.get("review_note", row.get("review_note")) or "").strip()
            if not note:
                raise ValueError("驳回线索必须填写依据")
        elif status != row["status"]:
            raise ValueError("不允许的线索状态转换")
        updates = {key: deepcopy(value) for key, value in changes.items()}
        if "summary" in updates:
            summary = str(updates["summary"] or "").strip()
            if not summary or len(summary) > MAX_SUMMARY_CHARS:
                raise ValueError(f"线索摘要需为 1 到 {MAX_SUMMARY_CHARS} 字")
            updates["summary"] = summary
        if "draft" in updates:
            updates["draft"] = _clean_draft(updates["draft"])
        if "edge_candidates" in updates:
            candidates = updates["edge_candidates"]
            if not isinstance(candidates, list) or len(candidates) > MAX_EDGE_CANDIDATES:
                raise ValueError("候选路段无效")
            updates["edge_candidates"] = [_clean_edge_candidate(item) for item in candidates]
        if "location" in updates and updates["location"] is not None:
            location = updates["location"]
            if (not isinstance(location, dict)
                    or not _valid_gcj(location.get("lng"), location.get("lat"))):
                raise ValueError("线索位置无效")
        for field in ("valid_from", "valid_to"):
            if field in updates:
                updates[field] = normalize_time(updates[field]) if updates[field] else None
        if "review_note" in updates:
            updates["review_note"] = str(updates["review_note"] or "")[:1000]
        if updates.get("status") == "rejected" and not str(updates.get("review_note") or "").strip():
            raise ValueError("驳回线索必须填写依据")
        row.update(updates)
        row["updated_at"] = _now_iso()
        row.setdefault("audit", []).append({
            "action": "rejected" if status == "rejected" else "updated",
            "actor": str(actor or "unknown")[:80], "at": row["updated_at"],
            "changes": sorted(updates),
        })
        _validate(row)
        _save(rows)
        return deepcopy(row)


def add_candidate(clue_id: str, lng: float, lat: float, snap: dict,
                  *, actor="unknown") -> dict:
    """Append a server-verified candidate road selected on the manager map."""
    candidate = _clean_edge_candidate({**snap, "distance_m": snap.get("dist_m", 0)})
    with clue_store_lock():
        rows, changed = _load_current()
        _persist_if_changed(rows, changed)
        row = next((item for item in rows if item["clue_id"] == clue_id), None)
        if row is None:
            raise KeyError(clue_id)
        if row["status"] in {"rejected", "expired", "adopted"}:
            raise ValueError("已结束的线索不能修改")
        identity = _edge_identity(candidate)
        candidates = [item for item in row.get("edge_candidates", [])
                      if _edge_identity(item) != identity]
        candidates.insert(0, candidate)
        row["edge_candidates"] = candidates[:MAX_EDGE_CANDIDATES]
        if not row.get("location"):
            row["location"] = {"name": "", "lng": float(lng), "lat": float(lat)}
        row["updated_at"] = _now_iso()
        row.setdefault("audit", []).append({
            "action": "candidate_added", "actor": str(actor or "unknown")[:80],
            "at": row["updated_at"], "changes": {"edge": list(identity)},
        })
        _validate(row)
        _save(rows)
        return deepcopy(row)


# ===================== 空间匹配与采集限流 =====================

def _haversine_m(lng1, lat1, lng2, lat2) -> float:
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(min(1.0, a)))


def _point_to_segment_m(lng, lat, a, b) -> float:
    scale = 111320.0 * math.cos(math.radians(lat))
    ax, ay = (a[0] - lng) * scale, (a[1] - lat) * 111320.0
    bx, by = (b[0] - lng) * scale, (b[1] - lat) * 111320.0
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / (dx * dx + dy * dy)))
    return math.hypot(ax + t * dx, ay + t * dy)


def _candidate_distance_m(candidate: dict, lng: float, lat: float):
    geometry = candidate.get("geometry_gcj")
    if _valid_geometry(geometry):
        points = [(float(point[0]), float(point[1])) for point in geometry]
        return min(_point_to_segment_m(lng, lat, points[index], points[index + 1])
                   for index in range(len(points) - 1))
    if candidate.get("snap_lng_gcj") is not None:
        return _haversine_m(lng, lat, float(candidate["snap_lng_gcj"]),
                            float(candidate["snap_lat_gcj"]))
    return None


def clues_near(rows: list, lng: float, lat: float, radius_m: float = 300.0) -> list:
    """Filter clues whose bound candidate road lies within the radius.

    Clues without a verifiable candidate geometry or snap point are excluded;
    an unbound clue is not a nearby result.
    """
    if not _valid_gcj(lng, lat):
        raise ValueError("附近线索查询坐标无效")
    radius_m = _finite(radius_m, "附近线索半径")
    if radius_m <= 0 or radius_m > 5000:
        raise ValueError("附近线索半径需在 0 到 5000 米之间")
    matched = []
    for row in rows:
        for candidate in row.get("edge_candidates", []):
            distance = _candidate_distance_m(candidate, float(lng), float(lat))
            if distance is not None and distance <= radius_m:
                matched.append(row)
                break
    return matched


def _load_rate() -> dict:
    try:
        with open(_rate_path(), encoding="utf-8") as stream:
            data = json.load(stream)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError):
        return {}


def _save_rate(data: dict) -> None:
    path = _rate_path()
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         prefix=".road-clue-rate-", suffix=".json",
                                         delete=False) as stream:
            temp_path = stream.name
            json.dump(data, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def claim_collect(place_key: str, *, window_seconds: int = COLLECT_RATE_WINDOW_SECONDS):
    """Cross-process per-place rate limit. Returns (allowed, retry_after_seconds)."""
    key = str(place_key or "").strip().casefold()
    if not key:
        raise ValueError("采集地点无效")
    with _file_lock(_rate_path()):
        data = _load_rate()
        now = time.time()
        cutoff = now - 24 * 3600
        data = {name: float(value) for name, value in data.items()
                if isinstance(value, (int, float)) and float(value) >= cutoff}
        last = data.get(key)
        if isinstance(last, (int, float)) and now - float(last) < window_seconds:
            return False, max(0, int(window_seconds - (now - float(last))))
        data[key] = now
        _save_rate(data)
        return True, 0


def claim_turn_attempt(key: str, *, window_seconds: int = COLLECT_TURN_WINDOW_SECONDS) -> bool:
    """Prevent the same question from triggering repeated collection in one turn."""
    now = time.time()
    with _turn_lock:
        for name, expires in list(_turn_attempts.items()):
            if expires <= now:
                _turn_attempts.pop(name, None)
        if key in _turn_attempts:
            return False
        _turn_attempts[key] = now + window_seconds
        return True


def reset_collect_attempts() -> None:
    with _turn_lock:
        _turn_attempts.clear()


# ===================== 采纳（跨文件一致性） =====================

def _merge_window(targets: list) -> tuple:
    starts = [_timestamp(row.get("valid_from")) for row in targets]
    ends = [_timestamp(row.get("valid_to")) for row in targets]
    starts = [value for value in starts if value is not None]
    ends = [value for value in ends if value is not None]
    start = max(starts) if starts else None
    end = min(ends) if ends else None
    if start is not None and end is not None and end <= start:
        raise ValueError("合并线索的时间窗没有重叠")
    return start, end


def _candidate_has_edge(candidate: dict, edge_ref: tuple) -> bool:
    u, v, key = edge_ref
    return (candidate["u"], candidate["v"], candidate["key"]) in {
        (u, v, key), (v, u, key),
    }


def _event_source(targets: list, primary_id: str, token: str) -> dict:
    return {
        "kind": "clue",
        "clue_ids": [target["clue_id"] for target in targets],
        "primary_clue_id": primary_id,
        "adoption_token": token,
        "clue_kinds": {target["clue_id"]: target["kind"] for target in targets},
        "source_labels": sorted({target.get("source_label", "") for target in targets
                                 if target.get("source_label")}),
    }


def _existing_condition_for(token: str, primary_id: str, clue_ids: set):
    from spatial.road_conditions import list_conditions
    conditions = list_conditions(include_inactive=True, strict=True)
    for condition in conditions:
        source = condition.get("source") if isinstance(condition, dict) else None
        if not isinstance(source, dict) or source.get("kind") != "clue":
            continue
        if source.get("adoption_token") == token:
            return condition
        if source.get("primary_clue_id") == primary_id:
            return condition
        if clue_ids & set(source.get("clue_ids") or []):
            return condition
    return None


def adopt_clues(primary_id: str, clue_ids: list, *, edge: dict, cond_type: str,
                name: str, description: str = "", start_time=None, end_time=None,
                action=None, affected_modes=None, cost_multiplier=None,
                blocked_modes=None, direction: str = "both",
                review_note: str, actor: str = "web") -> tuple:
    """Adopt reviewed clues into one road event, atomically where possible.

    Lock order is always clue store → road-condition store. A persisted
    adoption token makes a retry reuse the event written before a crash instead
    of creating a duplicate. The final clue write is never silently ignored.
    """
    from spatial.road_conditions import (
        ClueConditionAlreadyPublishedError, RoadConditionsUnavailableError,
        add_condition,
    )

    ids = []
    for clue_id in [primary_id, *(clue_ids or [])]:
        if isinstance(clue_id, str) and clue_id and clue_id not in ids:
            ids.append(clue_id)
    if not ids:
        raise ValueError("请选择要采纳的线索")
    note = str(review_note or "").strip()
    if not 8 <= len(note) <= 300:
        raise ValueError("采纳依据须为 8 到 300 字")
    if cond_type not in EVENT_TYPES:
        raise ValueError("路况事件类型无效")
    if not isinstance(direction, str) or direction not in DIRECTIONS:
        raise ValueError("路况影响方向无效")
    if not isinstance(edge, dict) or edge.get("u") is None or edge.get("v") is None:
        raise ValueError("缺少采纳道路的吸附结果")
    edge_ref = (_as_int(edge.get("u"), "道路起点"), _as_int(edge.get("v"), "道路终点"),
                _as_int(edge.get("key", 0), "道路边标识"))
    try:
        snap_distance = float(edge.get("dist_m", 0))
    except (TypeError, ValueError):
        raise ValueError("道路吸附距离无效") from None
    if not math.isfinite(snap_distance) or not 0 <= snap_distance <= SNAP_MAX_DIST_M:
        raise ValueError(f"道路吸附距离必须在 {SNAP_MAX_DIST_M:g} 米以内")
    if not _valid_gcj(edge.get("snap_lng_gcj"), edge.get("snap_lat_gcj")):
        raise ValueError("道路吸附点无效")
    if not _valid_geometry(edge.get("geometry_gcj")):
        raise ValueError("道路几何无效")

    now = time.time()
    start_ts = _timestamp(start_time)
    end_ts = _timestamp(end_time)
    if end_ts is None:
        raise ValueError("采纳线索时必须设置结束时间")
    if end_ts <= now:
        raise ValueError("结束时间必须晚于当前时间")
    if start_ts is not None and end_ts <= start_ts:
        raise ValueError("结束时间必须晚于开始时间")

    with clue_store_lock():
        rows, changed = _load_current()
        _persist_if_changed(rows, changed)
        targets = [next((row for row in rows if row["clue_id"] == clue_id), None)
                   for clue_id in ids]
        if any(target is None for target in targets):
            raise KeyError("线索不存在")
        if all(target["status"] == "adopted" for target in targets):
            linked_sets = [list(target.get("condition_ids") or []) for target in targets]
            # Only one shared event id is an unambiguous replay; several linked
            # events mean we cannot tell which one this request refers to.
            if all(items == linked_sets[0] for items in linked_sets) and len(linked_sets[0]) == 1:
                condition_id = linked_sets[0][0]
                from spatial.road_conditions import list_conditions
                condition = next((item for item in list_conditions(include_inactive=True)
                                  if item.get("id") == condition_id), None)
                if condition is not None:
                    return condition, deepcopy(targets), True
            raise ValueError("线索已采纳但关联事件不一致，请先在事件记录中核对")
        if any(target["status"] != "reviewing" for target in targets):
            raise ValueError("只有复核中的线索才能采纳")
        for target in targets:
            if target.get("vision_flags"):
                flags = target["vision_flags"]
                if flags.get("trial") or flags.get("replay") or flags.get("stale"):
                    raise ClueNotPublishable(
                        "该影像线索来自试用、回放或已超出有效拍摄时间，不能发布为当前路况")
        if any(target["kind"] == "vision" for target in targets):
            if action not in ACTIONS:
                raise ValueError("影像线索发布时必须明确提示、缓行或封路")
            if not _valid_modes(affected_modes):
                raise ValueError("影像线索发布时必须选择受影响的出行方式")
        if action is not None:
            if action not in ACTIONS:
                raise ValueError("路况影响动作无效")
            if not _valid_modes(affected_modes):
                raise ValueError("受影响的出行方式无效")
            if action == "slowdown" and (
                    cost_multiplier is None or not 1 <= float(cost_multiplier) <= 5):
                raise ValueError("缓行成本倍数必须在 1 到 5 之间")
        elif not _valid_modes(blocked_modes):
            raise ValueError("至少选择一种有效且不重复的禁行方式")

        merged_types = {target["draft"].get("event_type") for target in targets
                        if target["draft"].get("event_type")}
        if len(merged_types) > 1:
            raise ValueError("合并线索的事件类型不一致")
        for target in targets:
            if not any(_candidate_has_edge(candidate, edge_ref)
                       for candidate in target.get("edge_candidates", [])):
                raise ValueError("选定道路不在该线索的可核验候选路段中")
        if len(targets) > 1:
            shared = set.intersection(*[_edge_keys(target) for target in targets])
            if _edge_identity({"u": edge_ref[0], "v": edge_ref[1], "key": edge_ref[2]}) not in shared:
                raise ValueError("合并线索必须指向同一路段")

        window_start, window_end = _merge_window(targets)
        if start_ts is None:
            start_ts = window_start if window_start is not None else now
        if window_start is not None and start_ts < window_start - 0.5:
            raise ValueError("事件开始时间早于线索的有效时间窗")
        if window_end is not None and end_ts > window_end + 0.5:
            raise ValueError("事件结束时间超出线索的有效时间窗")

        token = next(((target.get("adoption") or {}).get("token")
                      for target in targets if isinstance(target.get("adoption"), dict)
                      and target["adoption"].get("token")), None)
        token = token or uuid.uuid4().hex
        existing = _existing_condition_for(token, primary_id, {target["clue_id"] for target in targets})
        if existing is None:
            for target in targets:
                target["adoption"] = {"token": token, "started_at": _now_iso()}
            _save(rows)
            event_source = _event_source(targets, primary_id, token)
            if any(target["kind"] == "vision" for target in targets):
                event_source["field_confirmation"] = note
            try:
                condition = add_condition(
                    cond_type=cond_type,
                    name=str(name or "").strip()[:30] or "线索采纳路况",
                    edge=edge,
                    click_point={"lng": float(edge["snap_lng_gcj"]),
                                 "lat": float(edge["snap_lat_gcj"])},
                    description=str(description or "").strip()[:200],
                    start_time=start_ts,
                    end_time=end_ts,
                    created_by=str(actor or "web")[:80],
                    source=event_source,
                    blocked_modes=blocked_modes,
                    action=action,
                    affected_modes=affected_modes,
                    cost_multiplier=cost_multiplier,
                    direction=direction,
                )
            except ClueConditionAlreadyPublishedError as exc:  # pragma: no cover - race guard
                condition = exc.condition
            except Exception:
                for target in targets:
                    target["adoption"] = None
                try:
                    _save(rows)
                except Exception:
                    pass
                raise
        else:
            condition = existing

        now_iso = _now_iso()
        for target in targets:
            linked = list(target.get("condition_ids") or [])
            if condition["id"] not in linked:
                linked.append(condition["id"])
            target["condition_ids"] = linked
            target["status"] = "adopted"
            target["adoption"] = None
            target["review_note"] = (f"{target.get('review_note')}；{note}" if target.get("review_note")
                                     else note)[:1000]
            target["updated_at"] = now_iso
            target.setdefault("audit", []).append({
                "action": "adopted", "actor": str(actor or "web")[:80],
                "at": now_iso, "changes": {"condition_id": condition["id"]},
            })
        _save(rows)
        return condition, deepcopy(targets), bool(existing is not None)


def detach_condition(condition_id: str, *, actor: str = "system") -> list:
    """Remove an event id from linked clues after a deletion.

    A clue with no remaining event returns to reviewing; a clue still linked to
    another valid event stays adopted.
    """
    changed_ids = []
    with clue_store_lock():
        rows, changed = _load_current()
        _persist_if_changed(rows, changed)
        now = _now_iso()
        for row in rows:
            linked = list(row.get("condition_ids") or [])
            if condition_id not in linked:
                continue
            row["condition_ids"] = [item for item in linked if item != condition_id]
            if not row["condition_ids"] and row.get("status") == "adopted":
                row["status"] = "reviewing"
                row["adoption"] = None
            row["updated_at"] = now
            row.setdefault("audit", []).append({
                "action": "event_deleted", "actor": str(actor or "system")[:80],
                "at": now, "changes": {"condition_id": condition_id},
            })
            changed_ids.append(row["clue_id"])
        if changed_ids:
            _save(rows)
    return changed_ids
