"""珞珈智行 — Agent 执行的溯源记录（ISO/W3C PROV 风格轻量实现）。

把一次 Agent run 记为「实体（输入/产出）— 活动 — 代理（模型）」，
并保留按序的工具活动，使一次推理链路可追溯、可回放。

与既有溯源出口的关系（同源不同出口，不重复采集）：
- agents/workflow.py 把工具调用事件写 PostgreSQL（仅后台 worker 路径）；
- api/routes.py 的 /api/telemetry 收前端行为埋点；
- 本模块在进程内把同一份工具回调规范化为 run 级 PROV 记录。

隐私与安全红线：
- 不落原始 query 文本，只落 sha256 前缀；
- 不落任何坐标，只落 has_gps / has_waypoints 这类布尔标志；
- 记录只允许管理员通过 /api/agent/provenance/<run_id> 读取。

失败即降级：任何持久化/序列化异常只记 warning，绝不阻断主链路。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agents import jsonl_store

logger = logging.getLogger(__name__)

PROVENANCE_PATH = Path(__file__).parent.parent / "data" / "provenance.jsonl"

# 追加写阈值：超过该大小时把旧记录裁剪掉一半，避免无限增长
MAX_FILE_BYTES = 8 * 1024 * 1024

PROV_NS = "http://www.w3.org/ns/prov#"


def _file_lock():
    """跨进程锁，作用于当前的 PROVENANCE_PATH（测试会替换该路径）。"""
    return jsonl_store.file_lock(PROVENANCE_PATH)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def query_fingerprint(query: str) -> str:
    """query 的不可逆指纹（只保留前 16 位十六进制）。"""
    return _digest(query or "")[:16]


def tool_set_fingerprint(tool_names) -> str:
    """工具集指纹：对排序后的工具名取哈希前缀，便于比对不同版本的能力集。"""
    joined = "\n".join(sorted(str(name) for name in (tool_names or [])))
    return _digest(joined)[:16]


class ProvenanceRecorder:
    """一次 Agent run 的溯源记录器（进程内，不落盘直到 finish）。"""

    def __init__(self, query: str, *, uid: str = None, model: str = None,
                 temperature: float = 0.0, tool_names=None,
                 has_gps: bool = False, has_waypoints: bool = False,
                 data_version: str = None, road_condition_version: str = None,
                 task_revision: int = None):
        self.run_id = str(uuid4())
        self._record = None
        self._started_at = _utc_now()
        self._started_monotonic = time.monotonic()
        self._model = model
        self._temperature = temperature
        self._tool_names = sorted(str(name) for name in (tool_names or []))
        self._uid = str(uid) if uid else None
        self._input = {
            "query_sha256": query_fingerprint(query),
            "has_gps": bool(has_gps),
            "has_waypoints": bool(has_waypoints),
            "data_version": data_version,
            "road_condition_version": road_condition_version,
            "task_revision": task_revision if isinstance(task_revision, int) else None,
        }
        self._activities = []
        self._influenced_by = []

    def add_activity(self, tool_name: str, argument_keys=None, outcome: str = None,
                     artifact_kind: str = None) -> None:
        """登记一次工具活动（与 tool_trace_callback 同源）。"""
        self._activities.append({
            "tool_name": str(tool_name or "unknown")[:80],
            "argument_keys": sorted({str(key) for key in (argument_keys or [])
                                     if str(key)})[:40],
            "outcome": outcome,
            "artifact_kind": artifact_kind,
            "at": _utc_now(),
        })

    def add_influence(self, reason: str) -> None:
        """登记一次反思-重规划依赖（wasInfluencedBy）。"""
        if reason and reason not in self._influenced_by:
            self._influenced_by.append(str(reason)[:80])

    def finish(self, *, response_kind: str = None, artifact_kinds=None,
               verification_status: str = None, route_state_digest: str = None,
               proof_id: str = None, turns: int = None) -> dict:
        """封口并返回 PROV 记录；重复调用返回同一记录。"""
        if self._record is not None:
            return self._record

        output_entity = {
            "id": f"entity:{self.run_id}:output",
            "role": "output",
            "response_kind": response_kind,
            "artifact_kinds": sorted({str(kind) for kind in (artifact_kinds or []) if kind}),
            "verification_status": verification_status,
            "route_state_digest": route_state_digest,
            "proof_id": proof_id,
            "turns": turns,
        }
        agent_id = f"agent:{tool_set_fingerprint(self._tool_names)}"
        activity = {
            "id": f"activity:{self.run_id}",
            "label": "campus_agent_run",
            "started_at": self._started_at,
            "ended_at": _utc_now(),
            "elapsed_ms": round((time.monotonic() - self._started_monotonic) * 1000, 1),
            "used": [f"entity:{self.run_id}:input"],
            "wasGeneratedBy": [output_entity["id"]],
            "wasAssociatedWith": [agent_id],
            "wasInfluencedBy": list(self._influenced_by),
            "tools_used": [item["tool_name"] for item in self._activities],
        }
        self._record = {
            "run_id": self.run_id,
            "prov": {"namespace": PROV_NS},
            "generated_at": activity["ended_at"],
            "session": {"uid": self._uid},
            "activity": activity,
            "agents": [{
                "id": agent_id,
                "role": "model",
                "model": self._model,
                "temperature": self._temperature,
                "tool_count": len(self._tool_names),
                "tools_version": tool_set_fingerprint(self._tool_names),
            }],
            "entities": [
                {"id": f"entity:{self.run_id}:input", "role": "input", **self._input},
                output_entity,
            ],
            "tool_activities": list(self._activities),
        }
        return self._record


def _prune(path: Path) -> None:
    """超过阈值时保留后半段记录。调用方必须已持有文件锁。"""
    jsonl_store.prune(path, MAX_FILE_BYTES)


def persist(record: dict) -> bool:
    """把一条 PROV 记录追加落到 JSONL；失败只 warning。

    追加与裁剪在同一把「进程内线程锁 + 跨进程文件锁」下完成，
    避免两个 Web 进程同时裁剪时互相截断对方的记录。
    """
    if not isinstance(record, dict) or not record.get("run_id"):
        return False
    try:
        jsonl_store.append_record(PROVENANCE_PATH, record, max_bytes=MAX_FILE_BYTES)
        return True
    except Exception as e:
        logger.warning("溯源记录写入失败（忽略）: %s", e)
        return False


def load(run_id: str) -> dict | None:
    """按 run_id 读取一条 PROV 记录；不存在返回 None。"""
    if not run_id or not isinstance(run_id, str):
        return None
    try:
        with open(PROVENANCE_PATH, "r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("run_id") == run_id:
                    return record
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("溯源记录读取失败: %s", e)
        return None
    return None
