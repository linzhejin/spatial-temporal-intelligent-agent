"""珞珈智行 — 经验知识层：执行编排的沉淀与召回。

OpenGA 的知识注入闭环里有一环是「知识沉淀与自进化」。本项目原先只有静态
任务卡（领域知识），执行结果不回流。本模块补上这一层：

- record_run：把一次已核验通过 / 明确成功的工具编排追加写到 JSONL（运行时，best-effort）；
- distill：离线把 JSONL 聚合成经验卡（scenario_key + 工具序列 → 成功/失败计数）；
- retrieve_experience：运行时按 scenario_key / 触发词召回高置信经验卡（只读）。

设计约束：
- 只沉淀「已核验通过或明确成功」的编排，绝不回灌 LLM 原文或失败样本；
- 运行时只读经验卡，不在线改写提示词（保确定性，避免 fail-open）；
- 不落原始 query 文本，场景键优先复用任务卡 id，零命中时退回查询指纹。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from agents import jsonl_store

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
EXPERIENCE_PATH = DATA_DIR / "agent_experience.jsonl"
CARDS_PATH = DATA_DIR / "experience_cards.json"
FLAT_CARDS_PATH = DATA_DIR / "task_cards.json"

# 召回门槛：样本够多且成功率够高才注入，避免用噪声模式误导模型
MIN_SUCCESS = 3
MIN_SUCCESS_RATIO = 0.8
MAX_EXPERIENCE_CARDS = 2

# 样本文件上限：超限保留后半段，避免按请求追加的文件无限增长
MAX_FILE_BYTES = 4 * 1024 * 1024

_cards_cache = None
_cards_cache_mtime = 0.0
_task_cards_cache = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def query_fingerprint(query: str) -> str:
    """归一化后的查询指纹（不可逆，仅用于兜底场景键）。"""
    return hashlib.sha256(_normalize(query).encode("utf-8")).hexdigest()[:16]


def _load_task_cards() -> list:
    """加载任务卡（进程内缓存）。

    该函数位于每个请求的热路径上（场景键推导），必须缓存；
    任务卡在运行时只读，进程内缓存不会造成可见性差异。
    """
    global _task_cards_cache
    if _task_cards_cache is None:
        try:
            data = json.loads(FLAT_CARDS_PATH.read_text(encoding="utf-8"))
            _task_cards_cache = data.get("cards", []) if isinstance(data, dict) else []
        except Exception as e:
            logger.warning("任务卡加载失败（经验层按空知识运行）: %s", e)
            return []
    return _task_cards_cache


def scenario_key(query: str) -> str:
    """场景键：优先复用任务卡 id（命中触发词），零命中时退回查询指纹。"""
    if not query:
        return "unknown"
    best = None
    best_hits = 0
    for card in _load_task_cards():
        hits = sum(1 for t in card.get("triggers", []) if t and t in query)
        if hits > best_hits:
            best, best_hits = card.get("id"), hits
    if best:
        return str(best)
    return "q:" + query_fingerprint(query)


def tool_sequence(tool_trace) -> list:
    """从工具回调序列里抽出工具名序列。"""
    sequence = []
    for item in tool_trace or []:
        if isinstance(item, dict) and item.get("tool_name"):
            sequence.append(str(item["tool_name"]))
    return sequence


def record_run(query: str, tool_trace, *, success: bool, verification_status: str = None,
               now: str = None) -> bool:
    """追加写一条已完成的编排样本；失败只 warning。

    调用方必须只在「核验通过 / 明确成功」时置 success=True；失败样本同样记录，
    但只用于把成功率拉下去（distill 会据此剔除低置信模式）。
    """
    try:
        sequence = tool_sequence(tool_trace)
        entry = {
            "at": now or _now_iso(),
            "scenario_key": scenario_key(query),
            "tool_sequence": sequence,
            "success": bool(success),
            "verification_status": verification_status,
        }
        jsonl_store.append_record(EXPERIENCE_PATH, entry, max_bytes=MAX_FILE_BYTES)
        return True
    except Exception as e:
        logger.warning("经验样本写入失败（忽略）: %s", e)
        return False


def _atomic_write(path: Path, payload) -> None:
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(directory),
                                         prefix=".experience-", suffix=".json",
                                         delete=False) as stream:
            temp_path = stream.name
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def distill(input_path: Path = None, output_path: Path = None) -> dict:
    """离线蒸馏：JSONL 样本 → 经验卡。返回写出的卡片字典。"""
    input_path = Path(input_path) if input_path else EXPERIENCE_PATH
    output_path = Path(output_path) if output_path else CARDS_PATH

    aggregates = {}
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    except Exception as e:
        logger.warning("经验样本读取失败: %s", e)
        lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("scenario_key") or "")
        sequence = [str(name) for name in (entry.get("tool_sequence") or []) if name]
        if not key or not sequence:
            continue
        bucket = aggregates.setdefault(
            (key, tuple(sequence)),
            {"scenario_key": key, "tool_sequence": list(sequence),
             "success_count": 0, "failure_count": 0, "last_seen": ""},
        )
        if entry.get("success"):
            bucket["success_count"] += 1
        else:
            bucket["failure_count"] += 1
        at = str(entry.get("at") or "")
        if at > bucket["last_seen"]:
            bucket["last_seen"] = at

    cards = sorted(aggregates.values(),
                   key=lambda c: (-c["success_count"], c["scenario_key"]))
    payload = {
        "generated_at": _now_iso(),
        "thresholds": {"min_success": MIN_SUCCESS, "min_success_ratio": MIN_SUCCESS_RATIO},
        "cards": cards,
    }
    _atomic_write(output_path, payload)
    return payload


def _load_cards() -> list:
    global _cards_cache, _cards_cache_mtime
    try:
        mtime = CARDS_PATH.stat().st_mtime_ns
    except OSError:
        return []
    if _cards_cache is None or mtime != _cards_cache_mtime:
        try:
            data = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
            cards = data.get("cards", []) if isinstance(data, dict) else []
            _cards_cache = [c for c in cards if isinstance(c, dict)]
            _cards_cache_mtime = mtime
        except Exception as e:
            logger.warning("经验卡加载失败，按空经验运行: %s", e)
            return []
    return _cards_cache


def _is_confident(card: dict) -> bool:
    success = card.get("success_count")
    failure = card.get("failure_count")
    if not isinstance(success, int) or not isinstance(failure, int):
        return False
    total = success + failure
    if success < MIN_SUCCESS or total <= 0:
        return False
    return success / total >= MIN_SUCCESS_RATIO


def retrieve_experience(query: str, limit: int = MAX_EXPERIENCE_CARDS) -> list:
    """按场景键召回高置信经验卡（运行时只读）。"""
    if not query:
        return []
    key = scenario_key(query)
    hits = []
    for card in _load_cards():
        if card.get("scenario_key") != key:
            continue
        if not _is_confident(card):
            continue
        hits.append(card)
    hits.sort(key=lambda c: (-c.get("success_count", 0), c.get("scenario_key", "")))
    return hits[:limit]


def reset_cache():
    """仅供测试使用：清空经验卡与任务卡缓存。"""
    global _cards_cache, _cards_cache_mtime, _task_cards_cache
    _cards_cache = None
    _cards_cache_mtime = 0.0
    _task_cards_cache = None
