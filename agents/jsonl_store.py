"""JSONL 运行数据存储的共享实现：跨进程锁 + 追加 + 超限裁剪。

Agent 侧有两处按请求追加的 JSONL（溯源记录、经验样本）。两者都要求：
- 多 Web 进程并发追加时记录不互相截断；
- 文件不能无限增长，超过阈值时保留后半段；
- 任何失败都只告警，不阻断主链路。

把这两件事放在一处，避免在不同模块里各自维护一份容易写错的锁实现。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# 同一进程内串行化所有 JSONL 写入（写入量很小，单锁足够）
_write_lock = threading.Lock()


@contextmanager
def file_lock(path):
    """跨进程独占锁：Windows 用 msvcrt，其余用 fcntl。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(path) + ".lock"
    with _write_lock, open(lock_path, "a+b") as lock_file:
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


def prune(path, max_bytes: int) -> None:
    """超过阈值时保留后半段记录。调用方必须已持有 file_lock。"""
    path = Path(path)
    try:
        if path.stat().st_size <= max_bytes:
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = lines[len(lines) // 2:]
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=str(path.parent),
                    prefix=f".{path.name}-", suffix=".tmp", delete=False) as stream:
                temp_path = stream.name
                stream.write("\n".join(kept) + ("\n" if kept else ""))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception as e:  # 裁剪失败不影响主链路
        logger.warning("JSONL 裁剪失败（忽略） %s: %s", path.name, e)


def append_record(path, record: dict, *, max_bytes: int | None = None) -> None:
    """在文件锁内追加一行 JSON；max_bytes 给定且在锁内裁剪。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with file_lock(path):
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        if max_bytes:
            prune(path, max_bytes)
