"""Fail-closed local guard for synthetic road-event development replay."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from psycopg.conninfo import conninfo_to_dict


ROOT = Path(__file__).resolve().parents[1]
REPLAY_ROOT = ROOT / "output/experiments/event_replay"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EPOCH = re.compile(r"[0-9]{1,12}\Z")


def assert_replay_database_target(database_url: str | None) -> None:
    """Reject any non-local or non-test database before enabling replay."""
    try:
        info = conninfo_to_dict(database_url or "")
    except Exception as exc:
        raise ValueError("event replay requires a local test database") from exc
    name = info.get("dbname") or ""
    host = (info.get("host") or "").casefold()
    if (not re.search(r"(?:^|_)test(?:_|$)", name, re.IGNORECASE)
            or host not in {"localhost", "127.0.0.1", "::1"}):
        raise ValueError("event replay requires a local test database")


def _certified_fixture(bound_path: str | Path) -> tuple[dict[str, object], list[dict]]:
    """Read, hash and validate one immutable snapshot for both proof and routing."""
    if (os.getenv("FLASK_ENV", "development") == "production"
            or os.getenv("AGENT_REPLAY_ENABLE_LOCAL") != "1"
            or os.getenv("AGENT_TOOL_TRACE_ENABLED") != "1"):
        raise ValueError("event replay is disabled")
    assert_replay_database_target(os.getenv("DATABASE_URL"))
    epoch_text = os.getenv("AGENT_REPLAY_EVENT_EPOCH", "")
    expected_sha = os.getenv("AGENT_REPLAY_EVENT_SHA256", "")
    if not _EPOCH.fullmatch(epoch_text) or not _SHA256.fullmatch(expected_sha):
        raise ValueError("frozen epoch and fixture digest are required")
    configured = os.getenv("ROAD_CONDITIONS_FILE", "")
    raw = Path(configured) if configured else None
    if raw is None or not raw.is_absolute() or not raw.is_file():
        raise ValueError("an existing absolute event fixture path is required")
    target = raw.resolve(strict=True)
    replay_root = REPLAY_ROOT.resolve(strict=True)
    if (target != raw.absolute() or not target.is_relative_to(replay_root)
            or target == replay_root or Path(bound_path).resolve() != target):
        raise ValueError("road-condition store is outside the isolated replay root")
    content = target.read_bytes()
    if len(content) > 2_000_000 or hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError("event fixture changed or is too large")
    try:
        rows = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("event fixture is invalid JSON") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("event fixture must contain records")
    from spatial import road_conditions as rc

    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("event fixture record is invalid")
        if row["id"] in seen or not row["id"]:
            raise ValueError("event fixture IDs must be unique")
        seen.add(row["id"])
        source = row.get("source")
        if (not isinstance(source, dict)
                or source.get("kind") != "research_simulated"
                or source.get("scope") != "synthetic_development_only"
                or source.get("campus_truth") is not False):
            raise ValueError("event fixture includes non-simulated records")
        rc._validate_condition_record(row)
    return {"as_of_epoch": int(epoch_text), "fixture_sha256": expected_sha}, rows


def certify_local_event_fixture(bound_path: str | Path) -> dict[str, object]:
    """Prove the exact file bound by routing is the frozen synthetic fixture."""
    proof, _ = _certified_fixture(bound_path)
    return proof


def certified_event_snapshot(bound_path: str | Path) -> tuple[int, list[dict]] | None:
    """Ordinary processes use the normal store; replay routes use checked bytes."""
    if (os.getenv("AGENT_REPLAY_EVENT_EPOCH") is None
            and os.getenv("AGENT_REPLAY_EVENT_SHA256") is None):
        return None
    proof, rows = _certified_fixture(bound_path)
    return int(proof["as_of_epoch"]), rows
