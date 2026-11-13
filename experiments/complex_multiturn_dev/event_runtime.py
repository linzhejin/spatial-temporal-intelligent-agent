"""Activate source-bound simulated event snapshots in a local isolated process."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_event_bundle(path, dataset, graph, pois):
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    if bundle.get("scope") != "synthetic_development_only" or bundle.get("campus_truth") is not False:
        raise ValueError("not a synthetic development event bundle")
    for name, source in (("dataset", dataset), ("graph", graph), ("poi", pois)):
        if bundle.get(name + "_sha256") != _sha(source):
            raise ValueError(name + " source differs from event bundle")
    fixtures = bundle.get("fixtures")
    if not isinstance(fixtures, list) or len({item.get("case_id") for item in fixtures}) != len(fixtures):
        raise ValueError("invalid event fixture list")
    return {item["case_id"]: item for item in fixtures}


def activate_event(fixture, stage, *, run_id, turn_number):
    """Bind an immutable stage file and let the project's guard certify it."""
    from experiments.event_replay_guard import REPLAY_ROOT, certify_local_event_fixture
    from spatial import road_conditions

    snapshot = fixture["stages"][stage]
    rows, epoch = snapshot["rows"], snapshot["epoch"]
    if not rows or type(epoch) is not int:
        raise ValueError("event stage has no frozen rows or time")
    target_dir = REPLAY_ROOT / run_id / fixture["case_id"] / str(turn_number)
    target_dir.mkdir(parents=True, exist_ok=False)
    path = target_dir / "road_conditions.json"
    raw = (json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("xb") as out:
        out.write(raw)
    path = path.resolve()
    os.environ.update(FLASK_ENV="development", AGENT_REPLAY_ENABLE_LOCAL="1",
                      AGENT_TOOL_TRACE_ENABLED="1", AMAP_POI_LOOKUP_ENABLED="false",
                      ROAD_CONDITIONS_FILE=str(path), AGENT_REPLAY_EVENT_EPOCH=str(epoch),
                      AGENT_REPLAY_EVENT_SHA256=hashlib.sha256(raw).hexdigest())
    road_conditions._CONDITIONS_FILE = str(path)
    proof = certify_local_event_fixture(path)
    if proof["as_of_epoch"] != epoch or proof["fixture_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("event guard did not certify the intended snapshot")
    active = road_conditions.list_conditions(strict=True)
    expected_ids = {row["id"] for row in rows if row["start_time"] <= epoch
                    and (not row["end_time"] or epoch < row["end_time"])
                    and (row.get("revoked_at") is None or epoch < row["revoked_at"])}
    if {row["id"] for row in active} != expected_ids:
        raise ValueError("active event rows do not match the frozen timeline")
    return {"stage": stage, "epoch": epoch, "fixture_sha256": proof["fixture_sha256"],
            "active_ids": sorted(expected_ids), "path": str(path)}


def deactivate_event(path):
    from spatial import road_conditions
    os.environ.pop("AGENT_REPLAY_EVENT_EPOCH", None)
    os.environ.pop("AGENT_REPLAY_EVENT_SHA256", None)
    os.environ["ROAD_CONDITIONS_FILE"] = str(path)
    road_conditions._CONDITIONS_FILE = str(path)
