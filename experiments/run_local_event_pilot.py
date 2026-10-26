"""Run the frozen synthetic road-event pilot through an isolated local Agent.

This is a development diagnostic, never a field or blind-test result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

from dotenv import dotenv_values
from psycopg.conninfo import conninfo_to_dict
import requests

from experiments.event_live_replay import (
    collect_event_case, prepare_event_fixture, require_isolated_event_service,
)
from experiments.event_replay_guard import REPLAY_ROOT, assert_replay_database_target
from experiments.scenario_live_adapter import build_poi_lookup
from tests.db_safety import verify_isolated_test_database


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/datasets/controlled_event_pilot_v3"
SCORE_ROOT = ROOT / "output/experiments/controlled_event_pilot_v3"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_PASS_ENV = {
    "PATH", "SystemRoot", "WINDIR", "TEMP", "TMP", "USERPROFILE",
    "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA",
    "COMSPEC", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy",
    "no_proxy", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
}


def load_poi_lookup(path: Path) -> dict:
    """Read the published POI envelope used by the live service."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pois = payload.get("pois") if isinstance(payload, dict) else None
    if not isinstance(pois, list) or any(not isinstance(poi, dict) for poi in pois):
        raise ValueError("invalid POI master data")
    return build_poi_lookup(pois)


def isolate_agent_sidecars(fixture_path: str | Path) -> dict[str, Path]:
    """Keep per-run experience and provenance writes beside the event fixture."""
    from agents import experience, provenance

    path = Path(fixture_path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("event fixture must be a file")
    sidecars = path.parent / "agent_sidecars"
    sidecars.mkdir(exist_ok=False)
    experience.EXPERIENCE_PATH = sidecars / "agent_experience.jsonl"
    provenance.PROVENANCE_PATH = sidecars / "provenance.jsonl"
    return {"experience": experience.EXPERIENCE_PATH,
            "provenance": provenance.PROVENANCE_PATH}


def disable_child_dotenv() -> None:
    """Prevent app and worker imports from reloading checkout credentials."""
    import dotenv

    dotenv.load_dotenv = lambda *_args, **_kwargs: False


def collection_source_versions(model_base_url: str | None,
                               model_name: str | None) -> dict[str, str]:
    """Record research inputs without persisting any API credential."""
    from experiments.controlled_event_replay_v3_run import _runtime_source_tree_sha

    def file_hash(relative: str) -> str:
        return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

    return {
        "model": model_name or "deepseek-chat",
        "provider_host": urlsplit(model_base_url or "https://api.deepseek.com/v1").hostname or "",
        "runtime_source_tree_sha256": _runtime_source_tree_sha(ROOT),
        "graph_sha256": file_hash("data/whu_road_network.graphml"),
        "poi_sha256": file_hash("data/pois.json"),
        "prompt_sha256": file_hash("agents/prompts/agent_system.txt"),
        "tool_code_sha256": file_hash("agents/tools.py"),
    }


def build_replay_environment(
    test_database_url: str, fixture: dict, *, model_key: str,
    base_env: dict[str, str] | None = None, model_base_url: str | None = None,
    model_name: str | None = None,
) -> dict[str, str]:
    """Pass only the local fixture, test DB and explicit model access to a child."""
    assert_replay_database_target(test_database_url)
    if not model_key:
        raise ValueError("a model key is required for real Agent replay")
    path = Path(fixture.get("path", ""))
    digest = fixture.get("fixture_sha256")
    epoch = fixture.get("as_of_epoch")
    if (not path.is_absolute() or not path.is_file()
            or not isinstance(digest, str)
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            or type(epoch) is not int):
        raise ValueError("event fixture is missing or changed")
    parent = os.environ if base_env is None else base_env
    pass_names = {name.casefold() for name in _PASS_ENV}
    child = {key: value for key, value in parent.items()
             if key.casefold() in pass_names and isinstance(value, str)}
    child.update(
        DATABASE_URL=test_database_url,
        ROAD_CONDITIONS_FILE=str(path.resolve()),
        AGENT_REPLAY_EVENT_EPOCH=str(epoch),
        AGENT_REPLAY_EVENT_SHA256=digest,
        AGENT_REPLAY_ENABLE_LOCAL="1",
        AGENT_TOOL_TRACE_ENABLED="1",
        AMAP_POI_LOOKUP_ENABLED="false",
        FLASK_ENV="development",
        AGENT_WORKER_CONCURRENCY="1",
        DEEPSEEK_API_KEY=model_key,
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
    )
    if model_base_url:
        child["OPENAI_BASE_URL"] = model_base_url
    if model_name:
        child["LLM_MODEL"] = model_name
    return child


def _local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(port: int) -> None:
    """Serve one fixture and one worker in the same disposable process."""
    import threading
    from werkzeug.serving import make_server

    disable_child_dotenv()
    from app import create_app
    from experiments.event_replay_guard import certify_local_event_fixture
    from jobs.worker import run_forever

    assert_replay_database_target(os.environ["DATABASE_URL"])
    certify_local_event_fixture(os.environ["ROAD_CONDITIONS_FILE"])
    isolate_agent_sidecars(os.environ["ROAD_CONDITIONS_FILE"])
    app = create_app()
    worker = threading.Thread(target=run_forever, kwargs={"concurrency": 1},
                              name="local-event-worker", daemon=True)
    worker.start()
    make_server("127.0.0.1", port, app, threaded=True).serve_forever()


def _wait_ready(session: requests.Session, base_url: str, fixture: dict,
                child: subprocess.Popen, *, timeout_s: float = 75) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError("local Agent service exited before readiness")
        try:
            require_isolated_event_service(session, base_url, fixture)
            return
        except (ValueError, requests.RequestException):
            time.sleep(0.5)
    raise TimeoutError("local Agent service did not prove isolation in time")


def _run(run_id: str, test_database_url: str, env_file: Path | None,
         *, timeout_s: float) -> Path:
    if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("invalid run ID")
    verify_isolated_test_database(test_database_url)
    root = REPLAY_ROOT / run_id
    scored = SCORE_ROOT / run_id
    if root.exists() or scored.exists():
        raise FileExistsError("run ID already has local output")
    source = dotenv_values(env_file) if env_file else {}
    model_key = os.environ.get("DEEPSEEK_API_KEY") or source.get("DEEPSEEK_API_KEY")
    if not model_key:
        raise ValueError("a model key is required for real Agent replay")
    model_url = os.environ.get("OPENAI_BASE_URL") or source.get("OPENAI_BASE_URL")
    model_name = os.environ.get("LLM_MODEL") or source.get("LLM_MODEL")
    source_versions = collection_source_versions(model_url, model_name)

    dataset = json.loads((PILOT / "scenarios_event_pilot_dev_v3.json")
                         .read_text(encoding="utf-8"))
    events = json.loads((PILOT / "event_snapshots_dev_v1.json")
                        .read_text(encoding="utf-8"))
    snapshots = {item["id"]: item for item in events["snapshots"]}
    lookup = load_poi_lookup(ROOT / "data/pois.json")
    root.mkdir(parents=True, exist_ok=False)
    replay_file = root / "replay.jsonl"
    metadata = []
    for case in dataset["cases"]:
        snapshot_id = case["gold_derivation"]["parameters"]["event_snapshot_id"]
        fixture = prepare_event_fixture(case, snapshots[snapshot_id], run_id)
        child_env = build_replay_environment(
            test_database_url, fixture, model_key=model_key,
            model_base_url=model_url, model_name=model_name,
        )
        port = _local_port()
        base_url = f"http://127.0.0.1:{port}"
        log_path = root / case["id"] / "service.log"
        with log_path.open("wb") as log, requests.Session() as session:
            child = subprocess.Popen(
                [sys.executable, "-m", "experiments.run_local_event_pilot",
                 "--serve", "--port", str(port)],
                cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                _wait_ready(session, base_url, fixture, child)
                row, evidence = collect_event_case(
                    case, session, base_url, lookup, fixture,
                    timeout_s=timeout_s,
                )
            finally:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with replay_file.open("a", encoding="utf-8") as stream:
            stream.write(line)
        metadata.append({"scenario_id": case["id"], **evidence,
                         "projection_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                         "service_log": str(log_path.relative_to(root))})
        if collection_source_versions(model_url, model_name) != source_versions:
            raise RuntimeError("runtime source or data changed during replay")
    (root / "collection_manifest.json").write_text(json.dumps({
        "run_type": "development_diagnostic", "formal_ready": False,
        "campus_truth": False, "observation_origin": "local_real_agent_unverified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_database_name": conninfo_to_dict(test_database_url)["dbname"],
        **source_versions,
        "cases": metadata,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from experiments.controlled_event_replay_v3_run import run_controlled_event_replay_v3

    return run_controlled_event_replay_v3(
        replay_file, SCORE_ROOT, run_id, observation_origin="unverified_projection",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--run-id")
    parser.add_argument("--test-database-url", default=os.environ.get("TEST_DATABASE_URL"))
    parser.add_argument("--env-file", type=Path,
                        help="Local credential file; only model settings are read")
    parser.add_argument("--timeout-s", type=float, default=120)
    args = parser.parse_args(argv)
    if args.serve:
        _serve(args.port)
        return 0
    if not args.run_id or not args.test_database_url or not 1 <= args.timeout_s <= 600:
        parser.error("run ID, test database URL and a 1-600 second timeout are required")
    result = _run(args.run_id, args.test_database_url, args.env_file,
                  timeout_s=args.timeout_s)
    print(json.dumps({"result_dir": str(result), "formal_ready": False},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
