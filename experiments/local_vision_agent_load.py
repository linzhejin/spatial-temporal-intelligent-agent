"""Small-N local Agent diagnostic with an independent real CPU video analysis.

Run with ``python -m experiments.local_vision_agent_load --help``. No app or
database imports occur in the coordinator. Only the inference child imports
vision.engine, with dotenv loading disabled and a minimal inherited environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from urllib.parse import quote, urlsplit
from uuid import uuid4

import psutil
import requests

from experiments.scenario_live_adapter import (
    TERMINAL, _data, require_isolated_replay_service, require_local_service,
)


ROOT = Path(__file__).resolve().parents[1]
REQUEST_BANK = (
    ("food_candidates", "我在桂园操场，附近有哪些吃饭的地方？"),
    ("simple_route", "从桂园操场步行到梅园操场，走最短路线。"),
    ("route_comparison", "比较从桂园操场到梅园操场步行的最短路线和风景优先路线。"),
    ("weather", "武汉大学现在天气怎么样？"),
    ("missing_origin", "带我去梅园操场。"),
)
PROGRESS_EVENTS = {"run_started", "node_started", "node_completed", "agent_tool_call"}


def canonical_service(url: str) -> str:
    """Reuse replay's policy and pin localhost to a literal loopback address."""
    require_local_service(url)
    parsed = urlsplit(url)
    port = parsed.port  # Reject malformed/out-of-range ports before network I/O.
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"http://{host}" + (f":{port}" if port is not None else "")


class LocalSession(requests.Session):
    """No environment proxies, .netrc credentials, or HTTP redirect traversal."""
    allow_redirects = False

    def __init__(self):
        super().__init__()
        self.trust_env = False

    def request(self, method, url, **kwargs):
        parsed = urlsplit(url)
        canonical_service(f"{parsed.scheme}://{parsed.netloc}")
        if parsed.username or parsed.password:
            raise ValueError("credentials are forbidden")
        kwargs["allow_redirects"] = False
        response = super().request(method, url, **kwargs)
        if 300 <= response.status_code < 400:
            raise ValueError("local service redirects are forbidden")
        return response


def distribution(values) -> dict:
    """Linear interpolation at (N-1)*q; keep original samples auditable."""
    raw = [value for value in values if isinstance(value, (int, float))
           and not isinstance(value, bool) and math.isfinite(value)]
    ordered = sorted(raw)

    def percentile(q):
        if not ordered:
            return None
        rank = (len(ordered) - 1) * q
        low, high = math.floor(rank), math.ceil(rank)
        return round(ordered[low] + (ordered[high] - ordered[low]) * (rank - low), 6)

    return {"n": len(raw), "values_s": raw, "p50_s": percentile(.5), "p95_s": percentile(.95)}


def summarize_metrics(rows: list[dict]) -> dict:
    return {
        "request_count": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
        "failure_count": sum(row["status"] not in {"completed", "needs_input"} for row in rows),
        **{key: distribution(row.get(key) for row in rows) for key in (
            "accepted_latency_s", "first_running_latency_s", "terminal_latency_s", "elapsed_s",
        )},
    }


def request_overlap(start: float, finish: float, vision_start, vision_finish) -> dict:
    if vision_start is None or vision_finish is None or finish <= start:
        return {"classification": "unknown", "seconds": None, "fraction": None}
    seconds = max(0, min(finish, vision_finish) - max(start, vision_start))
    fraction = seconds / (finish - start)
    classification = "none" if seconds == 0 else "full" if fraction >= 1 - 1e-9 else "partial"
    return {"classification": classification, "seconds": round(seconds, 6),
            "fraction": round(fraction, 6)}


def run_request(phase: str, index: int, case: tuple[str, str], base: str, *,
                poll_interval_s: float = .5, timeout_s: float = 180,
                session_factory=LocalSession, stop: threading.Event | None = None) -> dict:
    """Preserve terminal and transport failure observations without response text."""
    started = time.perf_counter()
    row = {"phase": phase, "index": index, "case": case[0], "query": case[1],
           "started_perf_s": started, "accepted_latency_s": None,
           "first_running_latency_s": None, "terminal_latency_s": None,
           "tool_names": [], "tool_events": [], "status": "client_error"}
    observed_events = set()
    try:
        base = canonical_service(base)
        with session_factory() as session:
            require_isolated_replay_service(session, base)
            conversation = _data(session.post(
                f"{base}/api/conversations", json={}, timeout=15,
            ))["conversation_id"]
            row["conversation_id"] = conversation
            message_started = time.perf_counter()
            row["message_started_perf_s"] = message_started
            accepted = _data(session.post(
                f"{base}/api/conversations/{quote(str(conversation), safe='')}/messages",
                json={"query": case[1], "request_id": str(uuid4())}, timeout=30,
            ))
            row["run_id"] = accepted["run_id"]
            row["accepted_latency_s"] = time.perf_counter() - message_started
            deadline = message_started + timeout_s
            while True:
                if stop is not None and stop.is_set():
                    row["status"] = "interrupted"
                    break
                run = _data(session.get(
                    f"{base}/api/runs/{quote(str(accepted['run_id']), safe='')}",
                    params={"conversation_id": conversation}, timeout=15,
                ))
                elapsed = time.perf_counter() - message_started
                row["last_observed_status"] = run.get("status")
                events = run.get("events") or []
                if not isinstance(events, list):
                    raise ValueError("invalid run events")
                events = [event for event in events if isinstance(event, dict)]
                has_progress = any(event.get("event_type") in PROGRESS_EVENTS for event in events)
                if (row["first_running_latency_s"] is None
                        and (run.get("status") == "running" or has_progress)):
                    row["first_running_latency_s"] = elapsed
                for event in events:
                    if event.get("event_type") != "agent_tool_call":
                        continue
                    payload = event.get("payload") or {}
                    if not isinstance(payload, dict):
                        continue
                    key = event.get("seq")
                    if key is None:
                        key = json.dumps(event, sort_keys=True)
                    if key in observed_events:
                        continue
                    observed_events.add(key)
                    name = payload.get("tool_name")
                    if isinstance(name, str):
                        row["tool_events"].append({"seq": event.get("seq"), "tool_name": name,
                                                   "outcome": payload.get("outcome")})
                        if name not in row["tool_names"]:
                            row["tool_names"].append(name)
                if run.get("status") in TERMINAL:
                    row["status"] = run["status"]
                    row["terminal_latency_s"] = elapsed
                    error = run.get("error")
                    if isinstance(error, dict):
                        row["run_error_code"] = error.get("code")
                    result = run.get("result")
                    if isinstance(result, dict):
                        row["response_kind"] = result.get("response_kind")
                    break
                if time.perf_counter() >= deadline:
                    raise TimeoutError("request deadline exceeded")
                if poll_interval_s:
                    time.sleep(min(poll_interval_s, max(0, deadline - time.perf_counter())))
    except (requests.RequestException, ValueError, KeyError, TypeError, TimeoutError) as error:
        row["status"] = "timeout" if isinstance(error, (TimeoutError, requests.Timeout)) else "client_error"
        row["error_type"] = type(error).__name__
        response = getattr(error, "response", None)
        if response is not None:
            row["http_status"] = response.status_code
    finally:
        row["finished_perf_s"] = time.perf_counter()
        row["elapsed_s"] = row["finished_perf_s"] - started
    return row


def write_json(path: Path, payload):
    with path.open("x", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)


def append_json(target, payload):
    target.write(json.dumps(payload, ensure_ascii=False) + "\n")
    target.flush()


def read_markers(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # Reader may see the writer's incomplete final line.
    return rows


def vision_child(model: Path, video: Path, marker_path: Path, result_path: Path) -> int:
    """Exact call-boundary markers; no HTTP, uploads, event or DB operations."""
    with marker_path.open("x", encoding="utf-8") as markers:
        def mark(event, **extra):
            append_json(markers, {"event": event, "perf_s": time.perf_counter(),
                                  "utc": datetime.now(timezone.utc).isoformat(), **extra})

        analysis_started = False
        try:
            # config.py calls load_dotenv at import; this standalone child must
            # never open repository .env files or inherit application credentials.
            import dotenv
            dotenv.load_dotenv = lambda *_args, **_kwargs: False
            import cv2
            cv2.setNumThreads(1)
            from vision.engine import OnnxDetector, analyze_media

            detector = OnnxDetector(str(model))
            if detector.cpu_threads != 1:
                raise ValueError("diagnostic requires one detector CPU thread")
            mark("analyze_started", pid=os.getpid(), detector_cpu_threads=detector.cpu_threads)
            analysis_started = True
            try:
                result = analyze_media(video, "video", None, detector=detector,
                                       accident_model=None, flood_segmenter=None)
            finally:
                mark("analyze_finished")
            write_json(result_path, {
                "status": "completed", "video_coverage": result.get("video_coverage"),
                "metrics": result.get("metrics"), "candidate_count": len(result.get("candidates", [])),
                "model": result.get("model"), "media": result.get("media"),
                "accuracy": "unlabeled_video_no_accuracy_claim",
            })
            mark("child_completed")
            return 0
        except Exception as error:
            mark("child_failed", error_type=type(error).__name__, analysis_started=analysis_started)
            return 1


class ResourceSampler:
    """Direct explicit PID samples; 100% CPU means one logical CPU."""
    def __init__(self, output: Path, service_pid, worker_pid, interval: float):
        self.path, self.interval = output, interval
        self.targets = {}
        self.errors = []
        self.phase = "setup"
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="diagnostic-resource-sampler", daemon=True)
        for label, pid in (("service", service_pid), ("worker", worker_pid)):
            if pid is not None:
                self.add(label, pid)

    def add(self, label, pid):
        try:
            process = psutil.Process(pid)
            process.cpu_percent(None)  # Prime; never report this first zero.
            with self.lock:
                self.targets[label] = process
        except psutil.Error as error:
            self.errors.append({"target": label, "pid": pid, "error_type": type(error).__name__})

    def _run(self):
        with self.path.open("x", encoding="utf-8") as target:
            while not self.stop.wait(self.interval):
                with self.lock:
                    targets = list(self.targets.items())
                for label, process in targets:
                    record = {"perf_s": time.perf_counter(), "phase": self.phase,
                              "target": label, "pid": process.pid}
                    try:
                        record.update(rss_bytes=process.memory_info().rss,
                                      cpu_percent=process.cpu_percent(None))
                    except psutil.Error as error:
                        record["error_type"] = type(error).__name__
                        with self.lock:
                            self.targets.pop(label, None)
                    append_json(target, record)


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def provenance(model: Path, video: Path) -> dict:
    versions = {}
    for package in ("requests", "psutil", "onnxruntime", "opencv-python", "numpy", "python-dotenv"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    files = ("experiments/local_vision_agent_load.py", "experiments/scenario_live_adapter.py",
             "vision/engine.py", "vision/analysis.py", "vision/tracking.py", "config.py",
             "agents/tools.py", "agents/planner.py", "agents/workflow.py", "api/routes.py")
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "python": sys.version,
            "cpu": {"processor": platform.processor(), "logical": psutil.cpu_count(),
                    "physical": psutil.cpu_count(logical=False)}, "versions": versions,
            "model": file_identity(model), "video": file_identity(video),
            "code": {name: file_identity(ROOT / name)["sha256"] for name in files},
            "caveat": "Small-N local diagnostic; not production capacity, Tencent Linux cgroups, or accuracy evidence."}


def child_environment() -> dict:
    # Explicit OS/runtime allowlist; do not enumerate or copy application secrets.
    env = {key: os.environ[key] for key in (
        "SystemRoot", "WINDIR", "PATH", "PATHEXT", "TEMP", "TMP", "SYSTEMDRIVE",
        "APPDATA",
    ) if key in os.environ}
    env.update(PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", PYTHON_DOTENV_DISABLED="1",
               VISION_CPU_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return env


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True, help="loopback isolated test service")
    parser.add_argument("--model", type=Path, required=True, help="existing external ONNX file")
    parser.add_argument("--video", type=Path, required=True, help="existing external video file")
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwritten")
    parser.add_argument("--requests-per-phase", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--poll-interval", type=float, default=.5)
    parser.add_argument("--sample-interval", type=float, default=.5)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--vision-timeout-seconds", type=float, default=900)
    parser.add_argument("--service-pid", type=int)
    parser.add_argument("--worker-pid", type=int)
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    base = canonical_service(args.service)
    for value in (args.requests_per_phase, args.concurrency, args.poll_interval,
                  args.sample_interval, args.timeout_seconds, args.vision_timeout_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("counts and timing intervals must be positive and finite")
    for name in ("model", "video"):
        path = getattr(args, name).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"{name} must be an existing file")
        if path.is_relative_to(ROOT):
            raise ValueError(f"{name} must live outside the repository")
        setattr(args, name, path)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, warmup, errors, child = [], None, [], None
    stop = threading.Event()
    sampler = ResourceSampler(args.output / "resources.jsonl", args.service_pid,
                              args.worker_pid, args.sample_interval)
    sampler.thread.start()
    markers_path = args.output / "vision-markers.jsonl"
    result_path = args.output / "vision-result.json"
    exit_code = 0
    try:
        write_json(args.output / "provenance.json", provenance(args.model, args.video))
        with LocalSession() as session:
            require_isolated_replay_service(session, base)
        if not args.skip_warmup:
            sampler.phase = "warmup"
            warmup = run_request("warmup", 0, REQUEST_BANK[1], base,
                                 poll_interval_s=args.poll_interval, timeout_s=args.timeout_seconds)
            write_json(args.output / "warmup.json", warmup)

        def phase(name, deadline=None):
            sampler.phase = name
            record_lock = threading.Lock()

            def request(index):
                row = run_request(
                    name, index + 1, REQUEST_BANK[index % len(REQUEST_BANK)], base,
                    poll_interval_s=args.poll_interval, timeout_s=args.timeout_seconds, stop=stop,
                )
                # Write in the task itself so interrupting as_completed cannot
                # lose already-started requests that return during cleanup.
                with record_lock:
                    rows.append(row)
                    with (args.output / "requests.jsonl").open("a", encoding="utf-8") as target:
                        append_json(target, row)

            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [executor.submit(request, index) for index in range(args.requests_per_phase)]
                try:
                    timeout = max(0, deadline - time.perf_counter()) if deadline is not None else None
                    for future in as_completed(futures, timeout=timeout):
                        future.result()
                except BaseException:
                    stop.set()
                    for future in futures:
                        future.cancel()
                    raise

        phase("baseline")
        sampler.phase = "vision_startup"
        with (args.output / "vision-child.log").open("x", encoding="utf-8") as child_log:
            child = subprocess.Popen(
                [sys.executable, "-m", "experiments.local_vision_agent_load", "_vision_child",
                 str(args.model), str(args.video), str(markers_path), str(result_path)],
                cwd=args.output, env=child_environment(), stdout=child_log, stderr=child_log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            sampler.add("vision_child", child.pid)
            vision_deadline = time.perf_counter() + args.vision_timeout_seconds
            while not any(row["event"] == "analyze_started" for row in read_markers(markers_path)):
                if child.poll() is not None:
                    raise RuntimeError("vision child failed before analysis")
                if time.perf_counter() >= vision_deadline:
                    raise TimeoutError("vision startup deadline exceeded")
                time.sleep(.05)
            phase("loaded", deadline=vision_deadline)
            sampler.phase = "vision_finish"
            while child.poll() is None:
                if time.perf_counter() >= vision_deadline:
                    raise TimeoutError("vision analysis deadline exceeded")
                time.sleep(.1)
            if child.returncode:
                raise RuntimeError("vision child failed")
    except KeyboardInterrupt:
        errors.append({"error_type": "KeyboardInterrupt"})
        exit_code = 130
    except Exception as error:
        errors.append({"error_type": type(error).__name__})
        exit_code = 1
    finally:
        stop.set()
        # Own Popen handle only. Service and worker PIDs are sampled, never stopped.
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        sampler.stop.set()
        sampler.thread.join(timeout=5)
        markers = read_markers(markers_path)
        vision_start = next((row["perf_s"] for row in markers if row["event"] == "analyze_started"), None)
        vision_end = next((row["perf_s"] for row in markers if row["event"] == "analyze_finished"), None)
        for row in rows:
            row["vision_overlap"] = request_overlap(
                row["message_started_perf_s"] if "message_started_perf_s" in row else row["started_perf_s"],
                row["finished_perf_s"], vision_start, vision_end,
            ) if row["phase"] == "loaded" else {"classification": "baseline_no_child"}
        phases = {phase: summarize_metrics([row for row in rows if row["phase"] == phase])
                  for phase in ("baseline", "loaded")}
        loaded = [row for row in rows if row["phase"] == "loaded"]
        fully_overlapped = (len(loaded) == args.requests_per_phase
                            and all(row["vision_overlap"]["classification"] == "full" for row in loaded)
                            and any(row["event"] == "child_completed" for row in markers))
        write_json(args.output / "report.json", {
            "service": base, "database_identity": "isolated_local_proof_required_per_request",
            "requests_per_phase": args.requests_per_phase, "concurrency": args.concurrency,
            "request_bank": [{"case": case, "query": query} for case, query in REQUEST_BANK],
            "poll_interval_s": args.poll_interval, "sample_interval_s": args.sample_interval,
            "latency_origin": "message POST start; elapsed includes readiness and fresh conversation setup",
            "first_running_semantics": "first poll observing running or an execution progress event; may be unobserved",
            "poll_caveat": "poll interval plus HTTP request time bounds observation granularity; no exact worker-start claim",
            "percentile_method": "linear interpolation at (N-1)*q; raw values in report",
            "cpu_percent_semantics": "psutil direct PID CPU; 100 percent is one logical CPU; primed first sample omitted",
            "caveat": "Small-N local diagnostic, not production capacity or Tencent Linux cgroup evidence. Unlabeled video has no accuracy score.",
            "fully_overlapped_loaded_phase": fully_overlapped,
            "overlap_counts": dict(Counter(row["vision_overlap"]["classification"] for row in loaded)),
            "vision_markers": markers, "vision_child_exit_code": child.returncode if child else None,
            "resource_target_errors": sampler.errors, "errors": errors,
            "phases": phases, "requests": sorted(rows, key=lambda row: (row["phase"], row["index"])),
        })
    print(json.dumps({"output": str(args.output), "exit_code": exit_code,
                      "fully_overlapped_loaded_phase": fully_overlapped}))
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "_vision_child":
        raise SystemExit(vision_child(*(Path(value) for value in sys.argv[2:])))
    raise SystemExit(main())
