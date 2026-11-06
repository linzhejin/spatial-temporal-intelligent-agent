"""PostgreSQL-leased worker for durable, checkpointed campus agent runs."""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

from storage import database
from storage.task_repository import (
    claim_next_run,
    finish_run,
    get_run_input,
    heartbeat_run,
)

logger = logging.getLogger(__name__)
_STOP = threading.Event()


def process_next_run(database_url, worker_id, graph, lease_seconds=30):
    """Claim and execute one run; a lost lease can never publish stale output."""
    claimed = claim_next_run(database_url, worker_id, lease_seconds)
    if not claimed:
        return None

    run_id = claimed["run_id"]
    lease_lost = threading.Event()
    heartbeat_stop = threading.Event()

    def keep_lease_alive():
        interval = max(1, lease_seconds // 3)
        while not heartbeat_stop.wait(interval):
            if not heartbeat_run(database_url, run_id, worker_id, lease_seconds):
                lease_lost.set()
                logger.warning("Worker lost lease for run %s", run_id)
                return

    heartbeat = threading.Thread(target=keep_lease_alive, name=f"lease-{run_id[:8]}", daemon=True)
    heartbeat.start()
    config = {"configurable": {"thread_id": run_id}}
    runtime_context = {"worker_id": worker_id}
    try:
        if claimed["attempt"] > 1 and hasattr(graph, "get_state"):
            snapshot = graph.get_state(config)
            if snapshot and snapshot.values:
                # The graph completed but the worker died before committing the result.
                # Reuse the checkpointed terminal state instead of repeating tools.
                state = snapshot.values if not snapshot.next else graph.invoke(
                    None, config, context=runtime_context,
                )
            else:
                state = graph.invoke(
                    {"run_id": run_id}, config, context=runtime_context,
                )
        else:
            state = graph.invoke(
                {"run_id": run_id}, config, context=runtime_context,
            )
        if lease_lost.is_set():
            return {"run_id": run_id, "status": "lease_lost"}
        result = state.get("result") if isinstance(state, dict) else None
        if not isinstance(result, dict):
            raise RuntimeError("workflow completed without a result")
        status = (
            "needs_input" if result.get("needs_input") or result.get("response_kind") == "clarify"
            else result.get("execution_status", "completed")
        )
        if status not in {"completed", "partial", "failed", "needs_input"}:
            status = "completed"
        if status == "failed" and result.get("response_kind") not in {
                "route", "route_comparison", "candidates", "no_route", "clarify"}:
            run_context = get_run_input(database_url, run_id) or {}
            result = _recoverable_failure_result(run_context, result)
            status = "needs_input"
        finished = finish_run(
            database_url, run_id, worker_id, status=status, result=result, return_status=True,
        )
        return {"run_id": run_id,
                "status": finished if isinstance(finished, str) else status if finished else "lease_lost"}
    except Exception as error:
        logger.exception("Agent run failed (%s)", run_id)
        if lease_lost.is_set():
            return {"run_id": run_id, "status": "lease_lost"}
        try:
            run_context = get_run_input(database_url, run_id) or {}
        except Exception:
            logger.exception("Could not load accepted context for failed run (%s)", run_id)
            run_context = {}
        recovery = _recoverable_failure_result(run_context)
        finished = finish_run(
            database_url, run_id, worker_id, status="needs_input",
            result=recovery, return_status=True,
        )
        return {
            "run_id": run_id,
            "status": finished if isinstance(finished, str) else "needs_input" if finished else "lease_lost",
            "error_type": type(error).__name__,
        }
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=2)


def _recoverable_failure_result(run_context, previous_result=None):
    """Keep accepted task state and turn internal failures into a conversational recovery step."""
    task = run_context.get("task") if isinstance(run_context, dict) else None
    task = task if isinstance(task, dict) else {}
    pending = task.get("place_request")
    previous_result = previous_result if isinstance(previous_result, dict) else {}
    if not isinstance(pending, dict):
        pending = previous_result.get("place_request")
    if isinstance(previous_result.get("candidates"), list) and previous_result["candidates"]:
        candidates = previous_result["candidates"]
        message = "我保留了这批已核对候选。你可以先选择其中一个，或说“换一批”继续。"
        return {**previous_result, "task_type": previous_result.get("task_type", "path_planning"),
                "response_kind": "candidates", "message": message, "reply": message,
                "place_request": pending or previous_result.get("place_request"),
                "execution_status": "partial"}
    question = "刚才这一步暂时没有完成，我保留了当前需求和已确认条件。你想按原条件继续，还是调整区域或类型？"
    recovery = {
        "task_type": "unknown", "response_kind": "clarify", "message": question,
        "reply": question,
        "clarify": {"question": question, "options": ["按原条件继续", "调整区域或类型"], "slot": "other"},
        "needs_input": True, "execution_status": "needs_input",
    }
    if isinstance(pending, dict):
        recovery["place_request"] = pending
    return recovery


def _worker_loop(database_url, idle_seconds, slot):
    from langgraph.checkpoint.postgres import PostgresSaver
    from agents.workflow import build_agent_workflow

    worker_id = f"whu-agent-{slot}-{uuid.uuid4()}"
    with PostgresSaver.from_conn_string(database_url) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, database_url=database_url)
        while not _STOP.is_set():
            try:
                outcome = process_next_run(database_url, worker_id, graph)
                if outcome is None:
                    _STOP.wait(idle_seconds)
            except Exception:
                logger.exception("Worker %s loop failed; retrying", slot)
                _STOP.wait(min(idle_seconds * 4, 5))


def run_forever(database_url=None, idle_seconds=0.5, concurrency=None):
    database_url = database.database_url(database_url)
    database.initialize(database_url)
    count = concurrency or int(os.getenv("AGENT_WORKER_CONCURRENCY", "3"))
    if not 1 <= count <= 16:
        raise ValueError("AGENT_WORKER_CONCURRENCY must be between 1 and 16")
    _supervise_workers(database_url, idle_seconds, count)


def _supervise_workers(database_url, idle_seconds, count, *, stop_event=None,
                       worker_target=None, poll_seconds=0.5,
                       restart_delay_seconds=0.5, max_restart_delay_seconds=30):
    """Keep every configured concurrency slot alive, including after startup failures."""
    stop_event = stop_event or _STOP
    worker_target = worker_target or _worker_loop
    workers = {}
    failures = {slot: 0 for slot in range(count)}
    restart_at = {slot: 0.0 for slot in range(count)}

    def start_slot(slot):
        def run():
            try:
                worker_target(database_url, idle_seconds, slot)
            except Exception:
                logger.exception("Agent worker slot %s exited unexpectedly", slot)

        thread = threading.Thread(target=run, name=f"agent-worker-{slot}", daemon=True)
        workers[slot] = thread
        thread.start()

    for slot in range(count):
        start_slot(slot)
    try:
        while not stop_event.wait(poll_seconds):
            now = time.monotonic()
            for slot, thread in list(workers.items()):
                if thread is None:
                    continue
                if thread.is_alive():
                    continue
                failures[slot] += 1
                delay = min(
                    restart_delay_seconds * (2 ** (failures[slot] - 1)),
                    max_restart_delay_seconds,
                )
                restart_at[slot] = now + delay
                workers[slot] = None
                logger.warning(
                    "Agent worker slot %s stopped; restarting in %.1f seconds (failure %s)",
                    slot, delay, failures[slot],
                )
            for slot, thread in list(workers.items()):
                if thread is None and now >= restart_at[slot] and not stop_event.is_set():
                    start_slot(slot)
    finally:
        for worker in workers.values():
            if worker is not None:
                worker.join(timeout=5)


def _request_stop(_signum, _frame):
    _STOP.set()


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    run_forever()


if __name__ == "__main__":
    main()
