"""Small real-provider smoke test against an explicitly isolated test database.

Credentials may be read from LUOJIA_PROVIDER_ENV; neither they nor raw provider
responses are printed. This creates synthetic conversations only in the test DB.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from dotenv import dotenv_values

for name, value in dotenv_values(os.environ.get("LUOJIA_PROVIDER_ENV", "/nonexistent")).items():
    if name in {"DEEPSEEK_API_KEY", "AMAP_KEY", "AMAP_WEB_KEY", "LLM_MODEL"} and value:
        os.environ[name] = value
os.environ["AGENT_DIALOGUE_MODE"] = "model"

from langgraph.checkpoint.memory import InMemorySaver
from agents import planner, workflow
from agents.dialogue import Dialogue
from jobs.worker import process_next_run
from storage import database
from storage.task_repository import create_conversation, create_task_run, get_run, get_run_input
from tests.db_safety import verify_isolated_test_database


def main():
    url = os.environ["TEST_DATABASE_URL"]
    verify_isolated_test_database(url)
    database.initialize(url)
    resume_task = os.environ.get("LUOJIA_SMOKE_RESUME_TASK")
    if resume_task:
        with database.connect(url) as conn:
            existing = conn.execute("SELECT c.owner_id,t.conversation_id,t.state FROM task t"
                                    " JOIN conversation c USING (conversation_id) WHERE t.task_id=%s",
                                    (resume_task,)).fetchone()
        if not existing:
            raise RuntimeError("resume task is not in the isolated test database")
        owner, conversation = str(existing["owner_id"]), str(existing["conversation_id"])
    else:
        owner, _, conversation = create_conversation(url)
    graph = workflow.build_agent_workflow(InMemorySaver(), database_url=url)
    captured = []
    presentation_errors = []
    factory = planner._make_client
    original_present = Dialogue.present

    def monitored_present(self, args):
        try:
            return original_present(self, args)
        except ValueError as error:
            presentation_errors.append(str(error).split(":", 1)[0])
            raise
    Dialogue.present = monitored_present

    def monitored_client():
        client = factory()
        original_create = client.chat.completions.create

        def create(**kwargs):
            captured.append(kwargs["messages"])
            return original_create(**kwargs)
        client.chat.completions.create = create
        return client

    planner._make_client = monitored_client

    def turn(query, **kwargs):
        start = time.monotonic()
        accepted = create_task_run(url, owner, conversation, query, str(uuid4()),
                                   datetime.now(timezone.utc) + timedelta(seconds=75), **kwargs)
        inputs = get_run_input(url, accepted["run_id"])
        expected = [{"role": row["role"], "content": row["content"]} for row in inputs["history"]]
        captured.clear()
        presentation_errors.clear()
        outcome = process_next_run(url, "smoke-" + uuid4().hex, graph, lease_seconds=30)
        run = get_run(url, owner, accepted["run_id"])
        result = run.get("result") or {}
        messages = ([{"role": row["role"], "content": row["content"]} for row in captured[0]
                     if row.get("role") in {"user", "assistant"}] if captured else [])
        pending = result.get("place_request") or result.get("completed_place_request") or {}
        events = [event["payload"] for event in run.get("events", [])
                  if event.get("event_type") == "agent_tool_call" and event.get("payload", {}).get("phase") == "started"]
        call_names, feedback = {}, []
        for item in captured[-1] if captured else []:
            if item.get("role") == "assistant":
                for call in item.get("tool_calls") or []:
                    call_names[call.get("id")] = (call.get("function") or {}).get("name")
            if item.get("role") != "tool":
                continue
            name = call_names.get(item.get("tool_call_id"))
            if name not in {"update_task", "browse_candidates", "present_places"}:
                continue
            try:
                value = json.loads(item.get("content") or "{}")
            except (ValueError, TypeError):
                continue
            page = value.get("candidate_page") or value
            feedback.append({"tool": name, "error": value.get("error"), "message": value.get("message") if value.get("error") else None,
                             "count": page.get("count"), "exhausted": page.get("exhausted")})
        record = {"seconds": round(time.monotonic() - start, 2), "status": run["status"],
                  "outcome": outcome and outcome["status"], "kind": result.get("response_kind"),
                  "failure_code": result.get("failure_code"), "model_calls": len(captured),
                  "all_original_messages_received": messages == [*expected, {"role": "user", "content": query}],
                  "history_count": len(expected), "task_id": run.get("task_id"),
                  "task_revision": run.get("current_task_revision"),
                  "start": (pending.get("start") or {}).get("name"),
                  "search_anchor": (pending.get("search_anchor") or {}).get("name"),
                  "search_radius_m": pending.get("search_radius_m"), "activities": pending.get("activities") or [],
                  "subcategories": pending.get("subcategories") or [],
                  "selected_poi_id": pending.get("selected_poi_id"),
                  "end_name": (result.get("end") or {}).get("name"),
                  "candidate_ids": [row.get("poi_id") for row in result.get("candidates") or []],
                  "route_verification": (result.get("route_verification") or {}).get("status"),
                  "presentation_errors": presentation_errors, "tool_feedback": feedback[-20:],
                  "tools": [event.get("tool_name") for event in events], "model_usage": result.get("model_usage")}
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return run, pending, record

    if resume_task:
        pending = existing["state"].get("place_request") or {}
        a = {"task_id": resume_task, "start": (pending.get("start") or {}).get("name"),
             "search_anchor": (pending.get("search_anchor") or {}).get("name"),
             "activities": pending.get("activities") or [], "subcategories": pending.get("subcategories") or [],
             "candidate_ids": (pending.get("semantic_request") or {}).get("shown_ids") or []}
        b = {"candidate_ids": []}
    else:
        first, pending, a = turn("我从信图出发，在总图附近找餐馆，别推荐食堂档口")
        if not a["candidate_ids"] or not a["all_original_messages_received"] or a["search_radius_m"] != 1000:
            return 1
        second, pending, b = turn("再换几家看看")
        if (not b["candidate_ids"] or not b["all_original_messages_received"] or b["task_id"] != a["task_id"]
                or set(a["candidate_ids"]) & set(b["candidate_ids"])
                or any(b[key] != a[key] for key in ("start", "search_anchor", "activities", "subcategories"))):
            return 2
    third, pending, c = turn("范围放宽些，其他要求都保持")
    if (not c["candidate_ids"] or not c["all_original_messages_received"] or c["task_id"] != a["task_id"]
            or c["search_radius_m"] != 2000 or any(c[key] != a[key] for key in ("start", "search_anchor", "activities", "subcategories"))
            or set(a["candidate_ids"] + b["candidate_ids"]) & set(c["candidate_ids"])):
        return 3
    selected = c["candidate_ids"][0]
    task_id, revision = third["task_id"], third["current_task_revision"]
    fourth, pending, d = turn("去这家", continuation_task_id=task_id, base_revision=revision,
                              selected_poi_id=selected,
                              ui_action={"type": "candidate_selection", "task_id": task_id,
                                         "task_revision": revision, "poi_id": selected})
    if (d["kind"] != "route" or d["route_verification"] != "pass" or not d["all_original_messages_received"]
            or d["selected_poi_id"] != selected or any(d[key] != a[key] for key in ("start", "search_anchor", "activities", "subcategories"))):
        return 4
    _, _, e = turn("刚才这条路线改成尽量平坦，还是从原来的地方出发")
    return 0 if (e["kind"] == "route" and e["route_verification"] == "pass"
                 and e["all_original_messages_received"] and e["selected_poi_id"] == selected
                 and e["end_name"] == d["end_name"]
                 and all(e[key] == a[key] for key in ("start", "search_anchor", "activities", "subcategories"))) else 5


if __name__ == "__main__":
    raise SystemExit(main())
