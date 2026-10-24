"""Owner-scoped, revision-checked persistence for conversations and tasks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re
import secrets
import uuid
from datetime import datetime
from typing import Any

from psycopg.types.json import Jsonb

from storage import database


class RevisionConflict(Exception):
    """The task changed after the caller read its base revision."""


def create_conversation(url: str | None = None) -> tuple[str, str, str]:
    owner_id, conversation_id = str(uuid.uuid4()), str(uuid.uuid4())
    token = secrets.token_urlsafe(32)
    credential_hash = hashlib.sha256(token.encode("utf-8")).digest()
    with database.connect(url) as conn:
        conn.execute(
            "INSERT INTO conversation(conversation_id, owner_id, credential_hash) "
            "VALUES (%s, %s, %s)",
            (conversation_id, owner_id, credential_hash),
        )
    return owner_id, token, conversation_id


def get_owner_by_conversation_token(url: str | None, token: str) -> str | None:
    """Authenticate a durable anonymous identity by its server-stored token hash."""
    if not isinstance(token, str) or not token:
        return None
    credential_hash = hashlib.sha256(token.encode("utf-8")).digest()
    with database.connect(url) as conn:
        rows = conn.execute(
            "SELECT owner_id FROM conversation WHERE credential_hash=%s LIMIT 2",
            (credential_hash,),
        ).fetchall()
    # Ambiguous credentials cannot establish an owner, even on old schemas.
    return str(rows[0]["owner_id"]) if len(rows) == 1 else None


def get_conversation(url: str | None, conversation_id: str, token: str) -> dict | None:
    credential_hash = hashlib.sha256((token or "").encode("utf-8")).digest()
    with database.connect(url) as conn:
        row = conn.execute(
            "SELECT conversation_id, owner_id, revision FROM conversation "
            "WHERE conversation_id = %s AND credential_hash = %s",
            (conversation_id, credential_hash),
        ).fetchone()
    return {**row, "conversation_id": str(row["conversation_id"]), "owner_id": str(row["owner_id"])} if row else None


def create_task(
    url: str | None, owner_id: str, conversation_id: str, state: dict[str, Any]
) -> str:
    from agents.state_models import TaskState

    task_id = str(uuid.uuid4())
    initial_model = TaskState.model_validate({
        **state, "task_id": task_id, "conversation_id": conversation_id, "revision": 0,
    })
    initial = initial_model.model_dump(mode="json")
    with database.connect(url) as conn:
        owned = conn.execute(
            "SELECT 1 FROM conversation WHERE owner_id = %s AND conversation_id = %s",
            (owner_id, conversation_id),
        ).fetchone()
        if not owned:
            raise PermissionError("Conversation is not owned by this identity")
        kind = str(state.get("task_kind", "route"))
        status = str(state.get("status", "draft"))
        conn.execute(
            "INSERT INTO task(task_id, conversation_id, task_kind, status, state) "
            "VALUES (%s, %s, %s, %s, %s)",
            (task_id, conversation_id, kind, status, Jsonb(initial)),
        )
        conn.execute(
            "INSERT INTO task_history(task_id, revision, state) VALUES (%s, 0, %s)",
            (task_id, Jsonb(initial)),
        )
    return task_id


def create_task_run(
    url: str | None, owner_id: str, conversation_id: str,
    message: str, idempotency_key: str, deadline_at: datetime,
    *, route_spec: dict[str, Any] | None = None,
    continuation_task_id: str | None = None,
    base_revision: int | None = None,
    selected_poi_id: str | None = None,
) -> dict:
    """Atomically accept a user message, task snapshot and queued run."""
    from agents.state_models import Requirement, RouteSpec, TaskState
    from agents.task_planner import build_task_plan

    message = message.strip()
    if not message or len(message) > 500:
        raise ValueError("message must contain 1-500 characters")
    if not idempotency_key or len(idempotency_key) > 200:
        raise ValueError("idempotency key is required and must be <= 200 characters")
    route_spec = route_spec or {}
    validated_route_spec = RouteSpec.model_validate(route_spec).model_dump(mode="json")
    if continuation_task_id is not None and (not isinstance(base_revision, int) or base_revision < 0):
        raise ValueError("continuation requires a nonnegative base_revision")
    submission = {"request_id": idempotency_key, "route_spec": route_spec,
                  "continuation_task_id": continuation_task_id}
    if selected_poi_id is not None:
        if not isinstance(selected_poi_id, str) or not re.fullmatch(r"poi_\d{3,}", selected_poi_id):
            raise ValueError("selected_poi_id must be a stable POI ID")
        if continuation_task_id is None:
            raise ValueError("selected_poi_id requires a clarification task")
        submission["selected_poi_id"] = selected_poi_id
    task_id, run_id = str(continuation_task_id or uuid.uuid4()), str(uuid.uuid4())
    with database.connect(url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 73518264))", (idempotency_key,))
        existing = conn.execute(
            "SELECT r.run_id, r.task_id, r.task_revision, r.status, m.content, m.metadata,"
            " t.state, c.owner_id, c.conversation_id "
            "FROM run r JOIN task t USING (task_id) JOIN conversation c USING (conversation_id) "
            "LEFT JOIN conversation_message m ON m.task_id=t.task_id AND m.role='user'"
            " AND m.metadata->>'request_id'=r.idempotency_key "
            "WHERE r.idempotency_key=%s",
            (idempotency_key,),
        ).fetchone()
        if existing:
            if (str(existing["owner_id"]) != owner_id
                    or str(existing["conversation_id"]) != conversation_id
                    or existing["content"] != message
                    or (existing["metadata"] or {}).get("request") != submission):
                raise RunConflict("request ID was already used for another submission")
            return {"task_id": str(existing["task_id"]), "run_id": str(existing["run_id"]),
                    "task_revision": existing["task_revision"], "status": existing["status"],
                    "created": False}
        owned = conn.execute(
            "SELECT 1 FROM conversation WHERE owner_id=%s AND conversation_id=%s FOR UPDATE",
            (owner_id, conversation_id),
        ).fetchone()
        if not owned:
            raise PermissionError("Conversation is not owned by this identity")
        if continuation_task_id is not None:
            row = conn.execute(
                "SELECT t.revision, t.status, t.state, t.task_kind FROM task t"
                " JOIN conversation c USING (conversation_id)"
                " WHERE c.owner_id=%s AND c.conversation_id=%s AND t.task_id=%s FOR UPDATE OF t",
                (owner_id, conversation_id, continuation_task_id),
            ).fetchone()
            if (not row or row["status"] != "needs_input" or row["revision"] != base_revision):
                raise TaskContinuationConflict("clarification task is no longer awaiting this answer")
            current = TaskState.model_validate(row["state"])
            merged_spec = current.route_spec.model_dump(mode="python")
            merged_spec.update(route_spec)
            new_spec = RouteSpec.model_validate(merged_spec)
            document = current.model_dump(mode="python")
            document.update({"status": "queued", "revision": current.revision + 1,
                             "updated_at": datetime.now().astimezone(), "pending_questions": [],
                             "route_spec": new_spec.model_dump(mode="python")})
            for requirement in document["requirements"]:
                if requirement["status"] == "needs_input":
                    requirement["status"] = "pending"
            document["requirements"].append(Requirement(
                requirement_id=f"answer-{uuid.uuid4().hex}", description=message,
                status="pending",
            ).model_dump())
            state = TaskState.model_validate(document).model_dump(mode="json")
            task_revision = state["revision"]
            conn.execute(
                "UPDATE task SET revision=%s, status='queued', state=%s, updated_at=now() WHERE task_id=%s",
                (task_revision, Jsonb(state), task_id),
            )
            conn.execute(
                "INSERT INTO task_history(task_id, revision, state) VALUES (%s, %s, %s)",
                (task_id, task_revision, Jsonb(state)),
            )
        else:
            plan = build_task_plan(
                message,
                has_route_context=bool(
                    validated_route_spec.get("start")
                    or validated_route_spec.get("end")
                    or validated_route_spec.get("stops")
                ),
            )
            task_kind = "route" if plan.kind == "route" else (
                "composite" if plan.kind == "composite" else "information"
            )
            state = TaskState(
                task_id=task_id,
                conversation_id=conversation_id,
                status="queued",
                task_kind=task_kind,
                route_spec=validated_route_spec,
                requirements=[Requirement(
                    requirement_id=item.requirement_id,
                    description=item.description,
                    status="pending",
                ) for item in plan.requirements],
            ).model_dump(mode="json")
            task_revision = 0
            conn.execute(
                "INSERT INTO task(task_id, conversation_id, revision, task_kind, status, state) "
                "VALUES (%s, %s, 0, %s, %s, %s)",
                (task_id, conversation_id, task_kind, "queued", Jsonb(state)),
            )
            conn.execute(
                "INSERT INTO task_history(task_id, revision, state) VALUES (%s, 0, %s)",
                (task_id, Jsonb(state)),
            )
        conversation = conn.execute(
            "UPDATE conversation SET revision=revision+1, updated_at=now() "
            "WHERE conversation_id=%s RETURNING revision", (conversation_id,),
        ).fetchone()
        conn.execute(
            "INSERT INTO conversation_message(conversation_id, seq, message_id, task_id, role, content, metadata) "
            "VALUES (%s, %s, %s, %s, 'user', %s, %s)",
            (conversation_id, conversation["revision"], str(uuid.uuid4()), task_id, message,
             Jsonb({"request_id": idempotency_key, "request": submission})),
        )
        conn.execute(
            "INSERT INTO run(run_id, task_id, task_revision, status, deadline_at, idempotency_key) "
            "VALUES (%s, %s, %s, 'queued', %s, %s)",
            (run_id, task_id, task_revision, deadline_at, idempotency_key),
        )
    return {"task_id": task_id, "run_id": run_id, "task_revision": task_revision,
            "status": "queued", "created": True}


def get_run_input(url: str | None, run_id: str) -> dict | None:
    """Read trusted task text and conversation context for a claimed run."""
    with database.connect(url) as conn:
        row = conn.execute(
            "SELECT r.run_id, r.task_id, r.task_revision, r.cancel_requested,"
            " t.conversation_id, t.revision, t.state, c.owner_id"
            " FROM run r JOIN task t USING (task_id)"
            " JOIN conversation c USING (conversation_id) WHERE r.run_id=%s",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        user_message = conn.execute(
            "SELECT seq, content, metadata FROM conversation_message"
            " WHERE task_id=%s AND role='user' ORDER BY seq DESC LIMIT 1",
            (row["task_id"],),
        ).fetchone()
        if not user_message:
            raise RuntimeError("queued task has no accepted user message")
        if row["task_revision"] > 0:
            origin_message = conn.execute(
                "SELECT content FROM conversation_message"
                " WHERE task_id=%s AND role='user' ORDER BY seq ASC LIMIT 1",
                (row["task_id"],),
            ).fetchone()
            history = conn.execute(
                "SELECT role, content FROM (SELECT seq, role, content"
                " FROM conversation_message WHERE task_id=%s AND seq<%s"
                " ORDER BY seq DESC LIMIT 8) recent ORDER BY seq",
                (row["task_id"], user_message["seq"]),
            ).fetchall()
        else:
            origin_message = None
            history = conn.execute(
                "SELECT role, content FROM (SELECT m.seq, m.role, m.content"
                " FROM conversation_message m LEFT JOIN task prior_task ON prior_task.task_id=m.task_id"
                " WHERE m.conversation_id=%s AND m.seq<%s"
                " AND (m.task_id IS NULL OR prior_task.status IN ('completed','partial'))"
                " ORDER BY seq DESC LIMIT 8) recent ORDER BY seq",
                (row["conversation_id"], user_message["seq"]),
            ).fetchall()
        previous = conn.execute(
            "SELECT prior.metadata->'route_state' AS route_state, prior.task_id"
            " FROM conversation_message prior"
            " WHERE prior.conversation_id=%s AND prior.role='assistant' AND prior.seq<%s"
            " AND prior.metadata ? 'route_state' ORDER BY prior.seq DESC LIMIT 1",
            (row["conversation_id"], user_message["seq"]),
        ).fetchone()
    return {
        "run_id": str(row["run_id"]), "task_id": str(row["task_id"]),
        "task_revision": row["task_revision"], "task_revision_current": row["revision"],
        "conversation_id": str(row["conversation_id"]), "owner_id": str(row["owner_id"]),
        "task": row["state"], "query": user_message["content"], "history": history,
        "selected_poi_id": ((user_message.get("metadata") or {}).get("request") or {}).get("selected_poi_id"),
        "task_origin_query": origin_message["content"] if origin_message else None,
        "previous_route_state": previous["route_state"] if previous else None,
        "previous_route_state_task_id": str(previous["task_id"]) if previous and previous["task_id"] else None,
        "cancel_requested": row["cancel_requested"],
    }


def get_task(url: str | None, owner_id: str, task_id: str) -> dict | None:
    with database.connect(url) as conn:
        row = conn.execute(
            "SELECT t.task_id, t.conversation_id, t.revision, t.task_kind, t.status, t.state "
            "FROM task t JOIN conversation c USING (conversation_id) "
            "WHERE c.owner_id = %s AND t.task_id = %s",
            (owner_id, task_id),
        ).fetchone()
    if not row:
        return None
    return {
        **row,
        "task_id": str(row["task_id"]),
        "conversation_id": str(row["conversation_id"]),
        "state": row["state"],
    }


def apply_task_update(
    url: str | None,
    owner_id: str,
    task_id: str,
    base_revision: int,
    operations: dict[str, Any],
) -> dict | None:
    if not operations or set(operations) - {"status", "task_kind", "route_spec", "requirements", "pending_questions", "artifact_ids"}:
        raise ValueError("Task update contains no fields or contains a protected field")
    with database.connect(url) as conn:
        row = conn.execute(
            "SELECT t.conversation_id, t.revision, t.task_kind, t.status, t.state "
            "FROM task t JOIN conversation c USING (conversation_id) "
            "WHERE c.owner_id = %s AND t.task_id = %s FOR UPDATE OF t",
            (owner_id, task_id),
        ).fetchone()
        if row is None:
            return None
        if row["revision"] != base_revision:
            raise RevisionConflict(f"expected revision {base_revision}, found {row['revision']}")
        revision = base_revision + 1
        state = {**row["state"], **operations, "revision": revision}
        task_kind = str(operations.get("task_kind", row["task_kind"]))
        status = str(operations.get("status", row["status"]))
        conn.execute(
            "UPDATE task SET revision=%s, task_kind=%s, status=%s, state=%s, updated_at=now() "
            "WHERE task_id=%s",
            (revision, task_kind, status, Jsonb(state), task_id),
        )
        conn.execute(
            "INSERT INTO task_history(task_id, revision, state) VALUES (%s, %s, %s)",
            (task_id, revision, Jsonb(state)),
        )
    return {"task_id": str(task_id), "conversation_id": str(row["conversation_id"]),
            "revision": revision, "task_kind": task_kind, "status": status, "state": state}


def get_task_history(url: str | None, owner_id: str, task_id: str) -> list[dict] | None:
    with database.connect(url) as conn:
        authorized = conn.execute(
            "SELECT 1 FROM task t JOIN conversation c USING (conversation_id) "
            "WHERE c.owner_id=%s AND t.task_id=%s", (owner_id, task_id),
        ).fetchone()
        if not authorized:
            return None
        rows = conn.execute(
            "SELECT revision, state, changed_at FROM task_history "
            "WHERE task_id=%s ORDER BY revision", (task_id,),
        ).fetchall()
    return rows


def list_tasks(url: str | None, owner_id: str, conversation_id: str) -> list[dict] | None:
    with database.connect(url) as conn:
        owned = conn.execute(
            "SELECT 1 FROM conversation WHERE owner_id=%s AND conversation_id=%s",
            (owner_id, conversation_id),
        ).fetchone()
        if not owned:
            return None
        rows = conn.execute(
            "SELECT task_id, conversation_id, revision, task_kind, status, state, updated_at "
            "FROM task WHERE conversation_id=%s ORDER BY updated_at DESC",
            (conversation_id,),
        ).fetchall()
    return [{**row, "task_id": str(row["task_id"]),
             "conversation_id": str(row["conversation_id"])} for row in rows]


def list_active_runs(url: str | None, owner_id: str, conversation_id: str) -> list[dict] | None:
    """Return accepted unfinished runs so a refreshed client can resume polling."""
    with database.connect(url) as conn:
        owned = conn.execute(
            "SELECT 1 FROM conversation WHERE owner_id=%s AND conversation_id=%s",
            (owner_id, conversation_id),
        ).fetchone()
        if not owned:
            return None
        rows = conn.execute(
            "SELECT r.run_id, r.task_id, r.status, r.deadline_at, t.revision AS task_revision,"
            " user_message.content AS query, user_message.seq AS message_seq"
            " FROM run r JOIN task t USING (task_id)"
            " JOIN conversation c USING (conversation_id)"
            " JOIN LATERAL (SELECT content, seq FROM conversation_message m"
            "   WHERE m.task_id=t.task_id AND m.role='user' ORDER BY seq DESC LIMIT 1) user_message ON true"
            " WHERE c.owner_id=%s AND c.conversation_id=%s"
            " AND r.status IN ('queued','running') ORDER BY r.created_at",
            (owner_id, conversation_id),
        ).fetchall()
    return [{**row, "run_id": str(row["run_id"]), "task_id": str(row["task_id"]),
             "deadline_at": row["deadline_at"].isoformat()} for row in rows]


def get_conversation_messages(
    url: str | None, owner_id: str, conversation_id: str, limit: int = 40,
) -> list[dict] | None:
    if not 1 <= limit <= 200:
        raise ValueError("message limit must be between 1 and 200")
    with database.connect(url) as conn:
        owned = conn.execute(
            "SELECT 1 FROM conversation WHERE owner_id=%s AND conversation_id=%s",
            (owner_id, conversation_id),
        ).fetchone()
        if not owned:
            return None
        rows = conn.execute(
            "SELECT message_id, task_id, seq, role, content, metadata, created_at"
            " FROM (SELECT message_id, task_id, seq, role, content, metadata, created_at"
            " FROM conversation_message WHERE conversation_id=%s"
            " ORDER BY seq DESC LIMIT %s) recent ORDER BY seq",
            (conversation_id, limit),
        ).fetchall()
    return [{**row, "message_id": str(row["message_id"]),
             "task_id": str(row["task_id"]) if row["task_id"] else None} for row in rows]


def apply_task_patch(url: str | None, owner_id: str, patch: dict[str, Any]):
    """Apply typed user changes and write state plus history in one transaction."""
    from agents.state_models import TaskPatch, TaskState
    from agents.task_patch import validate_and_apply_patch

    patch_model = TaskPatch.model_validate(patch)
    with database.connect(url) as conn:
        row = conn.execute(
            "SELECT t.conversation_id, t.revision, t.state "
            "FROM task t JOIN conversation c USING (conversation_id) "
            "WHERE c.owner_id=%s AND t.task_id=%s FOR UPDATE OF t",
            (owner_id, patch_model.task_id),
        ).fetchone()
        if row is None:
            return None
        if row["revision"] != patch_model.base_revision:
            raise RevisionConflict(
                f"expected revision {patch_model.base_revision}, found {row['revision']}"
            )
        history = []
        if patch_model.operations[0].op == "undo" and patch_model.base_revision > 0:
            previous = conn.execute(
                "SELECT state FROM task_history WHERE task_id=%s AND revision=%s",
                (patch_model.task_id, patch_model.base_revision - 1),
            ).fetchone()
            if previous:
                history.append(TaskState.model_validate(previous["state"]))
        current = TaskState.model_validate(row["state"])
        updated = validate_and_apply_patch(current, patch_model, trusted_history=history)
        serialized = updated.model_dump(mode="json")
        conn.execute(
            "UPDATE task SET revision=%s, task_kind=%s, status=%s, state=%s, updated_at=now() "
            "WHERE task_id=%s",
            (updated.revision, updated.task_kind, updated.status, Jsonb(serialized), patch_model.task_id),
        )
        conn.execute(
            "INSERT INTO task_history(task_id, revision, state) VALUES (%s, %s, %s)",
            (patch_model.task_id, updated.revision, Jsonb(serialized)),
        )
    return {"task_id": updated.task_id, "conversation_id": updated.conversation_id,
            "revision": updated.revision, "task_kind": updated.task_kind,
            "status": updated.status, "state": serialized}


class RunConflict(Exception):
    """The task revision or run idempotency key conflicts with this request."""


class TaskContinuationConflict(RunConflict):
    """The clarification task is no longer at the revision the user answered."""


def enqueue_run(
    url: str | None, owner_id: str, task_id: str, task_revision: int,
    idempotency_key: str, deadline_at: datetime,
) -> dict | None:
    if not idempotency_key or len(idempotency_key) > 200:
        raise ValueError("idempotency key is required and must be <= 200 characters")
    run_id = str(uuid.uuid4())
    with database.connect(url) as conn:
        existing = conn.execute(
            "SELECT r.run_id, r.task_id, r.task_revision, r.status, r.deadline_at "
            "FROM run r WHERE r.idempotency_key=%s",
            (idempotency_key,),
        ).fetchone()
        if existing:
            owned = conn.execute(
                "SELECT 1 FROM task t JOIN conversation c USING (conversation_id) "
                "WHERE c.owner_id=%s AND t.task_id=%s",
                (owner_id, existing["task_id"]),
            ).fetchone()
            if not owned or str(existing["task_id"]) != task_id or existing["task_revision"] != task_revision:
                raise RunConflict("idempotency key is already used by another task revision")
            return {**existing, "run_id": str(existing["run_id"]), "created": False}
        task = conn.execute(
            "SELECT t.task_id, t.revision FROM task t JOIN conversation c USING (conversation_id) "
            "WHERE c.owner_id=%s AND t.task_id=%s FOR UPDATE OF t",
            (owner_id, task_id),
        ).fetchone()
        if task is None:
            return None
        if task["revision"] != task_revision:
            raise RevisionConflict(f"expected revision {task_revision}, found {task['revision']}")
        conn.execute(
            "INSERT INTO run(run_id, task_id, task_revision, status, deadline_at, idempotency_key) "
            "VALUES (%s, %s, %s, 'queued', %s, %s)",
            (run_id, task_id, task_revision, deadline_at, idempotency_key),
        )
    return {"run_id": run_id, "task_id": task_id, "task_revision": task_revision,
            "status": "queued", "deadline_at": deadline_at, "created": True}


def _expire_due_runs(conn, run_id: str | None = None) -> None:
    """Close overdue runs even when no worker is available to claim a new one."""
    from agents.state_models import TaskState, utc_now

    expired = conn.execute(
        "SELECT r.run_id, r.task_id, r.task_revision, r.event_seq,"
        " t.revision AS current_revision, t.state, t.conversation_id"
        " FROM run r JOIN task t USING (task_id)"
        " WHERE r.status IN ('queued','running') AND r.deadline_at <= now()"
        + (" AND r.run_id=%s" if run_id is not None else "")
        + " FOR UPDATE OF r, t SKIP LOCKED",
        (run_id,) if run_id is not None else (),
    ).fetchall()
    for row in expired:
        actual_status = "failed" if row["current_revision"] == row["task_revision"] else "superseded"
        error = {"code": "deadline_exceeded", "message": "任务超过执行时限。"}
        conn.execute(
            "UPDATE run SET status=%s, error=%s, lease_until=NULL, updated_at=now() WHERE run_id=%s",
            (actual_status, Jsonb(error), row["run_id"]),
        )
        if actual_status == "failed":
            current = TaskState.model_validate(row["state"])
            serialized = current.model_copy(update={
                "status": "failed", "revision": current.revision + 1, "updated_at": utc_now(),
            }).model_dump(mode="json")
            conn.execute(
                "UPDATE task SET revision=%s, status='failed', state=%s, updated_at=now() WHERE task_id=%s",
                (serialized["revision"], Jsonb(serialized), row["task_id"]),
            )
            conn.execute(
                "INSERT INTO task_history(task_id, revision, state) VALUES (%s, %s, %s)",
                (row["task_id"], serialized["revision"], Jsonb(serialized)),
            )
            conversation = conn.execute(
                "UPDATE conversation SET revision=revision+1, updated_at=now()"
                " WHERE conversation_id=%s RETURNING revision", (row["conversation_id"],),
            ).fetchone()
            conn.execute(
                "INSERT INTO conversation_message(conversation_id, seq, message_id, task_id, role, content, metadata)"
                " VALUES (%s, %s, %s, %s, 'assistant', %s, %s)",
                (row["conversation_id"], conversation["revision"], str(uuid.uuid4()),
                 row["task_id"], error["message"], Jsonb({"status": "failed", "error": error})),
            )
        event_seq = row["event_seq"] + 1
        conn.execute("UPDATE run SET event_seq=%s WHERE run_id=%s", (event_seq, row["run_id"]))
        conn.execute(
            "INSERT INTO run_event(run_id, seq, event_type, payload) VALUES (%s, %s, 'run_completed', %s)",
            (row["run_id"], event_seq, Jsonb({"status": actual_status, "error": error})),
        )


def claim_next_run(url: str | None, worker_id: str, lease_seconds: int = 30) -> dict | None:
    if not worker_id or not 1 <= lease_seconds <= 300:
        raise ValueError("worker_id and a 1-300 second lease are required")

    with database.connect(url) as conn:
        _expire_due_runs(conn)
        row = conn.execute(
            "WITH candidate AS ("
            " SELECT run_id FROM run"
            " WHERE (status='queued' OR (status='running' AND lease_until < now()))"
            " AND NOT cancel_requested AND deadline_at > now()"
            " ORDER BY created_at, run_id FOR UPDATE SKIP LOCKED LIMIT 1"
            ") UPDATE run r SET status='running', worker_id=%s,"
            " lease_until=now() + (%s * interval '1 second'), attempt=attempt+1, updated_at=now()"
            " FROM candidate c WHERE r.run_id=c.run_id"
            " RETURNING r.run_id, r.task_id, r.task_revision, r.status, r.deadline_at,"
            " r.lease_until, r.attempt, r.cancel_requested",
            (worker_id, lease_seconds),
        ).fetchone()
    if not row:
        return None
    return {**row, "run_id": str(row["run_id"]), "task_id": str(row["task_id"])}


def heartbeat_run(url: str | None, run_id: str, worker_id: str, lease_seconds: int = 30) -> bool:
    with database.connect(url) as conn:
        row = conn.execute(
            "UPDATE run SET lease_until=now() + (%s * interval '1 second'), updated_at=now() "
            "WHERE run_id=%s AND worker_id=%s AND status='running' AND lease_until > now() "
            "AND NOT cancel_requested "
            "RETURNING run_id",
            (lease_seconds, run_id, worker_id),
        ).fetchone()
    return row is not None


def append_run_event(
    url: str | None, run_id: str, worker_id: str,
    event_type: str, payload: dict[str, Any], node_id: str | None = None,
) -> dict | None:
    with database.connect(url) as conn:
        run = conn.execute(
            "UPDATE run SET event_seq=event_seq+1, updated_at=now() "
            "WHERE run_id=%s AND worker_id=%s AND status='running' AND lease_until > now() "
            "AND NOT cancel_requested "
            "RETURNING event_seq",
            (run_id, worker_id),
        ).fetchone()
        if not run:
            return None
        row = conn.execute(
            "INSERT INTO run_event(run_id, seq, event_type, node_id, payload) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING run_id, seq, event_type, node_id, payload, created_at",
            (run_id, run["event_seq"], event_type, node_id, Jsonb(payload)),
        ).fetchone()
    return {**row, "run_id": str(row["run_id"])}


def _location_from_route_state(raw: dict | None) -> dict | None:
    from agents.state_models import LocationRef

    if not isinstance(raw, dict):
        return None
    name = raw.get("name") or raw.get("poi_name")
    poi_id = raw.get("poi_id") or raw.get("id")
    if poi_id:
        return LocationRef(poi_id=str(poi_id), name=name).model_dump(mode="json")
    if raw.get("type") == "coord":
        point = raw.get("coordinates") if isinstance(raw.get("coordinates"), dict) else raw
        lng, lat = point.get("lng", point.get("longitude")), point.get("lat", point.get("latitude"))
        if lng is not None and lat is not None:
            return LocationRef(
                name=name or "坐标位置",
                coordinates={"longitude": lng, "latitude": lat, "crs": point.get("crs", "WGS84")},
            ).model_dump(mode="json")
    if name:
        return LocationRef(name=str(name)).model_dump(mode="json")
    return None


def _route_spec_from_result(current, route_state: dict):
    from agents.state_models import HardConstraints, RouteSpec

    document = current.model_dump(mode="python")
    spec = document["route_spec"]
    start, end = _location_from_route_state(route_state.get("start")), _location_from_route_state(route_state.get("end"))
    if start:
        spec["start"] = start
    if end:
        spec["end"] = end
    via = route_state.get("via")
    via_ref = _location_from_route_state(via)
    if via_ref:
        spec["stops"] = [via_ref]
    if route_state.get("travel_mode") in {"walk", "bike", "drive"}:
        spec["travel_mode"] = route_state["travel_mode"]
    if route_state.get("route_kind") in {"direct", "via", "tour", "multimodal"}:
        spec["route_kind"] = route_state["route_kind"]
    strategy = route_state.get("strategy")
    if isinstance(strategy, dict):
        spec["strategy"] = strategy
    constraints = route_state.get("hard_constraints")
    if isinstance(constraints, dict):
        spec["hard_constraints"] = HardConstraints.model_validate({
            "avoid_slope": constraints.get("avoid_slope", constraints.get("slope") == "avoid"),
            "avoid_steps": constraints.get("avoid_steps", constraints.get("steps") == "avoid"),
            "require_verified_accessibility": constraints.get("require_verified_accessibility", False),
            "max_distance_m": constraints.get("max_distance_m"),
            "max_duration_s": constraints.get("max_duration_s"),
            "excluded_edge_ids": constraints.get("excluded_edge_ids", []),
        }).model_dump(mode="python")
    spec["data_version"] = route_state.get("data_version")
    spec["event_version"] = route_state.get("road_condition_version")
    document["route_spec"] = spec
    return RouteSpec.model_validate(spec), document


def finish_run(
    url: str | None, run_id: str, worker_id: str, *, status: str,
    result: dict | None = None, error: dict | None = None,
) -> bool:
    from agents.state_models import TaskState, utc_now

    if status not in {"completed", "partial", "failed", "needs_input", "cancelled", "superseded"}:
        raise ValueError("invalid terminal run status")
    if isinstance(result, dict):
        result = deepcopy(result)
        route_state = result.get("route_state")
        if isinstance(route_state, dict) and "reuse_certificate" in route_state:
            from agents.route_reuse import has_valid_reuse_certificate

            if not has_valid_reuse_certificate(route_state):
                # The proof is optional acceleration metadata. Drop a
                # malformed handle from every terminal result so it cannot
                # poison a later route refresh or historical replay.
                route_state.pop("reuse_certificate", None)
    with database.connect(url) as conn:
        run = conn.execute(
            "SELECT r.task_id, r.task_revision, r.event_seq, r.cancel_requested,"
            " r.status AS run_status, r.lease_until, r.deadline_at,"
            " t.revision AS current_revision, t.state, t.conversation_id, c.owner_id,"
            " (SELECT m.seq FROM conversation_message m"
            "  WHERE m.task_id=r.task_id AND m.role='user'"
            "    AND m.metadata->>'request_id'=r.idempotency_key"
            "  ORDER BY m.seq DESC LIMIT 1) AS source_message_seq"
            " FROM run r JOIN task t USING (task_id)"
            " JOIN conversation c USING (conversation_id)"
            " WHERE r.run_id=%s AND r.worker_id=%s FOR UPDATE OF r, t",
            (run_id, worker_id),
        ).fetchone()
        if (not run or run["run_status"] != "running" or run["cancel_requested"]
                or not run["lease_until"] or run["lease_until"] <= utc_now()
                or run["deadline_at"] <= utc_now()):
            return False
        actual_status = status
        if run["current_revision"] != run["task_revision"]:
            actual_status = "superseded"
        if actual_status != "superseded" and actual_status != "cancelled":
            current = TaskState.model_validate(run["state"])
            document = current.model_dump(mode="python")
            document.update({"status": actual_status, "revision": current.revision + 1,
                             "updated_at": utc_now()})
            if actual_status == "needs_input" and isinstance(result, dict):
                clarify = result.get("clarify") or {}
                question = clarify.get("question") if isinstance(clarify, dict) else None
                if not question and result.get("response_kind") == "candidates":
                    question = "请选择一个地点继续规划路线。"
                document["pending_questions"] = [question] if question else []
            if isinstance(result, dict) and isinstance(result.get("place_request"), dict):
                document["place_request"] = result["place_request"]
            elif actual_status in {"completed", "partial", "failed"}:
                document["place_request"] = None
            requirement_results = result.get("requirement_results", {}) if isinstance(result, dict) else {}
            for requirement in document.get("requirements", []):
                requirement_id = requirement["requirement_id"]
                status_for_requirement = requirement_results.get(requirement_id)
                if requirement_id.startswith("answer-"):
                    status_for_requirement = "satisfied" if actual_status == "completed" else None
                if status_for_requirement in {"satisfied", "failed", "needs_input"}:
                    requirement["status"] = status_for_requirement
                elif actual_status == "completed":
                    requirement["status"] = "satisfied"
                elif actual_status == "needs_input":
                    requirement["status"] = "needs_input"
                elif actual_status == "failed":
                    requirement["status"] = "failed"
            if isinstance(result, dict) and isinstance(result.get("route_state"), dict):
                route_state = result["route_state"]
                route_spec, _ = _route_spec_from_result(current, route_state)
                document["route_spec"] = route_spec.model_dump(mode="python")
            serialized = TaskState.model_validate(document).model_dump(mode="json")
            route_state = result.get("route_state") if isinstance(result, dict) else None
            handle = route_state.get("reuse_certificate") if isinstance(route_state, dict) else None
            if isinstance(handle, dict) and ("task_id" in handle or "task_revision" in handle):
                current_binding = (
                    handle.get("task_id") == str(run["task_id"])
                    and type(handle.get("task_revision")) is int
                    and handle.get("task_revision") == run["task_revision"]
                )
                rebound = False
                if current_binding:
                    from storage.route_proof_repository import rebind_verified_proof_task_revision

                    try:
                        rebound = rebind_verified_proof_task_revision(
                            conn,
                            proof_id=handle.get("proof_id"),
                            owner_id=str(run["owner_id"]),
                            task_id=str(run["task_id"]),
                            old_revision=run["task_revision"],
                            new_revision=serialized["revision"],
                        )
                    except (TypeError, ValueError):
                        # A malformed optional acceleration handle must not turn
                        # a valid route result into a failed user task.
                        rebound = False
                if rebound:
                    handle["task_revision"] = serialized["revision"]
                else:
                    route_state.pop("reuse_certificate", None)
            conn.execute(
                "UPDATE task SET revision=%s, status=%s, state=%s, updated_at=now() WHERE task_id=%s",
                (serialized["revision"], actual_status, Jsonb(serialized), run["task_id"]),
            )
            conn.execute(
                "INSERT INTO task_history(task_id, revision, state) VALUES (%s, %s, %s)",
                (run["task_id"], serialized["revision"], Jsonb(serialized)),
            )
            display_text = ""
            if result:
                for key in ("explanation", "reply", "message"):
                    if isinstance(result.get(key), str) and result[key].strip():
                        display_text = result[key].strip()
                        break
            if error and not display_text:
                display_text = str(error.get("message") or "任务执行失败，请稍后重试。")
            if display_text:
                conversation = conn.execute(
                    "UPDATE conversation SET revision=revision+1, updated_at=now()"
                    " WHERE conversation_id=%s RETURNING revision",
                    (run["conversation_id"],),
                ).fetchone()
                conn.execute(
                    "INSERT INTO conversation_message(conversation_id, seq, message_id, task_id, role, content, metadata) "
                    "VALUES (%s, %s, %s, %s, 'assistant', %s, %s)",
                    (run["conversation_id"], conversation["revision"], str(uuid.uuid4()),
                     run["task_id"], display_text, Jsonb({
                         "run_id": run_id,
                         "status": actual_status,
                         **({"source_message_seq": run["source_message_seq"]}
                            if run["source_message_seq"] is not None else {}),
                         **({"task_type": result["task_type"]} if isinstance(result, dict) and result.get("task_type") else {}),
                         **({"response_kind": result["response_kind"]} if isinstance(result, dict) and result.get("response_kind") else {}),
                         **({"route_state": result["route_state"]}
                            if isinstance(result, dict) and result.get("route_state") else {}),
                     })),
                )
        conn.execute(
            "UPDATE run SET status=%s, result=%s, error=%s, lease_until=NULL, updated_at=now() "
            "WHERE run_id=%s",
            (actual_status, Jsonb(result) if result is not None else None,
             Jsonb(error) if error is not None else None, run_id),
        )
        event_seq = run["event_seq"] + 1
        conn.execute("UPDATE run SET event_seq=%s WHERE run_id=%s", (event_seq, run_id))
        conn.execute(
            "INSERT INTO run_event(run_id, seq, event_type, payload) VALUES (%s, %s, %s, %s)",
            (run_id, event_seq, "run_completed", Jsonb({"status": actual_status})),
        )
    return True


def cancel_run(url: str | None, owner_id: str, run_id: str) -> dict | None:
    from agents.state_models import TaskState, utc_now

    with database.connect(url) as conn:
        row = conn.execute(
            "UPDATE run r SET cancel_requested=CASE WHEN r.status IN"
            " ('completed','partial','failed','needs_input','cancelled','superseded')"
            " THEN r.cancel_requested ELSE true END,"
            " status=CASE WHEN r.status IN ('completed','partial','failed','needs_input','cancelled','superseded')"
            " THEN r.status ELSE 'cancelled' END, lease_until=NULL, updated_at=now()"
            " FROM task t JOIN conversation c USING (conversation_id)"
            " WHERE r.task_id=t.task_id AND r.run_id=%s AND c.owner_id=%s"
            " RETURNING r.run_id, r.task_id, r.status, r.cancel_requested,"
            " t.revision AS task_revision, t.state, t.conversation_id",
            (run_id, owner_id),
        ).fetchone()
        if not row:
            return None
        if row["status"] == "cancelled" and row["state"].get("status") not in {"cancelled", "completed", "partial", "failed"}:
            current = TaskState.model_validate(row["state"])
            state = current.model_copy(update={
                "status": "cancelled", "revision": current.revision + 1, "updated_at": utc_now(),
            }).model_dump(mode="json")
            conn.execute(
                "UPDATE task SET revision=%s, status='cancelled', state=%s, updated_at=now() WHERE task_id=%s",
                (state["revision"], Jsonb(state), row["task_id"]),
            )
            conn.execute(
                "INSERT INTO task_history(task_id, revision, state) VALUES (%s, %s, %s)",
                (row["task_id"], state["revision"], Jsonb(state)),
            )
    return {key: row[key] for key in ("run_id", "task_id", "status", "cancel_requested")} | {
        "run_id": str(row["run_id"]), "task_id": str(row["task_id"]),
    }


def get_run(url: str | None, owner_id: str, run_id: str, after_seq: int = 0) -> dict | None:
    with database.connect(url) as conn:
        query = (
            "SELECT r.run_id, r.task_id, r.task_revision, t.revision AS current_task_revision,"
            " r.status, r.deadline_at, r.lease_until,"
            " r.attempt, r.cancel_requested, r.result, r.error FROM run r"
            " JOIN task t USING (task_id) JOIN conversation c USING (conversation_id)"
            " WHERE c.owner_id=%s AND r.run_id=%s"
        )
        run = conn.execute(query, (owner_id, run_id)).fetchone()
        if not run:
            return None
        if run["status"] in {"queued", "running"}:
            _expire_due_runs(conn, run_id)
            run = conn.execute(query, (owner_id, run_id)).fetchone()
        events = conn.execute(
            "SELECT seq, event_type, node_id, payload, created_at FROM run_event"
            " WHERE run_id=%s AND seq>%s ORDER BY seq",
            (run_id, after_seq),
        ).fetchall()
    return {**run, "run_id": str(run["run_id"]), "task_id": str(run["task_id"]),
            "events": events}
