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
    ui_action: dict[str, Any] | None = None,
    location_observation: dict[str, Any] | None = None,
) -> dict:
    """Atomically accept a user message, task snapshot and queued run."""
    from agents.state_models import Requirement, RouteSpec, TaskState
    from agents.task_planner import build_task_plan

    if not isinstance(message, str) or not message.strip() or len(message) > 500:
        raise ValueError("message must contain 1-500 characters")
    if not idempotency_key or len(idempotency_key) > 200:
        raise ValueError("idempotency key is required and must be <= 200 characters")
    route_spec = route_spec or {}
    validated_route_spec = RouteSpec.model_validate(route_spec).model_dump(mode="json")
    from agents.dialogue import model_dialogue_enabled
    if model_dialogue_enabled():
        return _create_model_run(
            url, owner_id, conversation_id, message, idempotency_key, deadline_at,
            route_spec=route_spec, continuation_task_id=continuation_task_id,
            base_revision=base_revision, selected_poi_id=selected_poi_id,
            ui_action=ui_action, location_observation=location_observation,
        )
    message = message.strip()
    if continuation_task_id is not None and (not isinstance(base_revision, int) or base_revision < 0):
        raise ValueError("continuation requires a nonnegative base_revision")
    submission = {"request_id": idempotency_key, "route_spec": route_spec,
                  "continuation_task_id": continuation_task_id}
    if selected_poi_id is not None:
        if not isinstance(selected_poi_id, str) or not re.fullmatch(
                r"(?:poi_\d{3,}|amap:[A-Za-z0-9_-]{3,80})", selected_poi_id):
            raise ValueError("selected_poi_id must be a stable local or provider ID")
        if continuation_task_id is None:
            raise ValueError("selected_poi_id requires a clarification task")
        submission["selected_poi_id"] = selected_poi_id
    task_id, run_id = str(continuation_task_id or uuid.uuid4()), str(uuid.uuid4())
    with database.connect(url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 73518264))", (idempotency_key,))
        existing = conn.execute(
            "SELECT r.run_id, r.task_id, r.task_revision, r.status, m.content, m.metadata,"
            " t.state, c.owner_id, c.conversation_id "
            "FROM run r JOIN task t USING (task_id) JOIN conversation c ON c.conversation_id=t.conversation_id "
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
            latest_user = conn.execute(
                "SELECT task_id FROM conversation_message WHERE conversation_id=%s AND role='user'"
                " ORDER BY seq DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if not latest_user or str(latest_user["task_id"]) != str(continuation_task_id):
                raise TaskContinuationConflict("a newer conversation request superseded this clarification")
            if selected_poi_id is not None:
                pending = current.place_request
                if (not pending or pending.pending_slot != "selection"
                        or selected_poi_id not in pending.candidate_poi_ids):
                    raise TaskContinuationConflict("selected place is not in the current candidate list")
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


def _create_model_run(url, owner_id, conversation_id, message, idempotency_key, deadline_at, *,
                      route_spec, continuation_task_id, base_revision, selected_poi_id,
                      ui_action, location_observation):
    """Persist the message and queue work without guessing its task relation."""
    if continuation_task_id is not None and (type(base_revision) is not int or base_revision < 0):
        raise ValueError("continuation requires a nonnegative base_revision")
    if selected_poi_id is not None and (not isinstance(selected_poi_id, str) or not re.fullmatch(
            r"(?:poi_\d{3,}|amap:[A-Za-z0-9_-]{3,80})", selected_poi_id)):
        raise ValueError("selected_poi_id must be a stable local or provider ID")
    submission = {"request_id": idempotency_key, "route_spec": route_spec,
                  "continuation_task_id": continuation_task_id, "base_revision": base_revision,
                  "selected_poi_id": selected_poi_id, "ui_action": ui_action,
                  "location_observation": location_observation}
    run_id = str(uuid.uuid4())
    with database.connect(url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 73518264))", (idempotency_key,))
        existing = conn.execute(
            "SELECT r.run_id,r.task_id,r.task_revision,r.status,r.conversation_id,c.owner_id,"
            " m.content,m.metadata FROM run r JOIN conversation c ON c.conversation_id=r.conversation_id"
            " LEFT JOIN conversation_message m ON m.conversation_id=r.conversation_id"
            " AND m.seq=r.input_message_seq WHERE r.idempotency_key=%s", (idempotency_key,),
        ).fetchone()
        if existing:
            if (str(existing["owner_id"]) != owner_id or str(existing["conversation_id"]) != conversation_id
                    or existing["content"] != message or (existing["metadata"] or {}).get("request") != submission):
                raise RunConflict("request ID was already used for another submission")
            return {"task_id": str(existing["task_id"]) if existing["task_id"] else None,
                    "task_revision": existing["task_revision"], "run_id": str(existing["run_id"]),
                    "status": existing["status"], "created": False}
        owned = conn.execute("SELECT 1 FROM conversation WHERE owner_id=%s AND conversation_id=%s FOR UPDATE",
                             (owner_id, conversation_id)).fetchone()
        if not owned:
            raise PermissionError("Conversation is not owned by this identity")
        if continuation_task_id is not None:
            task = conn.execute("SELECT revision,state FROM task WHERE conversation_id=%s AND task_id=%s",
                                (conversation_id, continuation_task_id)).fetchone()
            if not task or task["revision"] != base_revision:
                raise TaskContinuationConflict("the referenced card or task is no longer current")
            if selected_poi_id is not None:
                pending = (task["state"] or {}).get("place_request") or {}
                if selected_poi_id not in (pending.get("candidate_poi_ids") or []):
                    raise TaskContinuationConflict("selected place is not in the current candidate list")
                latest = conn.execute("SELECT task_id FROM conversation_message WHERE conversation_id=%s"
                                      " AND role='user' ORDER BY seq DESC LIMIT 1", (conversation_id,)).fetchone()
                if not latest or str(latest["task_id"]) != continuation_task_id:
                    raise TaskContinuationConflict("a newer message superseded this candidate card")
        elif selected_poi_id is not None:
            raise ValueError("selected_poi_id requires a referenced candidate task")
        conversation = conn.execute(
            "UPDATE conversation SET revision=revision+1,updated_at=now() WHERE conversation_id=%s RETURNING revision",
            (conversation_id,),
        ).fetchone()
        seq = conversation["revision"]
        conn.execute("INSERT INTO conversation_message(conversation_id,seq,message_id,role,content,metadata)"
                     " VALUES (%s,%s,%s,'user',%s,%s)",
                     (conversation_id, seq, str(uuid.uuid4()), message,
                      Jsonb({"request_id": idempotency_key, "request": submission})))
        conn.execute("INSERT INTO run(run_id,conversation_id,input_message_seq,model_led,status,"
                     "processing_stage,deadline_at,idempotency_key) VALUES (%s,%s,%s,true,'queued','queued',%s,%s)",
                     (run_id, conversation_id, seq, deadline_at, idempotency_key))
    return {"task_id": None, "task_revision": None, "run_id": run_id, "status": "queued", "created": True}


def _get_model_run_input(conn, row):
    message = conn.execute("SELECT seq,message_id,content,metadata FROM conversation_message"
                           " WHERE conversation_id=%s AND seq=%s AND role='user'",
                           (row["conversation_id"], row["input_message_seq"])).fetchone()
    if not message:
        raise RuntimeError("run has no accepted input message")
    history = conn.execute("SELECT seq,message_id,task_id,role,content,metadata FROM conversation_message"
                           " WHERE conversation_id=%s AND seq<%s AND role IN ('user','assistant') ORDER BY seq",
                           (row["conversation_id"], message["seq"])).fetchall()
    tasks = conn.execute("SELECT task_id,revision,task_kind,status,state FROM task"
                         " WHERE conversation_id=%s ORDER BY updated_at DESC,task_id",
                         (row["conversation_id"],)).fetchall()
    request = (message["metadata"] or {}).get("request") or {}
    return {"run_id": str(row["run_id"]), "conversation_id": str(row["conversation_id"]),
            "owner_id": str(row["owner_id"]), "task_id": str(row["task_id"]) if row["task_id"] else None,
            "task_revision": row["task_revision"], "query": message["content"],
            "message_id": str(message["message_id"]), "input_message_seq": message["seq"],
            "history": [{**item, "message_id": str(item["message_id"]),
                         "task_id": str(item["task_id"]) if item["task_id"] else None} for item in history],
            "tasks": [{**item, "task_id": str(item["task_id"])} for item in tasks],
            "task": row.get("state"), "request": request, "deadline_at": row["deadline_at"],
            "cancel_requested": row["cancel_requested"], "model_led": True}


def read_conversation_result(url, conversation_id, *, run_id, section, offset=0, limit=50):
    """Read a bounded page of a result from this conversation only."""
    try:
        run_id = str(uuid.UUID(str(run_id)))
    except (ValueError, TypeError, AttributeError):
        return {"error": "invalid_result_reference", "message": "结果引用格式无效。"}
    if (section not in {"route_summary", "route_geometry", "candidates"}
            or type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100):
        return {"error": "invalid_result_page", "message": "结果页参数无效。"}
    with database.connect(url) as conn:
        row = conn.execute("SELECT result FROM run WHERE run_id=%s AND conversation_id=%s"
                           " AND status IN ('completed','partial','needs_input')", (run_id, conversation_id)).fetchone()
    if not row or not isinstance(row["result"], dict):
        return {"error": "result_reference_unavailable", "message": "本会话没有可读取的该结果。"}
    result = row["result"]
    if section == "route_summary":
        return {"run_id": run_id, "section": section, "result": {
            key: result[key] for key in ("start", "end", "mode", "strategy", "distance_m",
                                         "recommended_length_m", "duration_min", "route_verification") if key in result}}
    rows = result.get("recommended") if section == "route_geometry" else result.get("candidates")
    rows = rows if isinstance(rows, list) else []
    page = rows[offset:offset + limit]
    return {"run_id": run_id, "section": section, "items": page, "offset": offset,
            "next_offset": offset + len(page), "count": len(rows), "exhausted": offset + len(page) >= len(rows)}


def bind_model_task(url, run_id, worker_id, *, relation, place_request, task_id=None, base_revision=None,
                    changed_fields=None):
    """Accept the model's proposed relation with lease, ownership and version checks."""
    from agents.state_models import FieldProvenance, PlaceRequestState, Requirement, RouteSpec, TaskState, utc_now

    if relation not in {"new", "continue", "resume_route"}:
        raise ValueError("invalid task relation")
    pending = PlaceRequestState.model_validate(place_request).model_dump(mode="json")
    with database.connect(url) as conn:
        run = conn.execute("SELECT r.*,c.owner_id FROM run r JOIN conversation c USING (conversation_id)"
                           " WHERE r.run_id=%s AND r.worker_id=%s FOR UPDATE OF r",
                           (run_id, worker_id)).fetchone()
        if (not run or not run["model_led"] or run["status"] != "running" or run["cancel_requested"]
                or not run["lease_until"] or run["lease_until"] <= utc_now() or run["deadline_at"] <= utc_now()):
            raise RunConflict("run lease expired or cancelled")
        conn.execute("SELECT 1 FROM conversation WHERE conversation_id=%s FOR UPDATE", (run["conversation_id"],))
        if conn.execute("SELECT 1 FROM conversation_message WHERE conversation_id=%s AND role='user'"
                        " AND seq>%s LIMIT 1", (run["conversation_id"], run["input_message_seq"])).fetchone():
            raise RunConflict("a newer input superseded this run before task binding")
        if run["task_id"]:
            task = conn.execute("SELECT revision,state FROM task WHERE task_id=%s AND conversation_id=%s FOR UPDATE",
                                (run["task_id"], run["conversation_id"])).fetchone()
            if (not task or task["revision"] != run["task_revision"]
                    or relation != "new" and str(task_id) != str(run["task_id"])):
                raise RunConflict("bound task changed during worker recovery")
            return {"task_id": str(run["task_id"]), "task_revision": run["task_revision"], "task": task["state"]}
        message = conn.execute("SELECT message_id,content,metadata FROM conversation_message"
                               " WHERE conversation_id=%s AND seq=%s AND role='user'",
                               (run["conversation_id"], run["input_message_seq"])).fetchone()
        if not message:
            raise RunConflict("input message is missing")
        request = (message["metadata"] or {}).get("request") or {}
        if request.get("selected_poi_id") and (
                relation == "new" or str(task_id) != str(request.get("continuation_task_id"))
                or base_revision != request.get("base_revision")):
            raise RunConflict("selected card must bind its referenced task and revision")
        field_provenance = {str(field): FieldProvenance(
            source="inference", scope="task", source_message_id=str(message["message_id"]), updated_at=utc_now(),
        ).model_dump(mode="json") for field in (changed_fields or []) if isinstance(field, str) and len(field) <= 150}
        if relation == "new":
            task_id = str(uuid.uuid4())
            state = TaskState(task_id=task_id, conversation_id=str(run["conversation_id"]),
                              task_kind="information", status="running",
                              route_spec=RouteSpec.model_validate(request.get("route_spec") or {}),
                              place_request=pending, field_provenance=field_provenance,
                              requirements=[Requirement(requirement_id="message-" + str(message["message_id"]),
                                                        description=message["content"])]).model_dump(mode="json")
            revision = 0
            conn.execute("INSERT INTO task(task_id,conversation_id,revision,task_kind,status,state)"
                         " VALUES (%s,%s,0,'information','running',%s)",
                         (task_id, run["conversation_id"], Jsonb(state)))
        else:
            if not task_id or type(base_revision) is not int:
                raise ValueError("continuation requires task_id and task_revision")
            task = conn.execute("SELECT revision,state FROM task WHERE task_id=%s AND conversation_id=%s FOR UPDATE",
                                (task_id, run["conversation_id"])).fetchone()
            if not task or task["revision"] != base_revision:
                raise RevisionConflict("task revision has changed")
            current = TaskState.model_validate(task["state"])
            document = current.model_dump(mode="python")
            revision = current.revision + 1
            document.update({"revision": revision, "status": "running", "updated_at": utc_now(),
                             "place_request": pending, "pending_questions": []})
            document["field_provenance"].update(field_provenance)
            state = TaskState.model_validate(document).model_dump(mode="json")
            conn.execute("UPDATE task SET revision=%s,status='running',state=%s,updated_at=now() WHERE task_id=%s",
                         (revision, Jsonb(state), task_id))
        conn.execute("INSERT INTO task_history(task_id,revision,state) VALUES (%s,%s,%s)",
                     (task_id, revision, Jsonb(state)))
        conn.execute("UPDATE conversation_message SET task_id=%s WHERE conversation_id=%s AND seq=%s",
                     (task_id, run["conversation_id"], run["input_message_seq"]))
        conn.execute("UPDATE run SET task_id=%s,task_revision=%s,processing_stage='tools',updated_at=now()"
                     " WHERE run_id=%s", (task_id, revision, run_id))
    return {"task_id": str(task_id), "task_revision": revision, "task": state}


def get_run_input(url: str | None, run_id: str) -> dict | None:
    """Read trusted task text and conversation context for a claimed run."""
    with database.connect(url) as conn:
        model_run = conn.execute(
            "SELECT r.*,c.owner_id,t.state FROM run r JOIN conversation c USING (conversation_id)"
            " LEFT JOIN task t ON t.task_id=r.task_id WHERE r.run_id=%s AND r.model_led",
            (run_id,),
        ).fetchone()
        if model_run:
            return _get_model_run_input(conn, model_run)
        row = conn.execute(
            "SELECT r.run_id, r.task_id, r.task_revision, r.cancel_requested,"
            " t.conversation_id, t.revision, t.state, c.owner_id"
            " FROM run r JOIN task t USING (task_id)"
            " JOIN conversation c ON c.conversation_id=t.conversation_id WHERE r.run_id=%s",
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
            "SELECT prior.metadata->'route_state' AS route_state, prior.task_id, source_user.seq AS source_seq"
            " FROM conversation_message prior"
            " LEFT JOIN conversation_message source_user ON source_user.conversation_id=prior.conversation_id"
            " AND source_user.role='user' AND source_user.task_id=prior.task_id"
            " AND source_user.seq::text=prior.metadata->>'source_message_seq'"
            " WHERE prior.conversation_id=%s AND prior.role='assistant' AND prior.seq<%s"
            " AND prior.metadata ? 'route_state'"
            " ORDER BY source_user.seq DESC NULLS LAST, prior.seq DESC LIMIT 1",
            (row["conversation_id"], user_message["seq"]),
        ).fetchone()
        # A newer failed or unfinished route request makes an older successful
        # route unsafe to inherit, even if its answer arrived more recently.
        # Explicitly unrelated questions such as weather remain pass-through.
        from agents.context_builder import is_implicit_destination_followup, is_route_followup
        from agents.task_planner import build_task_plan

        route_context_uncertain = False
        if is_route_followup(user_message["content"]):
            if not previous or previous.get("source_seq") is None:
                previous = None
                route_context_uncertain = True
            else:
                intervening = conn.execute(
                    "SELECT content FROM conversation_message"
                    " WHERE conversation_id=%s AND role='user' AND seq>%s AND seq<%s ORDER BY seq",
                    (row["conversation_id"], previous["source_seq"], user_message["seq"]),
                ).fetchall()
                if any(is_route_followup(item["content"])
                       or build_task_plan(item["content"], has_route_context=False).kind in {"route", "composite"}
                       for item in intervening):
                    previous = None
                    route_context_uncertain = True
        # Bind to the most recent *user request*, not the last response to
        # arrive: a slower, older task can finish after a newer one. If that
        # request has no completed answer, never fall back to an older trip.
        # The run result is server-owned; request JSON cannot provide it.
        latest_delivery = conn.execute(
            "SELECT r.status, r.result FROM conversation_message user_turn"
            " LEFT JOIN conversation_message prior ON prior.conversation_id=user_turn.conversation_id"
            " AND prior.role='assistant' AND prior.task_id=user_turn.task_id AND prior.seq<%s"
            " AND prior.metadata->>'source_message_seq'=user_turn.seq::text"
            " LEFT JOIN run r ON r.run_id::text=prior.metadata->>'run_id' AND r.task_id=prior.task_id"
            " WHERE user_turn.conversation_id=%s AND user_turn.role='user' AND user_turn.seq<%s"
            " ORDER BY user_turn.seq DESC, prior.seq DESC NULLS LAST LIMIT 1",
            (user_message["seq"], row["conversation_id"], user_message["seq"]),
        ).fetchone()
        # An elliptical "what about X?" refers to the immediately preceding
        # route, not an older route hidden behind another task or a failed run.
        if is_implicit_destination_followup(user_message["content"]):
            latest_result = (latest_delivery or {}).get("result") or {}
            latest_route = latest_result.get("route_state") if isinstance(latest_result, dict) else None
            verification = latest_result.get("route_verification") if isinstance(latest_result, dict) else None
            previous_state = (previous or {}).get("route_state")
            if (not latest_delivery or latest_delivery.get("status") not in {"completed", "partial"}
                    or not isinstance(latest_route, dict)
                    or latest_result.get("response_kind") != "route"
                    or not isinstance(verification, dict) or verification.get("status") != "pass"
                    or not isinstance(previous_state, dict)
                    or not previous_state.get("route_id")
                    or latest_route.get("route_id") != previous_state.get("route_id")):
                previous = None
                route_context_uncertain = True
        previous_place = None
        if (latest_delivery and latest_delivery["status"] == "completed"
                and isinstance(latest_delivery["result"], dict)):
            candidate = latest_delivery["result"].get("completed_place_request")
            if isinstance(candidate, dict):
                previous_place = candidate
    return {
        "run_id": str(row["run_id"]), "task_id": str(row["task_id"]),
        "task_revision": row["task_revision"], "task_revision_current": row["revision"],
        "conversation_id": str(row["conversation_id"]), "owner_id": str(row["owner_id"]),
        "task": row["state"], "query": user_message["content"], "history": history,
        "selected_poi_id": ((user_message.get("metadata") or {}).get("request") or {}).get("selected_poi_id"),
        "task_origin_query": origin_message["content"] if origin_message else None,
        "previous_route_state": previous["route_state"] if previous else None,
        "previous_route_state_task_id": str(previous["task_id"]) if previous and previous["task_id"] else None,
        "route_context_uncertain": route_context_uncertain,
        "previous_completed_place_request": previous_place,
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
            "SELECT r.run_id, r.task_id, r.status, r.deadline_at, r.task_revision, r.processing_stage,"
            " user_message.content AS query, user_message.seq AS message_seq"
            " FROM run r JOIN conversation c USING (conversation_id)"
            " LEFT JOIN LATERAL (SELECT content, seq FROM conversation_message m"
            "   WHERE m.conversation_id=r.conversation_id AND m.role='user'"
            "   AND (m.seq=r.input_message_seq OR (r.input_message_seq IS NULL AND m.task_id=r.task_id))"
            "   ORDER BY seq DESC LIMIT 1) user_message ON true"
            " WHERE c.owner_id=%s AND c.conversation_id=%s"
            " AND r.status IN ('queued','running') ORDER BY r.created_at",
            (owner_id, conversation_id),
        ).fetchall()
    return [{**row, "run_id": str(row["run_id"]), "task_id": str(row["task_id"]) if row["task_id"] else None,
             "deadline_at": row["deadline_at"].isoformat()} for row in rows]


def get_conversation_messages(
    url: str | None, owner_id: str, conversation_id: str, limit: int | None = None,
) -> list[dict] | None:
    if limit is not None and not 1 <= limit <= 200:
        raise ValueError("message limit must be between 1 and 200")
    with database.connect(url) as conn:
        owned = conn.execute(
            "SELECT 1 FROM conversation WHERE owner_id=%s AND conversation_id=%s",
            (owner_id, conversation_id),
        ).fetchone()
        if not owned:
            return None
        query = "SELECT message_id,task_id,seq,role,content,metadata,created_at FROM conversation_message WHERE conversation_id=%s"
        if limit is None:
            rows = conn.execute(query + " ORDER BY seq", (conversation_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM (" + query + " ORDER BY seq DESC LIMIT %s) recent ORDER BY seq",
                                (conversation_id, limit)).fetchall()
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

    unbound = conn.execute(
        "SELECT r.run_id,r.conversation_id,r.input_message_seq,r.event_seq FROM run r"
        " WHERE r.task_id IS NULL AND r.model_led AND r.status IN ('queued','running') AND r.deadline_at<=now()"
        + (" AND r.run_id=%s" if run_id is not None else "") + " FOR UPDATE SKIP LOCKED",
        (run_id,) if run_id is not None else (),
    ).fetchall()
    for item in unbound:
        error = {"code": "deadline_exceeded", "message": "本轮处理超时，请重试。"}
        conn.execute("SELECT 1 FROM conversation WHERE conversation_id=%s FOR UPDATE", (item["conversation_id"],))
        newer = conn.execute("SELECT 1 FROM conversation_message WHERE conversation_id=%s AND role='user'"
                             " AND seq>%s LIMIT 1", (item["conversation_id"], item["input_message_seq"])).fetchone()
        status = "superseded" if newer else "failed"
        if not newer:
            seq = conn.execute("UPDATE conversation SET revision=revision+1,updated_at=now()"
                               " WHERE conversation_id=%s RETURNING revision", (item["conversation_id"],)).fetchone()["revision"]
            conn.execute("INSERT INTO conversation_message(conversation_id,seq,message_id,role,content,metadata)"
                         " VALUES (%s,%s,%s,'assistant',%s,%s)",
                         (item["conversation_id"], seq, str(uuid.uuid4()), error["message"],
                          Jsonb({"run_id": str(item["run_id"]), "status": status,
                                 "source_message_seq": item["input_message_seq"], "error": error})))
        conn.execute("UPDATE run SET status=%s,error=%s,lease_until=NULL,processing_stage=%s,"
                     "event_seq=event_seq+1,updated_at=now() WHERE run_id=%s",
                     (status, Jsonb(error), status, item["run_id"]))
        conn.execute("INSERT INTO run_event(run_id,seq,event_type,payload) VALUES (%s,%s,'run_completed',%s)",
                     (item["run_id"], item["event_seq"] + 1, Jsonb({"status": status, "error": error})))
    expired = conn.execute(
        "SELECT r.run_id, r.task_id, r.task_revision, r.event_seq,r.model_led,r.input_message_seq,"
        " t.revision AS current_revision, t.state, t.conversation_id"
        " FROM run r JOIN task t USING (task_id)"
        " WHERE r.status IN ('queued','running') AND r.deadline_at <= now()"
        + (" AND r.run_id=%s" if run_id is not None else "")
        + " FOR UPDATE OF r, t SKIP LOCKED",
        (run_id,) if run_id is not None else (),
    ).fetchall()
    for row in expired:
        actual_status = "failed" if row["current_revision"] == row["task_revision"] else "superseded"
        if row["model_led"]:
            conn.execute("SELECT 1 FROM conversation WHERE conversation_id=%s FOR UPDATE", (row["conversation_id"],))
            newer = conn.execute("SELECT 1 FROM conversation_message WHERE conversation_id=%s AND role='user'"
                                 " AND seq>%s LIMIT 1", (row["conversation_id"], row["input_message_seq"])).fetchone()
            if newer:
                actual_status = "superseded"
        error = {"code": "deadline_exceeded", "message": "本轮处理超时，请重试。"}
        conn.execute(
            "UPDATE run SET status=%s, error=%s, lease_until=NULL, processing_stage=%s, updated_at=now() WHERE run_id=%s",
            (actual_status, Jsonb(error), actual_status, row["run_id"]),
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
                 row["task_id"], error["message"], Jsonb({"status": "failed", "error": error,
                    "run_id": str(row["run_id"]), "source_message_seq": row["input_message_seq"]})),
            )
        elif (actual_status == "superseded" and row["model_led"]
              and row["current_revision"] == row["task_revision"]):
            current = TaskState.model_validate(row["state"])
            serialized = current.model_copy(update={"status": "superseded", "revision": current.revision + 1,
                                                     "updated_at": utc_now()}).model_dump(mode="json")
            conn.execute("UPDATE task SET revision=%s,status='superseded',state=%s,updated_at=now() WHERE task_id=%s",
                         (serialized["revision"], Jsonb(serialized), row["task_id"]))
            conn.execute("INSERT INTO task_history(task_id,revision,state) VALUES (%s,%s,%s)",
                         (row["task_id"], serialized["revision"], Jsonb(serialized)))
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
            " SELECT r.run_id FROM run r JOIN conversation c ON c.conversation_id=r.conversation_id"
            " WHERE r.model_led AND (r.status='queued' OR (r.status='running' AND r.lease_until < now()))"
            " AND NOT r.cancel_requested AND r.deadline_at > now()"
            " AND (NOT r.model_led OR NOT EXISTS (SELECT 1 FROM run earlier"
            " WHERE earlier.conversation_id=r.conversation_id AND earlier.model_led"
            " AND earlier.status IN ('queued','running') AND NOT earlier.cancel_requested"
            " AND earlier.run_id<>r.run_id AND earlier.input_message_seq<r.input_message_seq))"
            " ORDER BY r.created_at, r.run_id FOR UPDATE OF c,r SKIP LOCKED LIMIT 1"
            ") UPDATE run r SET status='running', worker_id=%s,"
            " processing_stage='understanding',lease_until=now() + (%s * interval '1 second'),"
            " attempt=attempt+1, updated_at=now()"
            " FROM candidate c WHERE r.run_id=c.run_id"
            " RETURNING r.run_id, r.task_id, r.task_revision, r.status, r.deadline_at,"
            " r.lease_until, r.attempt, r.cancel_requested",
            (worker_id, lease_seconds),
        ).fetchone()
        if not row:
            # Legacy task-bound clients retain their original independent-run
            # behavior; model-led messages above are serialized by conversation.
            row = conn.execute(
                "WITH candidate AS (SELECT run_id FROM run"
                " WHERE NOT model_led AND (status='queued' OR (status='running' AND lease_until<now()))"
                " AND NOT cancel_requested AND deadline_at>now()"
                " ORDER BY created_at,run_id FOR UPDATE SKIP LOCKED LIMIT 1)"
                " UPDATE run r SET status='running',worker_id=%s,processing_stage='understanding',"
                " lease_until=now()+(%s * interval '1 second'),attempt=attempt+1,updated_at=now()"
                " FROM candidate c WHERE r.run_id=c.run_id"
                " RETURNING r.run_id,r.task_id,r.task_revision,r.status,r.deadline_at,"
                " r.lease_until,r.attempt,r.cancel_requested", (worker_id, lease_seconds),
            ).fetchone()
    if not row:
        return None
    return {**row, "run_id": str(row["run_id"]), "task_id": str(row["task_id"]) if row["task_id"] else None}


def heartbeat_run(url: str | None, run_id: str, worker_id: str, lease_seconds: int = 30) -> bool:
    with database.connect(url) as conn:
        row = conn.execute(
            "UPDATE run SET lease_until=now() + (%s * interval '1 second'), updated_at=now() "
            "WHERE run_id=%s AND worker_id=%s AND status='running' AND lease_until > now() "
            "AND NOT cancel_requested AND deadline_at > now() "
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
            "UPDATE run SET event_seq=event_seq+1, processing_stage=CASE WHEN %s='node_started'"
            " THEN COALESCE(%s,processing_stage) ELSE processing_stage END, updated_at=now() "
            "WHERE run_id=%s AND worker_id=%s AND status='running' AND lease_until > now() "
            "AND NOT cancel_requested AND deadline_at > now() "
            "RETURNING event_seq",
            (event_type, node_id, run_id, worker_id),
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


def _publishes_anchored_candidates(pending: object, result: dict) -> bool:
    return bool(
        isinstance(pending, dict)
        and (isinstance(pending.get("search_anchor"), dict) or isinstance(pending.get("start"), dict))
        and (pending.get("activities") or pending.get("subcategories"))
        and result.get("candidates")
    )


def _maintenance_snapshot_is_current(result: dict) -> bool:
    """Recheck graph and active events just before publishing an updated trip."""
    from agents import route_state, tools

    state = result.get("route_state")
    pending = result.get("place_request")
    maintenance = result.get("result_maintenance")
    record = maintenance.get("dependencies") if isinstance(maintenance, dict) else None
    if _publishes_anchored_candidates(pending, result):
        record = pending.get("result_dependencies")
    snapshot = record.get("candidate_snapshot") if isinstance(record, dict) else None
    try:
        data_version = route_state.current_data_version()
        event_version = route_state.current_road_condition_version(tools.list_conditions(strict=True))
    except Exception:
        return False
    if not isinstance(data_version, str) or data_version in {"", "unavailable"}:
        return False
    if isinstance(state, dict):
        if state.get("data_version") != data_version or state.get("road_condition_version") != event_version:
            return False
    elif not isinstance(snapshot, dict):
        return False
    if isinstance(record, dict):
        if (not isinstance(snapshot, dict) or snapshot.get("data_version") != data_version
                or snapshot.get("road_condition_version") != event_version):
            return False
    return True


def _requires_environment_snapshot(status: str, result: dict | None) -> bool:
    if status not in {"completed", "partial", "needs_input"} or not isinstance(result, dict):
        return False
    pending_result = result.get("place_request") if isinstance(result, dict) else None
    publishes_candidates = _publishes_anchored_candidates(pending_result, result)
    publishes_pending_route = (isinstance(pending_result, dict) and pending_result.get("completed_legs")
                               and isinstance(result.get("route_state"), dict))
    publishes_route_result = (isinstance(result, dict) and isinstance(result.get("route_state"), dict)
                              and result.get("response_kind") in {"route", "no_route"})
    return bool(isinstance(result.get("result_maintenance"), dict) or publishes_candidates
                or publishes_pending_route or publishes_route_result)


def finish_run(
    url: str | None, run_id: str, worker_id: str, *, status: str,
    result: dict | None = None, error: dict | None = None, return_status: bool = False,
) -> bool | str:
    if _requires_environment_snapshot(status, result):
        from spatial.road_conditions import condition_publication_lock

        # Road-event writers take the same lock. Hold it from the snapshot
        # check through database commit, so an accepted event cannot land in
        # between. This cannot freeze event expiry or out-of-band file writes.
        with condition_publication_lock():
            return _finish_run(url, run_id, worker_id, status=status, result=result,
                               error=error, return_status=return_status)
    return _finish_run(url, run_id, worker_id, status=status, result=result,
                       error=error, return_status=return_status)


def _finish_run(
    url: str | None, run_id: str, worker_id: str, *, status: str,
    result: dict | None = None, error: dict | None = None, return_status: bool = False,
) -> bool | str:
    from agents.state_models import FieldProvenance, TaskState, utc_now

    if status not in {"completed", "partial", "failed", "needs_input", "cancelled", "superseded"}:
        raise ValueError("invalid terminal run status")
    if _requires_environment_snapshot(status, result) and not _maintenance_snapshot_is_current(result):
        pending = deepcopy(result.get("completed_place_request") or result.get("place_request"))
        message = "路网、地点或路况在计算期间发生变化，旧路线未交付。请重新查询并选择下一站。"
        if isinstance(pending, dict):
            pending.update({"candidate_poi_ids": [], "candidate_places": [],
                            "candidate_pool": [], "selected_poi_id": None})
            if pending.get("pending_slot") in ("selection", "follow_up", None):
                pending["pending_slot"] = "follow_up"
            elif pending.get("pending_slot") == "destination":
                message = "路网或路况在计算期间发生变化，旧路线未交付。请补充下一站的具体地点，我会重新规划。"
            status = "needs_input"
            result = {"task_type": "unknown", "response_kind": "clarify", "message": message,
                      "clarify": {"question": message, "options": ["重新查询"]},
                      "place_request": pending, "needs_input": True,
                      "execution_status": "needs_input"}
        else:
            status = "failed"
            result = {"task_type": "chat", "response_kind": "chat", "reply": message,
                      "message": message, "execution_status": "failed"}
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
        unbound = conn.execute(
            "SELECT r.*,c.owner_id FROM run r JOIN conversation c USING (conversation_id)"
            " WHERE r.run_id=%s AND r.worker_id=%s AND r.model_led AND r.task_id IS NULL FOR UPDATE OF r",
            (run_id, worker_id),
        ).fetchone()
        if unbound:
            if (unbound["status"] != "running" or unbound["cancel_requested"] or not unbound["lease_until"]
                    or unbound["lease_until"] <= utc_now() or unbound["deadline_at"] <= utc_now()):
                return False
            conn.execute("SELECT 1 FROM conversation WHERE conversation_id=%s FOR UPDATE", (unbound["conversation_id"],))
            newer = conn.execute("SELECT 1 FROM conversation_message WHERE conversation_id=%s"
                                 " AND role='user' AND seq>%s LIMIT 1",
                                 (unbound["conversation_id"], unbound["input_message_seq"])).fetchone()
            actual_status = "superseded" if newer else status
            if actual_status != "superseded":
                display = next((result.get(k).strip() for k in ("assistant_text", "explanation", "reply", "message")
                                if isinstance(result, dict) and isinstance(result.get(k), str) and result[k].strip()), "")
                if display:
                    seq = conn.execute("UPDATE conversation SET revision=revision+1,updated_at=now()"
                                       " WHERE conversation_id=%s RETURNING revision", (unbound["conversation_id"],)).fetchone()["revision"]
                    conn.execute("INSERT INTO conversation_message(conversation_id,seq,message_id,role,content,metadata)"
                                 " VALUES (%s,%s,%s,'assistant',%s,%s)",
                                 (unbound["conversation_id"], seq, str(uuid.uuid4()), display,
                                  Jsonb({"run_id": run_id, "status": actual_status,
                                         "source_message_seq": unbound["input_message_seq"]})))
            conn.execute("UPDATE run SET status=%s,result=%s,error=%s,lease_until=NULL,processing_stage=%s,"
                         "event_seq=event_seq+1,updated_at=now() WHERE run_id=%s",
                         (actual_status, Jsonb(result) if result is not None and actual_status != "superseded" else None,
                          Jsonb(error) if error is not None else None, actual_status, run_id))
            conn.execute("INSERT INTO run_event(run_id,seq,event_type,payload) VALUES (%s,%s,'run_completed',%s)",
                         (run_id, unbound["event_seq"] + 1, Jsonb({"status": actual_status})))
            return actual_status if return_status else True
        run = conn.execute(
            "SELECT r.task_id, r.task_revision, r.event_seq, r.cancel_requested,"
            " r.status AS run_status, r.lease_until, r.deadline_at, r.model_led, r.input_message_seq,"
            " t.revision AS current_revision, t.state, t.conversation_id, c.owner_id,"
            " (SELECT m.seq FROM conversation_message m"
            "  WHERE m.task_id=r.task_id AND m.role='user'"
            "    AND m.metadata->>'request_id'=r.idempotency_key"
            "  ORDER BY m.seq DESC LIMIT 1) AS source_message_seq,"
            " (SELECT m.message_id FROM conversation_message m"
            "  WHERE m.conversation_id=r.conversation_id AND m.seq=r.input_message_seq) AS source_message_id"
            " FROM run r JOIN task t USING (task_id)"
            " JOIN conversation c ON c.conversation_id=t.conversation_id"
            " WHERE r.run_id=%s AND r.worker_id=%s FOR UPDATE OF r, t",
            (run_id, worker_id),
        ).fetchone()
        if (not run or run["run_status"] != "running" or run["cancel_requested"]
                or not run["lease_until"] or run["lease_until"] <= utc_now()
                or run["deadline_at"] <= utc_now()):
            return False
        actual_status = status
        if run["model_led"]:
            conn.execute("SELECT 1 FROM conversation WHERE conversation_id=%s FOR UPDATE", (run["conversation_id"],))
            newer = conn.execute("SELECT 1 FROM conversation_message WHERE conversation_id=%s"
                                 " AND role='user' AND seq>%s LIMIT 1",
                                 (run["conversation_id"], run["input_message_seq"])).fetchone()
            if newer:
                actual_status = "superseded"
        if run["current_revision"] != run["task_revision"]:
            actual_status = "superseded"
        if actual_status != "superseded" and actual_status != "cancelled":
            current = TaskState.model_validate(run["state"])
            document = current.model_dump(mode="python")
            document.update({"status": actual_status, "revision": current.revision + 1,
                             "updated_at": utc_now()})
            if run["model_led"] and run["source_message_id"] and isinstance(result, dict):
                for field in result.get("updated_task_fields") or []:
                    if isinstance(field, str) and len(field) <= 150:
                        document["field_provenance"][field] = FieldProvenance(
                            source="inference", scope="task", source_message_id=str(run["source_message_id"]),
                            updated_at=utc_now()).model_dump(mode="python")
            if actual_status == "needs_input" and isinstance(result, dict):
                clarify = result.get("clarify") or {}
                question = clarify.get("question") if isinstance(clarify, dict) else None
                if not question and result.get("response_kind") == "candidates":
                    question = "请选择一个地点继续规划路线。"
                document["pending_questions"] = [question] if question else []
            if isinstance(result, dict) and isinstance(result.get("place_request"), dict):
                document["place_request"] = result["place_request"]
            elif run["model_led"] and isinstance(result, dict) and isinstance(result.get("completed_place_request"), dict):
                document["place_request"] = result["completed_place_request"]
            elif actual_status in {"completed", "partial", "failed"} and not (run["model_led"] and actual_status == "failed"):
                document["place_request"] = None
            requirement_results = (result.get("requirement_results") or {}) if isinstance(result, dict) else {}
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
                for key in ("assistant_text", "explanation", "reply", "message"):
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
                         **({"clarify": result["clarify"]} if isinstance(result, dict) and isinstance(result.get("clarify"), dict) else {}),
                         **({"delivered_candidates": [
                             {key: item[key] for key in ("poi_id", "name", "subcategory", "source", "distance_m") if key in item}
                             for item in result["candidates"] if isinstance(item, dict)]}
                            if isinstance(result, dict) and isinstance(result.get("candidates"), list) else {}),
                         "task_revision": serialized["revision"],
                         **({"route_state": result["route_state"]}
                            if isinstance(result, dict) and result.get("route_state") else {}),
                     })),
                )
        elif (actual_status == "superseded" and run["model_led"]
              and run["current_revision"] == run["task_revision"]):
            current = TaskState.model_validate(run["state"])
            serialized = current.model_copy(update={"status": "superseded", "revision": current.revision + 1,
                                                     "updated_at": utc_now()}).model_dump(mode="json")
            conn.execute("UPDATE task SET revision=%s,status='superseded',state=%s,updated_at=now() WHERE task_id=%s",
                         (serialized["revision"], Jsonb(serialized), run["task_id"]))
            conn.execute("INSERT INTO task_history(task_id,revision,state) VALUES (%s,%s,%s)",
                         (run["task_id"], serialized["revision"], Jsonb(serialized)))
        conn.execute(
            "UPDATE run SET status=%s, result=%s, error=%s, lease_until=NULL, processing_stage=%s, updated_at=now() "
            "WHERE run_id=%s",
            (actual_status, Jsonb(result) if result is not None and actual_status != "superseded" else None,
             Jsonb(error) if error is not None else None, actual_status, run_id),
        )
        event_seq = run["event_seq"] + 1
        conn.execute("UPDATE run SET event_seq=%s WHERE run_id=%s", (event_seq, run_id))
        conn.execute(
            "INSERT INTO run_event(run_id, seq, event_type, payload) VALUES (%s, %s, %s, %s)",
            (run_id, event_seq, "run_completed", Jsonb({"status": actual_status})),
        )
    return actual_status if return_status else True


def cancel_run(url: str | None, owner_id: str, run_id: str) -> dict | None:
    from agents.state_models import TaskState, utc_now

    with database.connect(url) as conn:
        row = conn.execute(
            "SELECT r.run_id,r.task_id,r.task_revision AS run_task_revision,r.status,r.cancel_requested,r.conversation_id,"
            " r.model_led,r.input_message_seq,r.event_seq,"
            " t.revision AS task_revision,t.state FROM run r JOIN conversation c USING (conversation_id)"
            " LEFT JOIN task t ON t.task_id=r.task_id"
            " WHERE r.run_id=%s AND c.owner_id=%s FOR UPDATE OF r",
            (run_id, owner_id),
        ).fetchone()
        if not row:
            return None
        if row["status"] not in {"completed", "partial", "failed", "needs_input", "cancelled", "superseded"}:
            if row["model_led"]:
                text = "这项任务已停止，你之前确认的条件仍保留。需要继续时告诉我下一步。"
                conn.execute("SELECT 1 FROM conversation WHERE conversation_id=%s FOR UPDATE", (row["conversation_id"],))
                seq = conn.execute("UPDATE conversation SET revision=revision+1,updated_at=now()"
                                   " WHERE conversation_id=%s RETURNING revision", (row["conversation_id"],)).fetchone()["revision"]
                conn.execute("INSERT INTO conversation_message(conversation_id,seq,message_id,task_id,role,content,metadata)"
                             " VALUES (%s,%s,%s,%s,'assistant',%s,%s)",
                             (row["conversation_id"], seq, str(uuid.uuid4()), row["task_id"], text,
                              Jsonb({"run_id": str(row["run_id"]), "status": "cancelled",
                                     "source_message_seq": row["input_message_seq"], "ui_action": {"type": "cancel"}})))
                conn.execute("UPDATE run SET result=%s,event_seq=event_seq+1 WHERE run_id=%s",
                             (Jsonb({"task_type": "chat", "response_kind": "chat", "message": text,
                                     "assistant_text": text}), run_id))
                conn.execute("INSERT INTO run_event(run_id,seq,event_type,payload) VALUES (%s,%s,'run_completed',%s)",
                             (run_id, row["event_seq"] + 1, Jsonb({"status": "cancelled"})))
            conn.execute("UPDATE run SET status='cancelled',cancel_requested=true,lease_until=NULL,"
                         "processing_stage='cancelled',updated_at=now() WHERE run_id=%s", (run_id,))
            row["status"], row["cancel_requested"] = "cancelled", True
        if (row["status"] == "cancelled" and row["state"] and row["task_revision"] == row["run_task_revision"]
                and row["state"].get("status") not in {"cancelled", "completed", "partial", "failed"}):
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
        "run_id": str(row["run_id"]), "task_id": str(row["task_id"]) if row["task_id"] else None,
    }


def get_run(url: str | None, owner_id: str, run_id: str, after_seq: int = 0) -> dict | None:
    with database.connect(url) as conn:
        query = (
            "SELECT r.run_id, r.task_id, r.task_revision, t.revision AS current_task_revision,"
            " r.status, r.deadline_at, r.lease_until, r.processing_stage,"
            " r.attempt, r.cancel_requested, r.result, r.error FROM run r"
            " JOIN conversation c USING (conversation_id) LEFT JOIN task t ON t.task_id=r.task_id"
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
    return {**run, "run_id": str(run["run_id"]), "task_id": str(run["task_id"]) if run["task_id"] else None,
            "events": events}
