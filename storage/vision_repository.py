"""Durable, manager-scoped queue for isolated aerial media analysis."""
from __future__ import annotations

import uuid
from typing import Any

from psycopg.types.json import Jsonb

from storage import database


def create_job(url: str | None, *, created_by: str, original_name: str,
               media_kind: str, media_path: str, sha256: str,
               anchor_gcj: dict[str, float] | None, camera_stabilized: bool = False,
               captured_at=None, capture_time_source: str = "unknown",
               capture_timezone: str | None = None,
               observation_regions: list[dict] | None = None,
               observation_scene_id: str | None = None) -> dict:
    job_id = str(uuid.uuid4())
    with database.connect(url) as conn:
        row = conn.execute(
            "INSERT INTO manager_vision_job(job_id, created_by, original_name, media_kind,"
            " media_path, sha256, anchor_gcj, camera_stabilized, captured_at, capture_time_source,"
            " capture_timezone, observation_regions,"
            " observation_scene_id, status)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'queued')"
            " RETURNING job_id, status, created_at, capture_time_source, capture_timezone",
            (job_id, created_by, original_name, media_kind, media_path, sha256,
             Jsonb(anchor_gcj) if anchor_gcj is not None else None,
             bool(camera_stabilized), captured_at,
             capture_time_source, capture_timezone,
             Jsonb(observation_regions or []), observation_scene_id),
        ).fetchone()
    return {**row, "job_id": str(row["job_id"])}


def claim_next_job(url: str | None, worker_id: str, lease_seconds: int = 90) -> dict | None:
    with database.connect(url) as conn:
        conn.execute(
            "UPDATE manager_vision_job SET status='failed', lease_until=NULL,"
            " error=%s, updated_at=now() WHERE deleted_at IS NULL AND status='running'"
            " AND lease_until<now() AND attempts>=3",
            (Jsonb({"code": "worker_lease_exhausted", "message": "影像任务多次失去工作租约，请重新提交。"}),),
        )
        row = conn.execute(
            "WITH candidate AS (SELECT job_id FROM manager_vision_job"
            " WHERE deleted_at IS NULL AND attempts<3"
            " AND (status='queued' OR (status='running' AND lease_until < now()))"
            " ORDER BY created_at, job_id FOR UPDATE SKIP LOCKED LIMIT 1)"
            " UPDATE manager_vision_job j SET status='running', worker_id=%s,"
            " lease_until=now()+(%s * interval '1 second'), attempts=attempts+1, updated_at=now()"
            " FROM candidate c WHERE j.job_id=c.job_id"
            " RETURNING j.job_id, j.media_kind, j.media_path, j.anchor_gcj, j.camera_stabilized,"
            " j.observation_regions, j.captured_at, j.progress, j.checkpoint, j.attempts",
            (worker_id, lease_seconds),
        ).fetchone()
    return {**row, "job_id": str(row["job_id"])} if row else None


def heartbeat_job(url: str | None, job_id: str, worker_id: str, lease_seconds: int = 90) -> bool:
    with database.connect(url) as conn:
        row = conn.execute(
            "UPDATE manager_vision_job SET lease_until=now()+(%s * interval '1 second'), updated_at=now()"
            " WHERE job_id=%s AND worker_id=%s AND status='running' AND deleted_at IS NULL"
            " AND lease_until>now()"
            " RETURNING job_id", (lease_seconds, job_id, worker_id),
        ).fetchone()
    return row is not None


def get_job_cancel_state(url: str | None, job_id: str, worker_id: str) -> dict | None:
    """Read cancellation state without writing a progress checkpoint per frame."""
    with database.connect(url) as conn:
        row = conn.execute(
            "SELECT cancel_requested FROM manager_vision_job"
            " WHERE job_id=%s AND worker_id=%s AND status='running'"
            " AND deleted_at IS NULL AND lease_until>now()",
            (job_id, worker_id),
        ).fetchone()
    return dict(row) if row else None


def update_job_progress(url: str | None, job_id: str, worker_id: str, *,
                        progress: dict, checkpoint: dict | None = None) -> dict | None:
    """Persist live progress, optionally checkpointing a completed video segment."""
    with database.connect(url) as conn:
        row = conn.execute(
            "UPDATE manager_vision_job SET progress=%s, checkpoint=COALESCE(%s, checkpoint), updated_at=now()"
            " WHERE job_id=%s AND worker_id=%s AND status='running' AND lease_until>now()"
            " RETURNING cancel_requested",
            (Jsonb(progress), Jsonb(checkpoint) if checkpoint is not None else None,
             job_id, worker_id),
        ).fetchone()
    return dict(row) if row else None


def request_cancel_job(url: str | None, job_id: str) -> dict | None:
    """Cancel queued work immediately or ask a running worker to stop between sampled frames."""
    with database.connect(url) as conn:
        row = conn.execute(
            "UPDATE manager_vision_job SET"
            " status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,"
            " cancel_requested=CASE WHEN status='running' THEN true ELSE cancel_requested END,"
            " progress=CASE WHEN status='queued' THEN %s ELSE progress END,"
            " updated_at=now() WHERE job_id=%s AND status IN ('queued','running')"
            " RETURNING job_id, status, cancel_requested, progress",
            (Jsonb({"phase": "cancelled", "percent": 0}), job_id),
        ).fetchone()
    return {**row, "job_id": str(row["job_id"])} if row else None


def retry_job(url: str | None, job_id: str) -> dict | None:
    """Requeue failed/cancelled work; a failed video keeps its last complete segment checkpoint."""
    with database.connect(url) as conn:
        row = conn.execute(
            "UPDATE manager_vision_job SET status='queued', worker_id=NULL, lease_until=NULL, attempts=0,"
            " cancel_requested=false, error=NULL, review_status=NULL, review_candidate_index=NULL,"
            " review_note=NULL, reviewed_by=NULL, reviewed_at=NULL,"
            " progress=%s, updated_at=now()"
            " WHERE job_id=%s AND status IN ('failed','cancelled') AND review_status IS NULL"
            " RETURNING job_id, status, progress",
            (Jsonb({"phase": "queued", "percent": 0}), job_id),
        ).fetchone()
    return {**row, "job_id": str(row["job_id"])} if row else None


def finish_job(url: str | None, job_id: str, worker_id: str, *, status: str,
               result: dict | None = None, error: dict | None = None) -> str | None:
    if status not in {"needs_review", "completed", "failed", "cancelled"}:
        raise ValueError("invalid vision job terminal status")
    with database.connect(url) as conn:
        row = conn.execute(
            "UPDATE manager_vision_job SET"
            " status=CASE WHEN cancel_requested THEN 'cancelled' ELSE %s END,"
            " result=CASE WHEN cancel_requested THEN NULL ELSE %s END,"
            " error=CASE WHEN cancel_requested THEN %s ELSE %s END,"
            " progress=CASE WHEN cancel_requested OR %s='cancelled' THEN %s"
            " WHEN %s IN ('needs_review','completed') THEN %s ELSE progress END,"
            " lease_until=NULL, updated_at=now()"
            " WHERE job_id=%s AND worker_id=%s AND status='running'"
            " AND deleted_at IS NULL AND lease_until>now() RETURNING status",
            (status, Jsonb(result) if result is not None else None,
             Jsonb({"code": "vision_job_cancelled", "message": "管理员已取消影像分析；已完成分段可用于重试。"}),
             Jsonb(error) if error is not None else None,
             status, Jsonb({"phase": "cancelled", "percent": 0}),
             status, Jsonb({"phase": "complete", "percent": 100}), job_id, worker_id),
        ).fetchone()
    return str(row["status"]) if row else None


def get_job(url: str | None, job_id: str) -> dict | None:
    with database.connect(url) as conn:
        row = conn.execute(
            "SELECT job_id, created_by, original_name, media_kind, media_path, sha256, anchor_gcj, camera_stabilized,"
            " captured_at, capture_time_source, capture_timezone, observation_regions, observation_scene_id, progress, checkpoint, candidate_reviews, cancel_requested, status, attempts, result, error,"
            " review_status, review_candidate_index, review_note, reviewed_by, reviewed_at,"
            " created_at, updated_at FROM manager_vision_job WHERE job_id=%s AND deleted_at IS NULL", (job_id,),
        ).fetchone()
    return {**row, "job_id": str(row["job_id"])} if row else None


def list_jobs(url: str | None, limit: int = 50) -> list[dict[str, Any]]:
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    with database.connect(url) as conn:
        rows = conn.execute(
            "SELECT job_id, created_by, original_name, media_kind, anchor_gcj, camera_stabilized,"
            " captured_at, capture_time_source, capture_timezone, observation_regions, observation_scene_id, progress, candidate_reviews, cancel_requested, status, attempts,"
            " result, error, review_status, review_candidate_index, review_note, reviewed_by, reviewed_at,"
            " created_at, updated_at"
            " FROM manager_vision_job WHERE deleted_at IS NULL"
            " ORDER BY created_at DESC LIMIT %s", (limit,),
        ).fetchall()
    return [{**row, "job_id": str(row["job_id"])} for row in rows]


def review_candidate(url: str | None, job_id: str, *, candidate_index: int,
                     review_status: str, review_note: str, reviewed_by: str) -> dict | None:
    """Record a one-time, independently auditable decision for one candidate."""
    if review_status not in {"confirmed", "dismissed"}:
        raise ValueError("review_status must be confirmed or dismissed")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or candidate_index < 0:
        raise ValueError("candidate_index must be a non-negative integer")
    review = Jsonb({
        "status": review_status,
        "note": review_note[:1000],
        "reviewed_by": reviewed_by,
    })
    index_text = str(candidate_index)
    with database.connect(url) as conn:
        row = conn.execute(
            "UPDATE manager_vision_job SET candidate_reviews = candidate_reviews || "
            "jsonb_build_object(%s::text, %s::jsonb), updated_at=now() "
            "WHERE job_id=%s AND status='needs_review' AND review_status IS NULL "
            "AND jsonb_typeof(result->'candidates')='array' "
            "AND jsonb_array_length(result->'candidates') > %s "
            "AND NOT (candidate_reviews ? %s) "
            "RETURNING job_id, status, candidate_reviews",
            (index_text, review, job_id, candidate_index, index_text),
        ).fetchone()
    return {**row, "job_id": str(row["job_id"])} if row else None


def review_job(url: str | None, job_id: str, *, review_status: str,
               review_note: str, reviewed_by: str,
               candidate_index: int | None = None) -> dict | None:
    if review_status not in {"confirmed", "dismissed"}:
        raise ValueError("review_status must be confirmed or dismissed")
    if review_status == "confirmed" and (isinstance(candidate_index, bool)
                                          or not isinstance(candidate_index, int)
                                          or candidate_index < 0):
        raise ValueError("confirmed review requires a candidate index")
    if review_status == "dismissed":
        candidate_index = None
    with database.connect(url) as conn:
        row = conn.execute(
            "UPDATE manager_vision_job SET review_status=%s, review_candidate_index=%s,"
            " review_note=%s, reviewed_by=%s,"
            " reviewed_at=now(), updated_at=now() WHERE job_id=%s AND deleted_at IS NULL"
            " AND status='needs_review'"
            " AND review_status IS NULL"
            " AND (%s = 'dismissed' OR CASE WHEN jsonb_typeof(result->'candidates') = 'array'"
            " THEN jsonb_array_length(result->'candidates') > %s ELSE false END)"
            " RETURNING job_id, status, review_status, review_candidate_index, review_note, reviewed_by, reviewed_at",
            (review_status, candidate_index, review_note[:1000], reviewed_by, job_id,
             review_status, candidate_index if candidate_index is not None else -1),
        ).fetchone()
    return {**row, "job_id": str(row["job_id"])} if row else None


def begin_delete_jobs(url: str | None, job_id: str) -> list[dict[str, Any]]:
    """Tombstone the requested image and every visible exact-content duplicate.

    Tombstones are monotonic: a failed file removal must never make a deleted
    image visible again. The returned rows include prior tombstones for this
    content so a repeated delete can retry their physical cleanup.
    """
    with database.connect(url) as conn:
        target = conn.execute(
            "SELECT sha256 FROM manager_vision_job WHERE job_id=%s", (job_id,),
        ).fetchone()
        if not target:
            return []
        digest = target.get("sha256")
        identity_clause = "sha256=%s" if digest else "job_id=%s"
        identity = digest if digest else job_id
        locked_rows = conn.execute(
            "SELECT job_id FROM manager_vision_job WHERE " + identity_clause
            + " ORDER BY job_id FOR UPDATE", (identity,),
        ).fetchall()
        if not locked_rows:
            return []
        conn.execute(
            "UPDATE manager_vision_job SET deleted_at=COALESCE(deleted_at, now()),"
            " worker_id=NULL, lease_until=NULL, updated_at=now() WHERE " + identity_clause
            + " AND deleted_at IS NULL", (identity,),
        )
        rows = conn.execute(
            "SELECT job_id, media_path, sha256 FROM manager_vision_job WHERE " + identity_clause
            + " AND deleted_at IS NOT NULL ORDER BY updated_at, job_id", (identity,),
        ).fetchall()
    return [{**row, "job_id": str(row["job_id"])} for row in rows]


def list_pending_deletions(url: str | None, limit: int = 200) -> list[dict[str, Any]]:
    """Return hidden records whose media or database row still needs cleanup."""
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    with database.connect(url) as conn:
        rows = conn.execute(
            "SELECT job_id, media_path, sha256 FROM manager_vision_job"
            " WHERE deleted_at IS NOT NULL ORDER BY updated_at, job_id LIMIT %s", (limit,),
        ).fetchall()
    return [{**row, "job_id": str(row["job_id"])} for row in rows]


def mark_delete_retry_attempts(url: str | None, job_ids: list[str]) -> int:
    """Move unsuccessful cleanup attempts behind older untried tombstones."""
    if not job_ids:
        return 0
    identifiers = [uuid.UUID(job_id) for job_id in job_ids]
    with database.connect(url) as conn:
        rows = conn.execute(
            "UPDATE manager_vision_job SET updated_at=now()"
            " WHERE job_id=ANY(%s) AND deleted_at IS NOT NULL RETURNING job_id",
            (identifiers,),
        ).fetchall()
    return len(rows)


def finalize_delete_jobs(url: str | None, job_ids: list[str]) -> int:
    """Remove tombstoned rows only after their media files are absent."""
    if not job_ids:
        return 0
    identifiers = [uuid.UUID(job_id) for job_id in job_ids]
    with database.connect(url) as conn:
        rows = conn.execute(
            "DELETE FROM manager_vision_job WHERE job_id=ANY(%s) AND deleted_at IS NOT NULL"
            " RETURNING job_id", (identifiers,),
        ).fetchall()
    return len(rows)


def heartbeat_worker(url: str | None, worker_id: str, *, ready: bool,
                     status_detail: str | None = None) -> None:
    """Publish vision-process health; the web app never infers readiness from files alone."""
    with database.connect(url) as conn:
        conn.execute(
            "INSERT INTO manager_vision_worker(worker_id, ready, status_detail, heartbeat_at)"
            " VALUES (%s,%s,%s,now()) ON CONFLICT(worker_id) DO UPDATE SET"
            " ready=EXCLUDED.ready, status_detail=EXCLUDED.status_detail, heartbeat_at=now()",
            (worker_id, bool(ready), (status_detail or "")[:500]),
        )


def has_ready_worker(url: str | None, max_age_seconds: int = 30,
                     worker_id: str | None = None) -> bool:
    """Return true only when a healthy worker has renewed its heartbeat recently."""
    worker_filter = " AND worker_id=%s" if worker_id else ""
    parameters = (max(1, int(max_age_seconds)), worker_id) if worker_id else (max(1, int(max_age_seconds)),)
    with database.connect(url) as conn:
        row = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM manager_vision_worker WHERE ready=true"
            " AND heartbeat_at > now() - (%s * interval '1 second')" + worker_filter + ") AS ready",
            parameters,
        ).fetchone()
    return bool(row and row["ready"])


def ready_worker_status(url: str | None, max_age_seconds: int = 30) -> dict[str, Any] | None:
    """Return the newest fresh worker health details for manager capability reporting."""
    with database.connect(url) as conn:
        return conn.execute(
            "SELECT worker_id, status_detail, heartbeat_at FROM manager_vision_worker"
            " WHERE ready=true AND heartbeat_at > now() - (%s * interval '1 second')"
            " ORDER BY heartbeat_at DESC LIMIT 1",
            (max(1, int(max_age_seconds)),),
        ).fetchone()
