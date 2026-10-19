"""Persist server-verified route evidence; this boundary does not verify routes.

Only trusted server code may call create_verified_proof after verification. The
owner_id must be derived by the trusted API from a validated conversation
credential, never from route_state or an arbitrary caller UUID. No proof may be
issued or loaded without this authenticated anonymous identity. This module
checks owner existence and task ownership; cookie verification belongs to API.
Consumers supply bindings only and receive the stored edges, never caller edges.
Missing configuration and database errors propagate; no local fallback exists.
"""
from __future__ import annotations

import math
import re
import uuid

from psycopg.types.json import Jsonb

from storage import database


def _uuid(value, field):
    if not isinstance(value, str):
        raise ValueError(f'{field} must be a UUID string')
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        raise ValueError(f'{field} must be a UUID string') from None
    if str(parsed) != value:
        raise ValueError(f'{field} must be a canonical UUID string')
    return value


def _bindings(owner_id, semantic_digest, graph_version, event_version, task_id, task_revision):
    _uuid(owner_id, 'owner_id')
    if not isinstance(semantic_digest, str) or not re.fullmatch('[0-9a-f]{64}', semantic_digest):
        raise ValueError('semantic_digest must be a lowercase SHA-256 hex digest')
    for field, value in (('graph_version', graph_version), ('event_version', event_version)):
        if not isinstance(value, str) or not value.strip() or '\x00' in value:
            raise ValueError(f'{field} must be a nonempty string')
    if (task_id is None) != (task_revision is None):
        raise ValueError('task_id and task_revision must be supplied together')
    if task_id is not None:
        _uuid(task_id, 'task_id')
        if type(task_revision) is not int or not 0 <= task_revision <= 2**63 - 1:
            raise ValueError('task_revision must be a nonnegative bigint')


def _id(value):
    # Road graph IDs are strings or integers, preserving types through JSONB.
    # Limit integers to the interoperable JSON range used by browser clients.
    if type(value) is int and abs(value) <= 2**53 - 1:
        return value
    if isinstance(value, str) and value and '\x00' not in value:
        return value
    raise ValueError('edge IDs must be nonempty strings or JSON-safe integers')


def _edges(sequence):
    result = []
    for edge in sequence:
        if not isinstance(edge, (list, tuple)) or len(edge) != 3:
            raise ValueError('each edge must contain (u, v, key)')
        result.append([_id(value) for value in edge])
    return result


def _event_snapshot(value):
    if not isinstance(value, list):
        raise ValueError('event_snapshot must be a list')
    rows = []
    for item in value:
        if (not isinstance(item, dict) or set(item) != {'event_id', 'fingerprint'}
                or not isinstance(item['event_id'], str) or not item['event_id']
                or not re.fullmatch('[0-9a-f]{64}', str(item['fingerprint']))):
            raise ValueError('event_snapshot records are invalid')
        rows.append({'event_id': item['event_id'], 'fingerprint': item['fingerprint']})
    return rows


def create_verified_proof(
    url: str | None, *, owner_id: str, semantic_digest: str,
    graph_version: str, event_version: str, selected_edges, closed_edge_ids,
    ttl_seconds: float, task_id: str | None = None, task_revision: int | None = None,
    event_snapshot=None,
) -> str:
    """Store trusted evidence and return an opaque, server-generated proof UUID.

    A task must belong to this owner and still have the supplied revision.
    Raises ValueError for invalid inputs or stale/foreign task bindings.
    """
    _bindings(owner_id, semantic_digest, graph_version, event_version, task_id, task_revision)
    if not isinstance(selected_edges, (list, tuple)) or not selected_edges:
        raise ValueError('selected_edges must be a nonempty edge sequence')
    edges = _edges(selected_edges)
    if not isinstance(closed_edge_ids, (list, tuple)):
        raise ValueError('closed_edge_ids must be a sequence')
    closed = _edges(closed_edge_ids)
    snapshot = _event_snapshot(event_snapshot if event_snapshot is not None else [])
    if (type(ttl_seconds) not in (int, float) or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0):
        raise ValueError('ttl_seconds must be finite and positive')
    proof_id = str(uuid.uuid4())
    with database.connect(url) as conn:
        owner = conn.execute(
            'SELECT 1 FROM conversation WHERE owner_id=%s FOR SHARE', (owner_id,),
        ).fetchone()
        if not owner:
            raise ValueError('owner identity is not registered')
        if task_id is not None:
            task = conn.execute(
                'SELECT t.revision FROM task t JOIN conversation c USING(conversation_id) '
                'WHERE t.task_id=%s AND c.owner_id=%s FOR SHARE OF t, c',
                (task_id, owner_id),
            ).fetchone()
            if not task or task['revision'] != task_revision:
                raise ValueError('task binding is stale or does not belong to owner')
        conn.execute(
            'INSERT INTO route_proof (proof_id, owner_id, task_id, task_revision, '
            'semantic_digest, graph_version, event_version, selected_edges, closed_edge_ids, '
            'event_snapshot, verified, expires_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,true, '
            "clock_timestamp() + %s * interval '1 second')",
            (proof_id, owner_id, task_id, task_revision, semantic_digest, graph_version,
             event_version, Jsonb(edges), Jsonb(closed), Jsonb(snapshot), ttl_seconds),
        )
    return proof_id


def load_verified_proof(
    url: str | None, *, proof_id: str, owner_id: str, semantic_digest: str,
    graph_version: str, event_version: str, task_id: str | None = None,
    task_revision: int | None = None,
) -> dict | None:
    """Return stored evidence only when every binding is current; otherwise None."""
    _uuid(proof_id, 'proof_id')
    _bindings(owner_id, semantic_digest, graph_version, event_version, task_id, task_revision)
    with database.connect(url) as conn:
        row = conn.execute(
            'SELECT p.* FROM route_proof p WHERE p.proof_id=%s AND p.owner_id=%s '
            'AND p.semantic_digest=%s AND p.graph_version=%s AND p.event_version=%s '
            'AND p.task_id IS NOT DISTINCT FROM %s::uuid '
            'AND p.task_revision IS NOT DISTINCT FROM %s::bigint '
            'AND p.verified=true AND p.invalidated_at IS NULL AND p.expires_at>clock_timestamp() '
            'AND EXISTS (SELECT 1 FROM conversation c WHERE c.owner_id=p.owner_id) '
            'AND (p.task_id IS NULL OR EXISTS (SELECT 1 FROM task t '
            'JOIN conversation c USING(conversation_id) WHERE t.task_id=p.task_id '
            'AND t.revision=p.task_revision AND c.owner_id=p.owner_id))',
            (proof_id, owner_id, semantic_digest, graph_version, event_version, task_id, task_revision),
        ).fetchone()
    if row:
        for key in ('proof_id', 'owner_id', 'task_id'):
            if row[key] is not None:
                row[key] = str(row[key])
    return row


def rebind_verified_proof_task_revision(
    conn, *, proof_id: str, owner_id: str, task_id: str,
    old_revision: int, new_revision: int,
) -> bool:
    """Advance a worker proof with the result's atomic task-state revision.

    ``finish_run`` calls this in the same transaction that advances the task.
    The old revision must still be current at the time of rebinding, and only
    the immediately following revision is permitted. A later user edit still
    makes the proof unloadable through ``load_verified_proof``.
    """
    _uuid(proof_id, 'proof_id')
    _uuid(owner_id, 'owner_id')
    _uuid(task_id, 'task_id')
    if (type(old_revision) is not int or old_revision < 0
            or type(new_revision) is not int or new_revision != old_revision + 1):
        raise ValueError('proof revision transition must advance exactly once')
    cursor = conn.execute(
        'UPDATE route_proof p SET task_revision=%s '
        'WHERE p.proof_id=%s AND p.owner_id=%s AND p.task_id=%s '
        'AND p.task_revision=%s AND p.verified=true AND p.invalidated_at IS NULL '
        'AND p.expires_at>clock_timestamp() '
        'AND EXISTS (SELECT 1 FROM task t JOIN conversation c USING(conversation_id) '
        'WHERE t.task_id=p.task_id AND t.revision=%s AND c.owner_id=p.owner_id)',
        (new_revision, proof_id, owner_id, task_id, old_revision, old_revision),
    )
    return cursor.rowcount == 1
