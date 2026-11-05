"""Cookie-authenticated durable anonymous conversations and task snapshots."""
from __future__ import annotations

import logging
import math
import re
import uuid
from datetime import datetime, timedelta, timezone

from flask import Blueprint, current_app, jsonify, make_response, request
from pydantic import ValidationError

from agents.state_models import TaskState
from agents.state_models import RouteSpec
from storage import database
from storage.task_repository import (
    RevisionConflict,
    apply_task_patch,
    create_conversation,
    create_task,
    get_conversation,
    get_task,
    get_task_history,
    list_tasks,
    list_active_runs,
    get_conversation_messages,
    create_task_run,
    RunConflict,
    TaskContinuationConflict,
)

logger = logging.getLogger(__name__)
conversations_bp = Blueprint("conversations", __name__, url_prefix="/api/conversations")
COOKIE_NAME = "whu_conversation_token"


def _database_url():
    return current_app.config.get("DATABASE_URL")


def _failure(code: str, message: str, status: int):
    return jsonify({"error": code, "message": message}), status


def _token():
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        authorization = request.headers.get("Authorization", "")
        token = authorization[7:].strip() if authorization.startswith("Bearer ") else ""
    return token


def _owner(conversation_id: str):
    token = _token()
    if not token: return None
    conversation = get_conversation(_database_url(), conversation_id, token)
    return conversation.get("owner_id") if conversation else None


@conversations_bp.errorhandler(RuntimeError)
def _storage_unavailable(error):
    logger.error("持久会话存储不可用: %s", error)
    return _failure("conversation_storage_unavailable", "会话暂时无法保存，请稍后重试。", 503)


@conversations_bp.post("")
def new_conversation():
    body = request.get_json(silent=True) or {}
    if body:
        return _failure("invalid_request", "创建会话不接受额外字段。", 422)
    url = _database_url()
    database.initialize(url)
    owner_id, token, conversation_id = create_conversation(url)
    response = make_response(jsonify({"data": {
        "conversation_id": conversation_id,
        "revision": 0,
    }}), 201)
    response.set_cookie(
        COOKIE_NAME, token, max_age=60 * 60 * 24 * 365,
        httponly=True,
        secure=bool(current_app.config.get("SESSION_COOKIE_SECURE", False) or request.is_secure),
        samesite="Strict",
        path="/",
    )
    return response


@conversations_bp.get("/<conversation_id>")
def restore_conversation(conversation_id: str):
    owner_id = _owner(conversation_id)
    if owner_id is None:
        # Same response for missing IDs and invalid credentials prevents enumeration.
        return _failure("conversation_not_found", "会话不存在或已失效。", 404)
    conversation = get_conversation(_database_url(), conversation_id, _token())
    tasks = list_tasks(_database_url(), owner_id, conversation_id)
    active_runs = list_active_runs(_database_url(), owner_id, conversation_id)
    if tasks is None or active_runs is None:
        return _failure("conversation_not_found", "会话不存在或已失效。", 404)
    messages = get_conversation_messages(_database_url(), owner_id, conversation_id)
    return jsonify({"data": {
        "conversation_id": conversation_id,
        "revision": conversation["revision"] if conversation else 0,
        "tasks": tasks,
        "messages": messages or [],
        "active_runs": active_runs,
    }})


@conversations_bp.post("/<conversation_id>/tasks")
def new_task(conversation_id: str):
    owner_id = _owner(conversation_id)
    if owner_id is None:
        return _failure("conversation_not_found", "会话不存在或已失效。", 404)
    body = request.get_json(silent=True) or {}
    try:
        task = TaskState.model_validate({
            **body,
            "task_id": "pending-server-id",
            "conversation_id": conversation_id,
        })
    except ValidationError as error:
        return _failure("invalid_task", str(error), 422)
    task_id = create_task(
        _database_url(), owner_id, conversation_id, task.model_dump(mode="json")
    )
    return jsonify({"data": {
        "task_id": task_id,
        "conversation_id": conversation_id,
        "revision": 0,
        "task_kind": task.task_kind,
        "status": task.status,
        "state": {**task.model_dump(mode="json"), "task_id": task_id},
    }}), 201


@conversations_bp.get("/<conversation_id>/tasks/<task_id>")
def restore_task(conversation_id: str, task_id: str):
    owner_id = _owner(conversation_id)
    if owner_id is None:
        return _failure("task_not_found", "任务不存在。", 404)
    task = get_task(_database_url(), owner_id, task_id)
    if task is None or task["conversation_id"] != conversation_id:
        return _failure("task_not_found", "任务不存在。", 404)
    task["history"] = get_task_history(_database_url(), owner_id, task_id)
    return jsonify({"data": task})


@conversations_bp.patch("/<conversation_id>/tasks/<task_id>")
def patch_task(conversation_id: str, task_id: str):
    owner_id = _owner(conversation_id)
    if owner_id is None:
        return _failure("task_not_found", "任务不存在。", 404)
    existing = get_task(_database_url(), owner_id, task_id)
    if existing is None or existing["conversation_id"] != conversation_id:
        return _failure("task_not_found", "任务不存在。", 404)
    patch = request.get_json(silent=True) or {}
    if patch.get("task_id") != task_id:
        return _failure("invalid_task_patch", "任务标识不匹配。", 422)
    try:
        updated = apply_task_patch(_database_url(), owner_id, patch)
    except RevisionConflict as error:
        return _failure("revision_conflict", str(error), 409)
    except (ValidationError, ValueError) as error:
        return _failure("invalid_task_patch", str(error), 422)
    if updated is None:
        return _failure("task_not_found", "任务不存在。", 404)
    return jsonify({"data": updated})


@conversations_bp.post("/<conversation_id>/messages")
def submit_message(conversation_id: str):
    owner_id = _owner(conversation_id)
    if owner_id is None:
        return _failure("conversation_not_found", "会话不存在或已失效。", 404)
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    request_id = body.get("request_id") or request.headers.get("Idempotency-Key")
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        return _failure("invalid_message", "query 必须为 1 到 500 个字符。", 422)
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 200:
        return _failure("missing_request_id", "每条消息必须带唯一 request_id。", 422)
    if set(body) - {"query", "request_id", "travel_mode", "coord_start", "coord_end", "coord_waypoints",
                    "continuation_task_id", "base_revision", "selected_poi_id"}:
        return _failure("invalid_request", "消息包含不支持的字段。", 422)

    def location_ref(value, label):
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) - {"lng", "lat", "name", "crs"}:
            raise ValueError(f"{label}坐标格式无效")
        if value.get("crs", "WGS84") != "WGS84":
            raise ValueError(f"{label}坐标必须是 WGS-84")
        lng, lat = value.get("lng"), value.get("lat")
        if (isinstance(lng, bool) or not isinstance(lng, (int, float))
                or isinstance(lat, bool) or not isinstance(lat, (int, float))
                or not math.isfinite(lng) or not math.isfinite(lat)
                or not -180 <= lng <= 180 or not -90 <= lat <= 90):
            raise ValueError(f"{label}坐标超出有效范围")
        name = value.get("name", label)
        if not isinstance(name, str) or len(name) > 200:
            raise ValueError(f"{label}名称格式无效")
        return {"name": name, "coordinates": {
            "longitude": float(lng), "latitude": float(lat), "crs": "WGS84",
        }}

    deadline = datetime.now(timezone.utc) + timedelta(seconds=75)
    try:
        mode = body.get("travel_mode")
        if mode is not None and mode not in {"walk", "bike", "drive"}:
            raise ValueError("travel_mode 只支持 walk、bike 或 drive")
        route_spec = {}
        if mode is not None:
            route_spec["travel_mode"] = mode
        if "coord_start" in body:
            start = location_ref(body.get("coord_start"), "起点")
            if start is not None: route_spec["start"] = start
        if "coord_end" in body:
            end = location_ref(body.get("coord_end"), "终点")
            if end is not None: route_spec["end"] = end
        if "coord_waypoints" in body:
            raw_waypoints = body["coord_waypoints"]
            if not isinstance(raw_waypoints, list) or len(raw_waypoints) > 10:
                raise ValueError("途经点必须是最多 10 个坐标组成的列表")
            route_spec["stops"] = [location_ref(point, f"途经点{i + 1}") for i, point in enumerate(raw_waypoints)]
            route_spec["route_kind"] = "via" if route_spec["stops"] else "direct"
        RouteSpec.model_validate({**RouteSpec().model_dump(mode="python"), **route_spec})
        continuation_task_id = body.get("continuation_task_id")
        base_revision = body.get("base_revision")
        selected_poi_id = body.get("selected_poi_id")
        if selected_poi_id is not None:
            if not isinstance(selected_poi_id, str) or not re.fullmatch(
                    r"(?:poi_\d{3,}|amap:[A-Za-z0-9_-]{3,80})", selected_poi_id):
                raise ValueError("selected_poi_id 格式无效")
            if continuation_task_id is None:
                raise ValueError("selected_poi_id 只能用于续接当前候选任务")
        if continuation_task_id is not None:
            if not isinstance(continuation_task_id, str) or not isinstance(base_revision, int) or isinstance(base_revision, bool):
                raise ValueError("continuation_task_id 与 base_revision 格式无效")
            continuation_task_id = str(uuid.UUID(continuation_task_id))
        elif base_revision is not None:
            raise ValueError("base_revision 只能与 continuation_task_id 一起使用")
        accepted = create_task_run(
            _database_url(), owner_id, conversation_id, query,
            request_id.strip(), deadline, route_spec=route_spec,
            continuation_task_id=continuation_task_id, base_revision=base_revision,
            selected_poi_id=selected_poi_id,
        )
    except TaskContinuationConflict as error:
        return _failure("task_revision_conflict", str(error), 409)
    except RunConflict as error:
        return _failure("idempotency_conflict", str(error), 409)
    except (ValidationError, ValueError, PermissionError) as error:
        return _failure("invalid_message", str(error), 422)
    accepted["conversation_id"] = conversation_id
    return jsonify({"data": accepted}), 202
