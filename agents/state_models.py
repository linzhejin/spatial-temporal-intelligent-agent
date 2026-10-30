"""Typed, serializable accepted task state; never reconstruct it from chat text.

These contracts are deliberately separate from legacy client route snapshots.
Only a repository transaction may accept a new revision as authoritative.
"""

from datetime import datetime, timezone
from math import isclose
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be an ISO-8601 datetime") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(timezone.utc)


UtcDateTime = Annotated[datetime, BeforeValidator(_utc)]
Identifier = Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]
TravelMode = Literal["walk", "bike", "drive"]
MemoryScope = Literal["task", "session", "profile"]
PatchSource = Literal["user", "button", "system", "inference"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Coordinates(Contract):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    crs: Literal["WGS84", "GCJ02"]


class LocationRef(Contract):
    """A name may remain unresolved while the agent is asking for clarification."""

    poi_id: Identifier | None = None
    name: str | None = Field(default=None, max_length=500)
    coordinates: Coordinates | None = None

    @model_validator(mode="after")
    def has_resolvable_location(self):
        if not self.poi_id and self.coordinates is None and not self.name:
            raise ValueError("location requires a POI ID, a name, or coordinates with CRS")
        return self


class RouteWeights(Contract):
    distance: float = Field(default=1.0, ge=0, le=1)
    slope: float = Field(default=0.0, ge=0, le=1)
    scenery: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def unit_sum(self):
        if not isclose(self.distance + self.slope + self.scenery, 1.0, abs_tol=1e-8):
            raise ValueError("route weights must sum to one")
        return self


class RouteStrategy(Contract):
    name: Literal["shortest", "recommended", "scenery", "flat", "custom"] = "shortest"
    source: Identifier = "commute_default"
    task_class: Literal["commute", "leisure", "accessibility", "custom", "explicit_preference"] = "commute"
    weights: RouteWeights = Field(default_factory=RouteWeights)
    detour_cap: float = Field(default=1.0, ge=1, le=5)

    @model_validator(mode="after")
    def preserve_shortest_semantics(self):
        if self.source == "commute_default" and self.name != "shortest":
            raise ValueError("default commuting strategy must be shortest")
        if self.name == "shortest" and self.weights != RouteWeights():
            raise ValueError("shortest strategy requires weights (1, 0, 0)")
        return self


class HardConstraints(Contract):
    avoid_slope: bool = False
    avoid_steps: bool = False
    require_verified_accessibility: bool = False
    max_distance_m: float | None = Field(default=None, gt=0)
    max_duration_s: float | None = Field(default=None, gt=0)
    excluded_edge_ids: list[Identifier] = Field(default_factory=list, max_length=2000)


class TourSpec(Contract):
    duration_minutes: float | None = Field(default=None, gt=0)
    return_to_start: bool = True
    categories: list[Identifier] = Field(default_factory=list, max_length=50)


class RouteLeg(Contract):
    start: LocationRef
    end: LocationRef
    travel_mode: TravelMode = "walk"


class RouteSpec(Contract):
    # Incomplete tasks retain known fields while awaiting only the missing facts.
    start: LocationRef | None = None
    end: LocationRef | None = None
    stops: list[LocationRef] = Field(default_factory=list, max_length=10)
    travel_mode: TravelMode = "walk"
    objective: Literal["shortest_distance", "fastest_time", "balanced"] = "shortest_distance"
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    strategy: RouteStrategy = Field(default_factory=RouteStrategy)
    data_version: Identifier | None = None
    event_version: Identifier | None = None
    route_kind: Literal["direct", "via", "tour", "multimodal"] = "direct"
    tour: TourSpec | None = None
    legs: list[RouteLeg] = Field(default_factory=list, max_length=50)


class Requirement(Contract):
    requirement_id: Identifier
    description: str = Field(min_length=1, max_length=2000)
    status: Literal["pending", "satisfied", "failed", "needs_input"] = "pending"


class PlaceRequestState(Contract):
    """Durable intent slots for activity discovery and its follow-up answers."""

    request_id: Identifier
    original_query: str = Field(min_length=1, max_length=2000)
    root_query: str | None = Field(default=None, max_length=2000)
    activities: list[Identifier] = Field(default_factory=list, max_length=20)
    subcategories: list[Identifier] = Field(default_factory=list, max_length=50)
    follow_up_requests: list[dict[str, Any]] = Field(default_factory=list, max_length=10)
    completed_legs: list[dict[str, Any]] = Field(default_factory=list, max_length=10)
    pending_slot: Literal["activity", "start", "selection", "follow_up", "destination", "route_retry"] | None = None
    start: dict[str, Any] | None = None
    selected_poi_id: Identifier | None = None
    candidate_poi_ids: list[Identifier] = Field(default_factory=list, max_length=20)
    mode: TravelMode = "walk"
    strategy: Literal["shortest", "recommended", "scenery", "flat"] | None = None
    constraints: HardConstraints = Field(default_factory=HardConstraints)


class FieldProvenance(Contract):
    source: PatchSource
    scope: MemoryScope = "task"
    source_message_id: Identifier
    updated_at: UtcDateTime


class TaskState(Contract):
    task_id: Identifier
    conversation_id: Identifier
    revision: int = Field(default=0, ge=0)
    task_kind: Literal["route", "information", "composite", "comparison", "itinerary"] = "route"
    status: Literal["draft", "ready", "queued", "running", "needs_input", "completed", "partial", "failed", "cancelled", "superseded"] = "draft"
    route_spec: RouteSpec = Field(default_factory=RouteSpec)
    place_request: PlaceRequestState | None = None
    requirements: list[Requirement] = Field(default_factory=list, max_length=100)
    pending_questions: list[str] = Field(default_factory=list, max_length=100)
    artifact_ids: list[Identifier] = Field(default_factory=list, max_length=500)
    created_at: UtcDateTime = Field(default_factory=utc_now)
    updated_at: UtcDateTime = Field(default_factory=utc_now)
    field_provenance: dict[str, FieldProvenance] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_lifecycle(self):
        if self.updated_at < self.created_at:
            raise ValueError("updated_at precedes created_at")
        ids = [item.requirement_id for item in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement IDs must be unique within a task")
        return self


class PatchOperation(Contract):
    op: Literal["set", "add", "remove", "clear", "undo"]
    path: str | None = Field(default=None, max_length=150)
    value: Any = None
    scope: MemoryScope = "task"

    @model_validator(mode="after")
    def validate_operation_shape(self):
        if self.op == "undo":
            if self.path is not None or "value" in self.model_fields_set:
                raise ValueError("undo must not contain a path or client-supplied state")
        elif not self.path:
            raise ValueError("patch operation requires a path")
        if self.op in {"set", "add"} and "value" not in self.model_fields_set:
            raise ValueError("set and add require an explicit value")
        if self.op == "clear" and "value" in self.model_fields_set:
            raise ValueError("clear must not contain a value")
        return self


class TaskPatch(Contract):
    task_id: Identifier
    base_revision: int = Field(ge=0)
    operations: list[PatchOperation] = Field(min_length=1, max_length=100)
    source_message_id: Identifier
    source: PatchSource = "user"

    @model_validator(mode="after")
    def undo_is_atomic(self):
        if len(self.operations) != 1 and any(op.op == "undo" for op in self.operations):
            raise ValueError("undo must be the only operation")
        return self
