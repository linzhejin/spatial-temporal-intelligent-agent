from copy import deepcopy

import pytest

from agents.route_state import (
    apply_change,
    current_data_version,
    route_state_to_context,
    validate_route_state,
)


BASE = {
    "schema_version": 1,
    "route_id": "route-test",
    "route_kind": "via",
    "original_query": "从牌坊经樱顶到图书馆",
    "start": {"name": "牌坊", "type": "poi"},
    "end": {"name": "图书馆", "type": "poi"},
    "via": {"name": "樱顶", "type": "poi"},
    "tour": None,
    "legs": [],
    "travel_mode": "walk",
    "hard_constraints": {"slope": "normal"},
    "strategy": {
        "name": "shortest",
        "source": "commute_default",
        "task_class": "commute",
        "weights": {"distance": 1.0, "slope": 0.0, "scenery": 0.0},
        "detour_cap": 1.0,
    },
    "data_version": "data-v1",
    "road_condition_version": "roads-v1",
}


def test_mode_change_preserves_every_route_semantic_field():
    changed = apply_change(validate_route_state(BASE), {"travel_mode": "bike"})

    assert changed["travel_mode"] == "bike"
    for key in (
        "route_kind",
        "original_query",
        "start",
        "end",
        "via",
        "tour",
        "legs",
        "hard_constraints",
        "strategy",
    ):
        assert changed[key] == BASE[key]
    assert changed["route_id"] != BASE["route_id"]


def test_strategy_change_preserves_hard_constraints_and_route_shape():
    changed = apply_change(validate_route_state(BASE), {"strategy": "flat"})

    assert changed["strategy"] == {"name": "flat", "source": "button"}
    assert changed["hard_constraints"] == BASE["hard_constraints"]
    assert changed["start"] == BASE["start"]
    assert changed["end"] == BASE["end"]
    assert changed["via"] == BASE["via"]


def test_route_state_becomes_parser_context_without_losing_strategy_or_mode():
    context = route_state_to_context(BASE)

    assert context["start"] == BASE["start"]
    assert context["end"] == BASE["end"]
    assert context["constraints"] == BASE["hard_constraints"]
    assert context["weights"] == BASE["strategy"]["weights"]
    assert context["previous_intent"]["strategy_hint"] == "shortest"
    assert context["previous_intent"]["mode"] == "walk"


def test_itinerary_state_preserves_budget_and_stop_duration_across_changes():
    itinerary = deepcopy(BASE)
    itinerary.update(
        route_kind="itinerary",
        via={"type": "multi", "points":[
            {"name": "星湖园食堂", "type": "poi"},
            {"name": "图书馆", "type": "poi"},
        ]},
        itinerary={"time_budget_min": 45, "stop_duration_min": 8},
    )

    changed = apply_change(validate_route_state(itinerary), {"strategy": "flat"})

    assert changed["route_kind"] == "itinerary"
    assert changed["itinerary"] == {"time_budget_min": 45, "stop_duration_min": 8}
    assert changed["via"]["points"] == itinerary["via"]["points"]


def test_itinerary_state_rejects_missing_or_invalid_budget():
    itinerary = deepcopy(BASE)
    itinerary.update(
        route_kind="itinerary",
        via={"type": "multi", "points":[{"name": "圖書館", "type": "poi"}]},
        itinerary={"time_budget_min": -1, "stop_duration_min": 8},
    )

    with pytest.raises(ValueError, match="itinerary"):
        validate_route_state(itinerary)


def test_itinerary_state_rejects_boolean_budget_fields():
    itinerary = deepcopy(BASE)
    itinerary.update(
        route_kind="itinerary",
        via={"type": "multi", "points":[{"name": "图书馆", "type": "poi"}]},
        itinerary={"time_budget_min": True, "stop_duration_min": 8},
    )

    with pytest.raises(ValueError, match="itinerary"):
        validate_route_state(itinerary)


def test_validation_sanitizes_client_computed_fields_without_mutating_input():
    poisoned = deepcopy(BASE)
    poisoned.update(
        recommended_edge_ids=[[1, 2, 0]],
        distance_m=1,
        recommended=[1, 2],
    )

    clean = validate_route_state(poisoned)

    assert "recommended_edge_ids" not in clean
    assert "distance_m" not in clean
    assert "recommended" not in clean
    assert poisoned["recommended_edge_ids"] == [[1, 2, 0]]


def test_course_release_fingerprint_invalidates_existing_route_state(tmp_path, monkeypatch):
    import config
    import networkx as nx

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    before = nx.MultiDiGraph()
    before.add_edge(1, 2, key=0, length=10)
    before.graph["course_release_fingerprint"] = "release-a"
    after = before.copy()
    after.graph["course_release_fingerprint"] = "release-b"

    assert current_data_version(before) != current_data_version(after)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"travel_mode": "bike", "strategy": "flat"},
        {"unknown": "value"},
    ],
)
def test_rejects_zero_multiple_or_unknown_changes(change):
    with pytest.raises(ValueError, match="exactly one"):
        apply_change(BASE, change)


def test_road_condition_refresh_preserves_route_preferences_and_constraints():
    refreshed = apply_change(BASE, {"refresh_road_conditions": True})

    assert refreshed["route_id"] != BASE["route_id"]
    for field in ("start", "end", "via", "legs", "travel_mode", "hard_constraints", "strategy"):
        assert refreshed[field] == BASE[field]


def test_road_condition_refresh_requires_explicit_true():
    with pytest.raises(ValueError, match="refresh_road_conditions"):
        apply_change(BASE, {"refresh_road_conditions": False})


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("schema_version", 99, "schema_version"),
        ("route_kind", "other", "route_kind"),
        ("travel_mode", "fly", "travel_mode"),
        ("start", None, "endpoints"),
    ],
)
def test_rejects_invalid_core_state(field, value, message):
    raw = deepcopy(BASE)
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        validate_route_state(raw)
