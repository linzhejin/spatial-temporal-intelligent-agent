"""Publishing a nearby search must validate its actual search anchor."""

import pytest

from agents import route_state, tools
from storage.task_repository import _maintenance_snapshot_is_current, _requires_environment_snapshot


@pytest.mark.parametrize("status", ["completed", "partial", "needs_input"])
def test_nearby_candidates_without_trip_origin_are_checked_before_publication(monkeypatch, status):
    event_version = route_state.current_road_condition_version([])
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-current")
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [])
    snapshot = {"data_version": "data-current", "road_condition_version": event_version}
    result = {
        "response_kind": "candidates",
        "candidates": [{"poi_id": "poi_068", "name": "食堂"}],
        "place_request": {
            "start": None,
            "search_anchor": {"type": "poi", "name": "总图"},
            "activities": ["meal"],
            "subcategories": ["restaurant"],
            "candidate_poi_ids": ["poi_068"],
            "result_dependencies": {"candidate_snapshot": snapshot},
        },
    }

    assert _requires_environment_snapshot(status, result)
    assert _maintenance_snapshot_is_current(result)
    snapshot["data_version"] = "data-old"
    assert not _maintenance_snapshot_is_current(result)
    snapshot["data_version"] = "data-current"
    snapshot["road_condition_version"] = "roads-old"
    assert not _maintenance_snapshot_is_current(result)


def test_nearby_candidates_without_versions_fail_closed(monkeypatch):
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-current")
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [])
    result = {
        "response_kind": "candidates",
        "candidates": [{"poi_id": "poi_068"}],
        "place_request": {
            "start": None,
            "search_anchor": {"type": "poi", "name": "总图"},
            "activities": ["meal"],
            "result_dependencies": {},
        },
    }

    assert _requires_environment_snapshot("needs_input", result)
    assert not _maintenance_snapshot_is_current(result)
