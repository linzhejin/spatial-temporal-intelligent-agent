"""A long-running replan must not publish an already stale environment."""

from unittest.mock import patch

import pytest

import api.routes as routes
from spatial.road_conditions import RoadConditionsUnavailableError
from tests.test_replan_api import _state_for, _verified_replan_state, client  # noqa: F401


@pytest.mark.parametrize("changed", ["event", "graph", "unavailable"])
def test_replan_rechecks_environment_before_returning(client, changed):  # noqa: F811
    state = _state_for("direct")
    change = {"refresh_road_conditions": True}
    payload = {"recommended": [], "distance_m": 10, "mode": "walk",
               "route_state": _verified_replan_state(state, change, []),
               "route_verification": {"status": "pass"}}
    next_conditions = ([{"condition_id": "new-event", "action": "slowdown"}]
                       if changed == "event" else [])
    conditions = [[], RoadConditionsUnavailableError("unavailable") if changed == "unavailable"
                  else next_conditions]
    versions = [state["data_version"], "new-graph" if changed == "graph" else state["data_version"]]
    with patch.object(routes, "list_conditions", side_effect=conditions), \
         patch.object(routes, "_current_data_version", side_effect=versions), \
         patch("agents.tools.execute_tool", return_value=(payload, {"route": payload})):
        response = client.post("/api/route/replan", json={"route_state": state, "change": change})
    assert response.status_code == (503 if changed == "unavailable" else 409)
    assert "data" not in response.get_json()
