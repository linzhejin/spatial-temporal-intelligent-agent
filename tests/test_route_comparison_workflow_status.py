from agents.workflow import _agent_requirement_status


def _comparison(status, available, unavailable):
    strategies = ["shortest", "scenery", "flat"]
    alternatives = [
        {"strategy": strategy, "distance_m": 100 + index * 10}
        for index, strategy in enumerate(strategies[:available])
    ]
    missing = [
        {"strategy": strategy, "label": strategy, "reason": "fixture unavailable"}
        for strategy in strategies[available:available + unavailable]
    ]
    return {
        "response_kind": "route_comparison",
        "route_comparison": {
            "status": status,
            "start_name": "起点",
            "end_name": "终点",
            "mode": "walk",
            "alternatives": alternatives,
            "unavailable": missing,
            "evidence": {"data_version": "data-v1", "road_condition_version": "roads-v1"},
        },
    }


def test_complete_route_comparison_satisfies_a_route_request():
    assert _agent_requirement_status(_comparison("complete", 3, 0), "route") == "satisfied"


def test_partial_route_comparison_with_two_usable_options_satisfies_the_comparison():
    assert _agent_requirement_status(_comparison("partial", 2, 1), "route") == "satisfied"


def test_one_route_is_not_misreported_as_a_successful_comparison():
    assert _agent_requirement_status(_comparison("partial", 1, 2), "route") == "failed"


def test_comparison_with_an_invalid_distance_does_not_satisfy_the_request():
    response = _comparison("complete", 3, 0)
    response["route_comparison"]["alternatives"][0]["distance_m"] = float("nan")
    assert _agent_requirement_status(response, "route") == "failed"


def test_malformed_strategy_type_does_not_raise_during_status_validation():
    response = _comparison("complete", 3, 0)
    response["route_comparison"]["alternatives"][0]["strategy"] = []
    assert _agent_requirement_status(response, "route") == "failed"


def test_unrepresentably_large_distance_does_not_raise_during_status_validation():
    response = _comparison("complete", 3, 0)
    response["route_comparison"]["alternatives"][0]["distance_m"] = 10 ** 1000
    assert _agent_requirement_status(response, "route") == "failed"
