from agents.task_planner import build_task_plan


def test_route_and_weather_are_planned_as_independent_required_subtasks():
    plan = build_task_plan("从玉兰2门去图书馆，顺便查天气")

    assert plan.kind == "composite"
    assert {item.requirement_id for item in plan.requirements} == {"route", "weather"}
    assert {node.node_id for node in plan.nodes} == {"route-agent", "weather"}
    assert all(not node.depends_on for node in plan.nodes)


def test_same_task_destination_and_deictic_weather_followup_are_both_required():
    plan = build_task_plan("带我去桂园食堂 我在武汉大学科技门，查天气后再带我过去")
    assert plan.kind == "composite"
    assert {item.requirement_id for item in plan.requirements} == {"route", "weather"}


def test_via_first_route_request_keeps_route_and_weather_requirements():
    for query in (
        "经卓尔体育馆去樱顶，不走台阶，顺便查天气",
        "途经卓尔体育馆到樱顶，顺便查天气",
        "经过卓尔体育馆去樱顶，顺便查天气",
    ):
        plan = build_task_plan(query)
        assert plan.kind == "composite", query
        assert {item.requirement_id for item in plan.requirements} == {"route", "weather"}


def test_weather_only_task_does_not_require_an_llm_route_agent():
    plan = build_task_plan("查一下武大今天的天气")
    assert plan.kind == "weather"
    assert [node.node_id for node in plan.nodes] == ["weather"]


def test_non_weather_route_does_not_spawn_an_unrequested_weather_tool():
    plan = build_task_plan("从珞珈门到樱顶，走最短路径")
    assert plan.kind == "route"
    assert [node.node_id for node in plan.nodes] == ["route-agent"]
    assert [item.requirement_id for item in plan.requirements] == ["route"]


def test_weather_plus_poi_request_is_not_reduced_to_weather_only():
    plan = build_task_plan("今天下雨吗，顺便推荐附近的食堂")
    assert plan.kind == "agent"
    assert [node.tool for node in plan.nodes] == ["route_agent"]
    assert [item.requirement_id for item in plan.requirements] == ["answer"]


def test_weather_plus_road_condition_request_is_not_reduced_to_weather_only():
    plan = build_task_plan("今天下雨吗，珞珈山附近有没有封路")
    assert plan.kind == "agent"
    assert [node.tool for node in plan.nodes] == ["route_agent"]
