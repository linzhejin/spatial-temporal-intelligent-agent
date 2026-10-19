from agents.context_builder import build_agent_context, is_route_followup


def test_fresh_route_does_not_inherit_previous_route_slots():
    previous = {"start": {"name": "珞珈门"}, "end": {"name": "樱顶"}}
    context = build_agent_context("从信息学部到图书馆走最短路线", [], previous)
    assert "previous_route_state" not in context


def test_explicit_route_followup_inherits_route_slots():
    previous = {"start": {"name": "珞珈门"}, "end": {"name": "樱顶"}}
    assert is_route_followup("换一条更短的")
    context = build_agent_context("换一条更短的", [], previous)
    assert context["previous_route_state"] == previous


def test_unrelated_weather_question_does_not_inherit_route_slots():
    previous = {"start": {"name": "珞珈门"}, "end": {"name": "樱顶"}}
    context = build_agent_context("今天要带伞吗？", [], previous)
    assert "previous_route_state" not in context


def test_same_clarification_task_keeps_its_saved_context():
    previous = {"start": {"name": "珞珈门"}}
    context = build_agent_context("从南门", [], previous, same_task=True)
    assert context["previous_route_state"] == previous


def test_avoidance_constraint_alone_does_not_reuse_an_old_route():
    assert not is_route_followup("去卓尔体育馆，避开台阶")
    assert is_route_followup("刚才的路线避开台阶")


def test_explicit_mode_change_refers_to_previous_route():
    assert is_route_followup("刚才的路线改成骑行")
    assert is_route_followup("改为骑行")


def test_route_followup_after_interleaved_weather_keeps_previous_route_without_de():
    previous = {"start": {"name": "玉兰2门"}, "end": {"name": "樱顶"},
                "hard_constraints": {"avoid_steps": True}}
    query = "刚才路线改为平坦优先，别走台阶仍然保留"
    history = [{"role": "user", "content": "顺便查一下天气"},
               {"role": "assistant", "content": "当前天气多云"}]
    assert is_route_followup(query)
    context = build_agent_context(query, history, previous)
    assert context["previous_route_state"] == previous
