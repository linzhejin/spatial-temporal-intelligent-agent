"""核验失败反思-重规划（GAReflector 式）单测。"""

from agents import planner, reflection


def _report(*, checks=None, status=None, issues=None):
    report = {}
    if checks is not None:
        report["checks"] = checks
    if status is not None:
        report["status"] = status
    if issues is not None:
        report["issues"] = issues
    return report


def _group(status, issues=None):
    return {"status": status, "issues": issues or []}


class TestDiagnose:
    def test_constraint_conflict(self):
        diagnosis = reflection.diagnose(_report(checks={
            "mode_and_constraints": _group("fail", ["constraint_violation"]),
            "exact_edges": _group("pass"),
        }, status="fail"))
        assert diagnosis["primary_conflict"] == "constraint_vs_connectivity"
        assert diagnosis["failed_groups"] == ["mode_and_constraints"]
        assert diagnosis["should_give_up"] is False

    def test_budget_exceeded(self):
        diagnosis = reflection.diagnose(_report(checks={
            "duration_and_budget": _group("fail", ["distance_budget_exceeded"]),
        }, status="fail"))
        assert diagnosis["primary_conflict"] == "budget_exceeded"

    def test_stale_event_snapshot(self):
        diagnosis = reflection.diagnose(_report(checks={
            "road_conditions": _group("fail", ["event_version_mismatch"]),
        }, status="fail"))
        assert diagnosis["primary_conflict"] == "stale_event_snapshot"

    def test_endpoint_unresolved(self):
        diagnosis = reflection.diagnose(_report(checks={
            "endpoints": _group("fail", ["endpoint_unknown"]),
        }, status="fail"))
        assert diagnosis["primary_conflict"] == "endpoint_unresolved"

    def test_invalid_request_shape(self):
        diagnosis = reflection.diagnose(_report(checks={
            "route_structure": _group("fail", ["invalid_result"]),
        }, status="fail"))
        assert diagnosis["primary_conflict"] == "invalid_request_shape"

    def test_all_unknown_gives_up(self):
        diagnosis = reflection.diagnose(_report(checks={
            "mode_and_constraints": _group("unknown", ["accessibility_unverified"]),
            "duration_and_budget": _group("unknown", ["duration_context_unavailable"]),
        }, status="unknown"))
        assert diagnosis["primary_conflict"] == "evidence_insufficient"
        assert diagnosis["should_give_up"] is True

    def test_unclassified_failure(self):
        diagnosis = reflection.diagnose(_report(checks={
            "length_and_geometry": _group("fail", ["length_mismatch"]),
        }, status="fail"))
        assert diagnosis["primary_conflict"] == "unclassified"
        assert diagnosis["should_give_up"] is False

    def test_priority_prefers_root_cause(self):
        diagnosis = reflection.diagnose(_report(checks={
            "duration_and_budget": _group("fail", ["duration_budget_exceeded"]),
            "road_conditions": _group("fail", ["data_version_mismatch"]),
        }, status="fail"))
        assert diagnosis["primary_conflict"] == "stale_event_snapshot"

    def test_legacy_report_without_checks(self):
        diagnosis = reflection.diagnose(_report(status="fail", issues=["some_issue"]))
        assert diagnosis["primary_conflict"] == "unclassified"

    def test_empty_input_is_safe(self):
        diagnosis = reflection.diagnose({})
        assert diagnosis["primary_conflict"] is None
        assert diagnosis["should_give_up"] is False


class TestReflectionMessage:
    def test_first_attempt_is_targeted(self):
        diagnosis = reflection.diagnose(_report(checks={
            "mode_and_constraints": _group("fail", ["constraint_violation"]),
        }, status="fail"))
        text = reflection.build_reflection_message(diagnosis, 1)
        assert "硬约束与路网连通性冲突" in text
        assert "不要用同一组参数重复调用" in text

    def test_second_attempt_changes_strategy(self):
        diagnosis = reflection.diagnose(_report(checks={
            "duration_and_budget": _group("fail", ["duration_budget_exceeded"]),
        }, status="fail"))
        first = reflection.build_reflection_message(diagnosis, 1)
        second = reflection.build_reflection_message(diagnosis, 2)
        assert "最后一次" in second
        assert "更简单的一次调用" in second
        assert first != second

    def test_every_attempt_forbids_relaxing_constraints(self):
        diagnosis = reflection.diagnose(_report(checks={
            "exact_edges": _group("fail", ["excluded_edge_used"]),
        }, status="fail"))
        for attempt in (1, 2):
            text = reflection.build_reflection_message(diagnosis, attempt)
            assert "不要省略硬约束" in text or "不得省略或放宽" in text


class TestSummarize:
    def test_summary_names_the_conflict(self):
        diagnosis = reflection.diagnose(_report(checks={
            "duration_and_budget": _group("fail", ["distance_budget_exceeded"]),
        }, status="fail"))
        text = reflection.summarize(diagnosis)
        assert "预算" in text
        assert "时长与预算" in text

    def test_summary_of_empty_diagnosis(self):
        assert reflection.summarize({}) == ""


def test_route_failure_tells_user_when_slope_evidence_is_missing():
    report = _report(
        checks={"mode_and_constraints": _group("unknown", ["slope_evidence_unknown"])},
        status="unknown",
        issues=["slope_evidence_unknown"],
    )
    result = planner._route_verification_failure_response(
        {"route_verification": report, "message": "路线结果无法核对。"},
        message="路线未通过独立核验，没有展示路线。",
    )
    assert result["response_kind"] == "chat"
    assert result["route"] is None
    assert "坡度信息" in result["message"]
    assert "未展示新路线" in result["message"]
    assert result["route_verification"] == report


class TestGuardrails:
    def test_budget_constants_unchanged(self):
        assert planner.MAX_TURNS == 6
        assert planner.TIME_BUDGET_S == 40.0

    def test_reflection_attempts_bounded(self):
        assert planner.MAX_REFLECTION_ATTEMPTS == 2
        # 反思重试绝不能挤占主循环预算
        assert planner.MAX_REFLECTION_ATTEMPTS < planner.MAX_TURNS
