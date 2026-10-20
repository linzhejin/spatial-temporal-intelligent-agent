"""三层知识库与经验沉淀（WS2）单测。"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents import experience, knowledge, planner, tool_skills


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(experience, "EXPERIENCE_PATH", tmp_path / "agent_experience.jsonl")
    monkeypatch.setattr(experience, "CARDS_PATH", tmp_path / "experience_cards.json")
    experience.reset_cache()
    yield
    experience.reset_cache()


def _write_samples(entries):
    experience.EXPERIENCE_PATH.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )


class TestScenarioKey:
    def test_reuses_task_card_id(self):
        assert experience.scenario_key("我想看樱花") == "sakura_viewing"

    def test_falls_back_to_fingerprint(self):
        key = experience.scenario_key("zzzz 没有触发词的句子")
        assert key.startswith("q:")
        assert len(key) == 18  # "q:" + 16 位指纹

    def test_fingerprint_is_stable(self):
        assert (experience.scenario_key("随意一句话")
                == experience.scenario_key("随意一句话"))


class TestRecordRun:
    def test_appends_one_line_per_run(self):
        trace = [{"tool_name": "plan_route", "outcome": "success"}]
        assert experience.record_run("去食堂", trace, success=True) is True
        lines = experience.EXPERIENCE_PATH.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tool_sequence"] == ["plan_route"]
        assert entry["success"] is True

    def test_does_not_store_raw_query(self):
        experience.record_run("从珞珈门到图书馆", [{"tool_name": "plan_route"}], success=True)
        assert "珞珈门" not in experience.EXPERIENCE_PATH.read_text(encoding="utf-8")

    def test_never_raises_on_failure(self):
        with patch("builtins.open", side_effect=OSError("disk full")):
            assert experience.record_run("x", [{"tool_name": "t"}], success=True) is False

    def test_experience_file_is_bounded(self):
        """按请求追加的样本文件必须有上限，否则生产环境会无限增长。"""
        trace = [{"tool_name": "plan_route"}]
        for _ in range(8):
            experience.record_run("去食堂", trace, success=True)
        with patch.object(experience, "MAX_FILE_BYTES", 10):
            experience.record_run("去食堂", trace, success=True)
        remaining = experience.EXPERIENCE_PATH.read_text(encoding="utf-8").splitlines()
        assert 0 < len(remaining) < 9

    def test_task_cards_are_cached_on_hot_path(self):
        """场景键推导在每个请求上都会跑，不能每次都读盘。"""
        experience.reset_cache()
        original = Path.read_text
        seen = []

        def counting(path_self, *args, **kwargs):
            seen.append(str(path_self))
            return original(path_self, *args, **kwargs)

        with patch.object(Path, "read_text", counting):
            for _ in range(5):
                experience.scenario_key("我想看樱花")

        assert len([name for name in seen if name.endswith("task_cards.json")]) == 1

    def test_tool_sequence_ignores_malformed_items(self):
        assert experience.tool_sequence(
            [{"tool_name": "a"}, None, {}, {"tool_name": "b"}]
        ) == ["a", "b"]


class TestDistill:
    def test_aggregates_by_scenario_and_sequence(self):
        _write_samples([
            {"scenario_key": "sakura_viewing", "tool_sequence": ["plan_tour"],
             "success": True, "at": "2026-01-01T00:00:00+00:00"},
            {"scenario_key": "sakura_viewing", "tool_sequence": ["plan_tour"],
             "success": True, "at": "2026-01-02T00:00:00+00:00"},
            {"scenario_key": "sakura_viewing", "tool_sequence": ["plan_tour"],
             "success": False, "at": "2026-01-03T00:00:00+00:00"},
        ])
        payload = experience.distill()
        cards = payload["cards"]
        assert len(cards) == 1
        assert cards[0]["success_count"] == 2
        assert cards[0]["failure_count"] == 1
        assert cards[0]["last_seen"] == "2026-01-03T00:00:00+00:00"

    def test_missing_input_is_not_an_error(self):
        assert experience.distill()["cards"] == []


class TestRetrieveExperience:
    def _seed(self):
        _write_samples(
            [{"scenario_key": "meal_run", "tool_sequence": ["search_poi_candidates"],
              "success": True, "at": f"2026-01-0{i}T00:00:00+00:00"} for i in range(1, 6)]
            + [{"scenario_key": "meal_run", "tool_sequence": ["search_poi_candidates"],
                "success": False, "at": "2026-01-09T00:00:00+00:00"}]
        )
        experience.distill()

    def test_confident_pattern_is_recalled(self):
        self._seed()
        cards = experience.retrieve_experience("我想去食堂")
        assert cards and cards[0]["tool_sequence"] == ["search_poi_candidates"]

    def test_low_ratio_pattern_is_filtered(self):
        _write_samples([
            {"scenario_key": "meal_run", "tool_sequence": ["plan_route"], "success": True},
            {"scenario_key": "meal_run", "tool_sequence": ["plan_route"], "success": False},
        ])
        experience.distill()
        assert experience.retrieve_experience("我想去食堂") == []

    def test_below_min_samples_is_filtered(self):
        _write_samples([
            {"scenario_key": "meal_run", "tool_sequence": ["plan_route"], "success": True},
            {"scenario_key": "meal_run", "tool_sequence": ["plan_route"], "success": True},
        ])
        experience.distill()
        assert experience.retrieve_experience("我想去食堂") == []

    def test_unrelated_query_returns_nothing(self):
        self._seed()
        assert experience.retrieve_experience("zzzz 无关") == []

    def test_empty_query_returns_nothing(self):
        assert experience.retrieve_experience("") == []


class TestThreeLayers:
    def test_capability_layer_is_the_skill_manifest(self):
        assert knowledge.retrieve_capability_knowledge() == tool_skills.build_skill_manifest()

    def test_domain_layer_alias_still_works(self):
        cards = knowledge.retrieve_domain_knowledge("我想看樱花")
        assert cards and cards[0]["id"] == "sakura_viewing"
        assert knowledge.retrieve_cards("我想看樱花") == cards

    def test_experience_layer_reads_distilled_cards(self):
        _write_samples(
            [{"scenario_key": "sakura_viewing", "tool_sequence": ["plan_tour"],
              "success": True, "at": f"2026-01-0{i}T00:00:00+00:00"} for i in range(1, 6)]
        )
        experience.distill()
        cards = knowledge.retrieve_experience_knowledge("我想看樱花")
        assert cards and cards[0]["scenario_key"] == "sakura_viewing"

    def test_message_puts_experience_before_domain(self):
        _write_samples(
            [{"scenario_key": "sakura_viewing", "tool_sequence": ["plan_tour"],
              "success": True, "at": f"2026-01-0{i}T00:00:00+00:00"} for i in range(1, 6)]
        )
        experience.distill()
        msg = knowledge.build_knowledge_message("我想看樱花")
        assert msg is not None
        assert msg.index("历史成功编排") < msg.index("校园场景经验")

    def test_message_is_none_when_no_layer_hits(self):
        assert knowledge.build_knowledge_message("zzzz 完全无关") is None

    def test_domain_message_unchanged_without_experience(self):
        msg = knowledge.build_knowledge_message("膝盖不好，别爬坡")
        assert "无障碍出行" in msg
        assert "slope=0.6" in msg
        assert "历史成功编排" not in msg


class TestPlannerExperienceHook:
    def test_run_agent_records_experience_for_tool_runs(self, monkeypatch):
        recorded = []

        def fake_record(query, tool_trace, *, success, verification_status=None):
            recorded.append({"query": query, "success": success,
                             "tools": experience.tool_sequence(tool_trace)})
            return True

        monkeypatch.setattr("agents.planner.experience.record_run", fake_record)

        planner._record_experience_sample(
            "去食堂",
            {"response_kind": "candidates", "candidates": [{"id": "x"}],
             "provenance": {"tool_activities": [{"tool_name": "search_poi_candidates"}]}},
        )
        assert recorded == [{"query": "去食堂", "success": True,
                             "tools": ["search_poi_candidates"]}]

    def test_successful_route_reads_nested_verification(self, monkeypatch):
        """成功路线的核验报告随路线包返回，沉淀时必须读到它才算成功。"""
        recorded = []
        monkeypatch.setattr("agents.planner.experience.record_run",
                            lambda q, t, *, success, verification_status=None:
                            recorded.append((success, verification_status)) or True)
        response = {
            "response_kind": "route",
            "route": {"route_verification": {"status": "pass"}},
            "provenance": {"tool_activities": [{"tool_name": "plan_route"}]},
        }
        planner._record_experience_sample("从牌坊到图书馆", response)
        assert recorded == [(True, "pass")]

    def test_failed_route_is_recorded_as_failure(self, monkeypatch):
        recorded = []
        monkeypatch.setattr("agents.planner.experience.record_run",
                            lambda q, t, *, success, verification_status=None:
                            recorded.append((success, verification_status)) or True)
        response = {
            "response_kind": "chat",
            "route": None,
            "route_verification": {"status": "unknown"},
            "provenance": {"tool_activities": [{"tool_name": "plan_route"}]},
        }
        planner._record_experience_sample("从牌坊到图书馆", response)
        assert recorded == [(False, "unknown")]

    def test_sample_is_skipped_when_no_tool_called(self, monkeypatch):
        called = []
        monkeypatch.setattr("agents.planner.experience.record_run",
                            lambda *a, **k: called.append(True) or True)
        planner._record_experience_sample("你好", {"response_kind": "chat", "provenance": {}})
        assert called == []

    def test_experience_failure_never_breaks_response(self, monkeypatch):
        monkeypatch.setattr("agents.planner.experience.record_run",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        response = {"response_kind": "chat", "message": "hi",
                    "provenance": {"tool_activities": [{"tool_name": "plan_route"}]}}
        assert planner._record_experience_sample("x", response) is None
