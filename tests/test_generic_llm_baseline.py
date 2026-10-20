"""通用 LLM 直答基线（B2）的形状合同单测：不联网，只测确定性部分。"""

from experiments.baselines import generic_llm_baseline as baseline


class TestConstraintNormalization:
    def test_missing_constraints_get_system_defaults(self):
        assert baseline._normalize_constraints(None) == {
            "distance": "medium", "slope": "normal", "scenery": "normal",
        }

    def test_partial_constraints_are_filled_not_dropped(self):
        result = baseline._normalize_constraints({"slope": "avoid"})
        assert result == {"distance": "medium", "slope": "avoid", "scenery": "normal"}

    def test_invalid_enum_falls_back_to_default(self):
        result = baseline._normalize_constraints({
            "distance": "bogus", "slope": None, "scenery": 3,
        })
        assert result == {"distance": "medium", "slope": "normal", "scenery": "normal"}

    def test_no_none_values_reach_the_evaluator(self):
        """评测器会对约束取值排序；任何 None 都会让 macro-F1 崩溃。"""
        result = baseline._normalize_constraints({})
        assert all(value is not None for value in result.values())


class TestJsonExtraction:
    def test_plain_json(self):
        assert baseline._extract_json('{"task_type": "chat"}') == {"task_type": "chat"}

    def test_code_fenced_json(self):
        text = '```json\n{"task_type": "path_planning"}\n```'
        assert baseline._extract_json(text) == {"task_type": "path_planning"}

    def test_prose_wrapped_json(self):
        text = '好的，结果是：{"task_type": "poi_query"} 以上。'
        assert baseline._extract_json(text) == {"task_type": "poi_query"}

    def test_non_json_returns_none(self):
        assert baseline._extract_json("我不知道") is None
        assert baseline._extract_json("") is None


class TestNormalize:
    def test_route_is_never_executable_for_generic_baseline(self):
        result = baseline._normalize({"task_type": "path_planning"})
        assert result["route_executable"] is False
        assert result["geometry_verified"] is False

    def test_invalid_mode_falls_back_to_walk(self):
        assert baseline._normalize({"mode": "fly"})["mode"] == "walk"
        assert baseline._normalize({"mode": "bike"})["mode"] == "bike"

    def test_malformed_endpoints_are_dropped(self):
        result = baseline._normalize({"start": "珞珈门", "end": {"name": "樱顶"}})
        assert result["start"] is None
        assert result["end"] == {"name": "樱顶"}

    def test_unknown_task_type_when_absent(self):
        assert baseline._normalize({})["task_type"] == "unknown"


class TestFallback:
    def test_fallback_never_fabricates_places(self):
        result = baseline._fallback_intent("从珞珈门到樱顶")
        assert result["task_type"] == "unknown"
        assert result["start"] is None and result["end"] is None
        assert "未解析" in result["ambiguity"]

    def test_parse_without_client_uses_fallback(self, monkeypatch):
        monkeypatch.setattr(baseline, "_client", lambda: None)
        result = baseline.generic_parse("从珞珈门到樱顶")
        assert result["task_type"] == "unknown"
