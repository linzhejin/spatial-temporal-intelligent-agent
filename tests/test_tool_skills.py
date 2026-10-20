"""能力说明层（GeoSkill 式技能规范）单测。"""

from agents import planner, tool_skills
from agents.tools import TOOL_NAMES


class TestSkillRegistry:
    def test_coverage_matches_tool_registry(self):
        assert tool_skills.validate_skill_coverage() == []

    def test_every_tool_has_a_card(self):
        assert set(tool_skills.TOOL_SKILLS) == set(TOOL_NAMES)

    def test_produces_values_are_legal(self):
        for name, skill in tool_skills.TOOL_SKILLS.items():
            assert skill["produces"] in tool_skills.VALID_PRODUCES, name

    def test_categories_are_legal(self):
        for name, skill in tool_skills.TOOL_SKILLS.items():
            assert skill["category"] in tool_skills.VALID_CATEGORIES, name

    def test_confusable_targets_exist(self):
        for name, skill in tool_skills.TOOL_SKILLS.items():
            for other in skill.get("confusable_with") or []:
                assert other in TOOL_NAMES, f"{name} -> {other}"


class TestCardMatchesSchema:
    """能力卡不能与工具 schema 的必填项自相矛盾（模型会照卡片办）。"""

    def _schema(self, name):
        from agents.tools import TOOL_SCHEMAS
        return next(t["function"] for t in TOOL_SCHEMAS
                    if t["function"]["name"] == name)

    def test_plan_tour_start_is_documented_optional(self):
        schema = self._schema("plan_tour")
        assert "start" not in (schema.get("parameters") or {}).get("required", [])
        requires = tool_skills.TOOL_SKILLS["plan_tour"]["requires"]
        assert "可选" in requires
        assert not requires.startswith("起点")

    def test_plan_tour_tells_model_not_to_ask_for_location(self):
        skill = tool_skills.TOOL_SKILLS["plan_tour"]
        assert "不必先反问位置" in skill["use_when"]

    def test_plan_via_route_does_not_overstate_via_requirement(self):
        schema = self._schema("plan_via_route")
        assert "via_points" not in (schema.get("parameters") or {}).get("required", [])
        requires = tool_skills.TOOL_SKILLS["plan_via_route"]["requires"]
        assert "建议" in requires


class TestManifest:
    def test_manifest_lists_all_tools(self):
        manifest = tool_skills.build_skill_manifest()
        for name in TOOL_NAMES:
            assert name in manifest

    def test_manifest_carries_boundary_and_usage(self):
        manifest = tool_skills.build_skill_manifest()
        # 能力边界（何时不要用）与正向触发都必须出现在渲染结果里
        assert tool_skills.TOOL_SKILLS["plan_via_route"]["do_not_use_when"][:8] in manifest
        assert tool_skills.TOOL_SKILLS["plan_route"]["use_when"][:6] in manifest

    def test_manifest_is_cached(self):
        assert tool_skills.build_skill_manifest() is tool_skills.build_skill_manifest()


class TestPlannerInjection:
    def test_manifest_injected_right_after_system_prompt(self):
        msgs = planner._build_messages("从珞珈门到图书馆")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "system"
        assert "工具能力说明" in msgs[1]["content"]

    def test_manifest_contains_every_tool_name(self):
        msgs = planner._build_messages("从珞珈门到图书馆")
        manifest = msgs[1]["content"]
        for name in TOOL_NAMES:
            assert name in manifest
