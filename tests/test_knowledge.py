"""任务卡知识库检索注入单测。"""

from datetime import datetime

from agents import knowledge, planner


class TestRetrieveCards:
    def test_trigger_match(self):
        cards = knowledge.retrieve_cards("我想去看樱花，怎么走")
        assert cards and cards[0]["id"] == "sakura_viewing"

    def test_no_match_returns_empty(self):
        assert knowledge.retrieve_cards("今天天气不错") == []

    def test_limit(self):
        cards = knowledge.retrieve_cards("买完东西去上课，顺便吃饭", limit=2)
        assert len(cards) <= 2

    def test_season_boost(self):
        # 3 月 spring：sakura_viewing 有 season_hint=spring，应排在 campus_tour 前
        cards = knowledge.retrieve_cards("樱花 逛", now=datetime(2026, 3, 20))
        assert cards[0]["id"] == "sakura_viewing"

    def test_empty_query(self):
        assert knowledge.retrieve_cards("") == []


class TestBuildKnowledgeMessage:
    def test_message_contains_advice_and_weights(self):
        msg = knowledge.build_knowledge_message("膝盖不好，别爬坡")
        assert "无障碍出行" in msg
        assert "slope=0.6" in msg

    def test_no_match_returns_none(self):
        assert knowledge.build_knowledge_message("xyzzy 无命中词") is None


class TestPlannerInjection:
    def test_knowledge_injected_into_messages(self):
        msgs = planner._build_messages("我想看樱花")
        sys_texts = [m["content"] for m in msgs if m["role"] == "system"]
        assert any("赏樱" in t for t in sys_texts)

    def test_no_injection_when_no_match(self):
        msgs = planner._build_messages("zzzz 完全无关")
        # 主 system 提示词 + 固定注入的能力说明表
        assert len([m for m in msgs if m["role"] == "system"]) == 2
