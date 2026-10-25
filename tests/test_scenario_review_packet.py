"""A small, honest review packet for unresolved scenario semantics."""

import pytest

from experiments.scenario_review_packet import build_review_packet, render_review_packet


def _dataset():
    cases = []
    for category in ("place_recommendation", "mode_access"):
        for index in range(5):
            cases.append({
                "id": f"{category}_{index}",
                "category": category,
                "adjudication": {"status": "draft"},
                "messages": [{"role": "user", "content": f"问题 {category} {index}"}],
                "expected_turns": [{
                    "status": "completed", "kind": "route", "feasibility": "unknown",
                    "start_poi_id": "poi_001", "end_poi_id": "poi_002",
                    "via_poi_ids": [], "hard_constraints": {"avoid_steps": True},
                    "travel_mode": "walk", "strategy": "shortest",
                    "required_outputs": ["route"], "candidate_poi_ids": [],
                    "clarification_slot": None,
                }],
            })
    return {"protocol_version": "scenario-v1", "split": "development", "cases": cases}


def test_review_packet_is_stratified_repeatable_and_never_marks_gold_reviewed():
    dataset = _dataset()
    first = build_review_packet(dataset, per_category=2, seed=42)
    assert first == build_review_packet(dataset, per_category=2, seed=42)
    assert len(first["cards"]) == 4
    assert {category: sum(c["category"] == category for c in first["cards"])
            for category in ("place_recommendation", "mode_access")} == {
                "place_recommendation": 2, "mode_access": 2,
            }
    assert all(card["review_status"] == "pending" for card in first["cards"])
    assert all(card["human_decision"] is None for card in first["cards"])
    assert all(card["proposal_is_gold"] is False for card in first["cards"])
    assert dataset["cases"][0]["adjudication"]["status"] == "draft"


def test_review_packet_preserves_all_turns_and_rejects_short_or_reviewed_categories():
    dataset = _dataset()
    dataset["cases"][0]["messages"].append({"role": "user", "content": "再改一下"})
    dataset["cases"][0]["expected_turns"].append(
        {"status": "needs_input", "kind": "clarify", "clarification_slot": "destination"})
    packet = build_review_packet(dataset, per_category=5, seed=1)
    card = next(c for c in packet["cards"] if c["case_id"] == "place_recommendation_0")
    assert len(card["messages"]) == len(card["proposed_interpretation"]) == 2

    dataset["cases"][0]["adjudication"]["status"] = "adjudicated"
    with pytest.raises(ValueError, match="not enough draft"):
        build_review_packet(dataset, per_category=5, seed=1)


def test_review_text_shows_plain_names_but_keeps_answers_pending():
    packet = build_review_packet(_dataset(), per_category=1, seed=1)
    text = render_review_packet(packet, {"poi_001": "梅园综合楼", "poi_002": "卓尔体育馆"})
    assert "梅园综合楼（poi_001）" in text
    assert "卓尔体育馆（poi_002）" in text
    assert "待人工确认" in text
    assert "草稿建议，不是金标准" in text
    assert "预期呈现：route" in text
    assert "需要澄清：未指定" in text
