"""Review card export preserves the line between suggestion and gold."""
import json
from pathlib import Path

from scripts.analysis.prepare_review_cards_v2 import build_cards, render_cards


ROOT = Path(__file__).resolve().parents[1]


def test_review_pack_has_sixteen_distinct_pending_cases():
    dataset = json.loads((ROOT / "experiments/datasets/scenarios_test_v2_draft.json")
                         .read_text(encoding="utf-8"))
    cards = build_cards(dataset)
    assert len(cards) == len({card["id"] for card in cards}) == 16
    assert sum(card["group"] == "T5" for card in cards) == 6
    text = render_cards(cards, "a" * 64)
    assert "不是独立金标准" in text
    assert "pending" in text
    assert "v2_T5_sport_then_food_1" in text


def test_destination_weather_cards_keep_explicit_destination():
    dataset = json.loads((ROOT / "experiments/datasets/scenarios_test_v2_draft.json")
                         .read_text(encoding="utf-8"))
    cards = build_cards(dataset)
    selected = [card for card in cards if "destination_weather" in card["id"]]
    assert len(selected) == 3
    for card in selected:
        assert "目的地缺失" not in card["suggestion"]
        assert "追问起点" in card["suggestion"]
        assert "沿用原目的地" in card["suggestion"]


def test_sport_then_food_cards_require_complete_ordered_route():
    dataset = json.loads((ROOT / "experiments/datasets/scenarios_test_v2_draft.json")
                         .read_text(encoding="utf-8"))
    cards = build_cards(dataset)
    selected = [card for card in cards if "sport_then_food" in card["id"]]
    assert len(selected) == 3
    for card in selected:
        assert "餐饮候选" in card["suggestion"]
        assert "全程路线" in card["suggestion"]
        assert "原起点→篮球场→餐饮点" in card["suggestion"]


def test_food_before_route_cards_keep_specified_endpoint():
    dataset = json.loads((ROOT / "experiments/datasets/scenarios_test_v2_draft.json")
                         .read_text(encoding="utf-8"))
    cards = build_cards(dataset)
    selected = [card for card in cards if "food_before_route" in card["id"]]
    assert len(selected) == 3
    for card in selected:
        assert "起点→选定餐饮点→原终点" in card["suggestion"]
        assert "全程路线" in card["suggestion"]
        assert "任一段无法核验" in card["suggestion"]
