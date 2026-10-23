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
