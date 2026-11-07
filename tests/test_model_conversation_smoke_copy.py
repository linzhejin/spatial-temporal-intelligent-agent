import ast
from pathlib import Path

from scripts.model_conversation_smoke import (
    _accepted_history_present,
    _accepted_model_messages,
    _check_user_copy,
    _failure_frames,
    _has_restaurant_intent,
    _point_record,
)


def test_acceptance_copy_allows_short_plain_recommendation_without_repeating_cards():
    assert _check_user_copy(
        {"assistant_text": "总图附近可以看看这几家。"},
        cards=[{"name": "梅园食堂"}, {"name": "小肆川"}],
    )


def test_acceptance_copy_rejects_long_markdown_source_ids_and_card_repetition():
    for text in (
        "S5f44dc2b 是来源编号。",
        "参考来源：https://example.invalid/place",
        "**梅园食堂**适合吃饭。",
        "梅园食堂就在附近。",
        "这是一条" + "很长" * 51 + "的回复。",
    ):
        assert not _check_user_copy({"assistant_text": text}, cards=[{"name": "梅园食堂"}])


def test_place_point_reader_handles_provider_rows_and_normalized_coordinates():
    assert _point_record({"poi_id": "p1", "lng": 114.36, "lat": 30.53}) == {
        "lng": 114.36, "lat": 30.53, "crs": None,
    }
    assert _point_record({"poi_id": "p2", "coordinates": {"longitude": 114.37, "latitude": 30.54,
                                                               "crs": "GCJ-02"}}) == {
        "lng": 114.37, "lat": 30.54, "crs": "GCJ-02",
    }
    assert _point_record({"lng": True, "lat": 30.53}) == {}


def test_model_input_must_keep_the_complete_accepted_conversation_in_order():
    history = [
        {"role": "user", "content": "从信图去总图附近吃饭"},
        {"role": "assistant", "content": "总图附近可以看看这几家。"},
        {"role": "user", "content": "换几家"},
    ]
    messages = [
        {"role": "system", "content": "遵循任务和回复规则"},
        *history,
        {"role": "user", "content": "扩大范围"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
    ]

    assert _accepted_history_present(messages, history, "扩大范围")
    assert not _accepted_history_present([messages[0], *history[1:]], history, "扩大范围")
    reordered = [messages[0], history[1], history[0], history[2], *messages[4:]]
    assert not _accepted_history_present(reordered, history, "扩大范围")


def test_accepted_model_messages_count_only_chat_text_and_include_current_query():
    history = [
        {"role": "user", "content": "从信图去总图附近吃饭"},
        {"role": "assistant", "content": "总图附近可以看看这几家。"},
        {"role": "tool", "content": "内部工具结果"},
        {"role": "user", "content": None},
    ]

    assert _accepted_model_messages(history, "扩大范围") == [
        {"role": "user", "content": "从信图去总图附近吃饭"},
        {"role": "assistant", "content": "总图附近可以看看这几家。"},
        {"role": "user", "content": "扩大范围"},
    ]


def test_acceptance_failure_trace_records_locations_without_exception_text():
    try:
        raise RuntimeError("sensitive message content")
    except RuntimeError as error:
        frames = _failure_frames(error)

    assert frames[-1]["function"] == "test_acceptance_failure_trace_records_locations_without_exception_text"
    assert all(set(frame) == {"file", "line", "function"} for frame in frames)
    assert "sensitive message content" not in repr(frames)


def test_restaurant_intent_can_be_preserved_as_activity_or_place_category():
    assert _has_restaurant_intent({"activities": ["restaurant_meal"]})
    assert _has_restaurant_intent({"subcategories": ["restaurant"]})
    assert _has_restaurant_intent({"subcategories": ["fastfood"]})
    assert not _has_restaurant_intent({"activities": [], "subcategories": ["library"]})


def test_browser_acceptance_reads_the_playwright_response_status_property():
    source_path = Path(__file__).resolve().parents[1] / "scripts" / "model_conversation_smoke.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    accepted_response_properties = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "accepted_response"
    }

    assert "status" in accepted_response_properties
    assert "status_code" not in accepted_response_properties
