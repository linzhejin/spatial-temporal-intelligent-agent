from scripts.model_conversation_smoke import _accepted_history_present, _check_user_copy, _point_record


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
