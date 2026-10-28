from agents.explainer import _build_template_explanation


def test_route_explanation_is_compact_and_keeps_route_specific_reasons():
    text = _build_template_explanation(
        {
            "mode": "walk", "duration_min": 12, "distance_m": 1250,
            "shortest_distance_m": 820,
            "pois": [{"name": "樱顶"}, {"name": "老图书馆"}],
            "filter_status": "filtered",
            "filter_info": {"removed_edges": 2},
        },
        {"slope": "avoid", "scenery": "high"},
        None,
        weight_source="explicit_nl",
    )
    assert len(text) <= 90
    assert text.endswith("。")
    assert "偏好" in text
    assert "陡坡" in text
    assert "景观" in text
    assert "樱顶" in text


def test_relaxed_slope_is_reported_without_claiming_all_hills_were_avoided():
    text = _build_template_explanation(
        {"mode": "walk", "duration_min": 8, "filter_status": "degraded_slope"},
        {"slope": "avoid", "scenery": "normal"},
        None,
        weight_source="shortcut",
    )
    assert len(text) <= 90
    assert "放宽" in text
    assert "避开陡坡" not in text


def test_limited_road_annotations_are_disclosed_concisely():
    text = _build_template_explanation(
        {"mode": "walk", "filter_status": "degraded_annotations"},
        {"slope": "normal", "scenery": "normal"},
        None,
        weight_source="default",
    )
    assert len(text) <= 90
    assert "标注有限" in text


def test_step_avoidance_explains_unverified_edges():
    text = _build_template_explanation(
        {"mode": "walk", "filter_status": "steps_avoid+steps_unverified"},
        {"avoid_steps": True},
        None,
    )

    assert len(text) <= 90
    assert "已避开路网中已标注的台阶" in text
    assert "其余路段是否有台阶尚未核实" in text


def test_global_closure_is_not_described_as_a_cause_of_this_route():
    text = _build_template_explanation(
        {"mode": "walk", "filter_status": "road_closure"}, {}, None,
    )
    assert "规划时考虑了当前封闭路段" in text
    assert "因封闭绕行" not in text
