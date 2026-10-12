from vision import evaluation


def _evaluator():
    evaluator = getattr(evaluation, "evaluate_head_point_frame", None)
    assert callable(evaluator), "head-point evaluator must be implemented"
    return evaluator


def test_person_detection_box_matches_at_most_one_head_center():
    result = _evaluator()(
        [
            {"label": "pedestrian", "confidence": 0.9, "box": [0, 0, 50, 100]},
            {"label": "people", "confidence": 0.8, "box": [20, 0, 60, 100]},
            {"label": "car", "confidence": 0.99, "box": [0, 0, 50, 100]},
        ],
        [(25, 20)],
        confidence_threshold=0.3,
    )

    assert result == {"tp": 1, "fp": 1, "fn": 0, "predicted": 2, "ground_truth": 1}


def test_head_center_matching_uses_maximum_one_to_one_assignment():
    result = _evaluator()(
        [
            {"label": "pedestrian", "confidence": 0.9, "box": [0, 0, 100, 100]},
            {"label": "pedestrian", "confidence": 0.8, "box": [0, 0, 20, 20]},
        ],
        [(10, 10), (80, 80)],
    )

    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_moving_drone_rows_map_zero_based_frame_to_head_centers():
    parse = getattr(evaluation, "parse_moving_drone_rows", None)
    assert callable(parse), "MovingDroneCrowd annotation parser must be implemented"

    heads = parse(
        "0,7,10,20,4,6,-1,-1,-1,-1\n"
        "1,8,30,40,8,10,-1,-1,-1,-1\n",
        frame_index=0,
        image_width=100,
        image_height=100,
    )

    assert heads == [(12.0, 23.0)]


def test_moving_drone_rows_reject_nonpositive_head_boxes():
    parse = getattr(evaluation, "parse_moving_drone_rows", None)
    assert callable(parse), "MovingDroneCrowd annotation parser must be implemented"

    import pytest

    with pytest.raises(ValueError, match="annotation row"):
        parse("0,7,10,20,0,6,-1,-1,-1,-1", frame_index=0,
              image_width=100, image_height=100)


def test_moving_drone_rows_keep_edge_clipped_box_when_head_center_is_visible():
    parse = getattr(evaluation, "parse_moving_drone_rows", None)
    assert callable(parse), "MovingDroneCrowd annotation parser must be implemented"

    heads = parse(
        "0,7,96,10,6,6,-1,-1,-1,-1",
        frame_index=0,
        image_width=100,
        image_height=100,
    )

    assert heads == [(99.0, 13.0)]


def test_moving_drone_rows_exclude_head_center_outside_visible_image():
    parse = getattr(evaluation, "parse_moving_drone_rows", None)
    assert callable(parse), "MovingDroneCrowd annotation parser must be implemented"

    heads = parse(
        "0,7,98,10,6,6,-1,-1,-1,-1",
        frame_index=0,
        image_width=100,
        image_height=100,
    )

    assert heads == []
