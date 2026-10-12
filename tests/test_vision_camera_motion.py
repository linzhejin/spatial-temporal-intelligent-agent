import cv2
import numpy as np
import pytest

from vision.engine import (
    _estimate_camera_transform, _gray_for_motion, _relative_camera_transform,
)


def test_background_motion_estimator_recovers_a_known_image_translation():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    random = np.random.default_rng(127)
    for _ in range(180):
        x = int(random.integers(20, 620))
        y = int(random.integers(20, 460))
        radius = int(random.integers(3, 14))
        color = tuple(int(value) for value in random.integers(90, 255, size=3))
        cv2.circle(image, (x, y), radius, color, -1)
    matrix = np.asarray([[1, 0, 13], [0, 1, -7]], dtype=np.float32)
    moved = cv2.warpAffine(image, matrix, (640, 480))
    previous_gray, scale = _gray_for_motion(image, cv2)
    current_gray, current_scale = _gray_for_motion(moved, cv2)

    estimated = _estimate_camera_transform(previous_gray, current_gray, cv2, current_scale)

    assert scale == current_scale == 1.0
    assert estimated is not None
    assert estimated[0][2] == pytest.approx(13, abs=2)
    assert estimated[1][2] == pytest.approx(-7, abs=2)


def test_anchor_frame_transforms_derive_the_correct_adjacent_frame_motion():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    random = np.random.default_rng(284)
    for _ in range(180):
        x = int(random.integers(20, 580))
        y = int(random.integers(20, 460))
        radius = int(random.integers(3, 14))
        color = tuple(int(value) for value in random.integers(90, 255, size=3))
        cv2.circle(image, (x, y), radius, color, -1)
    first = cv2.warpAffine(image, np.asarray([[1, 0, 12], [0, 1, -3]], dtype=np.float32),
                           (640, 480))
    second = cv2.warpAffine(image, np.asarray([[1, 0, 26], [0, 1, -5]], dtype=np.float32),
                            (640, 480))
    anchor_gray, _ = _gray_for_motion(image, cv2)
    first_gray, scale = _gray_for_motion(first, cv2)
    second_gray, _ = _gray_for_motion(second, cv2)
    anchor_to_first = _estimate_camera_transform(anchor_gray, first_gray, cv2, scale)
    anchor_to_second = _estimate_camera_transform(anchor_gray, second_gray, cv2, scale)

    adjacent = _relative_camera_transform(anchor_to_first, anchor_to_second)

    assert adjacent is not None
    assert adjacent[0][2] == pytest.approx(14, abs=3)
    assert adjacent[1][2] == pytest.approx(-2, abs=3)
