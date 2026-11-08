"""Independent flat-ground ray truth, not a field calibration assessment."""
import math

import numpy as np
import pytest

from spatial.vision_footprint import pose_to_footprint
from spatial.vision_suggest import _pick_pose
from vision.registration import pose_to_homography_init


def pose(**changes):
    value = {
        "lng": 114.3623, "lat": 30.5363, "height_agl_m": 100.0,
        "gb_pitch_deg": -90.0, "gb_yaw_deg": 0.0, "gb_roll_deg": 0.0,
        "intrinsics": {"fx_px": 2000 / 3, "fy_px": 2000 / 3,
                       "cx_px": 500.0, "cy_px": 250.0},
    }
    value.update(changes)
    return value


def ground(H, x, y):
    result = H @ np.array([x, y, 1.0])
    return result[:2] / result[2]


def test_oblique_center_and_footprint_match_pinhole_truth():
    p = pose(gb_pitch_deg=-45.0)
    H = pose_to_homography_init(p, 1000, 500)["homography"]
    assert ground(H, 500, 250) == pytest.approx([0, 100], abs=1e-6)
    poly = pose_to_footprint(p, 1000, 500)["polygon"]
    half_fov = math.atan(250 / (2000 / 3))
    assert poly.bounds[1] == pytest.approx(100 * math.tan(math.pi / 4 - half_fov))
    assert poly.bounds[3] == pytest.approx(100 * math.tan(math.pi / 4 + half_fov))


def test_yaw_and_roll_change_ray_directions():
    H = pose_to_homography_init(pose(gb_pitch_deg=-45, gb_yaw_deg=90), 1000, 500)["homography"]
    assert ground(H, 500, 250) == pytest.approx([100, 0], abs=1e-6)
    H = pose_to_homography_init(pose(gb_roll_deg=90), 1000, 500)["homography"]
    assert ground(H, 600, 250) == pytest.approx([0, 15], abs=1e-6)


@pytest.mark.parametrize("change", [
    {"height_agl_m": None, "abs_alt_m": 100.0},
    {"height_agl_m": 0}, {"intrinsics": None}, {"gb_pitch_deg": None},
    {"gb_yaw_deg": None}, {"gb_roll_deg": None},
    {"gb_pitch_deg": -10.0},  # top of frame crosses horizon
])
def test_unusable_calibration_or_horizon_rejected(change):
    with pytest.raises(ValueError):
        pose_to_homography_init(pose(**change), 1000, 500)
    with pytest.raises(ValueError):
        pose_to_footprint(pose(**change), 1000, 500)


@pytest.mark.parametrize("time_seconds", [None, -0.01, 100, 5])
def test_mask_pose_must_be_timed_and_within_half_second(time_seconds):
    trajectory = {"trajectory": [dict(pose(), seconds=0), dict(pose(), seconds=10)]}
    with pytest.raises(ValueError):
        _pick_pose(trajectory, {"lng": 114.367, "lat": 30.534}, time_seconds=time_seconds,
                   require_time=True)


def test_mask_pose_accepts_half_second_match_inside_trajectory():
    trajectory = {"trajectory": [dict(pose(), seconds=0), dict(pose(), seconds=10)]}
    selected = _pick_pose(trajectory, {"lng": 114.367, "lat": 30.534},
                          time_seconds=9.5, require_time=True)
    assert selected["seconds"] == 10
