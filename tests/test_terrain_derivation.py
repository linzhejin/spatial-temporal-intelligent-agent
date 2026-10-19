import math

import pytest

from scripts.network.derive_terrain_attributes import (
    detect_facility_type,
    sample_profile,
    terrain_candidate,
)


def test_direction_reverses_grade_and_gain_loss():
    forward = terrain_candidate(
        [100.0, 102.0, 106.0], length_m=100.0, resolution_m=10.0
    )
    reverse = terrain_candidate(
        [106.0, 102.0, 100.0], length_m=100.0, resolution_m=10.0
    )
    assert forward["grade_signed_pct"] == pytest.approx(6.0)
    assert reverse["grade_signed_pct"] == pytest.approx(-6.0)
    assert forward["elevation_gain_m"] == reverse["elevation_loss_m"]
    assert forward["elevation_loss_m"] == reverse["elevation_gain_m"]


@pytest.mark.parametrize("length,confidence", [(60.0, "low"), (89.9, "low"), (90.0, "medium")])
def test_30m_dem_confidence_boundary(length, confidence):
    result = terrain_candidate([100.0, 104.0], length_m=length, resolution_m=30.0)
    assert result["confidence"] == confidence


def test_steps_remain_explicit_facility_fact():
    result = terrain_candidate(
        [100.0, 102.0], length_m=40.0, resolution_m=5.0,
        facility_type="steps",
    )
    assert result["facility_type"] == "steps"
    assert detect_facility_type({"highway": "steps"}) == "steps"
    assert detect_facility_type({"bridge": "yes"}) == "bridge"


def test_long_nodata_run_stays_unknown_instead_of_bridging_gap():
    result = terrain_candidate(
        [100.0, None, None, None, 110.0], length_m=100.0, resolution_m=10.0
    )
    assert result["slope_level"] is None
    assert result["confidence"] == "unknown"


class _FakeCRS:
    def to_string(self):
        return "EPSG:4326"


class _FakeDataset:
    crs = _FakeCRS()
    nodata = -9999.0

    def sample(self, coords):
        for x, y in coords:
            yield [100.0 + (x - 114.0) * 10000.0]


def test_sample_profile_uses_complete_geometry_and_spacing():
    profile = sample_profile(
        _FakeDataset(), [(114.0, 30.0), (114.001, 30.0)], spacing_m=20.0
    )
    assert len(profile) >= 5
    assert profile[0] == pytest.approx(100.0)
    assert profile[-1] == pytest.approx(110.0)
    assert all(math.isfinite(value) for value in profile if value is not None)


class _ProjectedDataset:
    crs = "EPSG:32650"
    nodata = None

    def __init__(self):
        self.coords = []

    def sample(self, coords):
        self.coords = list(coords)
        for _ in self.coords:
            yield [100.0]


def test_sample_profile_reprojects_wgs84_geometry_for_projected_dem():
    dataset = _ProjectedDataset()
    profile = sample_profile(
        dataset, [(114.0, 30.0), (114.001, 30.0)], spacing_m=30.0
    )

    assert len(profile) >= 4
    assert all(x > 100_000 and y > 1_000_000 for x, y in dataset.coords)
