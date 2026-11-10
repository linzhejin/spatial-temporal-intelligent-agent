from types import SimpleNamespace

import numpy as np
import pytest

from vision.analysis import build_specialized_candidates
from vision.specialized import DoodleverseFloodSegmenter


class FakeDoodleverseSession:
    def __init__(self, output, input_shape=(1, 768, 1024, 3)):
        self.output = output
        self.input_shape = input_shape
        self.feed = None
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="rgb", shape=self.input_shape)]

    def get_outputs(self):
        return [SimpleNamespace(name="probabilities", shape=list(self.output.shape))]

    def run(self, _names, feed):
        self.calls += 1
        self.feed = feed
        return [self.output]


def _region(region_id, x1, x2):
    return {
        "id": region_id,
        "kind": "road_surface",
        "polygon": [[x1, 0], [x2, 0], [x2, 1], [x1, 1]],
    }


def test_doodleverse_uses_raw_flood_probability_once_for_all_road_regions():
    output = np.zeros((1, 768, 1024, 10), dtype=np.float32)
    output[..., 0] = 1.0
    output[:, :, :513, 0] = 0.999
    output[:, :, :513, 3] = 0.001
    session = FakeDoodleverseSession(output)
    segmenter = DoodleverseFloodSegmenter(
        "flood-v7.onnx", session=session, probability_threshold=0.001,
        min_area_ratio=0.08,
    )
    frame = np.zeros((768, 1024, 3), dtype=np.uint8)

    predictions = segmenter.predict_regions(frame, [
        _region("west", 0.0, 0.5), _region("east", 0.5, 1.0),
    ])

    assert session.calls == 1
    assert set(predictions) == {"west", "east"}
    assert predictions["west"]["flooded_road_area_ratio"] == pytest.approx(1.0)
    assert predictions["east"]["flooded_road_area_ratio"] == pytest.approx(0.001953)
    assert predictions["west"]["flood_probability_threshold"] == pytest.approx(0.001)
    assert predictions["west"]["outline_polygons"]
    assert np.all(session.feed["rgb"][0, :, 0, 0] == 0)


def test_doodleverse_rejects_non_probability_output_instead_of_softmaxing_it():
    output = np.zeros((1, 768, 1024, 10), dtype=np.float32)
    output[..., 3] = 1.2
    session = FakeDoodleverseSession(output)
    segmenter = DoodleverseFloodSegmenter("flood-v7.onnx", session=session)

    with pytest.raises(ValueError, match="概率"):
        segmenter.predict_regions(np.zeros((768, 1024, 3), dtype=np.uint8), [
            _region("road", 0.0, 1.0),
        ])


def test_doodleverse_prepares_large_image_before_loading_model(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixture")
    segmenter = DoodleverseFloodSegmenter(str(model))
    calls = []
    output = np.zeros((1, 768, 1024, 10), dtype=np.float32)
    output[..., 0] = 1.0
    session = FakeDoodleverseSession(output)

    def prepare(_frame):
        calls.append("prepare")
        return np.zeros((1, 768, 1024, 3), dtype=np.float32)

    def load():
        calls.append("load")
        segmenter._validate_session(session)
        return session

    monkeypatch.setattr(segmenter, "_prepare_rgb", prepare)
    monkeypatch.setattr(segmenter, "_load_session", load)
    segmenter._predict_probabilities(np.zeros((10, 10, 3), dtype=np.uint8))
    assert calls == ["prepare", "load"]


def test_image_child_releases_model_before_resizing_probability_map(monkeypatch, tmp_path):
    from vision.assisted_image_child import _InProcessDoodleverse

    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixture")
    segmenter = _InProcessDoodleverse(str(model))
    segmenter.session = object()
    original = np.ones((2, 2), dtype=np.float32)
    monkeypatch.setattr(segmenter, "_predict_probabilities", lambda _frame: original)

    def finish(_frame, _regions, probabilities):
        assert segmenter.session is None
        assert not np.shares_memory(probabilities, original)
        return {"road": {"flooded_road_area_ratio": 1.0}}

    monkeypatch.setattr(segmenter, "_predictions_from_probability", finish)
    assert segmenter.predict_regions(np.zeros((10, 10, 3), dtype=np.uint8), [
        {"id": "road", "kind": "road_surface"},
    ])["road"]["flooded_road_area_ratio"] == 1.0


def test_assisted_flood_converts_the_decoded_image_in_place(monkeypatch, tmp_path):
    import cv2
    from vision.engine import analyze_media

    path = tmp_path / "color.png"
    frame = np.full((16, 16, 3), [10, 20, 30], dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)
    calls = []
    original = cv2.cvtColor

    def checked_color(source, code, *args, **kwargs):
        if code == cv2.COLOR_BGR2RGB:
            calls.append(kwargs.get("dst") is source)
        return original(source, code, *args, **kwargs)

    class Detector:
        def detect_regions(self, *_args, **_kwargs):
            return []

    class Flood:
        def predict_regions(self, rgb, _regions, **_kwargs):
            assert rgb[0, 0].tolist() == [30, 20, 10]
            return {}

    monkeypatch.setattr(cv2, "cvtColor", checked_color)
    analyze_media(path, "image", None, detector=Detector(), flood_segmenter=Flood(),
                  regions=[{"id": "road", "kind": "road_surface",
                            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
                  require_flood_success=True, inplace_flood_rgb=True)
    assert calls == [True]


def test_doodleverse_rejects_non_road_regions_without_running_model():
    output = np.zeros((1, 768, 1024, 10), dtype=np.float32)
    session = FakeDoodleverseSession(output)
    segmenter = DoodleverseFloodSegmenter("flood-v7.onnx", session=session)

    assert segmenter.predict_regions(np.zeros((768, 1024, 3), dtype=np.uint8), [
        {"id": "yard", "kind": "pedestrian", "polygon": [[0, 0], [1, 0], [1, 1]]},
    ]) == {}
    assert session.calls == 0


def test_doodleverse_preprocessing_rejects_empty_and_non_rgb_images():
    with pytest.raises(ValueError, match="uint8 RGB"):
        DoodleverseFloodSegmenter._prepare_rgb(np.zeros((0, 10, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8 RGB"):
        DoodleverseFloodSegmenter._prepare_rgb(np.zeros((10, 10), dtype=np.uint8))


@pytest.mark.parametrize("shape", [(73, 101, 3), (769, 1025, 3), (1200, 1601, 3)])
def test_doodleverse_preprocessing_preserves_official_resize_and_normalization(shape):
    from skimage.transform import resize

    frame = np.random.default_rng(42).integers(0, 256, size=shape, dtype=np.uint8)
    resized = resize(frame, (768, 1024, 3), preserve_range=True, clip=True).astype(np.uint8)
    std = max(float(np.std(resized)), 1 / np.sqrt(768 * 1024))
    expected = ((resized.astype(np.float64) - float(np.mean(resized))) / std)[None].astype(np.float32)
    assert np.array_equal(DoodleverseFloodSegmenter._prepare_rgb(frame), expected)


def test_assisted_flood_candidate_uses_eight_percent_single_image_threshold():
    observation = [{
        "time_seconds": 0.0,
        "region_id": "road",
        "flooded_road_area_ratio": 0.08,
        "water_evidence_type": "flooded_road",
        "outline_polygons": [[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]],
    }]

    candidates = build_specialized_candidates(
        observation, sample_interval_s=1.0, single_image=True,
        flood_area_threshold=0.08, flood_single_image_threshold=0.08,
    )

    assert [item["kind"] for item in candidates] == ["possible_flooding"]
