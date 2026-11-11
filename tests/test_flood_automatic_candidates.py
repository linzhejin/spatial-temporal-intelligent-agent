"""Automatic flood proposals must be model-derived and require human review."""

from types import SimpleNamespace

import numpy as np

from vision.analysis import build_specialized_candidates
from vision.specialized import DoodleverseFloodSegmenter


class _Session:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="rgb", shape=[1, 768, 1024, 3])]

    def get_outputs(self):
        return [SimpleNamespace(name="probabilities", shape=list(self.output.shape))]

    def run(self, _names, _feed):
        self.calls += 1
        return [self.output]


def _output():
    output = np.zeros((1, 768, 1024, 10), dtype=np.float32)
    output[..., 0] = 1.0
    return output


def test_automatic_proposals_use_flooded_road_model_class_not_water_or_whole_image():
    output = _output()
    # Two separate flooded-road regions, one dry road and one water body.
    for top, left in ((80, 80), (420, 700)):
        output[:, top:top + 80, left:left + 80, 0] = 0.1
        output[:, top:top + 80, left:left + 80, 3] = 0.9
    output[:, 180:300, 300:450, 0] = 0.1
    output[:, 180:300, 300:450, 4] = 0.9
    output[:, 500:650, 100:300, 0] = 0.099
    output[:, 500:650, 100:300, 3] = 0.001
    output[:, 500:650, 100:300, 5] = 0.9
    session = _Session(output)
    model = DoodleverseFloodSegmenter("injected.onnx", session=session)

    proposals = model.predict_automatic(np.zeros((512, 512, 3), dtype=np.uint8))

    assert session.calls == 1
    assert len(proposals) == 2
    for key, proposal in proposals.items():
        assert proposal["region_id"] == key
        assert proposal["region_origin"] == "automatic_model"
        assert proposal["outline_polygons"] == [proposal["region_polygon"]]
        assert len(proposal["region_polygon"]) >= 3
        assert all(0 <= value <= 1 for point in proposal["region_polygon"] for value in point)
        assert proposal["model_score"] > 0.8
        # A proposal is a localized model output, never a whole-image fake road.
        xs = [point[0] for point in proposal["region_polygon"]]
        ys = [point[1] for point in proposal["region_polygon"]]
        assert max(xs) - min(xs) < 0.2
        assert max(ys) - min(ys) < 0.2


def test_automatic_proposals_allow_no_detection_without_inventing_a_road():
    session = _Session(_output())
    model = DoodleverseFloodSegmenter("injected.onnx", session=session)

    assert model.predict_automatic(np.zeros((64, 64, 3), dtype=np.uint8)) == {}
    assert session.calls == 1


def test_automatic_proposals_respect_class_ties_and_reject_invalid_model_outputs():
    import pytest

    output = _output()
    output[:, 50:100, 50:100, 0] = 0.5
    output[:, 50:100, 50:100, 3] = 0.5
    model = DoodleverseFloodSegmenter("injected.onnx", session=_Session(output))
    assert model.predict_automatic(np.zeros((64, 64, 3), dtype=np.uint8)) == {}

    output[:, 50:100, 50:100, 3] = 1.5
    with pytest.raises(ValueError, match="概率"):
        model.predict_automatic(np.zeros((64, 64, 3), dtype=np.uint8))


def test_automatic_proposals_disclose_when_the_bounded_candidate_list_is_incomplete():
    output = _output()
    for row in range(5):
        for col in range(5):
            top, left = 30 + row * 80, 30 + col * 80
            output[:, top:top + 8, left:left + 8, 0] = 0.1
            output[:, top:top + 8, left:left + 8, 3] = 0.9
    model = DoodleverseFloodSegmenter("injected.onnx", session=_Session(output))

    proposals = model.predict_automatic(np.zeros((64, 64, 3), dtype=np.uint8))

    assert len(proposals) == 24
    assert all(item["automatic_candidates_truncated"] is True for item in proposals.values())


def test_automatic_proposal_becomes_an_individually_reviewable_candidate():
    polygon = [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]]
    observations = [{
        "time_seconds": 0.0,
        "region_id": "auto-flood-1",
        "region_origin": "automatic_model",
        "region_polygon": polygon,
        "outline_polygons": [polygon],
        "automatic_flood": True,
        "model_score": 0.9,
        "water_evidence_type": "flooded_road",
    }]

    candidates = build_specialized_candidates(observations, sample_interval_s=1.0, single_image=True)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["kind"] == "possible_flooding"
    assert candidate["review_required"] is True
    assert candidate["auto_publish"] is False
    assert candidate["evidence"]["summary"]["region_origin"] == "automatic_model"
    assert candidate["evidence"]["summary"]["region_polygon"] == polygon
    assert candidate["evidence"]["summary"]["outline_polygons"] == [polygon]


def test_image_analysis_runs_automatic_flood_without_user_regions(tmp_path):
    import cv2

    from vision.engine import analyze_media

    image = tmp_path / "public.png"
    assert cv2.imwrite(str(image), np.zeros((64, 96, 3), dtype=np.uint8))
    polygon = [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]]

    class Detector:
        def detect(self, _frame, *, tracking=False):
            return []

    class Flood:
        model_id = "test"
        model_version = "v1"

        def predict_automatic(self, _frame, *, cancel_check=None):
            return {"auto-flood-1": {
                "region_id": "auto-flood-1", "region_origin": "automatic_model",
                "region_polygon": polygon, "outline_polygons": [polygon],
                "automatic_flood": True, "model_score": 0.9,
                "water_evidence_type": "flooded_road",
            }}

    result = analyze_media(image, "image", None, detector=Detector(),
                           flood_segmenter=Flood(), regions=[],
                           automatic_flood=True, require_flood_success=True)

    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["evidence"]["summary"]["region_polygon"] == polygon
    assert result["metrics"]["specialized_model_frames"]["flood"] == 1


def test_automatic_no_detection_is_distinct_from_inference_failure(tmp_path):
    import cv2
    import pytest

    from vision.engine import analyze_media

    image = tmp_path / "empty.png"
    assert cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8))

    class Detector:
        def detect(self, _frame, *, tracking=False):
            return []

    class Flood:
        model_id = "test"
        model_version = "v1"

        def predict_automatic(self, _frame, *, cancel_check=None):
            return {}

    result = analyze_media(image, "image", None, detector=Detector(),
                           flood_segmenter=Flood(), automatic_flood=True,
                           require_flood_success=True)
    assert result["candidates"] == []
    assert result["metrics"]["specialized_model_frames"]["flood"] == 1
    assert result["safety"]["flood_segmentation_supported"] is True

    class Broken(Flood):
        def predict_automatic(self, _frame, *, cancel_check=None):
            raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="未完成"):
        analyze_media(image, "image", None, detector=Detector(),
                      flood_segmenter=Broken(), automatic_flood=True,
                      require_flood_success=True)
