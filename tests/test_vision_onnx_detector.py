import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vision.engine import OnnxDetector, VisionConfigurationError


class FakeSession:
    def __init__(self, outputs=None, input_shape=(1, 3, 960, 960)):
        self.outputs = outputs or [
            np.asarray([[3, 5]]),
            np.asarray([[[240, 120, 720, 600], [-10, 200, 40, 300]]], dtype=np.float32),
            np.asarray([[0.9, 0.1]], dtype=np.float32),
        ]
        self.feed = None
        self.input_shape = input_shape

    def get_inputs(self):
        return [
            SimpleNamespace(name="images", shape=self.input_shape),
            SimpleNamespace(name="orig_target_sizes", shape=(1, 2)),
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(name="labels"),
            SimpleNamespace(name="boxes"),
            SimpleNamespace(name="scores"),
        ]

    def run(self, output_names, input_feed):
        self.feed = input_feed
        return self.outputs


def test_onnx_detector_preprocesses_rgb_letterbox_and_restores_boxes():
    session = FakeSession()
    detector = OnnxDetector("model.onnx", session=session, confidence_threshold=0.25)
    # OpenCV camera frames are BGR. The blue source pixel moves to RGB channel 2.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[0, 0] = [255, 0, 0]

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0]["label"] == "car"
    assert detections[0]["confidence"] == pytest.approx(0.9)
    assert detections[0]["box"] == pytest.approx([160, 0, 480, 320])
    assert session.feed["images"].shape == (1, 3, 960, 960)
    assert session.feed["images"][0, 0, 120, 0] == pytest.approx(0.0)
    assert session.feed["images"][0, 2, 120, 0] == pytest.approx(1.0)
    assert session.feed["orig_target_sizes"].tolist() == [[960, 960]]


def test_onnx_detector_records_configured_cpu_thread_count(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fixture")
    created = {}

    class FakeSessionOptions:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0

    def make_session(path, *, sess_options, providers):
        created["path"] = path
        created["options"] = sess_options
        created["providers"] = providers
        return FakeSession()

    monkeypatch.setenv("VISION_CPU_THREADS", "4")
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=make_session,
    ))

    detector = OnnxDetector(str(model_path))

    assert detector.cpu_threads == 4
    assert created["options"].intra_op_num_threads == 4
    assert created["providers"] == ["CPUExecutionProvider"]


def test_onnx_detector_returns_empty_for_empty_valid_outputs():
    session = FakeSession(outputs=[
        np.empty((1, 0), dtype=np.int64),
        np.empty((1, 0, 4), dtype=np.float32),
        np.empty((1, 0), dtype=np.float32),
    ])
    assert OnnxDetector("model.onnx", session=session).detect(np.zeros((100, 100, 3), dtype=np.uint8)) == []


def test_onnx_detector_exposes_low_confidence_detections_only_for_tracking():
    session = FakeSession(outputs=[
        np.asarray([[3, 3, 3]]),
        np.asarray([[[100, 100, 180, 180], [200, 100, 280, 180], [300, 100, 380, 180]]], dtype=np.float32),
        np.asarray([[0.8, 0.2, 0.05]], dtype=np.float32),
    ])
    detector = OnnxDetector(
        "model.onnx", session=session, confidence_threshold=0.4,
        tracking_low_confidence_threshold=0.1,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert [item["confidence"] for item in detector.detect(frame)] == [0.8]
    assert [item["confidence"] for item in detector.detect(frame, tracking=True)] == [0.8, 0.2]


def test_onnx_detector_uses_annotated_crop_and_restores_full_frame_coordinates():
    session = FakeSession()
    detector = OnnxDetector("model.onnx", session=session, confidence_threshold=0.25)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect_regions(frame, [{
        "id": "lane", "kind": "vehicle_lane",
        "polygon": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
    }])

    assert len(detections) == 1
    assert detections[0]["box"] == pytest.approx([240, 120, 400, 280])
    assert session.feed["images"].shape == (1, 3, 960, 960)


@pytest.mark.parametrize("regions", [
    [],
    [{"id": "exclude-only", "kind": "exclude",
      "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
])
def test_onnx_detector_does_not_fall_back_to_full_frame_when_no_inference_region(regions):
    session = FakeSession()
    detector = OnnxDetector("model.onnx", session=session, confidence_threshold=0.25)

    with pytest.raises(ValueError, match="排除区域不能单独用于识别"):
        detector.detect_regions(np.zeros((480, 640, 3), dtype=np.uint8), regions)

    assert session.feed is None


def test_onnx_detector_tiles_large_observation_regions_and_merges_seam_duplicates():
    class TiledSession(FakeSession):
        def __init__(self):
            super().__init__()
            self.feeds = []

        def run(self, output_names, input_feed):
            self.feeds.append(input_feed)
            outputs = [
                ([3], [[640, 80, 704, 144]], [0.90]),
                ([3], [[160, 80, 224, 144]], [0.85]),
                ([3], [[720, 80, 768, 128]], [0.88]),
            ][len(self.feeds) - 1]
            labels, boxes, scores = outputs
            return [
                np.asarray([labels]),
                np.asarray([boxes], dtype=np.float32).reshape(1, len(boxes), 4),
                np.asarray([scores], dtype=np.float32),
            ]

    session = TiledSession()
    detector = OnnxDetector("model.onnx", session=session, confidence_threshold=0.25)
    frame = np.zeros((1200, 2400, 3), dtype=np.uint8)

    detections = detector.detect_regions(frame, [{
        "id": "lane", "kind": "vehicle_lane",
        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
    }])

    assert len(session.feeds) == 3, "large crops should be covered by overlapping higher-detail tiles"
    assert all(feed["images"].shape == (1, 3, 960, 960) for feed in session.feeds)
    assert len(detections) == 2, "the same vehicle detected on an overlap seam should appear once"
    assert detections[0]["box"] == pytest.approx([800, 100, 880, 180])
    assert detections[1]["box"] == pytest.approx([2100, 100, 2160, 160])


def test_dynamic_batch_model_batches_roi_tiles_and_restores_each_tile_coordinates(monkeypatch):
    class DynamicBatchSession(FakeSession):
        def __init__(self):
            super().__init__(input_shape=("N", 3, 960, 960))
            self.feeds = []

        def run(self, output_names, input_feed):
            self.feeds.append(input_feed)
            count = input_feed["images"].shape[0]
            return [
                np.tile(np.asarray([[3]], dtype=np.int64), (count, 1)),
                np.tile(np.asarray([[[80, 80, 160, 160]]], dtype=np.float32), (count, 1, 1)),
                np.tile(np.asarray([[0.9]], dtype=np.float32), (count, 1)),
            ]

    monkeypatch.setenv("VISION_TILE_BATCH_SIZE", "2")
    session = DynamicBatchSession()
    detector = OnnxDetector("model.onnx", session=session, confidence_threshold=0.25)
    frame = np.zeros((1200, 3000, 3), dtype=np.uint8)

    detections = detector.detect_regions(frame, [{
        "id": "lane", "kind": "vehicle_lane",
        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
    }])

    assert [feed["images"].shape[0] for feed in session.feeds] == [2, 1]
    assert all(feed["images"].shape[1:] == (3, 960, 960) for feed in session.feeds)
    assert [feed["orig_target_sizes"].tolist() for feed in session.feeds] == [
        [[960, 960], [960, 960]], [[960, 960]],
    ]
    np.testing.assert_allclose([item["box"] for item in detections], [
        [100, 100, 200, 200], [1000, 100, 1100, 200], [1900, 100, 2000, 200],
    ])


def test_dynamic_batch_model_defaults_to_serial_tiles(monkeypatch):
    monkeypatch.delenv("VISION_TILE_BATCH_SIZE", raising=False)
    detector = OnnxDetector(
        "model.onnx", session=FakeSession(input_shape=("N", 3, 960, 960)),
    )

    assert detector.tile_batch_size == 1


@pytest.mark.parametrize("batch_size", ["0", "5", "not-a-number"])
def test_dynamic_batch_model_rejects_unsafe_batch_configuration(monkeypatch, batch_size):
    monkeypatch.setenv("VISION_TILE_BATCH_SIZE", batch_size)

    with pytest.raises(VisionConfigurationError, match="1 到 4 之间的整数"):
        OnnxDetector("model.onnx", session=FakeSession(input_shape=("N", 3, 960, 960)))


def test_static_batch_one_model_keeps_roi_inference_serial(monkeypatch):
    monkeypatch.setenv("VISION_TILE_BATCH_SIZE", "4")

    class SerialSession(FakeSession):
        def __init__(self):
            super().__init__()
            self.feeds = []

        def run(self, output_names, input_feed):
            self.feeds.append(input_feed)
            return super().run(output_names, input_feed)

    session = SerialSession()
    detector = OnnxDetector("model.onnx", session=session, confidence_threshold=0.25)
    detector.detect_regions(np.zeros((1200, 2400, 3), dtype=np.uint8), [{
        "id": "lane", "kind": "vehicle_lane",
        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
    }])

    assert len(session.feeds) == 3
    assert all(feed["images"].shape == (1, 3, 960, 960) for feed in session.feeds)


def test_static_multi_item_batch_model_is_rejected():
    with pytest.raises(VisionConfigurationError, match="批次必须为 1 或动态维度"):
        OnnxDetector("model.onnx", session=FakeSession(input_shape=(2, 3, 960, 960)))


def test_onnx_detector_rejects_model_with_unexpected_signature():
    session = FakeSession(input_shape=(1, 3, "height", "width"))
    with pytest.raises(VisionConfigurationError, match="静态方形输入"):
        OnnxDetector("model.onnx", session=session)


def test_onnx_detector_rejects_inconsistent_model_output_shapes():
    session = FakeSession(outputs=[
        np.asarray([[3, 5]]),
        np.asarray([[[240, 120, 720, 600]]], dtype=np.float32),
        np.asarray([[0.9, 0.1]], dtype=np.float32),
    ])
    detector = OnnxDetector("model.onnx", session=session)
    with pytest.raises(VisionConfigurationError, match="输出维度"):
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
