"""Ignored pixels must never become false positives in FloodNet metrics."""
import numpy as np
import pytest
from PIL import Image
import json
import sys

from scripts.vision import calibrate_floodnet_threshold, evaluate_specialized
from scripts.vision.evaluate_specialized import _flood_pixel_counts


def test_full_image_counts_only_valid_label_pixels():
    labels = np.array([[3, 255], [4, 255]], dtype=np.uint8)
    prediction = np.ones((2, 2), dtype=bool)
    assert _flood_pixel_counts(prediction, labels) == (1, 1, 0, 2)


def test_roi_excludes_unknown_and_otherwise_valid_pixels():
    labels = np.array([[3, 255], [4, 255]], dtype=np.uint8)
    prediction = np.ones((2, 2), dtype=bool)
    roi = np.array([[True, True], [False, False]])
    assert _flood_pixel_counts(prediction, labels, roi=roi) == (1, 0, 0, 1)


@pytest.mark.parametrize('roi', [None, np.array([[True, True], [False, False]])])
def test_all_unknown_region_has_no_metric(roi):
    labels = np.array([[255, 255], [4, 4]], dtype=np.uint8)
    if roi is None:
        labels[:] = 255
    with pytest.raises(ValueError, match='valid.*pixels'):
        _flood_pixel_counts(np.ones((2, 2), dtype=bool), labels, roi=roi)


@pytest.mark.parametrize('all_unknown', [False, True])
def test_evaluator_excludes_unknown_pixels_and_rejects_empty_valid_roi(tmp_path, monkeypatch, all_unknown):
    Image.new('RGB', (2, 2)).save(tmp_path/'image.png')
    labels = np.full((2, 2), 255, dtype=np.uint8)
    if not all_unknown:
        labels[0, 0], labels[1, 0] = 3, 4
    Image.fromarray(labels).save(tmp_path/'mask.png')
    sample = {'image': 'image.png', 'mask': 'mask.png', 'group_id': 'g',
              'road_surface_polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}

    class StubSegmenter:
        flood_probability_threshold = None

        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return {'flooded_road_mask': np.ones((2, 2), dtype=bool)}

    monkeypatch.setattr(evaluate_specialized, 'OnnxFloodSegmenter', StubSegmenter)
    if all_unknown:
        with pytest.raises(ValueError, match='valid.*pixels'):
            evaluate_specialized._evaluate_flood({'samples': [sample]}, tmp_path, tmp_path/'m.onnx', 0.1)
    else:
        report = evaluate_specialized._evaluate_flood({'samples': [sample]}, tmp_path, tmp_path/'m.onnx', 0.1)
        assert (report['tp'], report['fp'], report['fn']) == (1, 1, 0)
        assert report['valid_pixel_count'] == 2
        assert report['ignored_pixel_count'] == 2


@pytest.mark.parametrize('all_unknown', [False, True])
def test_calibration_uses_same_validity_rule_and_rejects_empty_roi(tmp_path, monkeypatch, all_unknown):
    Image.new('RGB', (2, 2)).save(tmp_path/'image.png')
    labels = np.full((2, 2), 255, dtype=np.uint8)
    if not all_unknown:
        labels[0, 0], labels[1, 0] = 3, 4
    Image.fromarray(labels).save(tmp_path/'mask.png')
    sample = {'image': 'image.png', 'mask': 'mask.png', 'group_id': 'v1', 'split': 'validation',
              'road_surface_polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}
    manifest = {'schema_version': 1, 'task': 'flooded_road_segmentation', 'dataset': 'test',
                'split_unit': 'scene', 'split_groups': {'train': ['t1'], 'validation': ['v1'],
                                                        'test': ['x1']}, 'samples': [sample]}
    manifest_file = tmp_path/'manifest.json'
    manifest_file.write_text(json.dumps(manifest), encoding='utf8')
    model, output = tmp_path/'model.onnx', tmp_path/'report.json'
    model.write_bytes(b'dummy')

    class StubSegmenter:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return {'flooded_road_probability': np.ones((2, 2))}

    captured = {}

    def threshold(**kwargs):
        captured.update(kwargs)
        return {'threshold': 0.5}

    monkeypatch.setattr(calibrate_floodnet_threshold, 'OnnxFloodSegmenter', StubSegmenter)
    monkeypatch.setattr(calibrate_floodnet_threshold, 'select_probability_threshold_from_histograms', threshold)
    monkeypatch.setattr(calibrate_floodnet_threshold, 'ROOT', tmp_path/'other-repository')
    monkeypatch.setattr(sys, 'argv', ['calibrate', '--dataset-root', str(tmp_path), '--manifest', str(manifest_file),
                                     '--model', str(model), '--output', str(output), '--bins', '16'])
    if all_unknown:
        with pytest.raises(ValueError, match='valid.*pixels'):
            calibrate_floodnet_threshold.main()
        assert not output.exists()
    else:
        assert calibrate_floodnet_threshold.main() == 0
        assert sum(captured['positive_bins']) == 1
        assert sum(captured['negative_bins']) == 1
        report = json.loads(output.read_text(encoding='utf8'))
        assert report['valid_pixel_count'] == 2
        assert report['ignored_pixel_count'] == 2
