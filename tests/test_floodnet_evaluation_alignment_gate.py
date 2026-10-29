"""Pixel metrics must not be computed from guessed image/mask registration."""
import json
import sys

import pytest
from PIL import Image

from scripts.vision import calibrate_floodnet_threshold, evaluate_specialized


def test_matching_default_geometry_remains_readable_without_claiming_alignment(tmp_path):
    image, mask = tmp_path/'image.png', tmp_path/'mask.png'
    Image.new('RGB', (8, 8)).save(image)
    Image.new('L', (8, 8), color=3).save(mask)
    frame, labels, status = evaluate_specialized._read_flood_pair(image, mask)
    assert frame.shape == (8, 8, 3) and labels.shape == (8, 8)
    assert status == 'matching_dimensions'


@pytest.mark.parametrize('defect', ['dimensions', 'image_orientation', 'mask_orientation'])
@pytest.mark.parametrize('entry', ['evaluation', 'visible_water', 'calibration'])
def test_both_flood_entry_points_reject_unresolved_geometry(tmp_path, monkeypatch, defect, entry):
    image_exif, mask_exif = Image.Exif(), Image.Exif()
    image_exif[274] = 3 if defect == 'image_orientation' else 1
    mask_exif[274] = 3 if defect == 'mask_orientation' else 1
    image_size = (12, 8) if defect == 'dimensions' else (8, 8)
    Image.new('RGB', image_size).save(tmp_path / 'image.png', exif=image_exif)
    Image.new('L', (8, 8)).save(tmp_path / 'mask.png', exif=mask_exif)
    sample = {'image': 'image.png', 'mask': 'mask.png', 'group_id': 'v1', 'split': 'validation',
              'road_surface_polygon': [[0, 0], [1, 0], [1, 1], [0, 1]], 'flooded_road_class_id': 3}

    class NoInference:
        is_binary_visible_water = True

        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            pytest.fail('inference must not run on unverified image/mask geometry')

    expected = 'dimensions' if defect == 'dimensions' else 'orientation'
    if entry == 'evaluation':
        monkeypatch.setattr(evaluate_specialized, 'OnnxFloodSegmenter', NoInference)
        with pytest.raises(ValueError, match=expected):
            evaluate_specialized._evaluate_flood({'samples': [sample]}, tmp_path, tmp_path/'unused.onnx', 0.1)
    elif entry == 'visible_water':
        monkeypatch.setattr(evaluate_specialized, 'OnnxFloodSegmenter', NoInference)
        with pytest.raises(ValueError, match=expected):
            evaluate_specialized._evaluate_visible_water_road(
                {'samples': [sample]}, tmp_path, tmp_path/'unused.onnx', probability_threshold=0.5)
    else:
        monkeypatch.setattr(calibrate_floodnet_threshold, 'OnnxFloodSegmenter', NoInference)
        monkeypatch.setattr(calibrate_floodnet_threshold, 'ROOT', tmp_path/'other-repository')
        manifest = {'schema_version': 1, 'task': 'flooded_road_segmentation', 'dataset': 'test',
                    'split_unit': 'scene', 'split_groups': {'train': ['t1'], 'validation': ['v1'],
                                                            'test': ['x1']}, 'samples': [sample]}
        path = tmp_path / 'manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf8')
        model, output = tmp_path/'model.onnx', tmp_path/'report.json'
        model.write_bytes(b'dummy')
        monkeypatch.setattr(sys, 'argv', ['calibrate', '--dataset-root', str(tmp_path), '--manifest', str(path),
                                         '--model', str(model), '--output', str(output)])
        with pytest.raises(ValueError, match=expected):
            calibrate_floodnet_threshold.main()
        assert not output.exists()
