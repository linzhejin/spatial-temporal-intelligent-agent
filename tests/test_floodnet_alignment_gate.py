"""Training may not guess how unaligned or oriented source files correspond."""
import pytest
from PIL import Image

from vision.floodnet_training import FloodNetSegmentationDataset


def make_pair(tmp_path, *, image_size=(8, 8), image_orientation=1, mask_orientation=1):
    image_exif, mask_exif = Image.Exif(), Image.Exif()
    image_exif[274], mask_exif[274] = image_orientation, mask_orientation
    Image.new('RGB', image_size).save(tmp_path / 'image.png', exif=image_exif)
    Image.new('L', (8, 8)).save(tmp_path / 'mask.png', exif=mask_exif)
    return {'image': 'image.png', 'mask': 'mask.png', 'group_id': 'g', 'split': 'train'}


def test_training_rejects_unmatched_dimensions(tmp_path):
    sample = make_pair(tmp_path, image_size=(12, 8))
    with pytest.raises(ValueError, match='dimensions|geometry'):
        FloodNetSegmentationDataset(tmp_path, [sample], tile_size=8, overlap=0)


@pytest.mark.parametrize(('image_orientation', 'mask_orientation'), [(3, 1), (1, 3)])
def test_training_rejects_unresolved_orientation(tmp_path, image_orientation, mask_orientation):
    sample = make_pair(tmp_path, image_orientation=image_orientation, mask_orientation=mask_orientation)
    with pytest.raises(ValueError, match='orientation'):
        FloodNetSegmentationDataset(tmp_path, [sample], tile_size=8, overlap=0)


def test_training_checks_replaced_source_before_loading(tmp_path):
    sample = make_pair(tmp_path)
    dataset = FloodNetSegmentationDataset(tmp_path, [sample], tile_size=8, overlap=0)
    make_pair(tmp_path, image_orientation=3)
    with pytest.raises(ValueError, match='orientation'):
        dataset[0]


def test_matching_geometry_does_not_claim_pixel_alignment(tmp_path):
    dataset = FloodNetSegmentationDataset(tmp_path, [make_pair(tmp_path)], tile_size=8, overlap=0)
    assert 'not verified' in dataset.alignment_report['rule']
