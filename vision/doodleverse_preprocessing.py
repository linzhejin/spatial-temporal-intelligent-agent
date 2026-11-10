"""Reference-compatible input and output handling for FloodNet Res-UNet v7."""
from __future__ import annotations


INPUT_HEIGHT = 768
INPUT_WIDTH = 1024
CLASS_COUNT = 10
FLOODED_ROAD_CLASS = 3


def prepare_doodleverse_rgb(frame_rgb):
    """Resize and standardize one uint8 RGB image as the official pipeline does."""
    import math

    import numpy as np
    from skimage.transform import resize

    frame = np.asarray(frame_rgb)
    if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
            or frame.shape[0] < 1 or frame.shape[1] < 1):
        raise ValueError("Doodleverse 输入必须是 uint8 RGB 图片")
    # Resize each channel independently. The official resize is separable and
    # keeps the channel axis unchanged; this avoids three full-size float64
    # channels existing simultaneously for large source photographs.
    resized = np.empty((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
    for channel in range(3):
        resized[:, :, channel] = resize(
            frame[:, :, channel], (INPUT_HEIGHT, INPUT_WIDTH), preserve_range=True, clip=True,
        ).astype(np.uint8)
    pixels = float(resized.shape[0] * resized.shape[1])
    std_floor = 1.0 / math.sqrt(pixels)
    mean = float(np.mean(resized))
    std = max(float(np.std(resized)), std_floor)
    standardized = (resized.astype(np.float64) - mean) / std
    return standardized[None, ...].astype(np.float32)


def extract_flooded_road_probability(raw_output):
    """Validate the model's softmax output and return class-3 probability map."""
    import numpy as np

    values = np.asarray(raw_output, dtype=np.float32)
    expected = (1, INPUT_HEIGHT, INPUT_WIDTH, CLASS_COUNT)
    if values.shape != expected:
        raise ValueError(f"Doodleverse v7 概率输出形状错误：{values.shape}")
    if (not np.isfinite(values).all() or float(values.min()) < -1e-6
            or float(values.max()) > 1.000001):
        raise ValueError("Doodleverse v7 返回了无效概率，不会执行 softmax 修补")
    class_sums = values.sum(axis=-1, dtype=np.float32)
    if not 0.999 <= float(class_sums.mean()) <= 1.001:
        raise ValueError("Doodleverse v7 输出未通过 softmax 概率和校验")
    return values[0, :, :, FLOODED_ROAD_CLASS]
