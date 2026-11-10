"""Two-stage assisted image analysis, shared by the worker and its child."""
from __future__ import annotations

import gc
import logging
import sys

import config
from vision.engine import OnnxDetector, VisionAnalysisCancelled, analyze_media

logger = logging.getLogger(__name__)


class _TrialNoVehicleDetector:
    model_id = "trial-no-vehicle-detection"
    model_version = "1"

    def detect(self, _frame, *, tracking=False):
        return []

    def detect_regions(self, _frame, _regions, *, tracking=False):
        return []


def _release_inference_memory():
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            logger.info("The allocator does not support explicit memory trimming")


def _merge_assisted_flood_results(traffic_result, flood_result):
    merged = dict(traffic_result)
    merged["candidates"] = [*(traffic_result.get("candidates") or []),
                            *(flood_result.get("candidates") or [])]
    merged["flood_review"] = list(flood_result.get("flood_review") or [])
    metrics = dict(traffic_result.get("metrics") or {})
    specialized_frames = dict(metrics.get("specialized_model_frames") or {})
    specialized_frames.update((flood_result.get("metrics") or {}).get("specialized_model_frames") or {})
    metrics["specialized_model_frames"] = specialized_frames
    merged["metrics"] = metrics
    safety = dict(traffic_result.get("safety") or {})
    flood_safety = flood_result.get("safety") or {}
    if "flood_segmentation_supported" in flood_safety:
        safety["flood_segmentation_supported"] = flood_safety["flood_segmentation_supported"]
    merged["safety"] = safety
    model = dict(traffic_result.get("model") or {})
    flood_model = (flood_result.get("model") or {}).get("flood_model")
    if flood_model is not None:
        model["flood_model"] = flood_model
    merged["model"] = model
    errors, seen = [], set()
    for item in [*(traffic_result.get("optional_model_errors") or []),
                 *(flood_result.get("optional_model_errors") or [])]:
        if isinstance(item, dict):
            key = (item.get("model"), item.get("error_type"))
            if key not in seen:
                seen.add(key)
                errors.append(item)
    merged["optional_model_errors"] = errors
    return merged


def _analyze_assisted_flood_image(media_path, media_kind, anchor_gcj, *,
                                  camera_stabilized, regions, progress_callback,
                                  resume_state, cancel_check, flood_segmenter,
                                  detector_factory=None, traffic_required=True,
                                  max_frame_pixels=None):
    if media_kind != "image":
        raise ValueError("积水辅助识别仅支持图片")
    flood_regions = [region for region in regions or []
                     if isinstance(region, dict) and region.get("kind") == "road_surface"]
    if not flood_regions:
        raise ValueError("积水辅助识别需要先圈选路面范围")

    traffic_result = None
    if traffic_required:
        if detector_factory is None:
            detector_factory = lambda: OnnxDetector(
                config.VISION_MODEL_PATH,
                model_id=getattr(config, "VISION_MODEL_ID", "visdrone-rtdetrv4-s"),
                model_version=getattr(config, "VISION_MODEL_REVISION", "unspecified"),
            )
        traffic_detector = None
        try:
            traffic_detector = detector_factory()
            traffic_result = analyze_media(
                media_path, media_kind, anchor_gcj, camera_stabilized=camera_stabilized,
                detector=traffic_detector, accident_model=None, flood_segmenter=None,
                regions=regions or [], progress_callback=progress_callback,
                resume_state=resume_state, cancel_check=cancel_check,
                max_frame_pixels=max_frame_pixels,
            )
        finally:
            traffic_detector = None
            _release_inference_memory()
    if cancel_check and cancel_check():
        raise VisionAnalysisCancelled("管理员已取消该影像任务")
    flood_result = analyze_media(
        media_path, media_kind, anchor_gcj, camera_stabilized=camera_stabilized,
        detector=_TrialNoVehicleDetector(), accident_model=None, flood_segmenter=flood_segmenter,
        regions=flood_regions, progress_callback=progress_callback, resume_state=None,
        cancel_check=cancel_check, require_flood_success=True,
        max_frame_pixels=max_frame_pixels, inplace_flood_rgb=True,
    )
    return _merge_assisted_flood_results(traffic_result, flood_result) if traffic_required else flood_result
