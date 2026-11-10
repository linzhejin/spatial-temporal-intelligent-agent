"""Single-process traffic and flood analysis; the parent owns timeout and cancellation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from vision.assisted_image import _analyze_assisted_flood_image, _release_inference_memory
from vision.engine import OnnxDetector, analyze_media
from vision.specialized import DoodleverseFloodSegmenter


class _InProcessDoodleverse(DoodleverseFloodSegmenter):
    def predict_regions(self, frame_rgb, regions, *, cancel_check=None):
        if cancel_check and cancel_check():
            raise RuntimeError("积水检查已取消")
        road_regions = [region for region in regions or []
                        if isinstance(region, dict) and region.get("kind") == "road_surface"]
        if not road_regions:
            return {}
        # The returned class plane is a view of the full ONNX output. Copy it
        # before unloading the session so the arena can be returned to the OS.
        probabilities = self._predict_probabilities(frame_rgb).copy()
        self.session = None
        _release_inference_memory()
        return self._predictions_from_probability(frame_rgb, road_regions, probabilities)


def main(request_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if request["stage"] == "traffic":
        detector = OnnxDetector(
            request["traffic_model_path"], model_id=request["traffic_model_id"],
            model_version=request["traffic_model_version"],
        )
        result = analyze_media(
            request["media_path"], "image", request.get("anchor_gcj"),
            camera_stabilized=request.get("camera_stabilized", False), detector=detector,
            regions=request.get("regions") or [], resume_state=request.get("resume_state"),
            max_frame_pixels=request.get("max_frame_pixels"),
        )
    elif request["stage"] == "flood":
        flood = _InProcessDoodleverse(
            request["flood_model_path"], model_version=request["flood_model_version"],
            probability_threshold=request["probability_threshold"],
            min_area_ratio=request["min_area_ratio"],
            single_image_trigger_threshold=request["single_image_trigger_threshold"],
        )
        flood.model_sha256 = request.get("flood_model_sha256")
        result = _analyze_assisted_flood_image(
            request["media_path"], "image", request.get("anchor_gcj"),
            camera_stabilized=request.get("camera_stabilized", False),
            regions=request.get("regions") or [], progress_callback=None,
            resume_state=None, cancel_check=None, flood_segmenter=flood,
            traffic_required=False, max_frame_pixels=request.get("max_frame_pixels"),
        )
    else:
        raise ValueError("未知的积水图片分析阶段")
    Path(request["response_path"]).write_text(
        json.dumps({"status": "ok", "result": result}, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
