"""Video-level accident cues for isolated, human-reviewed trial jobs only."""
from __future__ import annotations

import math
import time
from pathlib import Path

import config
from vision.engine import VisionAnalysisCancelled, _validate_frame_dimensions


def _window_frame_ranges(*, total_frames: int, fps: float, window_seconds: float = 40,
                         stride_seconds: float = 20) -> list[tuple[int, int]]:
    if total_frames < 16 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("视频帧数或帧率无效。")
    if not 0 < stride_seconds <= window_seconds:
        raise ValueError("视频筛查窗口设置无效。")
    window = max(1, round(window_seconds * fps))
    stride = max(1, round(stride_seconds * fps))
    ranges = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + window)
        ranges.append((start, end))
        if end == total_frames:
            break
        start += stride
    return ranges


class OnnxTemporalAccident:
    """Score overlapping video windows; scores never imply a precise collision time."""

    model_id = "temporal-mobilenetv2-attention"
    trial_video_only = True

    def __init__(self, model_path: str, *, session=None, threshold: float = 0.5,
                 model_version: str = "unverified"):
        if session is None:
            if not model_path or not Path(model_path).is_file():
                raise ValueError("事故试用模型文件不存在。")
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = config.resolve_vision_cpu_threads()
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(model_path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or list(inputs[0].shape) != [1, 16, 3, 224, 224]:
            raise ValueError("事故试用模型必须接收 16 帧 224×224 的视频窗口。")
        if not math.isfinite(float(threshold)) or not 0 < float(threshold) <= 1:
            raise ValueError("事故试用模型阈值无效。")
        self.session = session
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.model_path = str(model_path)
        self.model_version = model_version
        self.threshold = float(threshold)

    def analyze_video(self, path, *, cancel_check=None, deadline=None, progress_callback=None) -> dict:
        import cv2
        import numpy as np

        def check() -> None:
            if cancel_check and cancel_check():
                raise VisionAnalysisCancelled("管理员已取消影像分析。")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("事故视频筛查超过配置时限。")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("视频无法解码。")
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _validate_frame_dimensions(width, height, max_pixels=config.VISION_TRIAL_MAX_FRAME_PIXELS)
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ranges = _window_frame_ranges(total_frames=total_frames, fps=fps)
            if total_frames / fps > config.VISION_TRIAL_MAX_VIDEO_SECONDS:
                raise ValueError(f"试用视频时长超过 {config.VISION_TRIAL_MAX_VIDEO_SECONDS:g} 秒，请先裁剪。")
            mean = np.asarray([.485, .456, .406], dtype=np.float32)
            std = np.asarray([.229, .224, .225], dtype=np.float32)
            windows = []
            for index, (start, end) in enumerate(ranges):
                check()
                frames = []
                fallback_count = 0
                for frame_index in np.linspace(start, end - 1, 16, dtype=np.int32):
                    check()
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    ok, frame = capture.read()
                    if not ok:
                        if not frames:
                            raise ValueError("视频窗口的首帧无法读取。")
                        fallback_count += 1
                        frame = frames[-1].copy()
                    else:
                        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
                    frames.append(frame)
                if fallback_count > 4:
                    raise ValueError("视频窗口读取失败的画面过多。")
                array = np.stack(frames).astype(np.float32) / 255.0
                tensor = np.transpose((array - mean) / std, (0, 3, 1, 2))[None].astype(np.float32)
                raw = np.asarray(self.session.run([self.output_name], {self.input_name: tensor})[0])
                if raw.size != 1 or not np.isfinite(raw).all() or not 0 <= float(raw.flat[0]) <= 1:
                    raise ValueError("事故试用模型返回了无效分数。")
                windows.append({"start_seconds": round(start / fps, 3),
                                "end_seconds": round(end / fps, 3),
                                "score": float(raw.flat[0]), "fallback_frames": fallback_count})
                if progress_callback:
                    progress_callback(index + 1, len(ranges))
            positive = [item for item in windows if item["score"] >= self.threshold]
            candidates = []
            if positive:
                candidates.append({
                    "kind": "possible_accident", "confidence": max(item["score"] for item in positive),
                    "reason": "这段视频可能出现碰撞，请查看原片。",
                    "status": "pending_review", "review_required": True, "auto_publish": False,
                    "evidence": {
                        "summary": {"model_scope": "temporal_window", "windows": len(positive)},
                        "segments": [{"start_seconds": item["start_seconds"],
                                      "end_seconds": item["end_seconds"]} for item in positive],
                    },
                })
            return {"windows_analyzed": len(windows), "windows": windows, "candidates": candidates}
        finally:
            capture.release()
