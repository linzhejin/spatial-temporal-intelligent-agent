"""Bounded, manager-only video evidence clip generation."""
from __future__ import annotations

import re
import uuid
import math
from pathlib import Path


def _clip_window(start, end, duration, *, padding_seconds, min_clip_seconds, max_clip_seconds):
    if not isinstance(start, (int, float)) or isinstance(start, bool):
        return None
    if not isinstance(end, (int, float)) or isinstance(end, bool):
        return None
    start, end = float(start), float(end)
    if start < 0 or end < start or start > duration:
        return None
    center = (start + end) / 2
    window = min(max_clip_seconds, max(min_clip_seconds, end - start + 2 * padding_seconds))
    clip_start = max(0.0, center - window / 2)
    clip_end = min(duration, clip_start + window)
    clip_start = max(0.0, clip_end - window)
    if clip_end <= clip_start:
        return None
    return round(clip_start, 3), round(clip_end, 3)


def _clip_reopens(cv2_module, path: Path) -> bool:
    """Reject containers that the installed decoder cannot actually read."""
    verification = None
    try:
        verification = cv2_module.VideoCapture(str(path))
        if not verification.isOpened():
            return False
        decoded, frame = verification.read()
        return bool(decoded and frame is not None)
    except Exception:
        return False
    finally:
        if verification is not None:
            verification.release()


def attach_evidence_clips(
    media_path: str | Path,
    candidates: list[dict],
    output_dir: str | Path,
    job_id: str,
    *,
    cv2_module=None,
    max_clips_per_candidate: int = 3,
    max_total_clips: int = 6,
    padding_seconds: float = 3.0,
    min_clip_seconds: float = 4.0,
    max_clip_seconds: float = 15.0,
    output_fps: float = 15.0,
) -> dict:
    """Create short WebM clips around candidate evidence without failing the job.

    The uploaded source remains untouched. Only generated filenames under the
    configured upload directory are returned for manager-only delivery.
    """
    try:
        normalized_job_id = str(uuid.UUID(str(job_id)))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("job_id must be a UUID") from error
    if not re.fullmatch(r"[a-f0-9-]{36}", normalized_job_id):
        raise ValueError("job_id must be a UUID")
    numeric_limits = (padding_seconds, min_clip_seconds, max_clip_seconds, output_fps)
    if (isinstance(max_clips_per_candidate, bool)
            or not isinstance(max_clips_per_candidate, int)
            or max_clips_per_candidate < 1
            or isinstance(max_total_clips, bool)
            or not isinstance(max_total_clips, int)
            or max_total_clips < 1
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in numeric_limits)
            or padding_seconds < 0 or min_clip_seconds <= 0
            or max_clip_seconds < min_clip_seconds or output_fps <= 0):
        raise ValueError("invalid evidence clip limits")
    try:
        if cv2_module is None:
            import cv2 as cv2_module
    except ImportError:
        for candidate in candidates:
            if isinstance(candidate.get("evidence"), dict):
                candidate["evidence"]["clips"] = []
                candidate["evidence"]["clip_status"] = "unavailable"
        return {"clips_created": 0, "clips_failed": 0, "clips_skipped": 0}

    source = Path(media_path)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for stale_clip in destination.glob(f"{normalized_job_id}-evidence-c*-s*.webm"):
        if stale_clip.parent == destination and not stale_clip.is_symlink():
            stale_clip.unlink(missing_ok=True)
    capture = cv2_module.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        for candidate in candidates:
            if isinstance(candidate.get("evidence"), dict):
                candidate["evidence"]["clips"] = []
                candidate["evidence"]["clip_status"] = "unavailable"
        return {"clips_created": 0, "clips_failed": 0, "clips_skipped": 0}

    created = failed = skipped = 0
    try:
        fps = float(capture.get(cv2_module.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("source video metadata is unavailable")
        duration = frame_count / fps
        frame_stride = max(1, int(round(fps / min(output_fps, fps))))
        encoded_fps = fps / frame_stride
        output_scale = min(1.0, 1280.0 / max(width, height))
        output_width = max(2, int(round(width * output_scale)) // 2 * 2)
        output_height = max(2, int(round(height * output_scale)) // 2 * 2)

        for candidate_index, candidate in enumerate(candidates):
            evidence = candidate.get("evidence") if isinstance(candidate, dict) else None
            if not isinstance(evidence, dict):
                continue
            segments = evidence.get("segments")
            segments = segments if isinstance(segments, list) else []
            summary = evidence.get("summary") if isinstance(evidence.get("summary"), dict) else {}
            if summary.get("model_scope") == "temporal_window":
                # A whole-window score cannot identify which shorter excerpt contains an event.
                evidence["clips"] = []
                evidence["clip_status"] = "window_only"
                skipped += len(segments)
                continue
            valid = []
            for original_index, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    continue
                window = _clip_window(
                    segment.get("start_seconds"), segment.get("end_seconds"), duration,
                    padding_seconds=padding_seconds,
                    min_clip_seconds=min_clip_seconds,
                    max_clip_seconds=max_clip_seconds,
                )
                if window is None:
                    skipped += 1
                    continue
                confidence = segment.get("confidence", 0)
                confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
                valid.append((confidence, window[0], original_index, window[1]))
            available = max(0, max_total_clips - created)
            clip_budget = min(max_clips_per_candidate, available)
            selected = sorted(valid, key=lambda item: (-item[0], item[1]))[:clip_budget]
            skipped += max(0, len(valid) - len(selected))
            selected.sort(key=lambda item: item[1])
            clips = []
            evidence_summary = evidence.get("summary") if isinstance(evidence.get("summary"), dict) else {}
            flood_outlines = evidence_summary.get("outline_polygons")
            flood_outline_time = evidence_summary.get("outline_time_seconds")
            for output_index, (_confidence, start, _original_index, end) in enumerate(selected):
                filename = f"{normalized_job_id}-evidence-c{candidate_index}-s{output_index}.webm"
                path = destination / filename
                if path.parent != destination or path.is_symlink():
                    failed += 1
                    continue
                start_frame = max(0, int(start * fps))
                max_window_frames = max(1, int(max_clip_seconds * fps + 0.999))
                end_frame = min(
                    frame_count,
                    start_frame + max_window_frames,
                    max(start_frame + 1, int(end * fps + 0.999)),
                )
                if not capture.set(cv2_module.CAP_PROP_POS_FRAMES, start_frame):
                    failed += 1
                    continue
                writer = cv2_module.VideoWriter(
                    str(path), cv2_module.VideoWriter_fourcc(*"VP90"), encoded_fps,
                    (output_width, output_height),
                )
                encoding = "vp9"
                if not writer.isOpened():
                    writer.release()
                    path.unlink(missing_ok=True)
                    writer = cv2_module.VideoWriter(
                        str(path), cv2_module.VideoWriter_fourcc(*"VP80"), encoded_fps,
                        (output_width, output_height),
                    )
                    encoding = "vp8"
                    if not writer.isOpened():
                        writer.release()
                        path.unlink(missing_ok=True)
                        failed += 1
                        continue
                wrote = 0
                frame_index = start_frame
                expected_frames = (end_frame - start_frame + frame_stride - 1) // frame_stride
                decode_failed = False
                try:
                    while frame_index < end_frame:
                        ok, frame = capture.read()
                        if not ok:
                            decode_failed = True
                            break
                        if (frame_index - start_frame) % frame_stride == 0:
                            output_frame = frame
                            if (isinstance(flood_outlines, list) and flood_outlines
                                    and isinstance(flood_outline_time, (int, float))
                                    and abs(frame_index / fps - float(flood_outline_time)) <= 0.75
                                    and hasattr(cv2_module, "polylines")):
                                output_frame = frame.copy()
                                height, width = output_frame.shape[:2]
                                contours = []
                                for polygon in flood_outlines[:8]:
                                    if not isinstance(polygon, list) or len(polygon) < 3:
                                        continue
                                    points = []
                                    for point in polygon[:128]:
                                        if not isinstance(point, (list, tuple)) or len(point) != 2:
                                            continue
                                        try:
                                            x, y = float(point[0]), float(point[1])
                                        except (TypeError, ValueError):
                                            continue
                                        if 0 <= x <= 1 and 0 <= y <= 1:
                                            points.append([round(x * (width - 1)), round(y * (height - 1))])
                                    if len(points) >= 3:
                                        contours.append(points)
                                if contours:
                                    import numpy as np
                                    cv2_module.polylines(
                                        output_frame,
                                        [np.asarray(points, dtype=np.int32) for points in contours],
                                        True, (255, 130, 32), max(2, round(width / 500)),
                                    )
                            if (output_width, output_height) != (width, height):
                                output_frame = cv2_module.resize(
                                    output_frame, (output_width, output_height),
                                    interpolation=getattr(cv2_module, "INTER_AREA", 3),
                                )
                            writer.write(output_frame)
                            wrote += 1
                        frame_index += 1
                finally:
                    writer.release()
                # A truncated source or unavailable encoder must not produce a
                # playable-looking clip whose timestamps imply complete evidence.
                artifact_ok = (
                    path.is_file() and path.stat().st_size > 0
                    and _clip_reopens(cv2_module, path)
                )
                if wrote == 0 or decode_failed or wrote != expected_frames or not artifact_ok:
                    path.unlink(missing_ok=True)
                    failed += 1
                    continue
                created += 1
                clips.append({
                    "filename": filename,
                    "start_seconds": round(start_frame / fps, 3),
                    "end_seconds": round(min(end_frame, frame_index) / fps, 3),
                    "duration_seconds": round(wrote / encoded_fps, 3),
                    "frames": wrote,
                    "container": "webm",
                    "encoding": encoding,
                })
            evidence["clips"] = clips
            evidence["clip_status"] = (
                "ready" if clips else "job_limit" if valid and not available else "unavailable"
            )
    finally:
        capture.release()
    return {"clips_created": created, "clips_failed": failed, "clips_skipped": skipped}
