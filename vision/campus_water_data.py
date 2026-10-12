"""Prepare flight-disjoint campus UAV frames for manual visible-water labeling."""
from __future__ import annotations

import json
import hashlib
import math
import re
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath


_SPLITS = ("train", "validation", "test")
_FLIGHT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_CAPTURE_TIME_SOURCES = {"unknown", "manual", "dji_srt"}
_TIMEZONE_NAME = re.compile(r"^[A-Za-z0-9_+./:-]{1,64}$")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


def _captured_at(value: object, flight_id: str) -> str:
    text = _required_text(value, f"flight {flight_id} captured_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"flight {flight_id} captured_at must be ISO 8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"flight {flight_id} captured_at must include a timezone")
    return parsed.isoformat()


def _file_fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (posix.is_absolute() or windows.is_absolute()
            or ".." in posix.parts or ".." in windows.parts):
        raise ValueError(f"{field} must stay inside the dataset")
    return posix.as_posix()


def build_campus_flight_plan(manifest: object, dataset_root) -> dict:
    """Validate one source video per flight and exact disjoint split membership."""
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("task") != "campus_visible_water_frame_extraction"
            or manifest.get("split_unit") != "flight_id"):
        raise ValueError("flight manifest must use schema v1 and flight_id splits")
    dataset = str(manifest.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("flight manifest dataset name is required")
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("dataset root must be a directory")

    groups = manifest.get("split_groups")
    if not isinstance(groups, dict):
        raise ValueError("split_groups is required")
    normalized_groups = {}
    for split in _SPLITS:
        values = groups.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"split_groups.{split} must contain at least one flight")
        normalized = [str(value).strip() for value in values]
        if (any(not value or not _FLIGHT_ID.fullmatch(value) for value in normalized)
                or len(normalized) != len(set(normalized))):
            raise ValueError(f"split_groups.{split} contains an invalid or duplicate flight ID")
        normalized_groups[split] = normalized
    all_groups = [value for values in normalized_groups.values() for value in values]
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("flight groups must not overlap")

    flights = manifest.get("flights")
    if not isinstance(flights, list) or not flights:
        raise ValueError("flight manifest must include at least one video")
    normalized_flights = []
    seen_ids, seen_paths, seen_hashes = set(), set(), {}
    present_groups = {split: set() for split in _SPLITS}
    site_splits = {}
    for index, flight in enumerate(flights):
        if not isinstance(flight, dict):
            raise ValueError(f"flight {index} must be an object")
        flight_id = str(flight.get("flight_id") or "").strip()
        split = str(flight.get("split") or "").strip()
        if not _FLIGHT_ID.fullmatch(flight_id):
            raise ValueError(f"flight {index} has an invalid flight_id")
        if flight_id in seen_ids:
            raise ValueError(f"duplicate flight_id: {flight_id}")
        if split not in _SPLITS or flight_id not in normalized_groups[split]:
            raise ValueError(f"flight {flight_id} conflicts with its declared split")
        site_id = _required_text(flight.get("site_id"), f"flight {flight_id} site_id")
        if not _FLIGHT_ID.fullmatch(site_id):
            raise ValueError(f"flight {flight_id} has an invalid site_id")
        previous_site_split = site_splits.setdefault(site_id, split)
        if previous_site_split != split:
            raise ValueError(f"site must not cross splits: {site_id}")
        location = _required_text(flight.get("location"), f"flight {flight_id} location")
        captured_at = _captured_at(flight.get("captured_at"), flight_id)
        capture_time_source = str(flight.get("capture_time_source") or "unknown").strip().lower()
        if capture_time_source not in _CAPTURE_TIME_SOURCES:
            raise ValueError(f"flight {flight_id} capture_time_source is invalid")
        capture_timezone = flight.get("capture_timezone")
        if capture_timezone is not None:
            capture_timezone = _required_text(
                capture_timezone, f"flight {flight_id} capture_timezone",
            )
            if not _TIMEZONE_NAME.fullmatch(capture_timezone):
                raise ValueError(f"flight {flight_id} capture_timezone is invalid")
        if capture_time_source in {"manual", "dji_srt"} and not capture_timezone:
            raise ValueError(f"flight {flight_id} capture_timezone is required")
        video = _relative_path(flight.get("video"), f"flight {flight_id} video")
        try:
            video_path = (root / Path(*PurePosixPath(video).parts)).resolve(strict=True)
            video_path.relative_to(root)
        except (OSError, ValueError) as error:
            raise ValueError(f"flight {flight_id} video is missing or outside the dataset") from error
        if not video_path.is_file():
            raise ValueError(f"flight {flight_id} video is not a file")
        canonical_path = str(video_path).casefold()
        if canonical_path in seen_paths:
            raise ValueError("video file is assigned to multiple flights")
        seen_ids.add(flight_id)
        seen_paths.add(canonical_path)
        present_groups[split].add(flight_id)
        source_bytes, source_sha256 = _file_fingerprint(video_path)
        duplicate_flight = seen_hashes.get(source_sha256)
        if duplicate_flight is not None:
            raise ValueError(
                f"identical source video assigned to {duplicate_flight} and {flight_id}"
            )
        seen_hashes[source_sha256] = flight_id
        sidecars = flight.get("sidecars", [])
        if not isinstance(sidecars, list):
            raise ValueError(f"flight {flight_id} sidecars must be a list")
        normalized_sidecars = []
        seen_sidecars = set()
        for sidecar_index, value in enumerate(sidecars):
            relative = _relative_path(value, f"flight {flight_id} sidecar {sidecar_index}")
            try:
                sidecar_path = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
                sidecar_path.relative_to(root)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"flight {flight_id} sidecar is missing or outside the dataset"
                ) from error
            if not sidecar_path.is_file():
                raise ValueError(f"flight {flight_id} sidecar is not a file")
            canonical_sidecar = str(sidecar_path).casefold()
            if canonical_sidecar in seen_sidecars:
                raise ValueError(f"flight {flight_id} contains a duplicate sidecar")
            seen_sidecars.add(canonical_sidecar)
            sidecar_bytes, sidecar_sha256 = _file_fingerprint(sidecar_path)
            normalized_sidecars.append({
                "path": relative,
                "bytes": sidecar_bytes,
                "sha256": sidecar_sha256,
            })
        normalized_flights.append({
            "flight_id": flight_id,
            "split": split,
            "site_id": site_id,
            "location": location,
            "captured_at": captured_at,
            "capture_time_source": capture_time_source,
            "capture_timezone": capture_timezone,
            "video": video,
            "video_path": video_path,
            "source_bytes": source_bytes,
            "source_sha256": source_sha256,
            "sidecars": normalized_sidecars,
        })

    declared = {split: set(values) for split, values in normalized_groups.items()}
    if present_groups != declared:
        raise ValueError("videos must match declared flight groups exactly")
    normalized_flights.sort(key=lambda item: (_SPLITS.index(item["split"]), item["flight_id"]))
    return {
        "dataset": dataset,
        "split_unit": "flight_id",
        "split_groups": normalized_groups,
        "site_splits": dict(sorted(site_splits.items())),
        "flights": normalized_flights,
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_campus_annotation_pack(plan: dict, output_dir, *, sample_fps: float = 1.0,
                                   jpeg_quality: int = 95, cv2_api=None) -> dict:
    """Extract frames and write editable train/test annotation templates locally."""
    if (isinstance(sample_fps, bool) or not isinstance(sample_fps, (int, float))
            or not math.isfinite(float(sample_fps)) or sample_fps <= 0):
        raise ValueError("sample_fps must be a finite positive number")
    if (isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int)
            or not 50 <= jpeg_quality <= 100):
        raise ValueError("jpeg_quality must be an integer from 50 to 100")
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError("output directory already exists; refusing to overwrite")
    staging = target.with_name(f".{target.name}.partial")
    if staging.exists():
        raise FileExistsError("partial output directory already exists; inspect it before retrying")
    if cv2_api is None:
        import cv2 as cv2_api

    target.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        samples = []
        flight_summaries = []
        sample_counts = {split: 0 for split in _SPLITS}
        for flight in plan["flights"]:
            flight_id = flight["flight_id"]
            split = flight["split"]
            frame_dir = staging / "frames" / split / flight_id
            mask_dir = staging / "masks" / split / flight_id
            frame_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            capture = cv2_api.VideoCapture(str(flight["video_path"]))
            try:
                if not capture.isOpened():
                    raise ValueError(f"cannot open flight video: {flight['video']}")
                fps = float(capture.get(cv2_api.CAP_PROP_FPS))
                reported_frames = int(capture.get(cv2_api.CAP_PROP_FRAME_COUNT))
                width = int(capture.get(cv2_api.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2_api.CAP_PROP_FRAME_HEIGHT))
                if not math.isfinite(fps) or fps <= 0 or reported_frames <= 0:
                    raise ValueError(f"flight video has invalid FPS or frame count: {flight['video']}")
                if width <= 0 or height <= 0:
                    raise ValueError(f"flight video has invalid dimensions: {flight['video']}")

                stride = max(1, round(fps / float(sample_fps)))
                decoded_frames = 0
                extracted = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_index = decoded_frames
                    decoded_frames += 1
                    if frame_index % stride:
                        continue
                    filename = f"frame_{frame_index:08d}"
                    image_rel = f"frames/{split}/{flight_id}/{filename}.jpg"
                    mask_rel = f"masks/{split}/{flight_id}/{filename}.png"
                    image_path = staging / Path(*PurePosixPath(image_rel).parts)
                    if not cv2_api.imwrite(
                        str(image_path), frame,
                        [cv2_api.IMWRITE_JPEG_QUALITY, jpeg_quality],
                    ):
                        raise OSError(f"could not write extracted frame: {image_rel}")
                    samples.append({
                        "sample_id": f"{flight_id}-{frame_index:08d}",
                        "split": split,
                        "group_id": flight_id,
                        "flight_id": flight_id,
                        "source_video": flight["video"],
                        "source_frame_index": frame_index,
                        "timestamp_seconds": round(frame_index / fps, 3),
                        "image": image_rel,
                        "width": width,
                        "height": height,
                        "mask": mask_rel,
                        "road_surface_polygon": None,
                        "annotation_status": "pending",
                    })
                    extracted += 1
                if not decoded_frames or not extracted:
                    raise ValueError(f"flight video yielded no decodable frames: {flight['video']}")
                if decoded_frames < reported_frames - 1:
                    raise ValueError(
                        "flight video ended before reported frame count: "
                        f"{flight['video']} ({decoded_frames}/{reported_frames})"
                    )
                sample_counts[split] += extracted
                flight_summaries.append({
                    "flight_id": flight_id,
                    "split": split,
                    "site_id": flight["site_id"],
                    "location": flight["location"],
                    "captured_at": flight["captured_at"],
                    "capture_time_source": flight["capture_time_source"],
                    "capture_timezone": flight["capture_timezone"],
                    "source_video": flight["video"],
                    "source_bytes": flight["source_bytes"],
                    "source_sha256": flight["source_sha256"],
                    "sidecars": flight["sidecars"],
                    "source_fps": fps,
                    "reported_frames": reported_frames,
                    "decoded_frames": decoded_frames,
                    "duration_seconds": round((decoded_frames - 1) / fps, 3),
                    "sampling_stride_frames": stride,
                    "extracted_frames": extracted,
                    "width": width,
                    "height": height,
                })
            finally:
                capture.release()

        common = {
            "schema_version": 1,
            "dataset": plan["dataset"],
            "split_unit": "flight_id",
            "split_groups": plan["split_groups"],
            "site_splits": plan["site_splits"],
            "mask_encoding": "binary_0_255",
            "label_source": "manual",
            "annotation_pending": True,
        }
        training_samples = [item for item in samples if item["split"] != "test"]
        testing_samples = [item for item in samples if item["split"] == "test"]
        worklist = {
            **common,
            "task": "visible_water_annotation_worklist",
            "sample_fps_target": float(sample_fps),
            "flights": flight_summaries,
            "samples": samples,
        }
        training_template = {
            **common,
            "task": "visible_water_road_segmentation",
            "flights": [item for item in flight_summaries if item["split"] != "test"],
            "samples": [
                {key: item[key] for key in (
                    "sample_id", "split", "group_id", "image", "mask", "road_surface_polygon",
                )}
                for item in training_samples
            ],
        }
        test_template = {
            **common,
            "task": "visible_water_road_segmentation",
            "flights": [item for item in flight_summaries if item["split"] == "test"],
            "samples": [
                {key: item[key] for key in (
                    "sample_id", "split", "group_id", "image", "mask", "road_surface_polygon",
                )}
                for item in testing_samples
            ],
        }
        detection_test_template = {
            "schema_version": 1,
            "task": "campus_object_detection",
            "dataset": plan["dataset"],
            "split_unit": "flight_id",
            "label_source": "manual",
            "split_groups": plan["split_groups"],
            "site_splits": plan["site_splits"],
            "annotation_pending": True,
            "flights": [item for item in flight_summaries if item["split"] == "test"],
            "samples": [
                {
                    "sample_id": item["sample_id"],
                    "group_id": item["group_id"],
                    "image": item["image"],
                    "width": item["width"],
                    "height": item["height"],
                    "observation_regions": [],
                    "objects": [],
                }
                for item in testing_samples
            ],
        }
        event_test_template = {
            "schema_version": 1,
            "task": "campus_event_detection",
            "dataset": plan["dataset"],
            "split_unit": "flight_id",
            "label_source": "manual",
            "event_kinds": ["congestion", "crowding", "accident", "flooding"],
            "split_groups": plan["split_groups"],
            "site_splits": plan["site_splits"],
            "annotation_pending": True,
            "samples": [
                {
                    "sample_id": item["flight_id"],
                    "video": item["source_video"],
                    "group_id": item["flight_id"],
                    "site_id": item["site_id"],
                    "location": item["location"],
                    "captured_at": item["captured_at"],
                    "capture_time_source": item["capture_time_source"],
                    "capture_timezone": item["capture_timezone"],
                    "source_sha256": item["source_sha256"],
                    "camera_stabilized": None,
                    "event_kinds": [],
                    "observation_regions": [],
                    "events": [],
                }
                for item in flight_summaries if item["split"] == "test"
            ],
        }
        _write_json(staging / "annotation-worklist.json", worklist)
        _write_json(staging / "training-template.json", training_template)
        _write_json(staging / "test-template.json", test_template)
        _write_json(staging / "detection-test-template.json", detection_test_template)
        _write_json(staging / "event-test-template.json", event_test_template)
        (staging / "ANNOTATION.md").write_text(
            """# 校园道路可见水体人工标注

1. 在 `training-template.json` 和 `test-template.json` 中，把每个样本的 `road_surface_polygon: null` 替换为原图坐标归一化到 0–1 的路面多边形。航次记录中的 `capture_time_source` 与 `capture_timezone` 是时间审计信息；DJI 字幕时间没有时区，必须核对拍摄设备时区后再登记。
2. 按 `mask` 路径保存与原图同宽高的灰度 PNG；路面范围内确认的可见水体标为 255，其余像素为 0。积水、湿路反光、阴影、湖面和干路要按标注规范区分；不确定样本应先剔除，不能猜标签。
3. 只把 train/validation 模板交给 `train_visible_water.py`；test 模板只交给独立评测脚本，不能参与训练、选模型或调阈值。保存原视频和本目录外的备份。
4. 模板含 `annotation_pending: true`，标注完成后可删除该提示字段；训练器仍会检查文件、尺寸、掩膜像素值和跨 split 的完全重复图像。
5. `detection-test-template.json` 用于校园人车检测评测。只保留准备人工框选的代表帧，在 `objects` 中逐个填写 `pedestrian`、`vehicle` 或 `ignore` 及像素坐标 `[x1,y1,x2,y2]`；不能把空数组当作尚未标注的占位结果。若还要评测管理端的道路 ROI 分块推理，在 `observation_regions` 中按归一化坐标填写 `vehicle_lane`、`pedestrian` 或 `road_surface` 多边形，例如 `{"id":"lane-1","kind":"vehicle_lane","polygon":[[0.1,0.2],[0.8,0.2],[0.8,0.6],[0.1,0.6]]}`；每个区域 ID 必须唯一，所有要计分的人车框完整落在某个 ROI 内，否则该帧不适合做两种路径的同口径比较。
6. `event-test-template.json` 指向原始测试航次视频。填写镜头是否稳定、已人工检查的事件类型、观察区域和事件时间段后，再删除 `annotation_pending`；运行评测时 `--dataset-root` 指向原始航拍数据目录。

水体分割结果是影像线索，不是现场封路结论；必须由管理员复核后才可影响路况。
""",
            encoding="utf-8",
        )
        staging.replace(target)
    except Exception:
        # Keep partial output for diagnosis; never overwrite or silently delete source work.
        raise
    return {
        "output_dir": str(target),
        "sample_counts": sample_counts,
        "flight_count": len(flight_summaries),
        "worklist": str(target / "annotation-worklist.json"),
        "training_template": str(target / "training-template.json"),
        "test_template": str(target / "test-template.json"),
        "detection_test_template": str(target / "detection-test-template.json"),
        "event_test_template": str(target / "event-test-template.json"),
    }
