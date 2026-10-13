"""Isolated worker for bounded aerial image/video inference jobs."""
from __future__ import annotations

import logging
import os
import signal
import importlib.util
import json
import threading
import time
import uuid
from pathlib import Path

import config
from dotenv import load_dotenv
from storage import database
from storage import vision_repository
from vision.evidence import attach_evidence_clips
from vision.engine import OnnxDetector, VisionAnalysisCancelled, analyze_media
from vision.model_release import check_optional_model_release
from vision.specialized import OnnxAiderClassifier, OnnxFloodSegmenter

logger = logging.getLogger(__name__)
_STOP = threading.Event()


def _optional_runtime_config(model_kind, *, accident_threshold, flood_min_area_ratio):
    sampling = {
        "sample_fps": float(config.VISION_SAMPLE_FPS),
        "specialized_sampling_policy": "detector_sample_grid_about_1hz_v2",
    }
    if model_kind == "accident":
        if accident_threshold is None:
            accident_threshold = config.VISION_ACCIDENT_THRESHOLD
        return {
            **sampling,
            "decision_threshold": float(accident_threshold),
            "dense_trigger_threshold": float(config.VISION_ACCIDENT_DENSE_TRIGGER_THRESHOLD),
            "dense_window_seconds": float(config.VISION_ACCIDENT_DENSE_WINDOW_SECONDS),
            "roi_policy": "all_vehicle_lanes_or_first_road_region_v1",
        }
    flood_probability = os.getenv("VISION_FLOOD_CLASS_PROBABILITY_THRESHOLD", "").strip()
    visible_probability = os.getenv("VISION_VISIBLE_WATER_CLASS_PROBABILITY_THRESHOLD", "").strip()
    flood_probability = float(flood_probability) if flood_probability else None
    visible_probability = float(visible_probability) if visible_probability else None
    if flood_probability != visible_probability:
        return None
    if flood_min_area_ratio is None:
        flood_min_area_ratio = config.VISION_FLOOD_MIN_AREA_RATIO
    return {
        **sampling,
        "min_area_ratio": float(flood_min_area_ratio),
        "probability_threshold": flood_probability,
        "roi_policy": "all_marked_road_surface_regions_v1",
    }


def load_optional_models(*, accident_path=None, accident_revision=None,
                         accident_threshold=None, accident_enabled=False, flood_path=None,
                         flood_enabled=False,
                         flood_revision=None, flood_min_area_ratio=None,
                         accident_loader=OnnxAiderClassifier,
                         flood_loader=OnnxFloodSegmenter,
                         release_manifest_path=None, readiness_report_path=None,
                         release_checker=check_optional_model_release):
    """Load specialized models independently; their failure must not disable traffic vision."""
    accident_model = flood_segmenter = None
    capabilities = {
        "accident_recognition_supported": False,
        "flood_segmentation_supported": False,
        "accident_recognition_release_status": "disabled",
        "flood_segmentation_release_status": "disabled",
    }

    def check_release(model_kind, model_path, revision):
        try:
            runtime_config = _optional_runtime_config(
                model_kind, accident_threshold=accident_threshold,
                flood_min_area_ratio=flood_min_area_ratio,
            )
            if runtime_config is None:
                return {"approved": False, "reason": "flood_threshold_configuration_ambiguous"}
            return release_checker(
                model_kind, model_path, revision,
                release_manifest_path, readiness_report_path, runtime_config,
            )
        except Exception:
            logger.exception("Optional %s model release check failed; capability disabled", model_kind)
            return {"approved": False, "reason": "release_check_failed"}

    if accident_enabled and accident_path:
        release = check_release("accident", accident_path, accident_revision)
        capabilities["accident_recognition_release_status"] = str(release.get("reason", "release_invalid"))
        if release.get("approved") is True:
            try:
                accident_model = accident_loader(
                    accident_path, model_version=accident_revision, threshold=accident_threshold,
                )
                capabilities["accident_recognition_supported"] = True
            except Exception:
                capabilities["accident_recognition_release_status"] = "model_load_failed"
                logger.exception("Optional accident scene model could not be loaded; capability disabled")
        else:
            logger.warning("Optional accident scene model remains disabled: %s",
                           capabilities["accident_recognition_release_status"])
    if flood_enabled and flood_path:
        release = check_release("flood", flood_path, flood_revision)
        capabilities["flood_segmentation_release_status"] = str(release.get("reason", "release_invalid"))
        if release.get("approved") is True:
            try:
                flood_segmenter = flood_loader(
                    flood_path, model_version=flood_revision, min_area_ratio=flood_min_area_ratio,
                )
                capabilities["flood_segmentation_supported"] = True
            except Exception:
                capabilities["flood_segmentation_release_status"] = "model_load_failed"
                logger.exception("Optional road-flood segmentation model could not be loaded; capability disabled")
        else:
            logger.warning("Optional road-flood model remains disabled: %s",
                           capabilities["flood_segmentation_release_status"])
    return accident_model, flood_segmenter, capabilities


class VisionJobCancelled(RuntimeError):
    """A manager cancelled a running job after its last saved checkpoint."""


def process_next_job(database_url: str, worker_id: str, lease_seconds: int = 90, *,
                     detector=None, accident_model=None, flood_segmenter=None):
    job = vision_repository.claim_next_job(database_url, worker_id, lease_seconds)
    if not job:
        return None
    job_id = job["job_id"]
    stop_heartbeat = threading.Event()
    lease_lost = threading.Event()

    def keep_lease_alive():
        while not stop_heartbeat.wait(max(1, lease_seconds // 3)):
            if not vision_repository.heartbeat_job(database_url, job_id, worker_id, lease_seconds):
                lease_lost.set()
                return

    heartbeat = threading.Thread(target=keep_lease_alive, name=f"vision-lease-{job_id[:8]}", daemon=True)
    heartbeat.start()
    try:
        path = Path(config.VISION_UPLOAD_DIR) / job["media_path"]

        def save_progress(snapshot):
            saved = vision_repository.update_job_progress(
                database_url, job_id, worker_id,
                progress=snapshot["progress"], checkpoint=snapshot.get("checkpoint"),
            )
            if saved is None:
                lease_lost.set()
                raise RuntimeError("影像任务工作租约已失效")
            if saved["cancel_requested"] and snapshot.get("checkpoint") is not None:
                raise VisionJobCancelled("管理员已取消该影像任务")

        last_cancel_poll = None
        cached_cancel_state = False

        def cancellation_requested():
            nonlocal last_cancel_poll, cached_cancel_state
            if lease_lost.is_set():
                return True
            now = time.monotonic()
            if last_cancel_poll is not None and now - last_cancel_poll < 1.0:
                return cached_cancel_state
            state = vision_repository.get_job_cancel_state(database_url, job_id, worker_id)
            last_cancel_poll = now
            if state is None:
                lease_lost.set()
                return True
            cached_cancel_state = bool(state.get("cancel_requested"))
            return cached_cancel_state

        result = analyze_media(
            path, job["media_kind"], job["anchor_gcj"],
            camera_stabilized=job["camera_stabilized"], detector=detector,
            accident_model=accident_model,
            flood_segmenter=flood_segmenter,
            regions=job.get("observation_regions") or [],
            progress_callback=save_progress,
            resume_state=job.get("checkpoint"),
            cancel_check=cancellation_requested,
        )
        finalizing_progress = {"phase": "finalizing", "percent": 99}
        if job["media_kind"] == "video":
            video_coverage = result.get("video_coverage", {})
            sampled_frames = video_coverage.get("sampled_frames")
            finalizing_progress.update({
                "frames_analyzed": sampled_frames,
                "total_sampled_frames": sampled_frames,
                "analyzed_through_seconds": video_coverage.get("analyzed_through_seconds"),
            })
        finalizing = vision_repository.update_job_progress(
            database_url, job_id, worker_id, progress=finalizing_progress,
        )
        if finalizing is None:
            lease_lost.set()
            return {"job_id": job_id, "status": "lease_lost"}
        if finalizing["cancel_requested"]:
            final_status = vision_repository.finish_job(
                database_url, job_id, worker_id, status="cancelled",
                error={"code": "vision_job_cancelled", "message": "管理员已取消影像分析；已完成分段可用于重试。"},
            )
            return {"job_id": job_id, "status": final_status or "lease_lost"}
        if job["media_kind"] == "video" and result.get("candidates"):
            try:
                clip_summary = attach_evidence_clips(
                    path, result["candidates"], config.VISION_UPLOAD_DIR, job_id,
                )
                result["evidence_clip_summary"] = clip_summary
            except Exception:
                # The original video and its time-coded evidence remain reviewable.
                # A codec issue must not turn an otherwise completed analysis into
                # a failed job.
                logger.exception("Evidence clip generation failed (%s)", job_id)
                result["evidence_clip_summary"] = {
                    "clips_created": 0, "clips_failed": 0, "clips_skipped": 0,
                }
                for candidate in result["candidates"]:
                    evidence = candidate.get("evidence") if isinstance(candidate, dict) else None
                    if isinstance(evidence, dict):
                        evidence.setdefault("clips", [])
                        evidence["clip_status"] = "unavailable"
        if lease_lost.is_set():
            return {"job_id": job_id, "status": "lease_lost"}
        status = "needs_review" if result.get("candidates") else "completed"
        final_status = vision_repository.finish_job(
            database_url, job_id, worker_id, status=status, result=result,
        )
        return {"job_id": job_id, "status": final_status or "lease_lost"}
    except (VisionJobCancelled, VisionAnalysisCancelled):
        if lease_lost.is_set():
            return {"job_id": job_id, "status": "lease_lost"}
        vision_repository.update_job_progress(
            database_url, job_id, worker_id,
            progress={"phase": "cancelled", "percent": 0},
        )
        final_status = vision_repository.finish_job(
            database_url, job_id, worker_id, status="cancelled",
            error={"code": "vision_job_cancelled", "message": "管理员已取消影像分析；已完成分段可用于重试。"},
        )
        return {"job_id": job_id, "status": final_status or "lease_lost"}
    except Exception as error:
        logger.exception("Vision job failed (%s)", job_id)
        if lease_lost.is_set():
            return {"job_id": job_id, "status": "lease_lost"}
        message = str(error)[:500] or "影像分析失败，请检查文件后重试。"
        final_status = vision_repository.finish_job(
            database_url, job_id, worker_id, status="failed",
            error={"code": "vision_analysis_failed", "message": message},
        )
        return {"job_id": job_id, "status": final_status or "lease_lost"}
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=2)


def run_forever(database_url=None, idle_seconds=1.0, heartbeat_seconds=10.0):
    database_url = database.database_url(database_url)
    database.initialize(database_url)
    worker_id = f"whu-vision-{uuid.uuid4()}"
    detector = None
    accident_model = None
    flood_segmenter = None
    ready = False
    status_detail = ""
    try:
        missing = [
            name for name in ("onnxruntime", "numpy", "cv2", "PIL", "scipy")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise RuntimeError("missing vision dependencies: " + ", ".join(missing))
        detector = OnnxDetector(
            config.VISION_MODEL_PATH,
            model_id=getattr(config, "VISION_MODEL_ID", "visdrone-rtdetrv4-s"),
            model_version=getattr(config, "VISION_MODEL_REVISION", "unspecified"),
        )
        accident_model, flood_segmenter, optional_capabilities = load_optional_models(
            accident_path=config.VISION_ACCIDENT_MODEL_PATH,
            accident_revision=config.VISION_ACCIDENT_REVISION,
            accident_threshold=config.VISION_ACCIDENT_THRESHOLD,
            accident_enabled=config.VISION_ACCIDENT_ENABLED,
            flood_path=config.VISION_FLOOD_MODEL_PATH,
            flood_enabled=config.VISION_FLOOD_ENABLED,
            flood_revision=config.VISION_FLOOD_REVISION,
            flood_min_area_ratio=config.VISION_FLOOD_MIN_AREA_RATIO,
            release_manifest_path=config.VISION_OPTIONAL_MODELS_RELEASE_MANIFEST,
            readiness_report_path=config.VISION_FIELD_READINESS_REPORT,
        )
        status_detail = json.dumps(optional_capabilities, separators=(",", ":"))
        ready = True
    except Exception as error:
        status_detail = f"{type(error).__name__}: {error}"[:500]
        logger.exception("Vision worker model initialization failed; uploads remain disabled")

    heartbeat_stop = threading.Event()

    def keep_worker_heartbeat():
        while not heartbeat_stop.is_set():
            try:
                vision_repository.heartbeat_worker(
                    database_url, worker_id, ready=ready, status_detail=status_detail,
                )
            except Exception:
                logger.exception("Could not publish vision worker readiness")
            heartbeat_stop.wait(heartbeat_seconds)

    heartbeat = threading.Thread(
        target=keep_worker_heartbeat, name="vision-worker-heartbeat", daemon=True,
    )
    heartbeat.start()
    try:
        while not _STOP.is_set():
            if not ready:
                _STOP.wait(idle_seconds)
                continue
            try:
                if process_next_job(
                    database_url, worker_id, detector=detector,
                    accident_model=accident_model, flood_segmenter=flood_segmenter,
                ) is None:
                    _STOP.wait(idle_seconds)
            except Exception:
                logger.exception("Vision worker loop failed; retrying")
                _STOP.wait(min(5, idle_seconds * 4))
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=2)
        try:
            vision_repository.heartbeat_worker(
                database_url, worker_id, ready=False, status_detail="worker_stopped",
            )
        except Exception:
            logger.exception("Could not mark vision worker stopped")


def _request_stop(_signum, _frame):
    _STOP.set()


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    run_forever()


if __name__ == "__main__":
    main()
