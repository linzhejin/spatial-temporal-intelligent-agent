"""Isolated worker for bounded aerial image/video inference jobs."""
from __future__ import annotations

import logging
import os
import signal
import importlib.util
import hashlib
import json
import sys
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
from vision.assisted_image import (
    _TrialNoVehicleDetector, _release_inference_memory,
    _analyze_assisted_flood_image, _merge_assisted_flood_results,
)
from vision.assisted_image_supervisor import run_supervised_assisted_image
from vision.model_release import assisted_flood_runtime_config, check_optional_model_release
from vision.specialized import DoodleverseFloodSegmenter, OnnxAiderClassifier, OnnxFloodSegmenter
from vision.temporal_trial import OnnxTemporalAccident

logger = logging.getLogger(__name__)
VISION_CAPACITY_LOCK_KEY = 735182640120
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


def load_trial_models(*, include_assisted_flood=False):
    """Load explicitly configured test weights without changing release capabilities."""
    capabilities = {
        "trial_accident_supported": False, "trial_flood_supported": False,
        "flood_assistance_supported": False,
        "flood_assistance_release_status": "disabled",
    }
    models = {"accident": None, "flood": None}
    # The assisted role serves public trials and campus image checks through
    # the same Doodleverse path; don't keep legacy comparison sessions resident.
    doodleverse_assisted_mode = bool(
        include_assisted_flood and config.VISION_FLOOD_ENABLED
        and config.VISION_FLOOD_RELEASE_MODE == "assisted_review"
    )
    trial_weights = (
        (("accident", config.VISION_TRIAL_ACCIDENT_MODEL_PATH),
         ("flood", config.VISION_TRIAL_FLOOD_MODEL_PATH))
        if config.VISION_TRIAL_ENABLED and not doodleverse_assisted_mode else ()
    )
    for kind, path in trial_weights:
        if not path:
            continue
        try:
            with Path(path).open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            if kind == "accident":
                accident_kind = config.VISION_TRIAL_ACCIDENT_KIND
                if accident_kind == "scene":
                    model = OnnxAiderClassifier(path, threshold=config.VISION_TRIAL_ACCIDENT_THRESHOLD,
                                                model_version="trial-sha256:" + digest)
                elif accident_kind == "temporal":
                    model = OnnxTemporalAccident(path, threshold=config.VISION_TRIAL_ACCIDENT_THRESHOLD,
                                                 model_version="trial-sha256:" + digest)
                    capabilities["trial_accident_media"] = "video"
                else:
                    raise ValueError("未知的事故试用模型类型。")
            else:
                model = OnnxFloodSegmenter(path, min_area_ratio=config.VISION_TRIAL_FLOOD_MIN_AREA_RATIO,
                                           flood_probability_threshold=config.VISION_TRIAL_FLOOD_PROBABILITY,
                                           model_version="trial-sha256:" + digest)
                if config.VISION_TRIAL_FLOOD_MODEL_ID:
                    model.model_id = config.VISION_TRIAL_FLOOD_MODEL_ID
            models[kind] = model
            capabilities["trial_" + kind + "_supported"] = True
        except Exception:
            logger.exception("Could not load %s trial model", kind)
    if include_assisted_flood:
        if not config.VISION_FLOOD_ENABLED:
            capabilities["flood_assistance_release_status"] = "disabled"
        elif config.VISION_FLOOD_RELEASE_MODE != "assisted_review":
            capabilities["flood_assistance_release_status"] = "unsupported_release_mode"
        elif not config.VISION_FLOOD_MODEL_PATH:
            capabilities["flood_assistance_release_status"] = "model_missing"
        elif not config.VISION_MODEL_PATH or not Path(config.VISION_MODEL_PATH).is_file():
            capabilities["flood_assistance_release_status"] = "traffic_detector_missing"
        else:
            try:
                model_path = config.VISION_FLOOD_MODEL_PATH
                with Path(model_path).open("rb") as file:
                    digest = hashlib.file_digest(file, "sha256").hexdigest()
                runtime_config = assisted_flood_runtime_config(
                    probability_threshold=config.VISION_FLOOD_PROBABILITY_THRESHOLD,
                    min_area_ratio=config.VISION_FLOOD_MIN_AREA_RATIO,
                )
                release = check_optional_model_release(
                    "flood", model_path, config.VISION_FLOOD_REVISION,
                    config.VISION_OPTIONAL_MODELS_RELEASE_MANIFEST, None,
                    runtime_config, release_mode="assisted_review",
                )
                capabilities["flood_assistance_release_status"] = str(
                    release.get("reason", "release_invalid"),
                )
                if release.get("approved") is True:
                    model = DoodleverseFloodSegmenter(
                        model_path,
                        model_version=config.VISION_FLOOD_REVISION,
                        probability_threshold=config.VISION_FLOOD_PROBABILITY_THRESHOLD,
                        min_area_ratio=config.VISION_FLOOD_MIN_AREA_RATIO,
                        single_image_trigger_threshold=config.VISION_FLOOD_MIN_AREA_RATIO,
                        runtime_python=config.VISION_FLOOD_RUNTIME_PYTHON,
                        timeout_seconds=config.VISION_FLOOD_TIMEOUT_SECONDS,
                    )
                    model.model_sha256 = digest
                    models["flood"] = model
                    capabilities.update({
                        "flood_assistance_supported": True,
                        "flood_assistance_release_status": "approved",
                        # The existing release evidence covers user-marked roads,
                        # not automatic whole-image proposals.
                        "flood_assistance_automatic_supported": False,
                        "flood_assistance_model_id": model.model_id,
                        "flood_assistance_model_version": model.model_version,
                        "flood_assistance_model_sha256": digest,
                        "trial_flood_supported": True,
                        "trial_flood_media": "image",
                        "trial_flood_automatic_supported": False,
                        "trial_flood_model_id": model.model_id,
                    })
            except Exception:
                capabilities["flood_assistance_release_status"] = "model_load_failed"
                logger.exception("Doodleverse assisted flood model could not be initialized")
    return models, capabilities


def _requires_vision_capacity_lock():
    return bool(
        config.VISION_TRIAL_ENABLED
        or (config.VISION_FLOOD_ENABLED and config.VISION_FLOOD_RELEASE_MODE == "assisted_review")
    )


def process_next_job(database_url: str, worker_id: str, lease_seconds: int = 90, *,
                     detector=None, accident_model=None, flood_segmenter=None,
                     trial_accident_model=None, trial_flood_segmenter=None, trial_only=False,
                     assisted_flood_supported=False,
                     automatic_flood_supported=False,
                     reserve_assisted=False,
                     on_locked_completion=None):
    options = {"detector": detector, "accident_model": accident_model, "flood_segmenter": flood_segmenter,
               "trial_accident_model": trial_accident_model, "trial_flood_segmenter": trial_flood_segmenter,
               "trial_only": trial_only,
               "assisted_flood_supported": assisted_flood_supported,
               "automatic_flood_supported": automatic_flood_supported,
               "reserve_assisted": reserve_assisted}
    if not _requires_vision_capacity_lock():
        return _process_next_job_unlocked(database_url, worker_id, lease_seconds, **options)
    # A single host may not have enough memory to run the campus and trial models together.
    # Keep a session-level lock alive independently of job leases, including across commits.
    with database.connect(database_url) as conn:
        acquired = conn.execute("SELECT pg_try_advisory_lock(%s) AS acquired",
                                (VISION_CAPACITY_LOCK_KEY,)).fetchone()["acquired"]
        conn.commit()
        if not acquired:
            return None
        stop_guard = threading.Event()
        lock_lost = threading.Event()

        def keep_capacity_lock():
            while not stop_guard.wait(1.0):
                try:
                    conn.execute("SELECT 1").fetchone()
                    conn.commit()
                except Exception:
                    lock_lost.set()
                    logger.exception("Vision capacity lock connection was lost")
                    return

        guard = threading.Thread(target=keep_capacity_lock, name="vision-capacity-lock", daemon=True)
        guard.start()
        try:
            try:
                outcome = _process_next_job_unlocked(database_url, worker_id, lease_seconds,
                                                      capacity_lock_lost=lock_lost, **options)
            except Exception:
                if on_locked_completion is not None:
                    options["trial_accident_model"] = options["trial_flood_segmenter"] = None
                    trial_accident_model = trial_flood_segmenter = None
                    on_locked_completion()
                raise
            if outcome is not None and on_locked_completion is not None:
                options["trial_accident_model"] = options["trial_flood_segmenter"] = None
                trial_accident_model = trial_flood_segmenter = None
                on_locked_completion()
            return outcome
        finally:
            stop_guard.set()
            guard.join(timeout=2)
            if not guard.is_alive():
                try:
                    conn.execute("SELECT pg_advisory_unlock(%s)", (VISION_CAPACITY_LOCK_KEY,))
                    conn.commit()
                except Exception:
                    logger.warning("Vision capacity lock was already lost", exc_info=True)


def _process_next_job_unlocked(database_url: str, worker_id: str, lease_seconds: int = 90, *,
                               detector=None, accident_model=None, flood_segmenter=None,
                               trial_accident_model=None, trial_flood_segmenter=None, trial_only=False,
                               assisted_flood_supported=False,
                               automatic_flood_supported=False,
                               reserve_assisted=False,
                               capacity_lock_lost=None):
    allow_trial = trial_accident_model is not None or trial_flood_segmenter is not None
    job = vision_repository.claim_next_job(database_url, worker_id, lease_seconds,
                                          allow_trial=allow_trial, trial_only=trial_only,
                                          allow_assisted=assisted_flood_supported,
                                          reserve_assisted=reserve_assisted)
    if not job:
        return None
    job_id = job["job_id"]
    job_started = time.monotonic()
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
        has_automatic_marker = any(
            isinstance(region, dict) and region.get("kind") == "automatic_flood_analysis"
            for region in job.get("observation_regions") or []
        )
        if has_automatic_marker and not automatic_flood_supported:
            raise RuntimeError("自动积水识别尚未通过验收，不能处理此任务。")
        is_assisted_flood_job = bool(
            assisted_flood_supported and job.get("purpose") == "operational"
            and job.get("media_kind") == "image"
            and any(isinstance(region, dict)
                    and region.get("flood_assistance_mode") == "assisted_review"
                    for region in job.get("observation_regions") or [])
        )
        specialized_job = job.get("purpose") == "trial" or is_assisted_flood_job
        trial_flood_video_unchecked = bool(
            job.get("purpose") == "trial" and job.get("media_kind") == "video"
            and getattr(trial_flood_segmenter, "single_image_trigger_threshold", None) is not None
        )
        if job.get("purpose") == "trial":
            if not allow_trial:
                raise RuntimeError("试用模型尚未就绪，无法分析这条测试影像。")
            accident_model = trial_accident_model
            flood_segmenter = None if trial_flood_video_unchecked else trial_flood_segmenter
        elif is_assisted_flood_job:
            if trial_flood_segmenter is None:
                raise RuntimeError("积水辅助识别未就绪，不能把任务当作未发现积水。")
            accident_model, flood_segmenter = None, trial_flood_segmenter
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
            if lease_lost.is_set() or (capacity_lock_lost is not None and capacity_lock_lost.is_set()):
                lease_lost.set()
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

        deadline = time.monotonic() + config.VISION_MAX_ANALYSIS_SECONDS
        temporal_accident = (accident_model if job.get("purpose") == "trial" and
                             getattr(accident_model, "trial_video_only", False) else None)
        job_regions = job.get("observation_regions") or []
        kinds = {region.get("kind") for region in job_regions}
        automatic_flood = bool(
            job.get("media_kind") == "image" and isinstance(flood_segmenter, DoodleverseFloodSegmenter)
            and automatic_flood_supported
            and (has_automatic_marker or (job.get("purpose") == "trial" and "road_surface" not in kinds))
        )
        analysis_regions = [region for region in job_regions
                            if region.get("kind") != "automatic_flood_analysis"]
        temporal_only = bool(temporal_accident is not None and job["media_kind"] == "video" and
                             not (flood_segmenter is not None and "road_surface" in kinds))
        supervised_trial_image = bool(
            job.get("purpose") == "trial" and job.get("media_kind") == "image"
            and isinstance(flood_segmenter, DoodleverseFloodSegmenter)
            and ("road_surface" in kinds or automatic_flood)
        )
        image_deadline = None
        if is_assisted_flood_job or supervised_trial_image:
            image_deadline = job_started + min(float(config.VISION_FLOOD_TIMEOUT_SECONDS), 120.0)
            save_progress({"progress": {"phase": "analyzing", "percent": 0}, "checkpoint": None})
            monitor_stop = threading.Event()
            cancel_pending = threading.Event()

            def monitor_cancellation():
                while not monitor_stop.is_set():
                    try:
                        state = vision_repository.get_job_cancel_state(database_url, job_id, worker_id)
                    except Exception:
                        logger.exception("Could not check image cancellation; ending the leased run")
                        lease_lost.set()
                        return
                    if state is None:
                        lease_lost.set()
                        return
                    if state.get("cancel_requested"):
                        cancel_pending.set()
                        return
                    monitor_stop.wait(0.5)

            monitor = threading.Thread(target=monitor_cancellation, name=f"vision-cancel-{job_id[:8]}", daemon=True)
            monitor.start()

            def supervised_cancelled():
                if capacity_lock_lost is not None and capacity_lock_lost.is_set():
                    lease_lost.set()
                return cancel_pending.is_set() or lease_lost.is_set()

            try:
                result = run_supervised_assisted_image({
                    "media_path": str(path), "anchor_gcj": job["anchor_gcj"],
                    "camera_stabilized": job["camera_stabilized"],
                    "regions": analysis_regions,
                    "automatic_flood": automatic_flood,
                    "resume_state": job.get("checkpoint"),
                    "traffic_required": is_assisted_flood_job,
                    "traffic_model_path": config.VISION_MODEL_PATH,
                    "traffic_model_id": config.VISION_MODEL_ID,
                    "traffic_model_version": config.VISION_MODEL_REVISION,
                    "flood_model_path": flood_segmenter.model_path,
                    "flood_model_version": flood_segmenter.model_version,
                    "flood_model_sha256": getattr(flood_segmenter, "model_sha256", None),
                    "probability_threshold": flood_segmenter.probability_threshold,
                    "min_area_ratio": flood_segmenter.min_area_ratio,
                    "single_image_trigger_threshold": flood_segmenter.single_image_trigger_threshold,
                    "runtime_python": flood_segmenter.runtime_python,
                    "max_frame_pixels": config.VISION_TRIAL_MAX_FRAME_PIXELS if supervised_trial_image else None,
                }, deadline=image_deadline, cancel_check=supervised_cancelled)
            finally:
                monitor_stop.set()
                monitor.join(timeout=0.25)
        elif temporal_only:
            # An accident-only trial has no need for vehicle tracking or ground-road matching.
            result = {"candidates": [], "metrics": {}, "safety": {"automatically_changes_routing": False},
                      "trial_temporal_only": True}
        else:
            is_trial = job.get("purpose") == "trial"
            trial_options = ({"target_sample_fps": min(1.0, config.VISION_TRIAL_SAMPLE_FPS),
                              "max_frame_pixels": config.VISION_TRIAL_MAX_FRAME_PIXELS,
                              "max_video_seconds": config.VISION_TRIAL_MAX_VIDEO_SECONDS} if is_trial else {})
            result = analyze_media(
                path, job["media_kind"], job["anchor_gcj"],
                camera_stabilized=job["camera_stabilized"],
                detector=_TrialNoVehicleDetector() if specialized_job else detector,
                accident_model=None if temporal_accident else accident_model,
                flood_segmenter=flood_segmenter,
                regions=job.get("observation_regions") or [],
                progress_callback=save_progress,
                resume_state=job.get("checkpoint"),
                cancel_check=cancellation_requested,
                require_flood_success=is_assisted_flood_job,
                **trial_options,
            )
        if automatic_flood:
            result["automatic_flood"] = True
        if job.get("purpose") == "trial":
            result["trial_specialized_only"] = True
        elif is_assisted_flood_job:
            result["flood_assistance"] = {
                "mode": "assisted_review",
                "field_validated": False,
                "human_confirmation_required": True,
                "automatic_flood": automatic_flood,
                "supported_media": ["image"],
                "model_id": getattr(flood_segmenter, "model_id", "doodleverse-floodnet-resunet-v7"),
                "model_version": getattr(flood_segmenter, "model_version", "unverified"),
                "model_sha256": getattr(flood_segmenter, "model_sha256", None),
                "probability_threshold": getattr(flood_segmenter, "probability_threshold", None),
                "min_area_ratio": getattr(flood_segmenter, "min_area_ratio", None),
            }
        temporal_result = None
        if temporal_accident is not None and job["media_kind"] == "video":
            temporal_result = temporal_accident.analyze_video(
                path, cancel_check=cancellation_requested, deadline=deadline,
            )
            result.setdefault("candidates", []).extend(temporal_result["candidates"])
            result.setdefault("trial_model_details", {})["accident"] = {
                "model_version": getattr(temporal_accident, "model_version", "unverified"),
                "scope": "temporal_window", "windows_analyzed": temporal_result["windows_analyzed"],
                "windows": temporal_result["windows"],
            }
        if job.get("purpose") == "trial":
            safety = result.get("safety") or {}
            counts = (result.get("metrics") or {}).get("specialized_model_frames") or {}
            result["trial_coverage"] = {
                "accident": ("not_configured" if accident_model is None else
                             "video_required" if temporal_accident is not None and job["media_kind"] != "video" else
                             "checked" if temporal_result is not None and temporal_result["windows_analyzed"] > 0 else
                             "region_required" if not kinds.intersection({"vehicle_lane", "road_surface"}) else
                             "failed" if safety.get("accident_recognition_supported") is False else
                             "checked" if counts.get("accident", 0) > 0 else "no_observation"),
                "flood": ("video_not_supported" if trial_flood_video_unchecked else
                          "not_configured" if flood_segmenter is None else
                          "region_required" if "road_surface" not in kinds and not automatic_flood else
                          "failed" if safety.get("flood_segmentation_supported") is False else
                          "checked" if counts.get("flood", 0) > 0 else "no_observation"),
            }
        if image_deadline is not None and time.monotonic() >= image_deadline:
            raise TimeoutError("积水图片分析超过 120 秒时限")
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
        if lease_lost.is_set() or (capacity_lock_lost is not None and capacity_lock_lost.is_set()):
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
    trial_models = {"accident": None, "flood": None}
    ready = False
    status_detail = ""
    try:
        role = config.VISION_WORKER_ROLE
        if role not in {"shared", "operational", "trial"}:
            raise RuntimeError("invalid vision worker role")
        missing = [
            name for name in ("onnxruntime", "numpy", "cv2", "PIL", "scipy")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise RuntimeError("missing vision dependencies: " + ", ".join(missing))
        if role == "trial":
            import cv2
            cv2.setNumThreads(1)
        optional_capabilities = {}
        if role != "trial":
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
                flood_enabled=(
                    config.VISION_FLOOD_ENABLED
                    and config.VISION_FLOOD_RELEASE_MODE != "assisted_review"
                ),
                flood_revision=config.VISION_FLOOD_REVISION,
                flood_min_area_ratio=config.VISION_FLOOD_MIN_AREA_RATIO,
                release_manifest_path=config.VISION_OPTIONAL_MODELS_RELEASE_MANIFEST,
                readiness_report_path=config.VISION_FIELD_READINESS_REPORT,
            )
        trial_capabilities = {}
        if role != "operational":
            trial_models, trial_capabilities = load_trial_models(include_assisted_flood=role == "trial")
        optional_capabilities.update(trial_capabilities)
        status_detail = json.dumps(optional_capabilities, separators=(",", ":"))
        ready = role != "trial" or any(trial_models.values()) or (
            trial_capabilities.get("flood_assistance_supported") is True
        )
    except Exception as error:
        role = role if role in {"shared", "operational", "trial"} else "shared"
        status_detail = f"{type(error).__name__}: {error}"[:500]
        logger.exception("Vision worker model initialization failed; uploads remain disabled")

    heartbeat_stop = threading.Event()

    def keep_worker_heartbeat():
        while not heartbeat_stop.is_set():
            try:
                vision_repository.heartbeat_worker(
                    database_url, worker_id, ready=ready, status_detail=status_detail,
                    role=role,
                )
            except Exception:
                logger.exception("Could not publish vision worker readiness")
            heartbeat_stop.wait(heartbeat_seconds)

    heartbeat = threading.Thread(
        target=keep_worker_heartbeat, name="vision-worker-heartbeat", daemon=True,
    )
    heartbeat.start()

    def release_trial_models():
        nonlocal trial_models
        trial_models = {"accident": None, "flood": None}
        _release_inference_memory()

    try:
        while not _STOP.is_set():
            if not ready:
                _STOP.wait(idle_seconds)
                continue
            try:
                outcome = process_next_job(
                    database_url, worker_id, detector=detector,
                    accident_model=accident_model, flood_segmenter=flood_segmenter,
                    trial_accident_model=trial_models["accident"],
                    trial_flood_segmenter=trial_models["flood"],
                    trial_only=role == "trial",
                    assisted_flood_supported=(
                        role == "trial" and optional_capabilities.get("flood_assistance_supported") is True
                    ),
                    automatic_flood_supported=(
                        role == "trial" and optional_capabilities.get("flood_assistance_automatic_supported") is True
                    ),
                    reserve_assisted=(
                        role != "trial" and config.VISION_FLOOD_ENABLED
                        and config.VISION_FLOOD_RELEASE_MODE == "assisted_review"
                    ),
                    on_locked_completion=release_trial_models if role == "trial" else None,
                )
                if outcome is None:
                    _STOP.wait(idle_seconds)
                elif role == "trial":
                    # systemd starts a fresh isolated process for the next trial job.
                    break
            except Exception:
                logger.exception("Vision worker loop failed; retrying")
                if role == "trial":
                    break
                _STOP.wait(min(5, idle_seconds * 4))
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=2)
        try:
            vision_repository.heartbeat_worker(
                database_url, worker_id, ready=False, status_detail="worker_stopped",
                role=role,
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
