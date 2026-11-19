"""Candidate-specific image-to-road association for operational vision jobs."""
from __future__ import annotations

import logging
from pathlib import Path

import config
from spatial import network
from spatial.vision_suggest import score_candidate_road_association
from vision.camera_profile import calibration_status, resolve_profile

logger = logging.getLogger(__name__)

_cached_graph = None
_cached_graph_signature = None


def _read_local_graph():
    """Load a local GraphML cache only; this path must never download OSM data."""
    global _cached_graph, _cached_graph_signature
    graph = network.get_network()
    if graph is not None:
        return graph
    cache_path = getattr(config, "ROAD_NETWORK_CACHE", None)
    if not cache_path:
        return None
    path = Path(cache_path)
    try:
        stat = path.stat()
    except OSError:
        return None
    signature = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if _cached_graph is None or _cached_graph_signature != signature:
        _cached_graph = network._load_graphml(str(path))
        _cached_graph_signature = signature
    return _cached_graph


def annotate_candidate_associations(job: dict, result: dict) -> None:
    """Attach a bounded, auditable road association to each operational candidate.

    Missing geometry, calibration, or local road data produces an explicit status;
    it never blocks the vision result and never triggers a network download.
    """
    if not isinstance(job, dict) or job.get("purpose", "operational") != "operational":
        return
    if not isinstance(result, dict) or result.get("research_replay"):
        return
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return

    try:
        graph = _read_local_graph()
    except Exception:
        logger.exception("Could not load cached road graph for candidate association")
        graph = None
    try:
        import vision.registration as registration
    except Exception:
        logger.exception("Candidate road projection runtime is unavailable")
        registration = None

    media = result.get("media") if isinstance(result.get("media"), dict) else {}
    media = dict(media)
    media.setdefault("kind", job.get("media_kind"))
    pose = job.get("camera_pose")
    anchor = job.get("anchor_gcj")
    regions = job.get("observation_regions")
    try:
        profile = resolve_profile()
        calibration = calibration_status(profile)
    except Exception:
        logger.exception("Camera calibration status is unavailable for candidate association")
        profile = None
        calibration = {"status": "unavailable", "residual_m": None}

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            candidate["road_association"] = score_candidate_road_association(
                candidate, graph, anchor, observation_regions=regions,
                camera_pose=pose, media=media, registration=registration,
                camera_profile=profile, calibration=calibration,
            )
        except Exception:
            logger.exception("Could not associate vision candidate with a road")
            candidate["road_association"] = {
                "status": "unavailable",
                "basis": "candidate_region_projection",
                "ranked_edges": [],
                "selected_edge": None,
                "reason_code": "association_failed",
            }
