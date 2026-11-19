import tempfile
import uuid
import math
from pathlib import Path

import networkx as nx
import pytest
from shapely.geometry import LineString

from spatial import network
from spatial.coord_transform import gcj02_to_wgs84
from vision import road_association


def test_operational_candidates_receive_candidate_specific_association(monkeypatch):
    calls = []
    graph = object()
    monkeypatch.setattr(road_association, "_read_local_graph", lambda: graph)
    monkeypatch.setattr(road_association, "resolve_profile", lambda: {"profile": "test"})
    monkeypatch.setattr(road_association, "calibration_status",
                        lambda profile: {"status": "calibrated", "residual_m": 1.2})

    def associate(candidate, received_graph, anchor, **kwargs):
        calls.append((candidate["kind"], received_graph, anchor, kwargs))
        return {"status": "matched", "selected_edge": {"road_name": candidate["kind"]}}

    monkeypatch.setattr(road_association, "score_candidate_road_association", associate)
    result = {
        "media": {"width": 640, "height": 480},
        "candidates": [{"kind": "possible_flooding"}, {"kind": "possible_congestion"}],
    }
    job = {
        "purpose": "operational",
        "media_kind": "video",
        "anchor_gcj": {"lng": 114.36, "lat": 30.53},
        "camera_pose": {"trajectory": []},
        "observation_regions": [{"id": "lane-east"}],
    }

    road_association.annotate_candidate_associations(job, result)

    assert len(calls) == 2
    assert all(call[1] is graph for call in calls)
    assert all(call[2] == job["anchor_gcj"] for call in calls)
    assert all(call[3]["media"] == {"width": 640, "height": 480, "kind": "video"} for call in calls)
    assert all(candidate["road_association"]["status"] == "matched"
               for candidate in result["candidates"])


def test_trial_candidates_are_not_associated_with_campus_roads(monkeypatch):
    monkeypatch.setattr(road_association, "_read_local_graph",
                        lambda: (_ for _ in ()).throw(AssertionError("must not load roads")))
    monkeypatch.setattr(network, "load_or_download_network",
                        lambda: (_ for _ in ()).throw(AssertionError("must not download roads")))
    result = {"candidates": [{"kind": "possible_accident"}]}

    road_association.annotate_candidate_associations({"purpose": "trial"}, result)

    assert "road_association" not in result["candidates"][0]


def test_missing_local_road_cache_is_reported_without_downloading(monkeypatch):
    monkeypatch.setattr(network, "get_network", lambda: None)
    missing_cache = Path(tempfile.gettempdir()) / f"codex-missing-road-{uuid.uuid4().hex}.graphml"
    monkeypatch.setattr(road_association.config, "ROAD_NETWORK_CACHE", str(missing_cache))
    monkeypatch.setattr(network, "load_or_download_network",
                        lambda: (_ for _ in ()).throw(AssertionError("must not download roads")))
    monkeypatch.setattr(road_association, "resolve_profile", lambda: {})
    monkeypatch.setattr(road_association, "calibration_status", lambda profile: {"status": "nominal"})
    result = {"candidates": [{"kind": "possible_accident"}]}

    road_association.annotate_candidate_associations(
        {"purpose": "operational", "media_kind": "image"}, result,
    )

    assert result["candidates"][0]["road_association"]["status"] == "unavailable"
    assert result["candidates"][0]["road_association"]["reason_code"] == "road_network_unavailable"


def test_operational_association_projects_candidate_region_with_real_scorer(monkeypatch):
    pytest.importorskip("numpy")
    anchor = {"lng": 114.363, "lat": 30.5365, "crs": "GCJ02"}
    origin_lng, origin_lat = gcj02_to_wgs84(anchor["lng"], anchor["lat"])
    metres_per_degree = 111320.0
    cos_lat = math.cos(math.radians(origin_lat))

    def wgs_point(east_m, north_m):
        return (origin_lng + east_m / (metres_per_degree * cos_lat),
                origin_lat + north_m / metres_per_degree)

    graph = nx.MultiDiGraph()
    forward = LineString([wgs_point(-80, 0), wgs_point(80, 0)])
    graph.add_edge(1, 2, key=0, geometry=forward, name="候选正下方道路", length=160.0)
    graph.add_edge(2, 1, key=0, geometry=LineString(list(reversed(forward.coords))),
                   name="候选正下方道路", length=160.0)
    monkeypatch.setattr(road_association, "_read_local_graph", lambda: graph)
    monkeypatch.setattr(road_association, "resolve_profile", lambda: {
        "fx_px": 2000 / 3, "fy_px": 2000 / 3,
    })
    monkeypatch.setattr(road_association, "calibration_status", lambda profile: {
        "status": "nominal", "residual_m": None,
    })

    result = {
        "media": {"width": 1000, "height": 1000},
        "candidates": [{
            "kind": "possible_congestion",
            "evidence": {"summary": {"region_id": "lane-east"}},
        }],
    }
    job = {
        "purpose": "operational",
        "media_kind": "image",
        "anchor_gcj": anchor,
        "camera_pose": {"trajectory": [{
            "lng": origin_lng,
            "lat": origin_lat,
            "height_agl_m": 100,
            "gb_yaw_deg": 0,
            "gb_pitch_deg": -90,
            "gb_roll_deg": 0,
            "intrinsics": {"fx_px": 2000 / 3, "fy_px": 2000 / 3,
                           "cx_px": 500, "cy_px": 500},
        }]},
        "observation_regions": [{
            "id": "lane-east",
            "kind": "vehicle_lane",
            "polygon": [[0.45, 0.45], [0.55, 0.45], [0.55, 0.55], [0.45, 0.55]],
        }],
    }

    road_association.annotate_candidate_associations(job, result)

    association = result["candidates"][0]["road_association"]
    assert association["status"] == "matched"
    assert association["calibration_status"] == "nominal"
    assert association["selected_edge"]["road_name"] == "候选正下方道路"
    assert association["selected_edge"]["score"] >= 0.05
