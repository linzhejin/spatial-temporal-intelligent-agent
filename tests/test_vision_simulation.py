"""Public footage can only get a clearly simulated campus association."""

import networkx as nx

from experiments.vision_simulation import build_manifest


def test_simulated_assignment_keeps_truth_boundary_and_detour(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_node("1", x=114.36, y=30.54)
    graph.add_node("2", x=114.361, y=30.54)
    graph.add_node("3", x=114.3605, y=30.541)
    graph.add_edge("1", "2", key=0, length=50, highway="footway")
    graph.add_edge("1", "3", key=0, length=70, highway="footway")
    graph.add_edge("3", "2", key=0, length=70, highway="footway")
    video = tmp_path / "public.mp4"
    video.write_bytes(b"public fixture video")
    manifest = build_manifest(graph, "a" * 64, [video], cases_per_video=1)
    case = manifest["videos"][0]["assigned_cases"][0]
    assert manifest["source_spatial_truth"] == "Songdo, not Wuhan"
    assert manifest["campus_location_status"] == "synthetic_random_assignment_only"
    assert manifest["publication_status"] == "never_publish_without_review"
    assert manifest["manager_review_status"] == "not_confirmed"
    assert case["direct_length_m"] == 50
    assert case["detour_after_hypothetical_closure_m"] == 140
    assert case["effect"]["status"] == "hypothetical_unpublished"
    assert len(manifest["videos"][0]["sha256"]) == 64
