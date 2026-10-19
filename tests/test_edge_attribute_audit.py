import networkx as nx

from scripts.network.build_edge_attribute_master import (
    graph_fingerprint,
    geometry_hash,
    segment_id,
)
from scripts.validate.audit_edge_attribute_master import audit_attribute_master
from spatial.edge_attributes import source_manifest_fingerprint


def _fixture(terrain_status="field_verified", scenery_status="field_verified"):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.0, y=30.0)
    graph.add_node(2, x=114.001, y=30.0)
    graph.add_edge(1, 2, key=0, length=96.0)

    sources = {"schema_version": 1, "sources": []}
    refs = [{"source": "test", "id": "edge-1"}]
    geom_hash = geometry_hash(graph, 1, 2, 0)
    master = {
        "schema_version": 1,
        "network_version": graph_fingerprint(graph),
        "source_manifest_fingerprint": source_manifest_fingerprint(sources),
        "records": [{
            "segment_id": segment_id(refs, geom_hash, "1>2:0"),
            "network_bindings": [{
                "edge_id": [1, 2, 0],
                "geometry_hash": geom_hash,
                "source_refs": refs,
            }],
            "terrain": {
                "grade_abs_pct": 2.5 if terrain_status == "field_verified" else None,
                "slope_level": 2 if terrain_status == "field_verified" else None,
                "confidence": "high" if terrain_status == "field_verified" else "unknown",
                "verification_status": terrain_status,
            },
            "scenery": {
                "scenery_score": 4 if scenery_status == "field_verified" else None,
                "scenery_level": 4 if scenery_status == "field_verified" else None,
                "confidence": "high" if scenery_status == "field_verified" else "unknown",
                "verification_status": scenery_status,
            },
        }],
    }
    return master, graph, sources


def test_attribute_audit_accepts_current_complete_verified_bindings():
    master, graph, sources = _fixture()

    report = audit_attribute_master(master, graph, sources)

    assert report["ready"] is True
    assert report["degraded"] is False
    assert report["coverage"]["missing_edges"] == 0
    assert report["attributes"]["terrain"]["verified_usable_edges"] == 1
    assert report["attributes"]["scenery"]["verified_usable_edges"] == 1


def test_attribute_audit_reports_unknown_scenery_without_calling_it_verified():
    master, graph, sources = _fixture(scenery_status="source_only")

    report = audit_attribute_master(master, graph, sources)

    assert report["ready"] is True
    assert report["degraded"] is True
    assert report["attributes"]["scenery"]["verified_usable_edges"] == 0
    assert {item["code"] for item in report["blockers"]} >= {
        "scenery_preference_evidence_missing",
    }


def test_attribute_audit_flags_stale_graph_binding():
    master, graph, sources = _fixture()
    master["network_version"] = "stale-network-version"

    report = audit_attribute_master(master, graph, sources)

    assert report["ready"] is False
    assert "network_version_mismatch" in {item["code"] for item in report["blockers"]}
