"""Regression tests for road network cache loading and annotation merge."""

import json
import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial import network


class _FakeGeometry:
    """Minimal stand-in for shapely geometry so tests stay dependency-light."""

    wkt = "LINESTRING (114.35 30.53, 114.36 30.53)"


def _make_graph(n_edges: int = 3) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for i in range(n_edges + 1):
        G.add_node(str(i), x=114.35 + i * 0.001, y=30.53 + i * 0.001)
    for i in range(n_edges):
        attrs = {"length": 100.0 + i, "highway": "footway"}
        if i == 0:
            attrs["geometry"] = _FakeGeometry()
        G.add_edge(str(i), str(i + 1), **attrs)
    return G


class TestNetworkCacheRecovery:
    @pytest.mark.parametrize("content", [b"", b"<graphml></graphml>", b"not a graphml file"])
    def test_broken_graphml_cache_is_rebuilt(self, tmp_path, monkeypatch, content):
        cache_file = tmp_path / "whu_road_network.graphml"
        cache_file.write_bytes(content)
        monkeypatch.setattr(network, "_cache_path", lambda: str(cache_file))
        monkeypatch.setattr(
            network,
            "_annotations_path",
            lambda: str(tmp_path / "missing_annotations.json"),
        )
        monkeypatch.setattr(network, "_download_network", lambda bbox=None: _make_graph())

        # 清除之前测试可能设置的全局状态
        network._G = None
        network._annotation_coverage_rate = None

        try:
            G = network.load_or_download_network()
            assert G.number_of_edges() == 3
            assert cache_file.stat().st_size > 0
            reloaded = nx.read_graphml(str(cache_file))
            assert reloaded.number_of_edges() == 3
        finally:
            network._G = None
            network._annotation_coverage_rate = None


class TestMergeAnnotations:
    def test_annotations_path_uses_explicit_environment_override(self, monkeypatch, tmp_path):
        custom = tmp_path / "oge-runtime.json"
        monkeypatch.setenv("ROAD_ANNOTATIONS_PATH", str(custom))
        assert network._annotations_path() == str(custom)

    def test_merge_annotations_applies_attributes_and_counts_coverage(self, tmp_path):
        G = _make_graph(4)
        ann_path = tmp_path / "road_annotations.json"
        ann_path.write_text(
            json.dumps(
                {
                    "edges": [
                        {
                            "edge_id": ["0", "1", 0],
                            "u": "0",
                            "v": "1",
                            "slope_level": 3,
                            "scenery_level": 4,
                            "name": "test road",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        rate = network._merge_annotations(G, str(ann_path))

        assert rate == pytest.approx(0.25)
        assert G["0"]["1"][0]["slope_level"] == 3
        assert G["0"]["1"][0]["scenery_level"] == 4
        assert G["0"]["1"][0]["name"] == "test road"

    def test_course_geometry_does_not_receive_stale_shape_annotations(self, tmp_path):
        G = _make_graph(1)
        edge = G["0"]["1"][0]
        edge.update({"name": "course name", "course_name_source": "course",
                     "course_geometry_replaced": True})
        ann_path = tmp_path / "road_annotations.json"
        ann_path.write_text(json.dumps({"edges": [{
            "edge_id": ["0", "1", 0], "u": "0", "v": "1",
            "slope_level": 5, "scenery_level": 5,
            "name": "stale OSM name", "walk_penalty": 2.5,
        }]}), encoding="utf-8")

        network._merge_annotations(G, str(ann_path))

        assert "slope_level" not in edge
        assert "scenery_level" not in edge
        assert edge["name"] == "course name"
        assert edge["walk_penalty"] == 2.5


def test_osm_provenance_merge_preserves_course_source_refs():
    G = _make_graph(1)
    course_ref = {"source": "WHU coursework", "id": "course:road:0001",
                  "license": "not_provided_with_source"}
    G["0"]["1"][0].update({"osmid": "123", "source_refs": [course_ref]})

    network._mark_osm_provenance(G)

    refs = G["0"]["1"][0]["source_refs"]
    assert course_ref in refs
    assert {"source": "OpenStreetMap", "id": "way/123", "license": "ODbL-1.0"} in refs


class TestGraphmlRoundTrip:
    def test_load_graphml_preserves_int_node_ids(self, tmp_path):
        G = nx.MultiDiGraph()
        G.add_node(13239152642, x=114.35, y=30.53)
        G.add_node(286074417, x=114.36, y=30.53)
        G.add_edge(13239152642, 286074417, length=10.0)
        path = str(tmp_path / "roundtrip.graphml")

        network._save_graphml(G, path)
        loaded = network._load_graphml(path)

        assert 13239152642 in loaded.nodes
        assert 286074417 in loaded.nodes
        assert all(isinstance(n, int) for n in loaded.nodes)

    def test_graphml_round_trip_omits_null_attributes_without_serializing_fake_values(self, tmp_path):
        G = nx.MultiDiGraph()
        G.graph.update(node_default={}, edge_default={})
        G.add_node(1, x=114.35, y=30.53, optional=None)
        G.add_node(2, x=114.36, y=30.53)
        G.add_edge(1, 2, length=10.0, course_surface=None)
        path = str(tmp_path / "null-attributes.graphml")

        network._save_graphml(G, path)
        loaded = network._load_graphml(path)

        assert "optional" not in loaded.nodes[1]
        assert all("course_surface" not in data for _, _, data in loaded.edges(data=True))


def test_routing_index_reuses_mode_graph_and_invalidates_on_reload():
    from spatial.routing_index import get_routing_index, invalidate_routing_index

    graph = _make_graph(4)
    first = get_routing_index(graph).for_mode("walk")
    second = get_routing_index(graph).for_mode("walk")
    assert first is second

    invalidate_routing_index()
    third = get_routing_index(graph).for_mode("walk")
    assert third is not first
