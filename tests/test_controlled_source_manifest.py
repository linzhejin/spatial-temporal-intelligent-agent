"""A formal source fingerprint must resolve to the bytes being judged."""

from pathlib import Path

import pytest

from experiments.controlled_source_manifest import (
    ROUTE_PILOT_INPUTS, make_source_manifest, verify_source_manifest,
)


INPUT_NAMES = (
    "graph", "poi", "boundary", "event_snapshot", "oracle_mode",
    "oracle_route", "oracle_existence", "generator", "templates", "information_snapshot",
    "coordinate_transform", "poi_taxonomy", "attribute_master",
    "anchor_builder", "anchors", "boundary_source", "spatial_snapshot_builder",
    "dataset_builder", "gold_contract", "source_manifest_builder",
)


def _files(root: Path) -> dict[str, str]:
    paths = {}
    for name in INPUT_NAMES:
        rel = f"inputs/{name}.txt"
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name, encoding="utf-8")
        paths[name] = rel
    return paths


def _meta():
    return {
        "graph_crs": "EPSG:4326", "boundary_crs": "EPSG:4326",
        "boundary_provenance": "config.CAMPUS_POLYS_GCJ via gcj02_to_wgs84",
        "event_scope": "controlled_synthetic",
    }


def test_source_manifest_is_deterministic_and_checks_actual_bytes(tmp_path):
    paths = _files(tmp_path)
    first = make_source_manifest(tmp_path, paths, _meta())
    second = make_source_manifest(tmp_path, dict(reversed(list(paths.items()))), _meta())
    assert first == second
    assert verify_source_manifest(first, tmp_path) == first["effective_inputs_sha256"]

    (tmp_path / paths["graph"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="graph.*sha256"):
        verify_source_manifest(first, tmp_path)


def test_manifest_rejects_missing_input_and_path_escape(tmp_path):
    paths = _files(tmp_path)
    paths.pop("event_snapshot")
    with pytest.raises(ValueError, match="event_snapshot"):
        make_source_manifest(tmp_path, paths, _meta())

    paths["event_snapshot"] = "../elsewhere.json"
    with pytest.raises(ValueError, match="outside"):
        make_source_manifest(tmp_path, paths, _meta())


def test_boundary_provenance_and_event_scope_are_required(tmp_path):
    paths = _files(tmp_path)
    metadata = _meta()
    metadata["boundary_provenance"] = ""
    with pytest.raises(ValueError, match="boundary_provenance"):
        make_source_manifest(tmp_path, paths, metadata)
    metadata = _meta()
    metadata.pop("event_scope")
    with pytest.raises(ValueError, match="event_scope"):
        make_source_manifest(tmp_path, paths, metadata)


def test_information_snapshot_is_required_for_compound_tool_cases(tmp_path):
    paths = _files(tmp_path)
    paths.pop("information_snapshot")
    with pytest.raises(ValueError, match="information_snapshot"):
        make_source_manifest(tmp_path, paths, _meta())


@pytest.mark.parametrize("name", ["coordinate_transform", "poi_taxonomy", "attribute_master",
                                  "anchor_builder", "anchors", "boundary_source",
                                  "spatial_snapshot_builder", "oracle_existence",
                                  "dataset_builder", "gold_contract",
                                  "source_manifest_builder"])
def test_generator_dependency_must_be_frozen(tmp_path, name):
    paths = _files(tmp_path)
    paths.pop(name)
    with pytest.raises(ValueError, match=name):
        make_source_manifest(tmp_path, paths, _meta())


def test_route_pilot_manifest_omits_unused_event_and_information_sources(tmp_path):
    paths = _files(tmp_path)
    pilot_paths = {name: paths[name] for name in ROUTE_PILOT_INPUTS}
    metadata = _meta()
    metadata["event_scope"] = "not_applicable"
    assert "event_snapshot" not in pilot_paths
    assert "information_snapshot" not in pilot_paths
    manifest = make_source_manifest(tmp_path, pilot_paths, metadata, profile="route_pilot")
    assert manifest["profile"] == "route_pilot"
    assert set(manifest["inputs"]) == ROUTE_PILOT_INPUTS
    assert verify_source_manifest(manifest, tmp_path) == manifest["effective_inputs_sha256"]
    with pytest.raises(ValueError, match="source inputs incomplete"):
        make_source_manifest(tmp_path, pilot_paths, metadata)


def test_text_source_hash_ignores_only_crlf_checkout_difference(tmp_path):
    paths = _files(tmp_path)
    graph_path = tmp_path / paths["graph"]
    graph_path.write_bytes(b"node-a\r\nnode-b\r\n")
    manifest = make_source_manifest(tmp_path, paths, _meta())
    assert manifest["inputs"]["graph"]["content_encoding"] == "utf8_lf_canonical_v1"
    graph_path.write_bytes(b"node-a\nnode-b\n")
    assert verify_source_manifest(manifest, tmp_path) == manifest["effective_inputs_sha256"]
    graph_path.write_bytes(b"node-a\nnode-c\n")
    with pytest.raises(ValueError, match="graph.*sha256"):
        verify_source_manifest(manifest, tmp_path)
