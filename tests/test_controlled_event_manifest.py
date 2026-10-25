"""The event pilot must bind graph, event, builder and scorer inputs."""

import pytest

from experiments.controlled_event_manifest import (
    EVENT_PILOT_INPUTS, make_event_manifest, verify_event_manifest,
)


def test_event_manifest_rejects_source_drift_and_mislabelled_scope(tmp_path):
    paths = {}
    for name in EVENT_PILOT_INPUTS:
        file = tmp_path / f"{name}.txt"
        file.write_text(f"{name}\n", encoding="utf-8")
        paths[name] = file.name
    metadata = {"graph_crs": "EPSG:4326", "boundary_crs": "EPSG:4326",
                "event_scope": "synthetic_development_only", "campus_truth": False}
    manifest = make_event_manifest(tmp_path, paths, metadata)
    assert verify_event_manifest(manifest, tmp_path) == manifest["effective_inputs_sha256"]
    (tmp_path / paths["event_snapshot"]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_event_manifest(manifest, tmp_path)
    with pytest.raises(ValueError, match="event scope"):
        make_event_manifest(tmp_path, paths, {**metadata, "campus_truth": True})
