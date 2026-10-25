"""A new source profile must not invalidate the frozen route pilot."""

from pathlib import Path

import pytest

from experiments.controlled_multi_turn_manifest import (
    MULTI_TURN_INPUTS, make_multi_turn_manifest, verify_multi_turn_manifest,
)


def _sources(root: Path):
    paths = {}
    for name in MULTI_TURN_INPUTS:
        path = root / "inputs" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        paths[name] = path.relative_to(root).as_posix()
    return paths


def test_multi_turn_manifest_freezes_helper_scorer_and_projection(tmp_path):
    paths = _sources(tmp_path)
    manifest = make_multi_turn_manifest(tmp_path, paths)
    assert manifest["profile"] == "multi_turn_pilot"
    assert verify_multi_turn_manifest(manifest, tmp_path) == manifest["effective_inputs_sha256"]
    missing = dict(paths)
    missing.pop("route_case_helpers")
    with pytest.raises(ValueError, match="incomplete"):
        make_multi_turn_manifest(tmp_path, missing)
    for name in ("route_case_helpers", "route_scorer", "observation_projection"):
        path = tmp_path / paths[name]
        path.write_text("changed", encoding="utf-8")
        with pytest.raises(ValueError, match=f"{name}.*sha256"):
            verify_multi_turn_manifest(manifest, tmp_path)
        path.write_text(name, encoding="utf-8")


def test_multi_turn_manifest_normalizes_checkout_newlines_and_rejects_escape(tmp_path):
    paths = _sources(tmp_path)
    path = tmp_path / paths["route_case_helpers"]
    path.write_bytes(b"line1\r\nline2\r\n")
    manifest = make_multi_turn_manifest(tmp_path, paths)
    path.write_bytes(b"line1\nline2\n")
    assert verify_multi_turn_manifest(manifest, tmp_path) == manifest["effective_inputs_sha256"]
    paths["route_case_helpers"] = "../outside.txt"
    with pytest.raises(ValueError, match="outside"):
        make_multi_turn_manifest(tmp_path, paths)
