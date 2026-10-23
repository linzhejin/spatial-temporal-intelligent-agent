"""The official benchmark selection must remain reproducible and unscored."""

import json

import pytest

from scripts.analysis.prepare_bfcl_official import CATEGORIES, select_subset


def _raw():
    return {category: b"".join(
        (json.dumps({"id": f"{category}_{index}", "question": [[{
            "role": "user", "content": "x"}]]}) + "\n").encode()
        for index in range(12)) for category in CATEGORIES}


def test_official_selection_is_deterministic_and_keeps_source_hashes():
    first = select_subset(_raw(), per_category=10)
    assert first == select_subset(_raw(), per_category=10)
    assert sum(len(group["selected_ids"]) for group in first["categories"].values()) == 40
    assert first["source_commit"]
    assert all(len(group["sha256"]) == 64 for group in first["categories"].values())
    assert "results" not in first


def test_official_selection_rejects_malformed_source():
    raw = _raw()
    raw[CATEGORIES[0]] = b'{"id":"fake_0","question":[]}\n'
    with pytest.raises(ValueError, match="invalid official"):
        select_subset(raw)
