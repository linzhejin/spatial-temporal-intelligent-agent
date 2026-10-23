"""Selection and scoring must never silently expand or shrink BFCL O1."""

import hashlib
import json

import pytest

from experiments.bfcl_official import (
    build_run_ids,
    validate_cached_sources,
    validate_official_scores,
    validate_result_subset,
)


def _fixture(tmp_path):
    categories = {}
    cache = tmp_path / "cache"
    cache.mkdir()
    for category in ("multi_turn_base", "multi_turn_miss_func"):
        data = b"".join((json.dumps({"id": f"{category}_{n}", "question": []})
                         + "\n").encode() for n in range(3))
        (cache / f"BFCL_v4_{category}.json").write_bytes(data)
        categories[category] = {"sha256": hashlib.sha256(data).hexdigest(),
                                "selected_ids": [f"{category}_0", f"{category}_2"]}
    return {"source_commit": "a" * 40, "categories": categories}, cache


def test_selection_verifies_raw_hash_and_exact_ids(tmp_path):
    manifest, cache = _fixture(tmp_path)
    validate_cached_sources(manifest, cache)
    assert build_run_ids(manifest) == {
        category: info["selected_ids"] for category, info in manifest["categories"].items()
    }
    (cache / "BFCL_v4_multi_turn_base.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_cached_sources(manifest, cache)


def test_missing_selected_source_id_is_rejected(tmp_path):
    manifest, cache = _fixture(tmp_path)
    manifest["categories"]["multi_turn_base"]["selected_ids"] = ["multi_turn_base_99"]
    with pytest.raises(ValueError, match="source IDs"):
        validate_cached_sources(manifest, cache)


def test_result_validation_requires_exact_subset(tmp_path):
    manifest, _ = _fixture(tmp_path)
    root = tmp_path / "result" / "DeepSeek-V3.2-Exp-FC"
    root.mkdir(parents=True)
    for category in manifest["categories"]:
        path = root / f"BFCL_v4_{category}_result.json"
        path.write_text("\n".join(json.dumps({"id": case_id, "result": []})
                                  for case_id in manifest["categories"][category]["selected_ids"]),
                        encoding="utf-8")
    report = validate_result_subset(manifest, tmp_path / "result", "DeepSeek-V3.2-Exp-FC")
    assert report["total"] == 4
    with pytest.raises(ValueError, match="generation metadata"):
        validate_result_subset(manifest, tmp_path / "result", "DeepSeek-V3.2-Exp-FC",
                               require_generation_metadata=True)
    (root / "BFCL_v4_multi_turn_base_result.json").write_text(
        json.dumps({"id": "multi_turn_base_0", "result": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="selected IDs"):
        validate_result_subset(manifest, tmp_path / "result", "DeepSeek-V3.2-Exp-FC")


def test_official_score_denominator_must_match_selection(tmp_path):
    manifest, _ = _fixture(tmp_path)
    root = tmp_path / "score" / "DeepSeek-V3.2-Exp-FC"
    root.mkdir(parents=True)
    for category in manifest["categories"]:
        (root / f"BFCL_v4_{category}_score.json").write_text(
            json.dumps({"accuracy": 0.5, "correct_count": 1, "total_count": 2}) + "\n",
            encoding="utf-8")
    report = validate_official_scores(manifest, tmp_path / "score", "DeepSeek-V3.2-Exp-FC")
    assert all(item["total"] == 2 for item in report.values())
    (root / "BFCL_v4_multi_turn_base_score.json").write_text(
        json.dumps({"accuracy": 1.0, "correct_count": 1, "total_count": 1}) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="denominator"):
        validate_official_scores(manifest, tmp_path / "score", "DeepSeek-V3.2-Exp-FC")
