"""Guard the pinned BFCL O1 subset before invoking its unmodified scorer.

This module does not implement BFCL scoring. The upstream evaluator owns that
definition; these checks prevent a partial evaluation from silently using a
different denominator than the registered 40-case selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_run_ids(manifest: dict) -> dict[str, list[str]]:
    categories = manifest.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("BFCL categories are required")
    selected = {}
    for category, info in categories.items():
        if not isinstance(category, str) or not category.startswith("multi_turn_"):
            raise ValueError("only BFCL multi_turn categories belong to O1")
        ids = info.get("selected_ids") if isinstance(info, dict) else None
        if not isinstance(ids, list) or not ids or any(
                not isinstance(case_id, str) or not case_id.startswith(category + "_")
                for case_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"invalid selected IDs for {category}")
        selected[category] = list(ids)
    return selected


def _read_jsonl_items(path: Path) -> list[dict]:
    items = []
    ids = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed BFCL JSONL at {path.name}:{number}") from exc
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"missing BFCL ID at {path.name}:{number}")
        ids.append(item["id"])
        items.append(item)
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {path.name}")
    return items


def validate_cached_sources(manifest: dict, cache: Path) -> dict[str, str]:
    selected = build_run_ids(manifest)
    digests = {}
    for category, ids in selected.items():
        path = cache / f"BFCL_v4_{category}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["categories"][category].get("sha256"):
            raise ValueError(f"BFCL source SHA-256 mismatch: {category}")
        source_ids = [item["id"] for item in _read_jsonl_items(path)]
        if not set(ids).issubset(source_ids):
            raise ValueError(f"selected source IDs missing: {category}")
        available = manifest["categories"][category].get("available_count")
        if available is not None and available != len(source_ids):
            raise ValueError(f"BFCL source count mismatch: {category}")
        digests[category] = digest
    return digests


def validate_result_subset(manifest: dict, result_dir: Path, model: str, *,
                           require_generation_metadata: bool = False) -> dict:
    selected = build_run_ids(manifest)
    if not model or "/" in model or "\\" in model or model in (".", ".."):
        raise ValueError("BFCL model must be one registry directory name")
    model_dir = result_dir / model
    counts = {}
    digests = {}
    for category, expected in selected.items():
        name = f"BFCL_v4_{category}_result.json"
        paths = list(model_dir.rglob(name)) if model_dir.is_dir() else []
        if len(paths) != 1:
            raise ValueError(f"expected one official result file for {category}")
        items = _read_jsonl_items(paths[0])
        found = [item["id"] for item in items]
        if set(found) != set(expected) or len(found) != len(expected):
            raise ValueError(f"BFCL result selected IDs mismatch: {category}")
        if require_generation_metadata and any(
                not ("traceback" in item or all(
                    key in item for key in ("inference_log", "input_token_count",
                                         "output_token_count", "latency")))
                for item in items):
            raise ValueError(f"BFCL official generation metadata missing: {category}")
        counts[category] = len(found)
        digests[category] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    unexpected = [path.name for path in model_dir.rglob("BFCL_v4_*_result.json")
                  if path.name not in {f"BFCL_v4_{name}_result.json" for name in selected}]
    if unexpected:
        raise ValueError(f"unexpected BFCL result categories: {unexpected}")
    return {"total": sum(counts.values()), "counts": counts,
            "result_sha256": digests}


def validate_official_scores(manifest: dict, score_dir: Path, model: str) -> dict:
    selected = build_run_ids(manifest)
    model_dir = score_dir / model
    reports = {}
    for category, expected in selected.items():
        paths = list(model_dir.rglob(f"BFCL_v4_{category}_score.json"))
        if len(paths) != 1:
            raise ValueError(f"expected one official score file for {category}")
        lines = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if not lines or lines[0].get("total_count") != len(expected):
            raise ValueError(f"official BFCL score denominator mismatch: {category}")
        reports[category] = {
            "correct": lines[0].get("correct_count"),
            "total": lines[0]["total_count"],
            "accuracy": lines[0].get("accuracy"),
            "score_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        }
    return reports
