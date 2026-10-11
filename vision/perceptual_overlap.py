"""Screen image collections for exact and perceptual overlap.

Perceptual hashes flag review candidates; they are not proof that two images
come from the same capture. Training/validation references are compared only
against candidate images, never against the manifest's test split.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


_PHASH_SIZE = 32
_PHASH_LOW_SIZE = 8
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _dct_basis(size: int = _PHASH_SIZE) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = np.arange(size, dtype=np.float64)[:, None]
    basis = np.cos(
        np.pi * (2 * positions + 1) * frequencies / (2 * size),
    )
    basis[0] *= 1 / math.sqrt(2)
    basis *= math.sqrt(2 / size)
    return basis


_DCT_BASIS = _dct_basis()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> int:
    """Return a deterministic 63-bit DCT perceptual hash for an image."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as source:
        image = source.convert("L").resize(
            (_PHASH_SIZE, _PHASH_SIZE), resampling,
        )
    pixels = np.asarray(image, dtype=np.float64)
    coefficients = _DCT_BASIS @ pixels @ _DCT_BASIS.T
    low_frequencies = coefficients[:_PHASH_LOW_SIZE, :_PHASH_LOW_SIZE].reshape(-1)
    median = float(np.median(low_frequencies[1:]))
    value = 0
    for index, coefficient in enumerate(low_frequencies[1:]):
        if coefficient > median:
            value |= 1 << index
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _resolve_inside(root: Path, relative: str, *, field: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} path escapes its root: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"{field} image does not exist: {relative}")
    return path


def audit_overlap(manifest_path: Path, reference_root: Path,
                  candidate_dir: Path, *, near_threshold: int = 2) -> dict:
    """Compare candidates with manifest train/validation images.

    Exact SHA-256 matches are conclusive byte duplicates. Perceptual matches
    are reported for human review and must not be interpreted as proof alone.
    """
    if near_threshold < 0 or near_threshold > 63:
        raise ValueError("near_threshold must be between 0 and 63")
    manifest_path = Path(manifest_path).resolve()
    reference_root = Path(reference_root).resolve()
    candidate_dir = Path(candidate_dir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest must contain a samples list")

    declared_splits = {}
    split_groups = manifest.get("split_groups")
    if split_groups is not None:
        if not isinstance(split_groups, dict):
            raise ValueError("manifest split_groups must be an object")
        for split, source_video_ids in split_groups.items():
            if not isinstance(source_video_ids, list):
                raise ValueError("manifest split_groups values must be lists")
            for source_video_id in source_video_ids:
                source_video_id = str(source_video_id)
                previous_split = declared_splits.get(source_video_id)
                if previous_split is not None and previous_split != split:
                    raise ValueError(
                        f"source video {source_video_id} assigned to multiple splits",
                    )
                declared_splits[source_video_id] = split
        for sample in samples:
            split = sample.get("split")
            if split not in {"train", "validation", "test"}:
                continue
            source_video_id = str(sample.get("source_video_id", ""))
            if declared_splits.get(source_video_id) != split:
                raise ValueError(
                    f"sample {sample.get('sample_id', '<unknown>')} conflicts with split_groups",
                )

    references = []
    for sample in samples:
        if sample.get("split") not in {"train", "validation"}:
            continue
        required = ("sample_id", "source_video_id", "image")
        if any(not sample.get(key) for key in required):
            raise ValueError("train/validation sample is missing identity or image fields")
        path = _resolve_inside(reference_root, sample["image"], field="reference")
        references.append({
            "sample_id": str(sample["sample_id"]),
            "source_video_id": str(sample["source_video_id"]),
            "split": str(sample["split"]),
            "path": path,
            "sha256": sha256_file(path),
            "perceptual_hash": perceptual_hash(path),
        })
    if not references:
        raise ValueError("manifest contains no train/validation reference images")
    reference_set_digest = hashlib.sha256()
    for reference in sorted(references, key=lambda item: (
            item["split"], item["source_video_id"], item["sample_id"],
    )):
        identity = (
            f"{reference['split']}\0{reference['source_video_id']}\0"
            f"{reference['sample_id']}\0"
            f"{reference['path'].relative_to(reference_root).as_posix()}\0"
            f"{reference['sha256']}\n"
        )
        reference_set_digest.update(identity.encode("utf-8"))

    candidate_paths = sorted(
        (path for path in candidate_dir.rglob("*")
         if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda path: path.as_posix().casefold(),
    )
    if not candidate_paths:
        raise ValueError("candidate directory contains no supported images")
    for path in candidate_paths:
        try:
            path.resolve().relative_to(candidate_dir)
        except ValueError as exc:
            raise ValueError(f"candidate image path escapes its root: {path}") from exc

    hashes_by_sha = {}
    for reference in references:
        hashes_by_sha.setdefault(reference["sha256"], []).append(reference)

    matches = []
    source_counts = Counter()
    distance_counts = Counter()
    exact_count = 0
    near_count = 0
    candidate_set_digest = hashlib.sha256()
    for path in candidate_paths:
        candidate_sha = sha256_file(path)
        candidate_relative_path = path.relative_to(candidate_dir).as_posix()
        candidate_set_digest.update(
            f"{candidate_relative_path}\0{candidate_sha}\n".encode("utf-8"),
        )
        candidate_hash = perceptual_hash(path)
        distances = [
            hamming_distance(candidate_hash, item["perceptual_hash"])
            for item in references
        ]
        nearest_distance = min(distances)
        nearest = [
            references[index] for index, distance in enumerate(distances)
            if distance == nearest_distance
        ]
        nearest.sort(key=lambda item: (
            item["source_video_id"], item["sample_id"], item["path"].as_posix(),
        ))
        selected = nearest[0]
        exact_references = hashes_by_sha.get(candidate_sha, [])
        is_exact = bool(exact_references)
        exact_count += int(is_exact)
        is_near = nearest_distance <= near_threshold
        near_count += int(is_near)
        if is_near:
            source_counts[selected["source_video_id"]] += 1
        distance_counts[str(nearest_distance)] += 1
        matches.append({
            "candidate": candidate_relative_path,
            "candidate_sha256": candidate_sha,
            "nearest_distance": nearest_distance,
            "nearest_tied_reference_count": len(nearest),
            "nearest_sample_id": selected["sample_id"],
            "nearest_source_video_id": selected["source_video_id"],
            "nearest_split": selected["split"],
            "nearest_reference": selected["path"].relative_to(reference_root).as_posix(),
            "nearest_reference_sha256": selected["sha256"],
            "exact_sha256_match": is_exact,
            "exact_reference_sample_ids": sorted(
                item["sample_id"] for item in exact_references
            ),
            "perceptual_match_requires_visual_review": is_near and not is_exact,
        })

    return {
        "schema_version": 1,
        "task": "perceptual_overlap_screen",
        "hash_method": "grayscale 32x32 DCT, 8x8 low-frequency coefficients, median threshold excluding DC",
        "near_threshold": near_threshold,
        "reference_split_policy": ["train", "validation"],
        "reference_manifest_sha256": sha256_file(manifest_path),
        "reference_image_set_sha256": reference_set_digest.hexdigest(),
        "candidate_image_set_sha256": candidate_set_digest.hexdigest(),
        "candidate_count": len(candidate_paths),
        "summary": {
            "reference_count": len(references),
            "candidate_count": len(candidate_paths),
            "exact_sha256_matches": exact_count,
            "near_threshold_count": near_count,
            "nearest_distance_histogram": dict(sorted(
                distance_counts.items(), key=lambda item: int(item[0]),
            )),
            "near_candidates_by_nearest_source_video": dict(sorted(source_counts.items())),
        },
        "limitations": [
            "Perceptual hash is a screening signal, not proof of duplicate capture.",
            "Near matches require visual review; this report does not establish capture provenance.",
            "Only manifest train/validation images were used as references; test images are excluded.",
        ],
        "matches": matches,
    }
