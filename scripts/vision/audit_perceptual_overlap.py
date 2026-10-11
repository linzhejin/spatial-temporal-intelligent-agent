"""Audit candidate images for exact or perceptual overlap with train/val data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.perceptual_overlap import audit_overlap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-threshold", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        parser.error(f"output already exists; choose another path or pass --overwrite: {output}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        parser.error(f"partial output already exists; inspect it before retrying: {partial}")

    report = audit_overlap(
        args.reference_manifest,
        args.reference_root,
        args.candidate_dir,
        near_threshold=args.near_threshold,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
