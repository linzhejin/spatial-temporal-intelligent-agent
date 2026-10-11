"""Prepare flight-disjoint campus UAV frames for road-water annotation.

The source videos, extracted frames, masks, and generated manifests belong in a
research data directory outside Git. A dry run validates split isolation and
records source hashes without decoding or writing video frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.campus_water_data import (  # noqa: E402
    build_campus_flight_plan,
    extract_campus_annotation_pack,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="directory containing original flight videos and optional sidecars")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON flight manifest with disjoint train/validation/test flights")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new annotation-pack directory; existing paths are never overwritten")
    parser.add_argument("--sample-fps", type=float, default=1.0,
                        help="target extraction rate in frames per second (default: 1.0)")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality from 50 to 100 (default: 95)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate provenance and split isolation without writing files")
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        plan = build_campus_flight_plan(manifest, args.dataset_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    split_counts = Counter(flight["split"] for flight in plan["flights"])
    summary = {
        "dataset": plan["dataset"],
        "flights_by_split": {
            split: split_counts[split]
            for split in ("train", "validation", "test")
        },
        "sites_by_split": {
            split: len({
                flight["site_id"] for flight in plan["flights"]
                if flight["split"] == split
            })
            for split in ("train", "validation", "test")
        },
        "source_bytes": sum(flight["source_bytes"] for flight in plan["flights"]),
        "source_sha256": {
            flight["flight_id"]: flight["source_sha256"]
            for flight in plan["flights"]
        },
    }
    if args.dry_run:
        print(json.dumps({**summary, "writes_files": False}, ensure_ascii=False, indent=2))
        return 0

    try:
        result = extract_campus_annotation_pack(
            plan,
            args.output_dir,
            sample_fps=args.sample_fps,
            jpeg_quality=args.jpeg_quality,
        )
    except (OSError, ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({**summary, **result, "writes_files": True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
