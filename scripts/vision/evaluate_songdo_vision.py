#!/usr/bin/env python3
"""Evaluate the campus vehicle detector on the annotated Songdo Vision test ZIP."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.songdo_vision_evaluation import (  # noqa: E402
    run_songdo_vision_evaluation,
    run_songdo_vision_paired_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True,
                        help="Songdo Vision v2 test ZIP; images are read in place without extraction")
    parser.add_argument("--model", type=Path, required=True,
                        help="local RT-DETRv4 ONNX model")
    parser.add_argument("--confidence", type=float, default=0.369,
                        help="fixed detector confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="class-aware box matching IoU threshold")
    parser.add_argument("--limit", type=int, default=0,
                        help="maximum images; 0 evaluates the full official test split")
    parser.add_argument("--sample-seed", type=int,
                        help="select a reproducible random subset instead of the first N images")
    parser.add_argument("--inference-path", choices=("whole_frame", "full_frame_tiled"),
                        default="whole_frame",
                        help="whole-frame resizing or full-image tiling proxy")
    parser.add_argument("--compare-full-frame-tiled", action="store_true",
                        help="run both paths on the same selected images and report metric deltas")
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="ONNX Runtime intra-op CPU threads (default: 1)")
    parser.add_argument("--output", type=Path,
                        help="optional UTF-8 JSON report path")
    args = parser.parse_args()

    if not args.archive.is_file() or not args.model.is_file():
        parser.error("archive and ONNX model files must exist")
    if args.limit < 0:
        parser.error("--limit must be zero or a positive image count")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be at least 1")

    # Set this before the lazy ONNX detector import/session construction.
    os.environ["VISION_CPU_THREADS"] = str(args.cpu_threads)
    started = time.monotonic()

    def report_progress(path_name: str, current: int, total: int, elapsed: float) -> None:
        rate = current / elapsed if elapsed else 0.0
        remaining = (total - current) / rate if rate else 0.0
        print(
            f"[{path_name}] evaluated {current}/{total} images; elapsed={elapsed:.1f}s; "
            f"estimated_remaining={remaining:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    try:
        if args.compare_full_frame_tiled:
            if args.inference_path != "whole_frame":
                parser.error("--inference-path cannot be combined with --compare-full-frame-tiled")
            report = run_songdo_vision_paired_comparison(
                args.archive,
                args.model,
                confidence_threshold=args.confidence,
                iou_threshold=args.iou,
                limit=args.limit,
                sample_seed=args.sample_seed,
                progress_callback=report_progress,
            )
        else:
            report = run_songdo_vision_evaluation(
                args.archive,
                args.model,
                confidence_threshold=args.confidence,
                iou_threshold=args.iou,
                limit=args.limit,
                sample_seed=args.sample_seed,
                inference_path=args.inference_path,
                progress_callback=lambda current, total, elapsed: report_progress(
                    args.inference_path, current, total, elapsed,
                ),
            )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    report["cpu_threads"] = args.cpu_threads
    report["cli_elapsed_seconds"] = round(time.monotonic() - started, 2)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
