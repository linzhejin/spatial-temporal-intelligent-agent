#!/usr/bin/env python3
"""Audit whether campus UAV data and evaluation evidence are ready."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.field_readiness import audit_campus_field_readiness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        help="root containing the source campus UAV videos")
    parser.add_argument("--flight-manifest", type=Path,
                        help="source-video inventory with flight, site, time, and split fields")
    parser.add_argument("--detection-manifest", type=Path,
                        help="manually reviewed campus detection test manifest")
    parser.add_argument("--detection-report", type=Path,
                        help="output from evaluate_campus_detection.py")
    parser.add_argument("--event-manifest", type=Path,
                        help="manually reviewed campus event test manifest")
    parser.add_argument("--event-report", type=Path,
                        help="output from evaluate_campus_events.py")
    parser.add_argument("--external-accident-manifest", type=Path,
                        help="public accident-event benchmark manifest with source and license metadata")
    parser.add_argument("--external-accident-report", type=Path,
                        help="held-out event report from evaluate_accident_events.py")
    parser.add_argument("--visible-water-manifest", type=Path,
                        help="manually labeled held-out campus road-water manifest")
    parser.add_argument("--visible-water-report", type=Path,
                        help="test report from evaluate_specialized.py --task visible_water_road_segmentation")
    parser.add_argument("--detector-model", type=Path,
                        help="production pedestrian/vehicle ONNX model")
    parser.add_argument("--accident-model", type=Path,
                        help="optional accident-scene classification ONNX model")
    parser.add_argument("--flood-model", type=Path,
                        help="optional road-water segmentation ONNX model")
    parser.add_argument("--output", type=Path,
                        help="optional JSON report path")
    parser.add_argument("--strict", action="store_true",
                        help="return exit code 1 unless campus accuracy claims are supported")
    args = parser.parse_args()
    try:
        report = audit_campus_field_readiness(
            dataset_root=args.dataset_root,
            flight_manifest=args.flight_manifest,
            detection_manifest=args.detection_manifest,
            detection_report=args.detection_report,
            event_manifest=args.event_manifest,
            event_report=args.event_report,
            external_accident_manifest=args.external_accident_manifest,
            external_accident_report=args.external_accident_report,
            visible_water_manifest=args.visible_water_manifest,
            visible_water_report=args.visible_water_report,
            detector_model=args.detector_model,
            accident_model=args.accident_model,
            flood_model=args.flood_model,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(args.strict and not report["accuracy_claim_allowed"])


if __name__ == "__main__":
    raise SystemExit(main())
