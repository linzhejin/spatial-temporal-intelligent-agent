"""One-shot isolated ONNX inference for Doodleverse FloodNet v7."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from vision.specialized import DoodleverseFloodSegmenter


def main(request_path: str) -> int:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    frame = np.load(request["image_path"], allow_pickle=False)
    model = DoodleverseFloodSegmenter(
        request["model_path"], model_version=request["model_version"],
        probability_threshold=request["probability_threshold"],
        min_area_ratio=request["min_area_ratio"],
        single_image_trigger_threshold=request["single_image_trigger_threshold"],
    )
    probabilities = model._predict_probabilities(frame)
    predictions = model._predictions_from_probability(
        frame, request.get("regions") or [], probabilities,
    )
    Path(request["response_path"]).write_text(json.dumps({
        "status": "ok", "predictions": predictions,
    }, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
