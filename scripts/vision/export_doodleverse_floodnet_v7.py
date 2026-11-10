"""Export the official FloodNet Res-UNet v7 H5 weights to FP32 ONNX.

Conversion dependencies belong in an isolated local environment. The exported
model artifact is intentionally kept outside the source repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INPUT_HEIGHT = 768
INPUT_WIDTH = 1024
CLASS_COUNT = 10
EXPECTED_SOURCE_SHA256 = "dbef76764f4a4449852dfa801964c2d5d50b0742a7d47bc78231710b88dd4b75"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_model(source_path: Path, output_path: Path) -> dict[str, object]:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import onnx
    import tensorflow as tf
    import tf2onnx

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_hash = sha256_file(source_path)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError("Source weights do not match the pinned official Doodleverse v7 file")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite exported model: {output_path}")
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    if metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite export metadata: {metadata_path}")
    source_model = tf.keras.models.load_model(source_path, compile=False)
    if tuple(source_model.input_shape[1:]) != (INPUT_HEIGHT, INPUT_WIDTH, 3):
        raise ValueError(f"Unexpected source input shape: {source_model.input_shape}")
    if tuple(source_model.output_shape[1:]) != (INPUT_HEIGHT, INPUT_WIDTH, CLASS_COUNT):
        raise ValueError(f"Unexpected source output shape: {source_model.output_shape}")

    def fp32_layer(layer):
        config = layer.get_config()
        if "dtype" in config:
            config["dtype"] = "float32"
        return layer.__class__.from_config(config)

    fp32_model = tf.keras.models.clone_model(source_model, clone_function=fp32_layer)
    fp32_model.set_weights(source_model.get_weights())
    non_fp32_layers = [
        layer.name for layer in fp32_model.layers if layer.compute_dtype != "float32"
    ]
    if non_fp32_layers:
        raise ValueError(f"FP32 clone contains non-FP32 compute layers: {non_fp32_layers[:8]}")
    if fp32_model.inputs[0].dtype != tf.float32 or fp32_model.outputs[0].dtype != tf.float32:
        raise ValueError("The FP32 clone must use float32 inputs and outputs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    signature = (tf.TensorSpec(
        (None, INPUT_HEIGHT, INPUT_WIDTH, 3), tf.float32, name="rgb_standardized",
    ),)
    tf2onnx.convert.from_keras(
        fp32_model, input_signature=signature, opset=17, output_path=str(output_path),
    )
    exported = onnx.load(str(output_path), load_external_data=True)
    onnx.checker.check_model(exported)
    graph_inputs = exported.graph.input
    graph_outputs = exported.graph.output
    if len(graph_inputs) != 1 or len(graph_outputs) != 1:
        raise ValueError("Export must have one image input and one probability output")
    if graph_inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX image input is not FP32")
    if graph_outputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX probability output is not FP32")

    metadata = {
        "source_model": str(source_path),
        "source_sha256": source_hash,
        "output_model": str(output_path),
        "output_sha256": sha256_file(output_path),
        "model_revision": "doodleverse-floodnet-resunet-v7",
        "model_doi": "10.5281/zenodo.7566797",
        "model_url": "https://zenodo.org/records/7566797",
        "model_license": "CC BY 4.0",
        "model_author": "Daniel Buscombe",
        "input_shape_nhwc": [None, INPUT_HEIGHT, INPUT_WIDTH, 3],
        "input_type": "float32 whole-image standardized RGB",
        "output_shape_nhwc": [None, INPUT_HEIGHT, INPUT_WIDTH, CLASS_COUNT],
        "output_type": "float32 softmax probabilities",
        "opset": 17,
        "tensorflow": tf.__version__,
        "tf2onnx": tf2onnx.__version__,
        "onnx": onnx.__version__,
        "field_validated": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official v7 fullmodel.h5")
    parser.add_argument("--output", type=Path, required=True, help="FP32 ONNX output outside Git")
    args = parser.parse_args()
    print(json.dumps(export_model(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
