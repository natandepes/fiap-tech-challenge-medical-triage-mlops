from __future__ import annotations

import json
from pathlib import Path

import joblib
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

from triage_api.config import MODEL_PATH, ONNX_LOCALE, ONNX_MODEL_PATH


def onnx_classes_path(onnx_path: Path) -> Path:
    return onnx_path.with_suffix(".classes.json")


def export_to_onnx(model_path: Path = MODEL_PATH, onnx_path: Path = ONNX_MODEL_PATH) -> Path:
    pipeline = joblib.load(model_path)
    onnx_model = to_onnx(
        pipeline,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={"zipmap": False, "tfidf__locale": ONNX_LOCALE},
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model.SerializeToString())
    onnx_classes_path(onnx_path).write_text(json.dumps([str(c) for c in pipeline.classes_]))
    return onnx_path


if __name__ == "__main__":
    print(export_to_onnx())
