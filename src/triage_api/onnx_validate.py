from __future__ import annotations

import json
import os
from pathlib import Path

from sklearn.metrics import classification_report

from triage_api.config import DATASET_PATH, METRICS_PATH, ONNX_MODEL_PATH
from triage_api.onnx_model import OnnxTriageModel
from triage_api.train import load_dataset, split_dataset

MAX_MACRO_F1_DRIFT_ENV = "TRIAGE_MAX_ONNX_F1_DRIFT"
DEFAULT_MAX_MACRO_F1_DRIFT = "0.02"


def evaluate_onnx(dataset_path: Path = DATASET_PATH, onnx_path: Path = ONNX_MODEL_PATH) -> dict:
    frame = load_dataset(dataset_path)
    _, x_test, _, y_test = split_dataset(frame)

    model = OnnxTriageModel.load(onnx_path)
    predictions = [model.predict(text).urgency.value for text in x_test]

    report = classification_report(y_test, predictions, output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
    }


def validate_onnx(
    sklearn_metrics: dict,
    dataset_path: Path = DATASET_PATH,
    onnx_path: Path = ONNX_MODEL_PATH,
) -> dict:
    onnx_metrics = evaluate_onnx(dataset_path, onnx_path)

    drift = abs(sklearn_metrics["macro_f1"] - onnx_metrics["macro_f1"])
    max_drift = float(os.getenv(MAX_MACRO_F1_DRIFT_ENV, DEFAULT_MAX_MACRO_F1_DRIFT))
    if drift > max_drift:
        raise ValueError(
            f"ONNX macro_f1 {onnx_metrics['macro_f1']:.4f} diverges from sklearn "
            f"{sklearn_metrics['macro_f1']:.4f} by {drift:.4f}, exceeding {max_drift:.4f}"
        )
    return onnx_metrics


if __name__ == "__main__":
    sklearn_metrics = json.loads(METRICS_PATH.read_text())
    print(json.dumps(validate_onnx(sklearn_metrics), indent=2))
