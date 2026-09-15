from __future__ import annotations

from triage_api.config import MODEL_BACKEND, MODEL_PATH, ONNX_MODEL_PATH
from triage_api.model import TriageModel
from triage_api.onnx_model import OnnxTriageModel


def load_model(backend: str = MODEL_BACKEND) -> TriageModel | OnnxTriageModel:
    if backend == "onnx":
        return OnnxTriageModel.load(ONNX_MODEL_PATH)
    if backend == "sklearn":
        return TriageModel.load(MODEL_PATH)
    raise ValueError(f"Unknown model backend: {backend!r}")
