from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as rt

from triage_api.enums import Urgency
from triage_api.model import Prediction
from triage_api.onnx_export import onnx_classes_path


class OnnxTriageModel:
    def __init__(self, session: rt.InferenceSession, classes: list[Urgency]) -> None:
        self._session = session
        self._classes = classes
        self._input_name = session.get_inputs()[0].name
        self._label_name, self._proba_name = (output.name for output in session.get_outputs())

    @classmethod
    def load(cls, path: Path) -> OnnxTriageModel:
        if not path.exists():
            raise FileNotFoundError(f"ONNX model artifact not found at {path}")
        classes_path = onnx_classes_path(path)
        if not classes_path.exists():
            raise FileNotFoundError(f"ONNX class labels not found at {classes_path}")

        classes = [Urgency(label) for label in json.loads(classes_path.read_text())]
        session = rt.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return cls(session, classes)

    def predict(self, text: str) -> Prediction:
        inputs = {self._input_name: np.array([[text]], dtype=object)}
        _, probabilities = self._session.run([self._label_name, self._proba_name], inputs)
        ranked = dict(
            sorted(
                zip(self._classes, (float(score) for score in probabilities[0]), strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        top_label, top_score = next(iter(ranked.items()))
        return Prediction(urgency=top_label, confidence=top_score, probabilities=ranked)
