from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Prediction:
    urgency: str
    confidence: float
    probabilities: dict[str, float]


class TriageModel:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline
        self._classes = list(pipeline.classes_)

    @classmethod
    def load(cls, path: Path) -> TriageModel:
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        return cls(joblib.load(path))

    def predict(self, text: str) -> Prediction:
        probabilities = self._pipeline.predict_proba([text])[0]
        ranked = dict(
            sorted(
                zip(self._classes, (float(p) for p in probabilities), strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        top_label, top_score = next(iter(ranked.items()))
        return Prediction(urgency=top_label, confidence=top_score, probabilities=ranked)
