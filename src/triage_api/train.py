from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from triage_api.config import (
    DATASET_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
)
from triage_api.dataset import build_dataset


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        build_dataset(path)
    return pd.read_csv(path)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "clf",
                LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced"),
            ),
        ]
    )


def train(dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH) -> dict:
    frame = _load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        frame["text"],
        frame["urgency"],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=frame["urgency"],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    report = classification_report(y_test, pipeline.predict(x_test), output_dict=True)
    metrics = {
        "samples": int(len(frame)),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class_f1": {
            label: report[label]["f1-score"] for label in sorted(frame["urgency"].unique())
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
