from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from triage_api.config import (
    CORPUS_BASE_URL,
    CORPUS_FILES,
    DATASET_PATH,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SAMPLE_DATASET_PATH,
    SAMPLE_SIZE,
)
from triage_api.enums import Condition, Urgency

TEXT_COLUMN = "medical_abstract"
LABEL_COLUMN = "condition_label"

URGENCY_BY_CONDITION = {
    Condition.NEOPLASMS: Urgency.ATTENTION,
    Condition.DIGESTIVE_SYSTEM: Urgency.NORMAL,
    Condition.NERVOUS_SYSTEM: Urgency.ATTENTION,
    Condition.CARDIOVASCULAR: Urgency.URGENT,
    Condition.GENERAL_PATHOLOGICAL: Urgency.NORMAL,
}


def download_corpus(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in CORPUS_FILES:
        destination = raw_dir / filename
        if not destination.exists():
            urllib.request.urlretrieve(f"{CORPUS_BASE_URL}/{filename}", destination)
        paths.append(destination)
    return paths


def load_corpus(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    splits = [pd.read_csv(path) for path in download_corpus(raw_dir)]
    return pd.concat(splits, ignore_index=True)


def to_triage_frame(corpus: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "text": corpus[TEXT_COLUMN].str.strip(),
            "urgency": corpus[LABEL_COLUMN].map(URGENCY_BY_CONDITION).map(str),
        }
    )
    frame = frame[frame["text"].str.len() > 0].dropna()
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_dataset(path: Path = DATASET_PATH, raw_dir: Path = RAW_DATA_DIR) -> Path:
    frame = to_triage_frame(load_corpus(raw_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_sample(
    path: Path = SAMPLE_DATASET_PATH,
    n_samples: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> Path:
    frame = to_triage_frame(load_corpus())
    per_label = n_samples // len(Urgency)
    sample = (
        frame.groupby("urgency")
        .head(per_label)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print(f"wrote {build_dataset()}")
