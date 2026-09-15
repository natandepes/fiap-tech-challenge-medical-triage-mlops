from __future__ import annotations

from pathlib import Path

import pandas as pd

from triage_api.config import SAMPLE_DATASET_PATH
from triage_api.enums import Urgency


def load_samples(path: Path = SAMPLE_DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Sample corpus not found at {path}; run `python -m triage_api.dataset` first"
        )
    return pd.read_csv(path)


def sample_reports(limit: int = 5, path: Path = SAMPLE_DATASET_PATH) -> list[str]:
    return load_samples(path)["text"].head(limit).tolist()


def labelled_samples(
    limit: int | None = None, path: Path = SAMPLE_DATASET_PATH
) -> list[tuple[str, Urgency]]:
    frame = load_samples(path) if limit is None else load_samples(path).head(limit)
    return [(row.text, Urgency(row.urgency)) for row in frame.itertuples()]
