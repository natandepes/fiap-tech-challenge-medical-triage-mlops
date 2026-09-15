import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = Path(os.getenv("TRIAGE_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.getenv("TRIAGE_MODEL_DIR", PROJECT_ROOT / "models"))

RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_PATH = DATA_DIR / "triage.csv"
SAMPLE_DATASET_PATH = PROJECT_ROOT / "data" / "sample_triage.csv"
MODEL_PATH = Path(os.getenv("TRIAGE_MODEL_PATH", MODEL_DIR / "model.joblib"))
METRICS_PATH = MODEL_DIR / "metrics.json"

CORPUS_COMMIT = "70a2d9106c724729be8b3c4ddb00d1b14ec300c8"
CORPUS_BASE_URL = os.getenv(
    "TRIAGE_CORPUS_BASE_URL",
    f"https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/{CORPUS_COMMIT}",
)
CORPUS_FILES = ("medical_tc_train.csv", "medical_tc_test.csv")
SAMPLE_SIZE = int(os.getenv("TRIAGE_SAMPLE_SIZE", "150"))

RANDOM_SEED = int(os.getenv("TRIAGE_RANDOM_SEED", "42"))
