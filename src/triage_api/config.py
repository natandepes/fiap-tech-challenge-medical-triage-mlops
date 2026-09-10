import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = Path(os.getenv("TRIAGE_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.getenv("TRIAGE_MODEL_DIR", PROJECT_ROOT / "models"))

DATASET_PATH = DATA_DIR / "triage.csv"
MODEL_PATH = Path(os.getenv("TRIAGE_MODEL_PATH", MODEL_DIR / "model.joblib"))
METRICS_PATH = MODEL_DIR / "metrics.json"

DATASET_SIZE = int(os.getenv("TRIAGE_DATASET_SIZE", "3000"))
RANDOM_SEED = int(os.getenv("TRIAGE_RANDOM_SEED", "42"))
