from __future__ import annotations

import logging
import os
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

MIN_MACRO_F1_ENV = "TRIAGE_MIN_MACRO_F1"
DEFAULT_MIN_MACRO_F1 = "0.80"

logger = logging.getLogger("airflow.task")


@dag(
    dag_id="triage_retraining",
    schedule="@weekly",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["triage", "mlops"],
    doc_md="Weekly retraining pipeline: ingest -> train -> evaluate -> publish, "
    "gated on a macro-F1 threshold before promoting the artifact to the registry.",
)
def triage_retraining():
    @task
    def ingest() -> str:
        from triage_api.dataset import build_dataset

        return str(build_dataset())

    @task
    def train(dataset_path: str) -> dict:
        from triage_api.train import train as run_training

        return run_training(Path(dataset_path))

    @task
    def evaluate(metrics: dict) -> dict:
        threshold = float(os.getenv(MIN_MACRO_F1_ENV, DEFAULT_MIN_MACRO_F1))
        if metrics["macro_f1"] < threshold:
            raise AirflowFailException(
                f"macro_f1 {metrics['macro_f1']:.4f} is below the {threshold:.4f} threshold"
            )
        return metrics

    @task
    def publish(metrics: dict) -> str:
        from triage_api.config import MODEL_DIR, MODEL_PATH
        from triage_api.registry import promote

        promoted_path = promote(MODEL_PATH, MODEL_DIR / "registry")
        logger.info("Promoted model version %s", promoted_path.name)
        return promoted_path.name

    publish(evaluate(train(ingest())))


triage_retraining()
