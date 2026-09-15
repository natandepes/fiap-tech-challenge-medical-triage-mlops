from __future__ import annotations

import logging
import os
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

MIN_MACRO_F1_ENV = "TRIAGE_MIN_MACRO_F1"
DEFAULT_MIN_MACRO_F1 = "0.55"

logger = logging.getLogger("airflow.task")


@dag(
    dag_id="triage_retraining",
    schedule="@weekly",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["triage", "mlops"],
    doc_md="Weekly retraining pipeline: ingest -> train -> evaluate -> export_onnx -> "
    "validate_onnx -> publish, gated on a macro-F1 threshold before exporting to ONNX and "
    "on sklearn/ONNX parity before promoting both artifacts to the registry.",
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
    def export_onnx(metrics: dict) -> dict:
        from triage_api.onnx_export import export_to_onnx

        export_to_onnx()
        return metrics

    @task
    def validate_onnx(metrics: dict) -> dict:
        from triage_api.onnx_validate import validate_onnx as check_onnx_parity

        try:
            check_onnx_parity(metrics)
        except ValueError as exc:
            raise AirflowFailException(str(exc)) from exc
        return metrics

    @task
    def publish(metrics: dict) -> dict:
        from triage_api.config import MODEL_DIR, MODEL_PATH, ONNX_MODEL_PATH
        from triage_api.registry import promote

        registry_dir = MODEL_DIR / "registry"
        joblib_path = promote(MODEL_PATH, registry_dir, latest_name="latest.txt")
        onnx_path = promote(ONNX_MODEL_PATH, registry_dir, latest_name="latest_onnx.txt")
        logger.info("Promoted model versions %s, %s", joblib_path.name, onnx_path.name)
        return {"joblib": joblib_path.name, "onnx": onnx_path.name}

    publish(validate_onnx(export_onnx(evaluate(train(ingest())))))


triage_retraining()
