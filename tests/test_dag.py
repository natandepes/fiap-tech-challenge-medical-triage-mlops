import pytest

pytest.importorskip("airflow")

from airflow.models import DagBag

DAG_ID = "triage_retraining"
EXPECTED_TASK_IDS = {"ingest", "train", "evaluate", "export_onnx", "validate_onnx", "publish"}


@pytest.fixture(scope="module")
def dag():
    dag_bag = DagBag(dag_folder="dags", include_examples=False)
    assert not dag_bag.import_errors, dag_bag.import_errors
    assert DAG_ID in dag_bag.dags
    return dag_bag.dags[DAG_ID]


def test_task_ids_match(dag):
    assert set(dag.task_ids) == EXPECTED_TASK_IDS


def test_dependency_chain_is_linear(dag):
    assert dag.get_task("ingest").upstream_task_ids == set()
    assert dag.get_task("train").upstream_task_ids == {"ingest"}
    assert dag.get_task("evaluate").upstream_task_ids == {"train"}
    assert dag.get_task("export_onnx").upstream_task_ids == {"evaluate"}
    assert dag.get_task("validate_onnx").upstream_task_ids == {"export_onnx"}
    assert dag.get_task("publish").upstream_task_ids == {"validate_onnx"}
