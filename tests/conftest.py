import pytest

from triage_api.config import MODEL_PATH
from triage_api.train import train


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    if not MODEL_PATH.exists():
        train()
    return MODEL_PATH


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from triage_api.main import app

    with TestClient(app) as test_client:
        yield test_client
