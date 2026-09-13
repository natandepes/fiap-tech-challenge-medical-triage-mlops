from triage_api.enums import Urgency
from triage_api.metrics import HEALTH_PATH


def test_health_reports_model_loaded(client):
    body = client.get(HEALTH_PATH).json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_a_valid_urgency_label(client):
    report = "Impression: acute intracranial hemorrhage. Immediate clinical attention required."
    response = client.post("/predict", json={"text": report})
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] in set(Urgency)
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == set(Urgency)


def test_predict_flags_a_clearly_urgent_report(client):
    report = (
        "Large pulmonary embolism with right heart strain. Patient is hemodynamically unstable."
    )
    response = client.post("/predict", json={"text": report})
    assert response.json()["urgency"] == Urgency.URGENT


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_rejects_whitespace_only_text(client):
    assert client.post("/predict", json={"text": "   \n\t "}).status_code == 422
