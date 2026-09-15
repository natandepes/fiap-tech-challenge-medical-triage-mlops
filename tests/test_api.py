from triage_api.enums import Urgency
from triage_api.metrics import HEALTH_PATH
from triage_api.samples import labelled_samples, sample_reports

SANITY_ACCURACY_FLOOR = 0.6


def test_health_reports_model_loaded(client):
    body = client.get(HEALTH_PATH).json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_a_valid_urgency_label(client):
    response = client.post("/predict", json={"text": sample_reports(1)[0]})
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] in set(Urgency)
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == set(Urgency)


def test_predict_agrees_with_the_corpus_labels_on_most_samples(client):
    samples = labelled_samples(limit=60)
    correct = sum(
        client.post("/predict", json={"text": text}).json()["urgency"] == expected
        for text, expected in samples
    )
    assert correct / len(samples) >= SANITY_ACCURACY_FLOOR


def test_predict_accepts_a_full_length_abstract(client):
    longest = max(sample_reports(limit=150), key=len)
    assert client.post("/predict", json={"text": longest}).status_code == 200


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_rejects_whitespace_only_text(client):
    assert client.post("/predict", json={"text": "   \n\t "}).status_code == 422
