def test_metrics_endpoint_exposes_expected_families(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "http_requests_total" in body
    assert "http_request_duration_seconds" in body
    assert "triage_predictions_total" in body


def test_request_counter_increments_after_predict(client):
    before = client.get("/metrics").text.count('http_requests_total{method="POST",path="/predict"')

    client.post("/predict", json={"text": "Routine follow-up, no acute findings."})

    after = client.get("/metrics").text.count('http_requests_total{method="POST",path="/predict"')
    assert after >= before


def test_prediction_counter_records_the_predicted_urgency(client):
    response = client.post(
        "/predict",
        json={"text": "Large pulmonary embolism with right heart strain."},
    )
    urgency = response.json()["urgency"]

    metrics_body = client.get("/metrics").text
    assert f'triage_predictions_total{{urgency="{urgency}"}}' in metrics_body
