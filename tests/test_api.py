import json
import logging


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["status"] == "ready"
    assert client.get("/model").json()["train_rows"] == 60


def test_score(client):
    response = client.post("/score", json={"text": "Отличный товар, качество хорошее"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert 0 <= body["positive_score"] <= 1
    assert body["confidence"] >= 0.5
    assert body["model_version"]
    assert response.headers["X-Request-ID"]


def test_negative(client):
    response = client.post("/score", json={"text": "Ужасный товар, плохое качество"})
    assert response.json()["label"] == "negative"


def test_batch_matches_single_and_counts_samples(client):
    text = "Прекрасный сервис"
    single = client.post("/score", json={"text": text}).json()
    batch = client.post("/score/batch", json={"texts": [text, text]}).json()
    assert batch == [single, single]
    assert client.get("/drift").json()["samples"] == 3


def test_validation(client):
    for body in ({"text": "   "}, {"text": "x" * 3001}, {}, {"text": 123}, {"text": "ok", "extra": 1}):
        assert client.post("/score", json=body).status_code == 422
    for body in ({"texts": []}, {"texts": ["x"] * 33}, {"texts": [""]}):
        assert client.post("/score/batch", json=body).status_code == 422
    assert client.get("/drift").json()["samples"] == 0


def test_monitoring(client):
    assert client.get("/drift").json()["status"] == "insufficient_data"
    client.post("/score/batch", json={"texts": ["xyzzy " * 200] * 5})
    report = client.get("/drift").json()
    assert report["status"] == "ready"
    assert report["drift_detected"] is True
    metrics = client.get("/metrics").text
    assert "http_request_duration_seconds_bucket" in metrics
    assert "texts_scored_total" in metrics
    assert "data_drift_detected 1.0" in metrics


def test_logs_do_not_contain_text_or_query(client):
    records = []
    class Collector(logging.Handler):
        def emit(self, record):
            records.append(json.loads(record.getMessage()))
    handler = Collector()
    logger = logging.getLogger("text_scoring")
    logger.addHandler(handler)
    try:
        client.post("/score?secret=PRIVATE_QUERY", json={"text": "PRIVATE_TEXT"})
    finally:
        logger.removeHandler(handler)
    assert records[-1]["event"] == "http_request"
    assert "PRIVATE" not in json.dumps(records)


def test_unknown_paths_use_one_metric_label(client):
    client.get("/unknown-one")
    client.get("/unknown-two")
    metrics = client.get("/metrics").text
    assert 'route="unmatched"' in metrics
    assert "/unknown-one" not in metrics
