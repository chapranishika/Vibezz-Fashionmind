"""/metrics exposes Prometheus text and the middleware counts requests."""


def test_metrics_is_prometheus_text(client):
    client.get("/health")                    # generate one observed request
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "app_models_loaded" in body
    assert "http_request_duration_seconds_bucket" in body


def test_request_counter_increments(client):
    client.get("/health")
    client.get("/health")
    body = client.get("/metrics").text
    line = next(l for l in body.splitlines()
                if l.startswith('http_requests_total{') and 'path="/health"' in l)
    # at least the two /health calls above
    assert float(line.rsplit(" ", 1)[1]) >= 2.0


def test_recommend_increments_recs_served(client):
    client.post("/recommend", json={"customer_id": "x", "n": 3})
    body = client.get("/metrics").text
    assert any(l.startswith("recommendations_served_total{") and float(l.rsplit(" ", 1)[1]) > 0
               for l in body.splitlines())
