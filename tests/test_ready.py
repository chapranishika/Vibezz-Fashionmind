"""
/ready must be a real readiness gate: 503 unless models loaded AND DB answers.
/health stays liveness-only (200 as long as the process serves).
"""


def test_health_is_liveness_only(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_503_when_model_count_below_min(client, monkeypatch):
    # the fake registry has ~22 objects; MIN_MODELS is 30
    import api.main as main
    monkeypatch.setattr(main, "MIN_MODELS", 30)
    r = client.get("/ready")
    assert r.status_code == 503
    b = r.json()
    assert b["ready"] is False and b["models_ok"] is False
    assert b["db_ok"] is True                       # fake DB answers fine


def test_ready_200_when_models_and_db_ok(client, monkeypatch):
    import api.main as main
    monkeypatch.setattr(main, "MIN_MODELS", 1)
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_ready_503_when_db_unreachable(client, monkeypatch, fake_db):
    import api.main as main
    monkeypatch.setattr(main, "MIN_MODELS", 1)

    def boom(*a, **k):
        raise ConnectionError("supabase unreachable")
    fake_db.table = boom
    r = client.get("/ready")
    assert r.status_code == 503
    b = r.json()
    assert b["ready"] is False and b["db_ok"] is False
    assert "db_error" in b
