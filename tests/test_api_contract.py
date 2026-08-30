"""Request/response contract tests for the FastAPI surface.

Model loading and Supabase are patched (see conftest.py) so these run in
milliseconds and need no trained artefacts.
"""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] > 0


def test_recommend_cold_start_contract(client):
    r = client.post("/recommend", json={"customer_id": "cust_cold", "n": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["is_cold_start"] is True
    assert len(body["recommendations"]) == 4
    first = body["recommendations"][0]
    assert {"article_id", "score", "reasons", "product_type_name"} <= set(first)


def test_trends_contract(client):
    r = client.get("/trends")
    assert r.status_code == 200
    body = r.json()
    assert "trending" in body and len(body["trending"]) >= 1
    assert "trend_score" in body["trending"][0]


def test_outfit_contract(client):
    r = client.post("/outfit", json={"upper_article_id": "1"})
    assert r.status_code == 200
    body = r.json()
    assert body["upper"] == "1" and body["lower"] == "2"


def test_explain_contract(client):
    r = client.post("/explain", json={"customer_id": "cust_known", "article_id": "2"})
    assert r.status_code == 200
    assert len(r.json()["reasons"]) == 3


def test_visual_search_unknown_article_404(client):
    r = client.post("/visual-search", json={"article_id": "0000999", "k": 5})
    assert r.status_code == 404


def test_text_search_contract(client):
    r = client.post("/text-search", json={"query": "what to wear to the beach", "k": 3})
    assert r.status_code == 200
    assert isinstance(r.json()["knowledge_chunks"], list)


def test_chat_demo_mode_without_api_key(client):
    r = client.post("/chat", json={"message": "help me pick an outfit",
                                    "customer_id": "cust_cold"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "demo"
    assert "Demo mode" in body["response"]


def test_rate_limiter_wired():
    # limiter is attached to app.state at import time
    import api.main as main
    assert getattr(main.app.state, "limiter", None) is not None


# ── auth + cart flow ────────────────────────────────────────────────────────
def test_signup_login_and_authed_cart(client):
    su = client.post("/auth/signup", json={
        "email": "a@b.com", "password": "hunter2hunter2", "name": "Ada"})
    assert su.status_code == 200, su.text
    token = su.json()["token"]
    assert su.json()["customer_id"].startswith("FM")

    li = client.post("/auth/login", json={"email": "a@b.com", "password": "hunter2hunter2"})
    assert li.status_code == 200
    assert li.json()["email"] == "a@b.com"

    # cart requires auth
    assert client.get("/cart").status_code in (401, 403)

    ok = client.get("/cart", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200
    assert ok.json() == {"items": [], "total": 0, "count": 0}


def test_login_wrong_password_rejected(client):
    client.post("/auth/signup", json={
        "email": "c@d.com", "password": "correct-horse", "name": "Bob"})
    bad = client.post("/auth/login", json={"email": "c@d.com", "password": "nope"})
    assert bad.status_code == 401
