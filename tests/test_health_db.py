"""
/health/db — the RLS/trigger/cron watchdog. Two things worth pinning:
  1. unauthenticated callers get ONLY {"ok": bool} — never the per-invariant
     detail, which is a map of how to attack the DB while it's broken;
  2. a broken posture is a 503 (so the CI smoke test fails and GitHub emails).
"""


def _token(client):
    r = client.post("/auth/signup", json={
        "email": "posture@test.com", "password": "hunter2hunter2", "name": "P"})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def test_unauth_gets_bare_ok_only(client):
    r = client.get("/health/db")
    assert r.status_code == 200
    assert r.json() == {"ok": True}          # no anon_grants / cron / trigger keys


def test_authed_gets_full_detail(client):
    r = client.get("/health/db", headers={"Authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "anon_grants" in body and "event_trigger" in body and "cron" in body


def test_drift_is_503_and_still_bare_for_unauth(client, fake_db):
    fake_db.rpc_return = {
        "ok": False, "anon_grants": 3, "event_trigger": False,
        "rls_all_tables": False, "cron": {"purge": False}}
    r = client.get("/health/db")
    assert r.status_code == 503
    assert r.json() == {"ok": False}


def test_drift_detail_visible_when_authed(client, fake_db):
    tok = _token(client)
    fake_db.rpc_return = {
        "ok": False, "anon_grants": 3, "event_trigger": False,
        "rls_all_tables": False, "cron": {"purge": False}}
    r = client.get("/health/db", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 503
    assert r.json()["anon_grants"] == 3 and r.json()["event_trigger"] is False


def test_rpc_failure_is_503_not_500(client, fake_db):
    def boom(*a, **k):
        raise RuntimeError("security_posture() does not exist")
    fake_db.rpc = boom
    r = client.get("/health/db")
    assert r.status_code == 503
    assert r.json()["ok"] is False
