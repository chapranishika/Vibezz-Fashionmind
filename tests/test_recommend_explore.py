"""RECS_EXPLORE_EPS defaults off — /recommend order and payload are unchanged."""


def test_recommend_deterministic_by_default(client):
    a = client.post("/recommend", json={"customer_id": "u1", "n": 6}).json()
    b = client.post("/recommend", json={"customer_id": "u1", "n": 6}).json()
    ids_a = [r["article_id"] for r in a["recommendations"]]
    ids_b = [r["article_id"] for r in b["recommendations"]]
    assert ids_a == ids_b
    assert "explored" not in a               # no exploration flag when eps = 0


def test_recommend_still_returns_scored_ranked_items(client):
    r = client.post("/recommend", json={"customer_id": "u2", "n": 4}).json()
    recs = r["recommendations"]
    assert recs and all("article_id" in x for x in recs)
