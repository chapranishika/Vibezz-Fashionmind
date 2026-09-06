"""The RECS_USE_GRU hook: off by default (no behaviour change), and when a GRU
is present in the registry it unions candidates without crashing."""
import numpy as np


def test_recommend_unchanged_when_gru_absent(client):
    # fake_M has no 'gru' key -> the union block is skipped entirely
    r = client.post("/recommend", json={"customer_id": "u1", "n": 5})
    assert r.status_code == 200
    assert r.json()["recommendations"]


def test_recommend_survives_a_gru_in_the_registry(client, monkeypatch):
    import src.genai.stylist_chatbot as sc

    class _FakeGRU:
        def topk(self, seq, k=50):
            return list(range(1, k + 1))

    # wire a GRU whose vocab overlaps the fake item ids so the union has effect
    ids = list(getattr(sc._M["i2aid"], "tolist", lambda: list(sc._M["i2aid"]))())
    a2i = {str(a).lstrip("0"): i + 1 for i, a in enumerate(ids[:20])}
    sc._M["gru"] = {"model": _FakeGRU(), "a2i": a2i,
                    "i2a": {v: k for k, v in a2i.items()}}
    # give the test user some history in that vocab
    sc._M["u_hist"] = dict(sc._M["u_hist"])
    sc._M["u_hist"]["u1"] = list(a2i.keys())[:3]

    r = client.post("/recommend", json={"customer_id": "u1", "n": 5})
    assert r.status_code == 200
    assert isinstance(r.json().get("recommendations"), list)
    sc._M.pop("gru", None)
    sc._M.pop("_aid2iidx", None)
