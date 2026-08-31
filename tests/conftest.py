"""
Shared pytest fixtures for FashionMind.

The API loads ~200 MB of trained artifacts at startup. The tests here do NOT
need trained models — they check request/response contracts and pure logic —
so `fake_models()` builds a tiny, coherent `_M` registry and the `client`
fixture patches model loading and the Supabase client before the app starts.
"""
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.pop("GEMINI_API_KEY", None)          # force demo mode in chat tests
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-use-only")
# Keep the test suite fully offline: an empty value survives load_dotenv(
# override=False), so no live SerpApi / Pinterest calls (and no quota spend).
os.environ["SERPAPI_KEY"] = ""
os.environ["PINTEREST_ACCESS_TOKEN"] = ""
os.environ["PINTEREST_ALLOW_UNOFFICIAL"] = ""
os.environ["OPENROUTER_API_KEY"] = ""          # chat stays in demo mode in tests


def build_fake_M():
    """A minimal `_M` that satisfies the cold-start / trends / outfit / explain
    code paths without any FAISS index or ALS model."""
    art = pd.DataFrame({
        "article_id": ["1", "2", "3", "4"],
        "prod_name": ["Tee", "Jeans", "Dress", "Coat"],
        "product_type_name": ["T-shirt", "Trousers", "Dress", "Coat"],
        "product_type_name_idx": [0, 1, 2, 3],
        "product_group_name": ["Garment Upper body", "Garment Lower body",
                                "Garment Full body", "Garment Upper body"],
        "colour_group_name": ["Black", "Blue", "Red", "Camel"],
        "colour_group_name_idx": [0, 1, 2, 3],
        "garment_group_name": ["Jersey Basic", "Denim", "Dresses", "Outerwear"],
        "garment_group_name_idx": [0, 1, 2, 3],
        "detail_desc": ["a", "b", "c", "d"],
    })
    pop_seg = pd.DataFrame({
        "age_bucket": ["26-35"] * 4,
        "club": ["ACTIVE"] * 4,
        "article_id": ["1", "2", "3", "4"],
        "score": [1.0, 0.9, 0.8, 0.7],
    })
    trend = pd.DataFrame({
        "week": pd.to_datetime(["2020-09-07"] * 3),
        "product_type_name": ["Dress", "T-shirt", "Coat"],
        "trend_score": [0.9, 0.5, 0.3],
        "sales": [100, 80, 20],
        "predicted": [98.0, 82.0, 22.0],
    })
    outfit = pd.DataFrame({
        "upper_article_id": ["1"],
        "lower_article_id": ["2"],
        "compatibility_score": [0.87],
    })
    shap_df = pd.DataFrame({
        "customer_id": ["cust_known"],
        "article_id": ["2"],
        "reason_1": ["↑ Personalised match"],
        "reason_2": ["↑ Trending this week"],
        "reason_3": ["↑ Matches price range"],
    })
    cust = pd.DataFrame({
        "customer_id": ["cust_cold"],
        "age": [30.0], "age_norm": [0.3], "club": ["ACTIVE"],
        "engagement_score": [0.5],
    }).set_index("customer_id")

    M = {
        "art": art,
        "art_lu": art.set_index("article_id"),
        "pop_seg": pop_seg,
        "pop": pop_seg,
        "pop_s": {"1": 1.0, "2": 0.9, "3": 0.8, "4": 0.7},
        "pop12": ["1", "2", "3", "4"],
        "trend": trend,
        "lt_map": {"Dress": 0.9, "T-shirt": 0.5, "Coat": 0.3},
        "outfit_df": outfit,
        "shap_df": shap_df,
        "cust": cust,
        "a_price": {"1": 0.01, "2": 0.03, "3": 0.05, "4": 0.09},
        "u_price": {},
        "u_hist": {},
        "cid2u": {},              # no known CF users -> everyone is cold-start
        "i2aid": {},
        "u_top_pt": {},
        "vis_a2i": {},            # nothing in the visual index -> 404 path
        "vis_aids": np.array([], dtype=object),
        "nlp_a2i": {},
        # tiny RAG stand-in so retrieve_context returns something deterministic
        "kb": None,
        "kb_embs": None,
    }
    return M


class _FakeTable:
    """Chainable no-op stand-in for a supabase-py query builder."""
    def __init__(self, store, name):
        self._store, self._name = store, name
        self._rows = list(store.get(name, []))

    def select(self, *a, **k): return self
    def insert(self, row, *a, **k):
        rows = row if isinstance(row, list) else [row]
        processed = []
        for r in rows:
            r = dict(r)
            bucket = self._store.setdefault(self._name, [])
            r.setdefault("id", str(len(bucket) + 1))   # supabase ids are string UUIDs
            bucket.append(r)
            processed.append(r)
        self._rows = processed
        return self
    def upsert(self, row, *a, **k): return self.insert(row)
    def update(self, *a, **k): return self
    def delete(self, *a, **k): return self
    def eq(self, field, value):
        self._rows = [r for r in self._store.get(self._name, []) if r.get(field) == value]
        return self
    def ilike(self, *a, **k): return self
    def gte(self, *a, **k): return self
    def lte(self, *a, **k): return self
    def order(self, *a, **k): return self
    def range(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def execute(self):
        return types.SimpleNamespace(data=list(self._rows))


class _FakeDB:
    def __init__(self): self._store = {}
    def table(self, name): return _FakeTable(self._store, name)


@pytest.fixture()
def fake_db():
    return _FakeDB()


@pytest.fixture()
def client(monkeypatch, fake_db):
    """FastAPI TestClient with model loading + Supabase patched out."""
    import src.genai.stylist_chatbot as sc

    fake_M = build_fake_M()
    monkeypatch.setattr(sc, "_M", fake_M, raising=False)
    monkeypatch.setattr(sc, "_load", lambda: None)
    # retrieve_context needs the RAG artefacts; stub it for contract tests
    monkeypatch.setattr(sc, "retrieve_context",
                        lambda q, k=3: ["Pick breathable fabrics for warm weather."])

    import api.db as db
    monkeypatch.setattr(db, "get_db", lambda: fake_db)

    import api.routes.auth as auth
    import api.routes.cart as cart
    import api.routes.products as products
    for mod in (auth, cart, products):
        if hasattr(mod, "get_db"):
            monkeypatch.setattr(mod, "get_db", lambda: fake_db)

    import api.main as main
    monkeypatch.setattr(main, "_load", lambda: None)
    monkeypatch.setattr(main, "retrieve_context",
                        lambda q, k=3: ["Pick breathable fabrics for warm weather."])
    monkeypatch.setattr(main, "_MODELS", fake_M)

    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c
