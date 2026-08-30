"""Logic tests for the recommendation code paths that do not need trained models."""
import numpy as np
import pandas as pd
import pytest

import src.genai.stylist_chatbot as sc
from tests.conftest import build_fake_M


@pytest.fixture()
def fake_models(monkeypatch):
    M = build_fake_M()
    monkeypatch.setattr(sc, "_M", M, raising=False)
    monkeypatch.setattr(sc, "_load", lambda: None)
    return M


def test_cold_start_uses_age_club_segment(fake_models):
    out = sc.get_recommendations("cust_cold", n=4)
    assert out["is_cold_start"] is True
    assert len(out["recommendations"]) == 4
    # highest-scoring segment item first
    assert out["recommendations"][0]["article_id"] == "1"
    for r in out["recommendations"]:
        assert {"article_id", "product_type_name", "colour_group_name",
                "score", "reasons"} <= set(r)


def test_cold_start_dataframe_truthiness_regression(fake_models):
    """`pop_seg = _M.get('pop_seg') or _M.get('pop')` used to raise
    'truth value of a DataFrame is ambiguous'. Guard against a regression."""
    assert isinstance(fake_models["pop_seg"], pd.DataFrame)
    out = sc.get_recommendations("anyone_unknown", n=3)          # must not raise
    assert out["is_cold_start"] is True
    assert len(out["recommendations"]) == 3


def test_cold_start_falls_back_to_global_pop_when_no_profile(fake_models):
    # unknown cid that is NOT in _M['cust'] -> global pop12 fallback
    out = sc.get_recommendations("no_such_customer", n=2)
    assert out["is_cold_start"] is True
    ids = [r["article_id"] for r in out["recommendations"]]
    assert ids == ["1", "2"]                                     # pop12 order


def test_trend_report_shape_and_ordering(fake_models):
    rep = sc.get_trend_report()
    assert "week" in rep and "trending" in rep
    scores = [t["trend_score"] for t in rep["trending"]]
    assert scores == sorted(scores, reverse=True)
    assert rep["trending"][0]["product_type_name"] == "Dress"


def test_trend_report_category_filter(fake_models):
    rep = sc.get_trend_report("shirt")
    assert all("shirt" in t["product_type_name"].lower() for t in rep["trending"])


def test_outfit_suggestion_returns_pair(fake_models):
    out = sc.get_outfit_suggestion("1")
    assert out["upper"] == "1"
    assert out["lower"] == "2"
    assert 0.0 <= out["compatibility_score"] <= 1.0


def test_explain_recommendation_precomputed_hit(fake_models):
    out = sc.explain_recommendation("cust_known", "2")
    assert out["source"] == "pre-computed"
    assert len(out["reasons"]) == 3


def test_explain_recommendation_miss_returns_fallback(fake_models):
    out = sc.explain_recommendation("nobody", "nothing")
    assert "reasons" in out and len(out["reasons"]) == 3
    assert "note" in out
