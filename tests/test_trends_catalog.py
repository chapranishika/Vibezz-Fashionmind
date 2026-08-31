"""Pinterest trend provider + aggregator catalogue + trend-shop tools."""
import pytest

from src.trends.pinterest_trends import get_pinterest_trends
from src.trends.trend_map import map_trend
from src.catalog import aggregate_search, get_product, CuratedTrendSource


# ── Pinterest provider ─────────────────────────────────────────────────────
def test_pinterest_falls_back_to_cache_offline(monkeypatch):
    # no token, unofficial disabled -> cache seed must still answer
    monkeypatch.delenv("PINTEREST_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("PINTEREST_ALLOW_UNOFFICIAL", raising=False)
    rows = get_pinterest_trends(limit=10)
    assert len(rows) == 10
    assert all(0 <= r["score"] <= 100 for r in rows)
    assert rows == sorted(rows, key=lambda r: -r["score"])
    assert rows[0]["source"] == "cache"


def test_pinterest_category_filter():
    rows = get_pinterest_trends(category="jeans", limit=25)
    assert any("jean" in r["keyword"] for r in rows)


# ── keyword -> catalogue mapping ──────────────────────────────────────────
def test_map_trend_known_and_unknown():
    m = map_trend("barrel jeans")
    assert "Trousers" in m["product_types"]
    assert "barrel-jeans" in m["tags"]
    # unknown term degrades to bare tokens, never raises
    u = map_trend("some brand new microtrend")
    assert u["product_types"] == [] and "brand" in u["tags"]


def test_map_trend_does_not_mutate_across_calls():
    """regression: a shared-list bug made every call return the union of all
    previous mappings."""
    map_trend("balletcore")
    map_trend("quiet luxury")
    m = map_trend("barrel jeans")
    assert m["product_types"] == ["Trousers"]        # only its own mapping


# ── curated source + aggregator ──────────────────────────────────────────
def test_curated_source_loads_seed():
    c = CuratedTrendSource()
    assert len(c._items) >= 30
    assert all(i["buy_url"].startswith("http") for i in c._items)


def test_aggregate_search_by_tag_returns_products():
    rows = aggregate_search(tags=["balletcore"], limit=8)
    assert rows
    assert all("buy_url" in r and "retailer" in r for r in rows)


def test_aggregate_search_query_space_vs_hyphen():
    # "quiet luxury" must still hit the "quiet-luxury" tagged items
    rows = aggregate_search(query="quiet luxury",
                            product_types=["Sweater", "Trousers", "Blazer", "Coat"],
                            tags=["quiet-luxury", "minimal", "neutral"], limit=8)
    assert rows


def test_get_product_roundtrip():
    c = CuratedTrendSource()
    pid = c._items[0]["id"]
    p = get_product(pid)
    assert p and p["id"] == pid


# ── stylist trend tools ──────────────────────────────────────────────────
def test_shop_the_trend_shape():
    from src.genai.stylist_chatbot import shop_the_trend
    out = shop_the_trend("cargo pants", n=5)
    assert out["trend"] == "cargo pants"
    assert 1 <= len(out["products"]) <= 5
    assert all(p["buy_url"].startswith("http") for p in out["products"])


def test_build_trend_outfit_assembles_full_look():
    from src.genai.stylist_chatbot import build_trend_outfit
    out = build_trend_outfit("office siren", budget_max=8000)
    slots = {x["slot"] for x in out["look"]}
    assert {"shoes"} <= slots and len(out["look"]) >= 3
    assert out["estimated_total"]["min"] > 0


def test_trend_report_carries_pinterest_rising(monkeypatch):
    import src.genai.stylist_chatbot as sc
    from tests.conftest import build_fake_M
    monkeypatch.setattr(sc, "_M", build_fake_M(), raising=False)
    monkeypatch.setattr(sc, "_load", lambda: None)
    rep = sc.get_trend_report()
    assert "pinterest_rising" in rep and len(rep["pinterest_rising"]) >= 1
