"""
Aggregator routes — trend-driven discovery that links out to real retailers.

  GET /catalog/trends              Pinterest rising fashion searches (India)
  GET /catalog/shop?trend=...      products for a trend keyword (curated + live)
  GET /catalog/outfit?trend=...    a full assembled look for a trend
  GET /catalog/products?...        filtered aggregator search
  GET /catalog/product/{id}        one product (any source)

Products always carry a `buy_url` that opens the item on the retailer's own
site; `price_is_estimate=true` items are category estimates, not live prices.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.genai.stylist_chatbot import (
    get_pinterest_trends, shop_the_trend, build_trend_outfit,
)
from src.catalog import aggregate_search, get_product

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/trends")
def catalog_trends(category: Optional[str] = None, region: str = "IN"):
    return get_pinterest_trends(category=category, region=region)


@router.get("/shop")
def catalog_shop(trend: str = Query(..., description="trend keyword, e.g. 'barrel jeans'"),
                 budget_max: Optional[float] = None, n: int = 12):
    return shop_the_trend(trend, budget_max=budget_max, n=n)


@router.get("/outfit")
def catalog_outfit(trend: str = Query(..., description="trend keyword"),
                   budget_max: Optional[float] = None):
    return build_trend_outfit(trend, budget_max=budget_max)


@router.get("/products")
def catalog_products(q: Optional[str] = None, product_type: Optional[str] = None,
                     tag: Optional[str] = None, colour: Optional[str] = None,
                     source: Optional[str] = Query(None, description="curated|hm-demo|serpapi"),
                     limit: int = 24):
    rows = aggregate_search(
        query=q,
        product_types=[product_type] if product_type else None,
        tags=[t.strip() for t in tag.split(",")] if tag else None,
        colours=[colour] if colour else None,
        limit=limit,
        sources=[source] if source else None,
    )
    return {"products": rows, "count": len(rows)}


@router.get("/product/{product_id}")
def catalog_product(product_id: str):
    p = get_product(product_id)
    if not p:
        raise HTTPException(404, "Product not found")
    return p


@router.get("/metrics", tags=["insights"])
def metrics():
    """Real evaluation numbers for the Dashboard — read from the training
    outputs, never invented. Missing files simply omit their block."""
    import csv, json, os
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent.parent / "data" / "features"
    out: dict = {}

    def _csv(name):
        p = base / name
        if not p.exists():
            return []
        with open(p, newline="") as f:
            return [ {k: (float(v) if _isnum(v) else v) for k, v in row.items()}
                     for row in csv.DictReader(f) ]

    def _isnum(v):
        try:
            float(v); return True
        except (TypeError, ValueError):
            return False

    out["held_out"] = _csv("final_metrics.csv")          # popularity / ALS / full pipeline, held-out test users
    out["candidate_eval"] = _csv("cf_results.csv")        # popularity / ALS / BPR, 3k users
    ab = _csv("ab_test_results.csv")
    out["paired_comparison"] = ab[0] if ab else {}
    mc = base / "model_card.json"
    if mc.exists():
        out["model_card"] = json.loads(mc.read_text(encoding="utf-8"))
    return out
