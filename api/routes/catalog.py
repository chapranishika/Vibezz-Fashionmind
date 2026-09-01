"""
Secondary surface — kept for reference, not part of the core recommender product.

  GET /catalog/metrics            REAL model metrics for the Dashboard (used)
  GET /catalog/trends             Pinterest rising fashion searches (experimental)
  GET /catalog/shop?trend=...     products for a trend keyword (experimental)
  GET /catalog/outfit?trend=...   an assembled look for a trend (experimental)
  GET /catalog/products?...       aggregator search (experimental)
  GET /catalog/product/{id}       one aggregator product (experimental)

The trend-shopping aggregator was an earlier direction. The project's product is
the personalised H&M recommender (see /recommend, /explain, /trends). Only
/catalog/metrics is wired into the UI. The rest stay so the code still runs.
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
    out = get_pinterest_trends(category=category, region=region)
    try:
        from src.trends.festivals import current_festive
        out["festive"] = current_festive()      # None when nothing is near
    except Exception:
        out["festive"] = None
    return out


@router.get("/shop")
def catalog_shop(trend: str = Query(..., description="trend keyword, e.g. 'barrel jeans'"),
                 budget_max: Optional[float] = None, n: int = 60):
    return shop_the_trend(trend, budget_max=budget_max, n=n)


@router.get("/outfit")
def catalog_outfit(trend: str = Query(..., description="trend keyword"),
                   budget_max: Optional[float] = None):
    return build_trend_outfit(trend, budget_max=budget_max)


@router.get("/photos")
def catalog_photos(q: str = Query(..., description="plain query, e.g. 'black trousers women'"),
                   n: int = 10):
    """Real product photos for a plain query — used to illustrate catalogue items
    (H&M SKUs) that have no image of their own. Backed by SerpApi's 24h disk
    cache, so a given (type, colour) query costs one lookup per day at most."""
    try:
        from src.catalog.sources import SerpApiShoppingSource
        rows = SerpApiShoppingSource(gl="in").search(query=q, limit=max(1, min(n, 20)))
    except Exception:
        rows = []
    return {"photos": [{"image": r["image"], "title": r.get("title", ""),
                        "buy_url": r.get("buy_url", ""),
                        "price_min": r.get("price_min"), "retailer": r.get("retailer", "")}
                       for r in rows if r.get("image")]}


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
