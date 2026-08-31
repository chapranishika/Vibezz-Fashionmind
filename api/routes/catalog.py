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
