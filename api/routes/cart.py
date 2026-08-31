"""
Cart routes: /cart  GET/POST/DELETE
Wishlist routes: /wishlist  GET/POST/DELETE
Order routes: /orders  GET/POST
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from api.db import get_db
from api.routes.auth import get_current_user
from datetime import datetime, timezone

router = APIRouter(tags=["cart"])

# ── Cart ─────────────────────────────────────────────────────────────────────
# Two kinds of cart line:
#   · demo   — a trained-catalogue (H&M) item that can go through fake checkout
#   · compare — an aggregator item from a real retailer; "buy" deep-links out,
#     the cart is just a cross-site price/compare list
class CartAddReq(BaseModel):
    article_id: str
    size: str = "M"
    quantity: int = 1
    price: float = None
    mode: str = "demo"                 # "demo" | "compare"
    source: str = None                 # curated | hm-demo | serpapi
    retailer: str = None
    buy_url: str = None
    title: str = None
    image: str = None
    look: str = None
    currency: str = "INR"

_CART_COLS = {"customer_id", "article_id", "size", "quantity", "price"}

@router.get("/cart")
def get_cart(user=Depends(get_current_user)):
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    items = db.table("cart_items").select("*").eq("customer_id", cid).execute()
    data = items.data or []
    total = sum((i.get("price") or 0) * i.get("quantity", 1)
                for i in data if (i.get("mode") or "demo") == "demo")
    return {"items": data, "total": round(total, 2), "count": len(data)}

@router.post("/cart")
def add_to_cart(req: CartAddReq, user=Depends(get_current_user)):
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    row = {"customer_id": cid, "article_id": req.article_id, "size": req.size,
           "quantity": req.quantity, "price": req.price, "mode": req.mode,
           "source": req.source, "retailer": req.retailer, "buy_url": req.buy_url,
           "title": req.title, "image": req.image, "look": req.look,
           "currency": req.currency}
    try:
        db.table("cart_items").upsert(row, on_conflict="customer_id,article_id,size").execute()
    except Exception:
        # cart_items has not been migrated with the aggregator columns — fall
        # back to the original minimal schema so demo checkout still works.
        db.table("cart_items").upsert({k: row[k] for k in _CART_COLS},
                                      on_conflict="customer_id,article_id,size").execute()
    return {"status": "added", "article_id": req.article_id, "mode": req.mode}

@router.get("/cart/compare")
def compare_cart(user=Depends(get_current_user)):
    """Aggregator view: cross-site items grouped by look, with a price range and
    a deep link to each retailer. `total` here is only the demo-mode subtotal."""
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    items = (db.table("cart_items").select("*").eq("customer_id", cid).execute().data) or []
    agg = [i for i in items if (i.get("mode") or "demo") == "compare"]
    looks: dict = {}
    for i in agg:
        looks.setdefault(i.get("look") or "saved", []).append({
            "article_id": i.get("article_id"), "title": i.get("title"),
            "retailer": i.get("retailer"), "buy_url": i.get("buy_url"),
            "image": i.get("image"), "price": i.get("price"),
            "currency": i.get("currency", "INR"), "source": i.get("source"),
        })
    groups = []
    for name, its in looks.items():
        prices = [x["price"] for x in its if x.get("price")]
        groups.append({"look": name, "items": its, "item_count": len(its),
                       "retailers": sorted({x["retailer"] for x in its if x.get("retailer")}),
                       "price_estimate": {"min": sum(prices) if prices else None,
                                          "currency": its[0]["currency"] if its else "INR"}})
    demo_total = sum((i.get("price") or 0) * i.get("quantity", 1)
                     for i in items if (i.get("mode") or "demo") == "demo")
    return {"looks": groups, "compare_count": len(agg),
            "demo_subtotal": round(demo_total, 2),
            "note": "Aggregator items link out to each retailer to buy. Prices are "
                    "estimates unless a live source is configured."}

@router.delete("/cart/{article_id}")
def remove_from_cart(article_id: str, size: str = "M", user=Depends(get_current_user)):
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    db.table("cart_items").delete().eq("customer_id", cid).eq("article_id", article_id).eq("size", size).execute()
    return {"status": "removed"}

@router.delete("/cart")
def clear_cart(user=Depends(get_current_user)):
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    db.table("cart_items").delete().eq("customer_id", cid).execute()
    return {"status": "cleared"}

# ── Wishlist ──────────────────────────────────────────────────────────────────
@router.get("/wishlist")
def get_wishlist(user=Depends(get_current_user)):
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    items = db.table("wishlists").select("article_id,added_at").eq("customer_id", cid).execute()
    return {"items": items.data or [], "count": len(items.data or [])}

@router.post("/wishlist/{article_id}")
def add_to_wishlist(article_id: str, user=Depends(get_current_user)):
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    try:
        db.table("wishlists").insert({"customer_id": cid, "article_id": article_id}).execute()
    except Exception:
        pass  # already wishlisted
    return {"status": "wishlisted", "article_id": article_id}

@router.delete("/wishlist/{article_id}")
def remove_from_wishlist(article_id: str, user=Depends(get_current_user)):
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    db.table("wishlists").delete().eq("customer_id", cid).eq("article_id", article_id).execute()
    return {"status": "removed"}

# ── Orders ────────────────────────────────────────────────────────────────────
class PlaceOrderReq(BaseModel):
    address: dict
    payment_method: str = "cod"

@router.post("/orders")
def place_order(req: PlaceOrderReq, user=Depends(get_current_user)):
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    # Get cart
    cart = db.table("cart_items").select("*").eq("customer_id", cid).execute()
    if not cart.data:
        raise HTTPException(status_code=400, detail="Cart is empty")
    total = sum((i.get("price") or 0) * i.get("quantity", 1) for i in cart.data)
    # Create order
    order_res = db.table("orders").insert({
        "customer_id": cid, "status": "confirmed",
        "total_amount": round(total, 2),
        "address": req.address, "payment_method": req.payment_method,
    }).execute()
    order_id = order_res.data[0]["id"]
    # Create order items
    items = [{"order_id": order_id, "article_id": i["article_id"],
              "size": i.get("size"), "quantity": i.get("quantity", 1),
              "price": i.get("price")} for i in cart.data]
    db.table("order_items").insert(items).execute()
    # Clear cart
    db.table("cart_items").delete().eq("customer_id", cid).execute()
    return {"order_id": order_id, "status": "confirmed", "total": round(total, 2)}

@router.get("/orders")
def get_orders(user=Depends(get_current_user)):
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    orders = db.table("orders").select("*").eq("customer_id", cid).order("placed_at", desc=True).execute()
    return {"orders": orders.data or []}
