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
class CartAddReq(BaseModel):
    article_id: str
    size: str = "M"
    quantity: int = 1
    price: float = None

@router.get("/cart")
def get_cart(user=Depends(get_current_user)):
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    items = db.table("cart_items").select("*").eq("customer_id", cid).execute()
    total = sum((i.get("price") or 0) * i.get("quantity", 1) for i in (items.data or []))
    return {"items": items.data or [], "total": round(total, 2), "count": len(items.data or [])}

@router.post("/cart")
def add_to_cart(req: CartAddReq, user=Depends(get_current_user)):
    db = get_db()
    res = db.table("user_auth").select("customer_id").eq("id", user["sub"]).execute()
    cid = res.data[0]["customer_id"]
    row = {"customer_id": cid, "article_id": req.article_id,
           "size": req.size, "quantity": req.quantity, "price": req.price}
    db.table("cart_items").upsert(row, on_conflict="customer_id,article_id,size").execute()
    return {"status": "added", "article_id": req.article_id}

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
