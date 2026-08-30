"""
FashionMind — Supabase database layer
======================================
Project : fashionmind (ap-south-1)
URL     : https://cxfsiotzfshrjdrctmge.supabase.co

Uses supabase-py (service-role key) for all server-side writes.
Anon key is exposed to the frontend only.
"""
import os
from functools import lru_cache
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://cxfsiotzfshrjdrctmge.supabase.co")

@lru_cache(maxsize=1)
def get_db() -> Client:
    """Return a cached Supabase client using the service-role key."""
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not key:
        raise RuntimeError(
            "SUPABASE_SERVICE_KEY not set. "
            "Get it from Supabase dashboard → Settings → API → service_role key."
        )
    return create_client(SUPABASE_URL, key)


# ── Helpers ──────────────────────────────────────────────────────────────────

def log_recommendation(
    customer_id: str,
    article_id: str,
    score: float,
    rank: int,
    is_cold_start: bool = False,
    reason_1: str = None,
    reason_2: str = None,
    reason_3: str = None,
    occasion: str = None,
    model_version: str = "v1",
) -> dict:
    """Insert one recommendation row. Returns the inserted row."""
    db = get_db()
    row = dict(
        customer_id=customer_id, article_id=article_id,
        score=round(float(score), 6), rank=rank,
        is_cold_start=is_cold_start,
        reason_1=reason_1, reason_2=reason_2, reason_3=reason_3,
        occasion=occasion, model_version=model_version,
    )
    res = db.table("recommendations").insert(row).execute()
    return res.data[0] if res.data else {}


def log_recommendations_bulk(rows: list[dict]) -> int:
    """Bulk-insert a list of recommendation dicts. Returns count inserted."""
    if not rows:
        return 0
    db = get_db()
    res = db.table("recommendations").insert(rows).execute()
    return len(res.data or [])


def log_chat_message(
    session_id: str,
    role: str,
    content: str = None,
    tool_name: str = None,
    tool_input: dict = None,
    tool_result: dict = None,
    tokens_in: int = None,
    tokens_out: int = None,
    latency_ms: int = None,
) -> dict:
    """Insert a chat message row."""
    db = get_db()
    row = dict(
        session_id=session_id, role=role, content=content,
        tool_name=tool_name, tool_input=tool_input, tool_result=tool_result,
        tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms,
    )
    res = db.table("chat_messages").insert(row).execute()
    return res.data[0] if res.data else {}


def create_chat_session(customer_id: str = None, metadata: dict = None) -> str:
    """Create a new chat session. Returns the session UUID."""
    db = get_db()
    row = dict(customer_id=customer_id, metadata=metadata or {})
    res = db.table("chat_sessions").insert(row).execute()
    return res.data[0]["id"] if res.data else None


def record_click(rec_id: int) -> None:
    """Mark a recommendation as clicked."""
    db = get_db()
    from datetime import datetime, timezone
    db.table("recommendations").update(
        {"clicked": True, "clicked_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", rec_id).execute()


def record_purchase(rec_id: int) -> None:
    """Mark a recommendation as purchased."""
    db = get_db()
    from datetime import datetime, timezone
    db.table("recommendations").update(
        {"purchased": True, "purchased_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", rec_id).execute()


def get_customer_history(customer_id: str, limit: int = 20) -> list[dict]:
    """Fetch the most recent interactions for a customer."""
    db = get_db()
    res = (
        db.table("interactions")
        .select("article_id, t_dat, price")
        .eq("customer_id", customer_id)
        .order("t_dat", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def upsert_trend_scores(rows: list[dict]) -> int:
    """Upsert weekly trend score rows (week + product_type_name as unique key)."""
    if not rows:
        return 0
    db = get_db()
    res = db.table("trend_scores").upsert(rows, on_conflict="week,product_type_name").execute()
    return len(res.data or [])


def save_ab_result(row: dict) -> dict:
    """Insert an A/B test result row."""
    db = get_db()
    res = db.table("ab_test_results").insert(row).execute()
    return res.data[0] if res.data else {}
