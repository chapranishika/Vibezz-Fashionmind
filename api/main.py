"""
FashionMind API — Production FastAPI Application
================================================
All models loaded once at startup via lifespan context.
Endpoints: /health /recommend /trends /outfit /visual-search /explain /chat /chat/stream
"""
import os, sys, json, warnings
from pathlib import Path

# Load .env before anything reads os.getenv (auth secret, Supabase keys, API
# keys). A bare `uvicorn api.main:app` does not otherwise pick it up.
_BASE_DIR = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_BASE_DIR / ".env", override=False)
except Exception:
    pass

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from api.db import (log_recommendations_bulk, create_chat_session,
                    log_chat_message)
from api.routes.auth     import router as auth_router
from api.routes.cart     import router as cart_router
from api.routes.products import router as products_router
from api.routes.catalog  import router as catalog_router

# Anchor all relative model/data paths to the project root regardless of CWD.
BASE_DIR = _BASE_DIR
os.chdir(BASE_DIR)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.genai.stylist_chatbot import (
    _M as _MODELS, _load,
    chat, stream_chat, retrieve_context,
    get_recommendations, get_trend_report,
    get_outfit_suggestion, explain_recommendation,
)

# Public model registry. Shares the SAME objects as stylist_chatbot._M — the
# models are loaded once, not once per module, so there is no duplicate copy
# of the ALS model / FAISS indexes / feature arrays in memory.
M = {}
# minimum object count a healthy load produces (currently 33); readiness fails
# below this so a model-less container never gets traffic.
MIN_MODELS = 30
_LOAD_ERROR: str | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _LOAD_ERROR
    print("Loading FashionMind models...")
    import time
    t0 = time.time()
    try:
        _load()               # populate the shared registry once
        M.update(_MODELS)     # expose the same objects here — no second copy
        print(f"All models loaded in {time.time()-t0:.1f}s ✓  ({len(M)} objects)")
    except Exception as e:
        _LOAD_ERROR = f"{type(e).__name__}: {e}"
        print(f"Model loading error: {_LOAD_ERROR}")
    yield
    M.clear()

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app = FastAPI(title="FashionMind API",
              description="Multi-stage fashion recommender with Gemini stylist",
              version="1.0.0", lifespan=lifespan)

app.state.limiter = limiter
app.include_router(auth_router)
app.include_router(cart_router)
app.include_router(products_router)
app.include_router(catalog_router)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

from api.metrics import (MetricsMiddleware, metrics_response, record_recs,
                         record_chat, record_tool_call, record_llm_tokens)
app.add_middleware(MetricsMiddleware)

# Serve the extracted subset of real H&M product photos, if present.
from fastapi.staticfiles import StaticFiles
_IMG_DIR = BASE_DIR / "data" / "raw" / "images"
IMAGES_MOUNTED = _IMG_DIR.is_dir() and any(_IMG_DIR.iterdir())
if IMAGES_MOUNTED:
    app.mount("/images", StaticFiles(directory=str(_IMG_DIR)), name="images")

# ── Request models ─────────────────────────────────────────────────
# Cold-start recommendations are handled inside
# src.genai.stylist_chatbot.get_recommendations (age_bucket × club segment
# with a global-popularity fallback) — there is no separate helper here.

class RecommendReq(BaseModel):
    customer_id: str
    occasion:    Optional[str]   = None
    max_price:   Optional[float] = None
    n:           int             = 12

class ChatReq(BaseModel):
    message:     str
    customer_id: str  = "anonymous"
    history:     list = []

class OutfitReq(BaseModel):
    upper_article_id: Optional[str] = None
    style:            Optional[str] = None

class ExplainReq(BaseModel):
    customer_id: str
    article_id:  str

class VisualSearchReq(BaseModel):
    article_id: str
    k:          int = 10

class TextSearchReq(BaseModel):
    query: str
    k:     int = 10

# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness only — the process is up and serving. Says nothing about whether
    models loaded or the DB is reachable; use /ready for that."""
    return {"status": "ok", "models_loaded": len(M), "service": "FashionMind",
            "images_mounted": IMAGES_MOUNTED}

@app.get("/ready")
def ready():
    """Readiness: models actually loaded AND the DB answers. 503 otherwise, so
    the platform healthcheck (Dockerfile HEALTHCHECK -> here) refuses to route
    traffic to a container that booted broken — the "prod ran for months with
    zero models" failure mode becomes impossible."""
    checks = {"models": len(M), "min_models": MIN_MODELS,
              "models_ok": len(M) >= MIN_MODELS, "load_error": _LOAD_ERROR}
    try:
        from api.db import get_db
        get_db().table("recommendations").select("id").limit(1).execute()
        checks["db_ok"] = True
    except Exception as e:
        checks["db_ok"] = False
        checks["db_error"] = f"{type(e).__name__}: {e}"
    ready = checks["models_ok"] and checks["db_ok"]
    return JSONResponse({"ready": ready, **checks}, status_code=200 if ready else 503)

@app.get("/metrics")
def metrics():
    """Prometheus exposition. Scrape target for a dashboard / alert rules."""
    db_ok = True
    try:
        from api.db import get_db
        get_db().table("recommendations").select("id").limit(1).execute()
    except Exception:
        db_ok = False
    return metrics_response(models_loaded=len(M),
                            ready=(len(M) >= MIN_MODELS and db_ok))

@app.get("/health/db")
def health_db(request: Request):
    """Security-posture watchdog: whether the RLS lockdown, the auto-revoke
    event trigger, and the pg_cron maintenance jobs are all still in place
    (db/migrations/002-004). smoke_prod.py asserts `ok` here, so a failed daily
    CI run — which GitHub emails the repo owner about — is the alert channel if
    any of them silently disappears.

    Unauthenticated callers (the smoke test) get only `{"ok": bool}` + status —
    the per-invariant detail (grant counts, which trigger is missing) is a map
    of how to attack the DB when it's broken, so it needs a valid JWT."""
    try:
        from api.db import get_db
        from api.routes.auth import SECRET, ALGO
        import jwt as _jwt
        res = get_db().rpc("security_posture", {}).execute()
        posture = res.data if isinstance(res.data, dict) else (res.data or {})
        ok = bool(posture.get("ok"))
        code = 200 if ok else 503

        authed = False
        hdr = request.headers.get("authorization", "")
        if hdr.lower().startswith("bearer "):
            try:
                _jwt.decode(hdr[7:], SECRET, algorithms=[ALGO])
                authed = True
            except Exception:
                authed = False
        return JSONResponse(posture if authed else {"ok": ok}, status_code=code)
    except Exception as e:
        # detail of an *exception* is fine to expose — it's not posture data
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"},
                            status_code=503)

@app.post("/recommend")
def recommend(req: RecommendReq):
    result = get_recommendations(req.customer_id, req.occasion, req.max_price, req.n)
    record_recs(len(result.get("recommendations", [])), result.get("is_cold_start", False))
    # ── Log to Supabase (fire-and-forget, never blocks response) ──
    try:
        recs = result.get("recommendations", [])
        db_rows = [dict(
            customer_id=req.customer_id,
            article_id=r["article_id"],
            score=float(r.get("score", 0)),
            rank=i + 1,
            is_cold_start=result.get("is_cold_start", False),
            reason_1=(r.get("reasons") or [None])[0],
            reason_2=(r.get("reasons") or [None, None])[1] if len(r.get("reasons") or []) > 1 else None,
            reason_3=(r.get("reasons") or [None]*3)[2]     if len(r.get("reasons") or []) > 2 else None,
            occasion=req.occasion,
            model_version="v1",
        ) for i, r in enumerate(recs)]
        log_recommendations_bulk(db_rows)
    except Exception:
        pass
    return result

@app.get("/trends")
def trends(category: Optional[str] = None):
    return get_trend_report(category)

@app.post("/outfit")
def outfit(req: OutfitReq):
    return get_outfit_suggestion(req.upper_article_id, req.style)

@app.post("/explain")
def explain(req: ExplainReq):
    return explain_recommendation(req.customer_id, req.article_id)

@app.post("/visual-search")
def visual_search(req: VisualSearchReq):
    if not M: raise HTTPException(503, "Models not loaded")
    aid = req.article_id.lstrip('0')
    if aid not in M['vis_a2i']:
        raise HTTPException(404, f"Article {aid} not in visual index")
    idx   = M['vis_a2i'][aid]
    D, I  = M['vis_index'].search(M['vis_feats'][idx:idx+1], req.k+1)
    results = []
    for sim, i in zip(D[0][1:], I[0][1:]):
        if i < 0 or i >= len(M['vis_aids']): continue
        a    = M['vis_aids'][i]
        info = {'article_id': a, 'visual_similarity': round(float(sim), 4)}
        if a in M['art_lu'].index:
            r = M['art_lu'].loc[a]
            info.update({'product_type': str(r['product_type_name']),
                         'colour':       str(r['colour_group_name'])})
        results.append(info)
    return {"query_article": aid, "similar_items": results}

@app.post("/text-search")
def text_search(req: TextSearchReq):
    if not M: raise HTTPException(503, "Models not loaded")
    chunks = retrieve_context(req.query, k=req.k)
    return {"query": req.query, "knowledge_chunks": chunks}

@app.post("/chat")
async def chat_endpoint(req: ChatReq):
    import time
    _t0 = time.time()
    # Create or reuse a chat session in Supabase
    try:
        session_id = create_chat_session(customer_id=req.customer_id)
    except Exception:
        session_id = None
    api_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        # Demo mode — works without any LLM key
        chunks = retrieve_context(req.message, k=2)
        recs   = get_recommendations(req.customer_id, n=4)
        demo = (
            "**[Demo mode — set OPENROUTER_API_KEY (or GEMINI_API_KEY) for live replies]**\n\n"
            + "**Style tips for your query:**\n"
            + "\n".join(f"• {c[:100]}" for c in chunks[:2])
            + "\n\n**Top picks:**\n"
            + "\n".join(
                f"• {r['article_id']} — {r.get('product_type_name','item')} "
                f"({r.get('colour_group_name','')}) | {(r.get('reasons') or ['—'])[0]}"
                for r in recs.get('recommendations', [])[:3])
        )
        record_chat("demo")
        return {"response": demo, "history": req.history, "mode": "demo"}
    import time as _time
    try:
        response, updated = chat(req.message, req.customer_id, req.history, api_key)
    except Exception:
        record_chat("error"); raise
    record_chat("live")
    # ── Log to Supabase ──────────────────────────────────────────
    try:
        latency = int((_time.time() - _t0) * 1000)
        if session_id:
            log_chat_message(session_id, role="user",      content=req.message)
            log_chat_message(session_id, role="assistant", content=response, latency_ms=latency)
    except Exception:
        pass
    return {"response": response, "history": updated, "mode": "live", "session_id": session_id}

@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    api_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        async def demo_stream():
            import asyncio
            msg   = "Demo mode: set OPENROUTER_API_KEY for live streaming. "
            recs  = get_recommendations(req.customer_id, n=3)
            items = "  ".join(
                f"{r['article_id']}({r.get('product_type_name','item')})"
                for r in recs.get('recommendations', [])[:3])
            full  = msg + "Top picks: " + items
            for w in full.split(' '):
                await asyncio.sleep(0.04)
                yield f"data: {json.dumps({'token': w+' ', 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True, 'full': full})}\n\n"
        return StreamingResponse(demo_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    async def _instrumented():
        record_chat("live")
        async for chunk in _aiter(stream_chat(req.message, req.customer_id, req.history, api_key)):
            if '"tool_call"' in chunk:
                try:
                    record_tool_call(json.loads(chunk[5:]).get("tool_call", ""))
                except Exception:
                    pass
            yield chunk

    return StreamingResponse(_instrumented(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _aiter(gen):
    """stream_chat is a sync generator; iterate it without blocking the loop hard."""
    for item in gen:
        yield item