"""
Prometheus metrics for the API — one /metrics endpoint so degradation is
visible on a dashboard instead of only via the daily CI smoke or a user
complaint.

Exposed series:
  http_request_duration_seconds   histogram   {method,path,status}
  http_requests_total             counter     {method,path,status}
  recommendations_served_total    counter     {cold_start}
  chat_tool_calls_total           counter     {tool}
  llm_tokens_total                counter     {direction}          (in|out)
  chat_requests_total             counter     {mode}               (live|demo|error)
  app_models_loaded               gauge
  app_ready                       gauge                            (1|0)

Path label is the route *template* ("/products/{id}/reviews"), not the raw URL,
so cardinality stays bounded.
"""
from __future__ import annotations

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQ_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency",
    ["method", "path", "status"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
REQ_TOTAL = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])

RECS_SERVED = Counter("recommendations_served_total", "Ranked items returned", ["cold_start"])
TOOL_CALLS = Counter("chat_tool_calls_total", "Stylist tool invocations", ["tool"])
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["direction"])
CHAT_REQS = Counter("chat_requests_total", "Chat requests", ["mode"])

MODELS_LOADED = Gauge("app_models_loaded", "Objects in the model registry")
APP_READY = Gauge("app_ready", "1 if /ready would pass, else 0")


def record_recs(n: int, cold_start: bool) -> None:
    RECS_SERVED.labels(cold_start=str(bool(cold_start)).lower()).inc(max(n, 0))


def record_tool_call(name: str) -> None:
    TOOL_CALLS.labels(tool=name or "unknown").inc()


def record_llm_tokens(tokens_in: int | None, tokens_out: int | None) -> None:
    if tokens_in:
        LLM_TOKENS.labels(direction="in").inc(tokens_in)
    if tokens_out:
        LLM_TOKENS.labels(direction="out").inc(tokens_out)


def record_chat(mode: str) -> None:
    CHAT_REQS.labels(mode=mode).inc()


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        t0 = time.perf_counter()
        status = 500
        try:
            resp = await call_next(request)
            status = resp.status_code
            return resp
        finally:
            path = _route_template(request)
            labels = (request.method, path, str(status))
            REQ_LATENCY.labels(*labels).observe(time.perf_counter() - t0)
            REQ_TOTAL.labels(*labels).inc()


def metrics_response(models_loaded: int, ready: bool) -> Response:
    MODELS_LOADED.set(models_loaded)
    APP_READY.set(1 if ready else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
