# Serving image for the FastAPI API (CPU-only).
# Uses requirements-api.txt — the offline pipeline's torch/transformers/plotting
# stack is intentionally excluded (image ~13 GB -> ~2 GB).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc g++ libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

# Hugging Face Spaces inject $PORT (7860); keep a fallback for plain `docker run`.
ENV PORT=7860 \
    ENVIRONMENT=production \
    HF_HOME=/tmp/hf
EXPOSE 7860

# Readiness, not liveness: /ready is 503 until models loaded AND DB reachable,
# so an orchestrator won't route to a container that booted broken.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-7860}/ready" || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
