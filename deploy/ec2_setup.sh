#!/bin/bash
# FashionMind — EC2 Setup Script
# Ubuntu 22.04 LTS | t2.large (8GB RAM) recommended
# Usage: bash deploy/ec2_setup.sh
set -e

echo "=== FashionMind EC2 Setup ==="

# ── System deps ───────────────────────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv git nginx curl openjdk-17-jre-headless
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# ── Python env ────────────────────────────────────────────────────────────
python3 -m venv env && source env/bin/activate
pip install --upgrade pip

# CPU torch (saves 3GB vs CUDA)
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# ── .env check ────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  echo "ERROR: .env file not found. Copy .env.example → .env and fill in keys."
  exit 1
fi
source .env

# ── Train models (skip if already trained) ────────────────────────────────
if [ ! -f "models/als_model.npz" ]; then
  echo "=== Training models (one-time, ~20 min on t2.large) ==="
  python3 src/ingestion/etl.py
  python3 src/recsys/collaborative_filtering.py
  python3 src/vision/visual_features.py
  python3 src/trends/trend_forecasting.py
  python3 src/ranker/reranker.py
  python3 src/genai/rag_knowledge_base.py
else
  echo "=== Models already trained, skipping ==="
fi

# ── CLIP encoding (EC2 only) ──────────────────────────────────────────────
if [ ! -f "models/clip_embeddings.npy" ]; then
  echo "=== Running CLIP encoding ==="
  python3 src/vision/clip_embeddings.py
fi

# ── Start API ─────────────────────────────────────────────────────────────
EC2_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_IP")
pkill -f "uvicorn api.main" 2>/dev/null || true
nohup env/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2 > logs/api.log 2>&1 &
sleep 4
curl -sf http://localhost:8000/health && echo " API healthy ✓" || echo " API may still be starting"

# ── Configure nginx to serve frontend ────────────────────────────────────
# Patch API URL in frontend to use EC2 IP
cp frontend/index.html /tmp/fm_index.html
sed -i "s|http://localhost:8000|http://${EC2_IP}:8000|g" /tmp/fm_index.html
sudo mkdir -p /var/www/fashionmind
sudo cp /tmp/fm_index.html /var/www/fashionmind/index.html

sudo tee /etc/nginx/sites-available/fashionmind > /dev/null << NGINX
server {
    listen 80;
    server_name _;
    root /var/www/fashionmind;
    index index.html;

    # Serve frontend
    location / {
        try_files \$uri /index.html;
    }

    # Proxy API (avoids CORS in production)
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/fashionmind /etc/nginx/sites-enabled/fashionmind
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

echo ""
echo "============================================"
echo "  FashionMind Deployed!"
echo "  Frontend : http://${EC2_IP}"
echo "  API      : http://${EC2_IP}:8000"
echo "  Swagger  : http://${EC2_IP}:8000/docs"
echo "============================================"
