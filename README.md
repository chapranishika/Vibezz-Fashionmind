---
title: Vibezz Fashionmind
emoji: 👗
colorFrom: pink
colorTo: indigo
sdk: docker
app_port: 7860
---

# FashionMind 🛍️

> Multi-stage personalised fashion recommender with a Gemini-powered stylist chatbot.  
> Built on the H&M Kaggle corpus — **31.8M transactions over 2 years** · **105,542 articles** · **1,371,980 customers**.
> The CF and trend models train on a recent multi-week slice of that corpus (see Dataset).

### 🚀 Live Production Links
* **Frontend Website (Vercel)**: [https://vibezz-fashionmind.vercel.app](https://vibezz-fashionmind.vercel.app)
* **FastAPI Server (Hugging Face)**: [https://nishika1202-vibezz-fashionmind-api.hf.space](https://nishika1202-vibezz-fashionmind-api.hf.space)
* **Interactive API Swagger Docs**: [https://nishika1202-vibezz-fashionmind-api.hf.space/docs](https://nishika1202-vibezz-fashionmind-api.hf.space/docs)
* **Health Status**: [https://nishika1202-vibezz-fashionmind-api.hf.space/health](https://nishika1202-vibezz-fashionmind-api.hf.space/health)

---

## ☁️ Production Cloud Architecture

The application is deployed on a **high-availability, hybrid cloud architecture** designed for production performance, scalability, and cost optimization:

| Tier | Cloud Service | Host/Role | Cost | Status |
|---|---|---|---|---|
| **Frontend UI** | **Vercel** | Free static CDN hosting | `$0.00` | **Active** |
| **Backend API** | **Hugging Face Spaces** | Docker container instance (2 vCPUs, 16GB RAM) | `$0.00` | **Active** |
| **Database** | **Supabase** | Cloud Postgres database (user profiles & RAG history) | `$0.00` | **Active** |
| **GPU Runner** | **AWS EC2** | `g4dn.xlarge` instance (NVIDIA T4 GPU) for CLIP visual embeddings | On-Demand | **Stopped** *(to prevent billing)* |

### 🛠️ Key Architectural Decisions:
1. **Zero-Cost Production Hosting**: By leveraging Vercel for static distribution and Hugging Face Spaces for Docker container execution, the system gets access to 16GB of container memory (essential for loading large FAISS indices and collaborative filtering sparse matrices) completely for free.
2. **On-Demand GPU Pipelines (AWS EC2)**: Heavy deep learning workloads, such as extracting 512-dimensional CLIP visual embeddings from H&M product images, are routed to an AWS EC2 `g4dn.xlarge` GPU instance on-demand. Once visual feature extraction is completed and embeddings are exported, the instance is safely **Stopped** to guarantee zero idle billing charges.
3. **Database & RAG Storage**: Supabase serves as a real-time serverless database to persist customer interaction history, styling preferences, and chatbot conversation logs.

---


![Python](https://img.shields.io/badge/python-3.11-blue?style=flat-square)
![pandas](https://img.shields.io/badge/pandas-2.x-orange?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-brightgreen?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-1.7-red?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square)

---

## Results

All numbers below are printed by the training scripts and re-derivable from
`data/features/*.csv`. Training window: 2020-06-01 → 2020-09-08; evaluation is a
**strict temporal holdout** (the two weeks after).

**Phase 2 — candidate generation** (3,000 users with ≥5 purchases, next-2-weeks holdout):

| Model | Recall@10 | NDCG@10 | MAP@12 |
|---|---|---|---|
| Popularity baseline | 0.0066 | 0.0060 | 0.0033 |
| **ALS** | **0.0129** | **0.0100** | **0.0052** |
| BPR | 0.0070 | 0.0049 | 0.0022 |

ALS ≈ **2× the popularity baseline** on Recall@10. BPR did not converge on this
window (train AUC ≈ 0.5) and is kept only as a reference — ALS is what feeds the
re-ranker.

**Phase 5 — LambdaRank re-ranker** (500 held-out test users, never seen in training):

| Model | Recall@10 | NDCG@10 | MAP@12 |
|---|---|---|---|
| Popularity baseline | 0.0003 | 0.0003 | 0.0003 |
| ALS retrieval | 0.0106 | 0.0087 | 0.0044 |
| **Full pipeline (rerank ALS top-50)** | **0.0146** | **0.0109** | **0.0064** |

The re-ranker lifts NDCG@10 **+25%** and MAP@12 **+45%** over ALS by re-ordering
the retrieved set. **But** on a paired per-user hit@12 test it is **not**
distinguishable from ALS (McNemar exact p = 1.0; 95% bootstrap CI on the
difference [−0.022, +0.018]) — reranking 50 candidates cannot recover a relevant
item ALS never retrieved, so *coverage* is bounded by retrieval and only
*ordering* improves. Absolute values are low because a 14-week window is sparse
and recent-popularity nearly vanishes on held-out users.

**Trend forecaster:** MAE 166, **MAPE 8.9%** on weekly sales per product type
(LightGBM, lag + seasonality features).  
**SHAP (global):** the re-ranker leans hardest on *trend score* (0.10), then
*CF rank* (0.03), *price affinity*, *popularity*, and *visual similarity*.

> This is an offline study, not a live A/B test — there is no production traffic.
> The earlier "+83% lift" figure was an artifact of scoring the re-ranker on its
> own training users and has been removed.

---

## Architecture

```
Raw data (H&M Kaggle — 31.8M transactions, 2018-09 to 2020-09)
  │
  ▼
Phase 1 ── pandas ETL  (chunked CSV; a Spark path is not used)
           · Article features: type, colour, garment, appearance → category indices
           · Customer features: age buckets, engagement score, club membership
           · 4 Parquet feature files output
  │
  ├──────────────────────────────────────────────┐
  ▼                                              ▼
Phase 2 ── Collaborative Filtering         Phase 3 ── Visual Features
  ALS (implicit library, 64 factors)         SVD 64-dim metadata embeddings
  310k users × 42k items sparse matrix       FAISS IVFFlat index (<5ms retrieval)
  (≥5 purchases, 14-week window,             Outfit compatibility pairs
   0.024% density)                           → Phase 3B on EC2: CLIP 512-dim
  ALS top-N candidate retrieval
  Cold-start: age-bucket × club popularity
  │                                          │
  ├──────────────────────────────────────────┘
  │
  ▼
Phase 4 ── NLP + Trend Forecasting
  TF-IDF (15k vocab, bigrams) → LSA 64-dim text embeddings
  LightGBM time-series: lag features, seasonality, 8.8% MAPE
  Google Trends API fusion (runs on AWS EC2)
  Trend scores per product_type per week
  │
  ▼
Phase 5 ── LightGBM LambdaRank Re-ranker
  13 signals fused: CF score · visual sim · NLP sim · trend score
                    price affinity · category match · popularity ···
  SHAP TreeExplainer: per-recommendation reasons cached from shap_explanations.parquet
  A/B test framework: t-test + 95% CI (results pending full eval run)
  │
  ▼
Phase 6 ── Gemini 2.0 Stylist Chatbot
  RAG: 37 fashion knowledge chunks (TF-IDF + SVD retrieval)
  Tool calling: 4 ML pipeline tools (recommendations, trends, outfit, explain)
  SSE streaming: word-by-word token delivery
  Multi-turn conversation history
  │
  ▼
Phase 7 ── FastAPI · Docker · single-file vanilla-JS SPA · Vercel + HF Spaces
```

---

## How This Maps to Myntra's Stack

| FashionMind | Myntra equivalent |
|---|---|
| ALS candidate generation | Recsys retrieval layer |
| LightGBM LambdaRank re-ranker | Personalised ranking model |
| FAISS IVFFlat ANN search | Real-time candidate retrieval |
| CLIP visual embeddings (EC2) | "Shop the Look" / visual search |
| LightGBM trend forecasting | Demand planning / SCM team |
| SHAP explanations (cached) | Model interpretability for business |
| Gemini tool-calling chatbot | Conversational AI features |
| A/B test + hypothesis testing | Experimentation platform |
| Chunked pandas ETL | Data engineering pipeline |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | pandas (chunked CSV), NumPy, Parquet |
| Collaborative filtering | ALS (`implicit`), FAISS IVFFlat |
| Re-ranking | LightGBM LambdaRank, 13 signals |
| Visual search | SVD metadata (local) · CLIP ViT-B/32 (EC2) |
| NLP | TF-IDF → LSA 64-dim embeddings |
| Trend forecasting | LightGBM, lag features, Google Trends API |
| Explainability | SHAP TreeExplainer (pre-computed, cached) |
| GenAI | Gemini 2.0 Flash, RAG, SSE streaming, tool-calling |
| API | FastAPI, Python 3.11, SlowAPI rate limiting |
| Frontend | HTML/JS (Myntra-style pink editorial theme) |
| Infrastructure | Docker, docker-compose, AWS EC2 |

---

## Dataset

| Metric | Value |
|---|---|
| Source | H&M Personalised Fashion Recommendations (Kaggle 2022) |
| Transactions | 31,788,324 purchase records |
| Customers | 1,371,980 (age 16–99, mean 36.4) |
| Articles | 105,542 across 131 product types, 50 colour groups |
| Date range | 2018-09-20 → 2020-09-22 (full corpus) |

> Phase 1 processes the full articles/customers tables. Phases 2 and 4 stream
> `transactions_train.csv` in chunks but **train on a recent multi-week window**
> — CF/BPR on **2020-06-01 → 2020-09-08** (≈14 weeks) with a strict temporal
> holdout of **2020-09-09 → 2020-09-22**; the interaction matrix is built only
> from users with ≥5 purchases in the train window. The trend forecaster uses
> the full history up to the holdout. Numbers above are corpus totals, not the
> trained slice.

---

## Quick Start

```bash
git clone https://github.com/chapranishika/fashionmind
cd fashionmind

# 1. Add Kaggle data to data/raw/
#    transactions_train.csv, articles.csv, customers.csv

# 2. Environment
cp .env.example .env && nano .env   # add GEMINI_API_KEY

# 3. Run pipeline phases in order
python3 src/ingestion/etl.py                    # Phase 1: pandas ETL
python3 src/recsys/collaborative_filtering.py   # Phase 2: ALS
python3 src/vision/visual_features.py           # Phase 3A: Visual FAISS
python3 src/trends/trend_forecasting.py         # Phase 4: NLP + trends
python3 src/ranker/reranker.py                  # Phase 5: LambdaRank + SHAP
python3 src/genai/rag_knowledge_base.py         # Phase 6A: RAG
python3 src/genai/stylist_chatbot.py            # Phase 6B: Test tools

# 4. Start API
uvicorn api.main:app --reload
# → http://localhost:8000/docs

# 5. Open UI
open frontend/index.html

# 6. Full stack (Docker)
docker-compose up -d
# API: http://localhost:8000  |  UI: http://localhost:3000
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/recommend` | Personalised recs + SHAP reasons |
| GET | `/trends` | Trending product types (LightGBM) |
| POST | `/outfit` | Outfit compatibility pairing |
| POST | `/visual-search` | FAISS visual similarity |
| POST | `/text-search` | NLP text similarity |
| POST | `/explain` | SHAP explanation for an item (cached) |
| POST | `/chat` | Gemini stylist (single-turn) |
| POST | `/chat/stream` | SSE streaming chat |
| GET | `/catalog/trends` | Rising Pinterest fashion searches (India) + catalogue mapping |
| GET | `/catalog/shop?trend=` | Products for a trend keyword, across curated retailers + live source |
| GET | `/catalog/outfit?trend=` | A full assembled look (top/dress + bottom + shoes + accessory) |
| GET | `/catalog/products` | Filtered aggregator search (`q`, `product_type`, `tag`, `colour`, `source`) |
| GET | `/catalog/product/{id}` | One aggregator product |
| GET | `/cart/compare` | Cross-site compare view of `mode=compare` cart lines |

---

## Trend-driven aggregator (Pinterest → products)

Alongside the trained recommender, FashionMind runs a **trend aggregator**: it
pulls rising fashion searches from **Pinterest**, maps each keyword to catalogue
filters, and returns buyable products that **link out to the retailer's own
site** (Myntra, Ajio, Nykaa Fashion, H&M India, Zara, Urbanic, Snitch, FabIndia).
The cart has two modes — `demo` (trained H&M catalogue, fake checkout) and
`compare` (aggregator items, price-compare list, buy on the retailer).

| Layer | File | Falls back to |
|---|---|---|
| Pinterest signal | `src/trends/pinterest_trends.py` | official API → unofficial endpoint → shipped cache (`data/features/pinterest_trends_cache.json`) |
| Keyword → catalogue | `src/trends/trend_map.py` | bare tokens as tags |
| Product sources | `src/catalog/sources.py` | curated seed (`data/catalog/trend_products.json`, ~50 IN items) + trained H&M catalogue; **SerpApi Google Shopping** adapter activates when `SERPAPI_KEY` is set |
| Stylist tools | `get_pinterest_trends`, `shop_the_trend`, `build_trend_outfit` in `src/genai/stylist_chatbot.py` | — |
| UI | `frontend/index.html` → **Trending** tab | — |

Everything works with **zero configuration** (cache + curated). Set
`PINTEREST_ACCESS_TOKEN` and/or `SERPAPI_KEY` in `.env` for live data — see
`.env.example`. The curated seed's `buy_url`s are retailer **search** URLs and its
prices are category estimates (`price_is_estimate: true`); a live product API
replaces both.

---

## AWS EC2 Deployment (Phase 3B + Google Trends)

```bash
# Launch t2.large Ubuntu 22.04 | open ports 22, 80, 8000
scp -i key.pem -r models/ ubuntu@EC2_IP:~/fashionmind/
scp -i key.pem -r data/   ubuntu@EC2_IP:~/fashionmind/

# On EC2:
bash deploy/ec2_setup.sh
# → Installs torch, runs CLIP encoding, fetches Google Trends, deploys API
```

---

## Project Structure

```
fashionmind/
├── src/
│   ├── ingestion/    etl.py                    Phase 1
│   ├── recsys/       collaborative_filtering.py Phase 2
│   ├── vision/       visual_features.py         Phase 3A
│   │                 clip_embeddings.py          Phase 3B (EC2)
│   ├── trends/       trend_forecasting.py        Phase 4
│   ├── ranker/       reranker.py                 Phase 5
│   └── genai/        rag_knowledge_base.py       Phase 6A
│                     stylist_chatbot.py          Phase 6B
├── api/
│   └── main.py       FastAPI (9 endpoints, SlowAPI rate limiting)
├── frontend/
│   └── index.html    Myntra-style UI
├── deploy/
│   ├── ec2_setup.sh  One-command EC2 deploy
│   └── git_push.sh   Semantic GitHub push
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---


