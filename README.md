---
title: Vibezz
emoji: 👗
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
---

# Vibezz — a personalised H&M recommender

> Ranks the H&M catalogue for a shopper and **tells them why** each pick was made.
> ALS candidate retrieval → LightGBM LambdaRank re-ranking over 13 signals → SHAP explanations.

* **Frontend**: [vibezz-fashionmind.vercel.app](https://vibezz-fashionmind.vercel.app)
* **API**: [nishika1202-vibezz-fashionmind-api.hf.space](https://nishika1202-vibezz-fashionmind-api.hf.space) · [/docs](https://nishika1202-vibezz-fashionmind-api.hf.space/docs)

---

## What it does

| | |
|---|---|
| **Picked for you** | Personalised ranked feed. ALS retrieves ~100 candidates, a LightGBM LambdaRank model re-orders them using 13 signals. |
| **Explain** | Every recommendation carries its top-3 SHAP reasons (`↑ Trending this week`, `↑ Favourite category`…). The Dashboard shows global SHAP importance. |
| **Trending product types** | A LightGBM weekly demand-forecast model (8.9% MAPE); its output is one of the 13 ranking signals. |
| **Dashboard** | Model performance read **live from the training outputs** — held-out metrics, the ALS-vs-re-ranker paired test, the retrieval ceiling, SHAP, the pipeline. No hand-entered numbers. |
| **Stylist** | An OpenRouter/Gemini chatbot with 4 tools (`get_recommendations`, `explain_recommendation`, `get_trend_report`, `get_outfit_suggestion`). Falls back to a demo mode with no key. |

An earlier trend-shopping aggregator (Pinterest → SerpApi → outfit builder) still
lives under `src/catalog/` and `/catalog/*` but is **not** part of the product;
only `/catalog/metrics` (the Dashboard feed) is wired into the UI.

---

## Results

All numbers come from the training scripts and are re-derivable from
`data/features/*.csv` + `model_card.json`. **Strict temporal holdout**: train on
2020-05-01 → 2020-09-08, predict the two weeks after. Held-out test users are
never seen during training.

**Held-out NDCG@10** (1,000 users), mean with 95% bootstrap CI
(1,000 resamples over users — `python scripts/eval_slices.py`):

| Model | NDCG@10 | 95% CI |
|---|---|---|
| Popularity baseline | 0.0022 | [0.0011, 0.0034] |
| ALS retrieval | 0.0065 | [0.0042, 0.0093] |
| **Full pipeline (re-ranked)** | **0.0087** | **[0.0059, 0.0115]** |

* **The pipeline's and ALS's marginal CIs overlap** (0.0059–0.0115 vs
  0.0042–0.0093). The re-ranker's edge is real *paired* but not resolvable from
  1,000 users on the unpaired margins — the honest read is "the re-ranker helps
  on the users it helps, and the aggregate lift needs more users or an online
  test to call with confidence."
* **Paired hit@12** on the same 1,000 users: ALS 0.033 → re-ranked **0.050**,
  McNemar exact **p = 0.014**, 95% bootstrap CI **[0.004, 0.030]** — the paired
  test *does* clear zero because it cancels per-user variance the margins carry.
* **Where the re-ranker helps vs hurts** (NDCG@10, `data/features/eval_slices.csv`):

  | slice | ALS | re-ranked | |
  |---|---|---|---|
  | age = mature | 0.0073 | **0.0144** | re-ranker ~2× |
  | age = mid | **0.0064** | 0.0046 | re-ranker *worse* |
  | price tier = high | 0.0053 | **0.0092** | |
  | price tier = low | **0.0106** | 0.0088 | ALS wins |

  It's not a uniform lift — the re-ranker trades mid-age / budget-shopper
  performance for gains on mature / higher-spend users. A product decision, not
  just a metric.
* The held-out set is drawn from users with ≥1 future purchase, so every test
  user already has history — **cold-start is a code path with no offline
  coverage**. That's a gap, not a result.
* **Retrieval ceiling — candidate recall@100 = 0.036.** ALS only puts 3.6% of
  held-out ground-truth into the candidate pool, so 0.036 is the hard cap on
  recall@10. The pipeline reaches 0.012, i.e. **~34% of what is retrievable**.
  Absolute numbers are low because next-basket prediction on a ~4.5-month,
  0.034%-dense matrix is genuinely sparse — the honest lever left is better
  *retrieval*, not a better re-ranker.
* BPR is a reference model (train AUC 0.97) but underperforms ALS on this task
  and is not carried into the pipeline.
* Trend forecast: **MAPE 8.9%**, MAE 166, 106 weeks.

Interaction matrix after ≥5-purchase users and ≥3-purchase items:
**374,335 × 35,329**, density **0.0342%**, from a 6.5M-transaction window.

---

## Pipeline

```
transactions_train.csv (31.8M rows, 2018-09 → 2020-09)
        │  chunked pandas ETL — filter to the training window, stream in 1M-row chunks
        ▼
[2] ALS collaborative filtering            implicit, 64 factors, 150 iters
        │  + BPR (reference) + segmented cold-start (age-bucket × club)
        ▼
[3] Visual features                        one-hot → SVD 64-d → FAISS IVFFlat
[4] NLP + demand forecast                  TF-IDF → LSA · LightGBM weekly sales (8.9% MAPE)
        ▼
[5] LightGBM LambdaRank re-ranker          13 signals: als_score · visual_sim · nlp_sim ·
        │                                  trend_score · price_affinity · category_match ·
        │                                  popularity · CF rank · engagement · age · type/colour/garment
        │  SHAP TreeExplainer → cached per-recommendation reasons
        │  candidate-recall diagnostic + McNemar paired comparison
        ▼
[6] TF-IDF / SVD RAG                        ~190 style-knowledge chunks for the stylist
        ▼
FastAPI · Supabase (auth + cart) · OpenRouter (stylist) · vanilla-JS SPA
```

Run it (needs the H&M Kaggle CSVs in `data/raw/`):

```bash
python -m src.ingestion.etl                 # [1] ETL
python src/recsys/collaborative_filtering.py # [2] ALS + BPR + cold-start
python src/vision/visual_features.py         # [3] visual FAISS
python src/trends/trend_forecasting.py       # [4] NLP + demand forecast
python src/ranker/reranker.py                # [5] LambdaRank + SHAP + eval + model_card.json
python src/genai/rag_knowledge_base.py       # [6] RAG
uvicorn api.main:app --reload                # serve
```

---

## API

| Method | Endpoint | |
|---|---|---|
| GET | `/health` | liveness — process is serving (says nothing about models/DB) |
| GET | `/ready` | readiness — 503 unless models loaded AND DB reachable; Dockerfile `HEALTHCHECK` points here |
| GET | `/health/db` | RLS / trigger / cron watchdog (503 if drifted); see SECURITY.md |
| GET | `/metrics` | Prometheus exposition (latency histogram, req/rec/tool-call/token counters); dashboard: `deploy/grafana_dashboard.json` |
| POST | `/recommend` | `{customer_id, n}` → ranked items + SHAP reasons |
| POST | `/explain` | `{customer_id, article_id}` → the reasons for one pick |
| GET | `/trends` | trending product types (demand forecast) |
| GET | `/products` | catalogue browse (`category`, `colour`, `q`, `page`) |
| GET | `/catalog/metrics` | real model metrics for the Dashboard |
| POST | `/chat` · `/chat/stream` | stylist (OpenRouter → Gemini → demo) |
| POST | `/auth/signup` · `/auth/login` | bcrypt + JWT, Supabase-backed |
| — | `/cart`, `/products/{id}/reviews` | cart + reviews (Supabase) |

Real H&M product photos for the ~55k articles present in the raw zips are served
at `/images/<prefix>/<id>.jpg` (`python scripts/extract_hm_images.py`, ~1 GB).
Cards fall back to a colour-derived tile when a photo is missing — never a stock
mismatch.

---

## Config

Copy `.env.example` → `.env`. Everything has a safe default:

* `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` — stylist LLM (free models exist). Without it, chat runs in demo mode.
* `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` — auth + cart. The demo login works without a DB.
* `JWT_SECRET` — `openssl rand -hex 32`.

---

## Tests

`pytest -q` — offline suite (ranking metrics, cold-start logic, API contract,
auth + cart, `/ready` + `/health/db` + `/metrics`, data contracts, eval-slice
stats). Tiny fixtures, no dataset needed.

`python scripts/smoke_prod.py` — end-to-end check against a running deployment
(`PROD_API_URL`, defaults to the live Space): `/ready`, security posture,
`/recommend` priced in ₹, chat live and returning the right garment category.

`python scripts/validate_features.py` — data contracts on the feature parquets
(`src/data/contracts.py`): uniqueness, ranges, `is_*`/`*_idx` domains,
referential integrity (price/popularity → catalogue), cross-table vocabulary.
**Run after the ETL and before deploying a retrained model.** Exit ≠ 0 on any
`error`-severity violation. Currently: 0 errors, 1 warn (a known age-bucket
vocabulary split between two tables — documented, non-fatal).

CI (`.github/workflows/ci.yml`) runs the offline suite on every push, the smoke
test on `main`, and cheap smoke daily. `uptime.yml` pings `/ready` every ~10 min
and opens an issue on failure (see [OPS.md](OPS.md)).

---

## Deploy

Frontend → **Vercel** (static, `vercel.json` rewrites to `/frontend`).
Backend → **Hugging Face Space** (Docker SDK, `Dockerfile` + `requirements-api.txt`,
port 7860). The API image deliberately omits the pipeline's torch/transformers
stack — nothing under `api.main` imports it (~13 GB → ~2 GB).

The trained artifacts (`models/`, `data/features/`, ~1 GB) are **git-ignored** —
too big for GitHub and they'd bloat every clone. They ship straight to the Space
via LFS:

```
set HF_TOKEN=hf_xxx            # write token for the Space owner
python deploy/hf_space_sync.py   # code + secrets (from .env) + models + features + restart
```

A GitHub-only sync would boot a container with zero models loaded — always run
the script, or check `/health` shows `models_loaded > 0` after any deploy.

---

## Dataset

H&M Personalised Fashion Recommendations (Kaggle 2022) — 31,788,324 transactions
(2018-09-20 → 2020-09-22), 1,371,980 customers, 105,542 articles. The recommender
trains on a recent ~4.5-month slice; the trend model uses the full history.
