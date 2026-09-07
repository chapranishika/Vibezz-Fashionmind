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

> **Point-in-time leak — found, fully fixed, retrained (2026-09-06 → 09-07).**
> Two leaks in `trend_score`: (a) it was read from `trend_scores`' latest week
> (2020-09-21, *inside* the holdout), and (b) the trend *forecaster* was itself
> fit through the holdout (`FORECAST_END = 2020-09-22`). Both fixed —
> `reranker.py` caps the trend week at the 2020-09-08 split, and
> `trend_forecasting.py` now trains with `FORECAST_END = split` (MAPE 8.9% →
> 12.3%, the honest cost). `scripts/check_leakage.py` guards it in CI; the
> re-ranker was retrained on the clean feature. An intermediate write-up here
> said "leak-free the re-ranker doesn't beat ALS" — that was a train/serve-skew
> artifact (scoring the old leak-trained model with the capped feature), not a
> real measurement. Everything below is the fully leak-free re-measurement.

**Held-out (1,000 users), fully leak-free** (`data/features/final_metrics.csv`;
NDCG@10 also with a 1,000-sample bootstrap CI from `scripts/eval_slices.py`):

| Model | Recall@10 | NDCG@10 | 95% CI | MAP@12 |
|---|---|---|---|---|
| Popularity | 0.0037 | 0.0022 | [0.0011, 0.0034] | 0.0008 |
| ALS retrieval | 0.0078 | 0.0065 | [0.0042, 0.0093] | 0.0032 |
| **Full pipeline (re-ranked)** | **0.0131** | **0.0100** | **[0.0070, 0.0131]** | **0.0049** |

* **The re-ranker beats ALS, cleanly and significantly** — paired hit@12 on the
  same 1,000 users: ALS 0.033 → pipeline **0.057, +72.7%**, McNemar exact
  **p = 0.0004**, 95% CI on the difference **[0.011, 0.037]** (pipeline wins 34,
  ALS wins 10, 956 ties). The paired test cancels the per-user variance the
  unpaired margins carry. Removing the leak did *not* weaken this — it
  strengthened it (was +57.6%, p = 0.0066 with the leak).
* **Feature ablation** on the fully-clean model (`scripts/ablate_features.py`,
  zero each feature → ΔNDCG@10, 95% bootstrap CI, `data/features/feature_ablation.csv`):

  | carries it (CI excludes 0) | ~marginal (CI touches 0) | dead (Δ ≈ 0) |
  |---|---|---|
  | `ptype_idx` −34% · `nlp_sim` −29% · `popularity_score` −25% | `visual_sim` −15% · `rank_norm` −14% | `als_score` · **`trend_score`** · `price_affinity` · `colour_idx` · `garment_idx` · `category_match` · `age_norm` · `engagement_score` |

  Only 3 of 13 features carry the model. **`trend_score` drops from "−35%,
  matters" to "−7%, CI includes 0" once the forecaster leak is gone** — its
  earlier prominence *was* the leak. `als_score` also contributes ~nothing: the
  re-ranker barely uses the retrieval score it re-ranks. The real content
  signals are product-type, text similarity, and popularity.
* **Retrieval ceiling — and how to raise it** (`scripts/eval_retrieval.py`,
  candidate recall@100 = fraction of held-out ground-truth in the 100-candidate
  pool; repeat purchases filtered from every source so it matches ALS's
  `filter_already_liked_items`):

  | retrieval source | recall@100 | 95% CI | |
  |---|---|---|---|
  | ALS (current) | 0.036 | [0.029, 0.044] | co-purchase |
  | content two-tower | 0.009 | [0.006, 0.013] | **worse than ALS** — content similarity to history barely helps once restocks are removed |
  | **GRU4Rec (sequence)** | **0.076** | **[0.066, 0.087]** | **2.1× ALS**, CIs disjoint — recency/order is the signal that's missing |
  | round-robin union | 0.049 | [0.040, 0.058] | 1:1:1 interleave, dragged by the dead two-tower; a GRU-weighted union → ~0.076+ |

  The re-ranker improves *ordering* within the ALS pool (+72.7% paired hit@12
  above), but it can't recommend an item retrieval never surfaced — and ALS
  surfaces only 3.6% of what users buy. A GRU sequence model **doubles** that
  ceiling (0.076). Re-ranking is worth ~1.7× on hit@12; retrieval is worth 2×
  on what's reachable at all — so the highest-value change is **swap / augment
  ALS retrieval with GRU4Rec** (wired behind `RECS_USE_GRU`, pending a reranker
  re-fit on the mixed candidates — `docs/GRU_RERANK_REFIT.md`), and drop the
  content two-tower.
  `src/recsys/two_tower.py`, `src/recsys/sequence.py`;
  `scripts/train_{two_tower,sequence}.py`.
* The held-out set is drawn from users with ≥1 future purchase, so every test
  user already has history — **cold-start is a code path with no offline
  coverage**. That's a gap, not a result.
* **Off-policy evaluation** (`src/eval/offpolicy.py`, `scripts/offpolicy_eval.py`):
  IPS / SNIPS / doubly-robust estimators to answer "what would a ranker change
  do to CTR?" from logged data. The serving policy is deterministic, so its
  propensities are degenerate — `/recommend` grows an ε-greedy slate shuffle
  (`RECS_EXPLORE_EPS`, default 0 = off) that logs `p_logged` per impression to
  make the log usable. Estimators are validated against a synthetic bandit with
  a known target value (IPS unbiased, SNIPS lower-variance, DR robust to a
  wrong reward model *or* wrong propensities); real-data OPE waits on collected
  exploration traffic.
* BPR is a reference model (train AUC 0.97) but underperforms ALS and is not
  carried into the pipeline.
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
