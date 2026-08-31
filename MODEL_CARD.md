# Model card — Vibezz recommender

_Generated 2026-08-31. Machine-readable copy: `data/features/model_card.json`._

## Task
Given a customer's purchase history, rank the H&M catalogue and return the top-N,
each with the SHAP reasons behind it. Evaluated as **next-basket prediction**:
train up to a cutoff, predict the two weeks after.

## Data
* Source: H&M Personalised Fashion Recommendations (Kaggle 2022), 31.8M transactions.
* Training window: **2020-05-01 → 2020-09-08** (~4.5 months, 6.5M transactions).
* Holdout: **2020-09-09 → 2020-09-22**, strictly after the cutoff.
* Interaction matrix after filtering to users with ≥5 purchases and items with
  ≥3 purchases: **374,335 × 35,329**, density **0.0342%**.

## Models
| Stage | Model | Notes |
|---|---|---|
| Retrieval | ALS (`implicit`) | 64 factors, 150 iters, L2 0.01 |
| Reference | BPR | 120 iters, train AUC 0.97 — underperforms ALS here, not used downstream |
| Re-rank | LightGBM LambdaRank | 13 features, early-stopped (best iter 3), trained on 6,000 users × 100 candidates |
| Explain | SHAP TreeExplainer | top-3 directional reasons cached per user–item pair |
| Trend signal | LightGBM regressor | weekly sales per product type, lag + rolling + seasonality |
| Cold start | segmented popularity | age-bucket × club membership, global fallback |

## Held-out results (1,000 users, never seen in training)
| Model | Recall@10 | NDCG@10 | MAP@12 |
|---|---|---|---|
| Popularity | 0.0037 | 0.0022 | 0.0008 |
| ALS | 0.0078 | 0.0065 | 0.0032 |
| Full pipeline | **0.0123** | **0.0087** | **0.0041** |

* Paired hit@12: ALS 0.033 → pipeline **0.050** (+51%). McNemar exact
  **p = 0.014**; 95% bootstrap CI on the difference **[0.004, 0.030]**.
* **Candidate recall@100 = 0.036** — the ceiling. ALS only retrieves 3.6% of the
  held-out ground-truth into the pool; the re-ranker reaches 34% of that ceiling
  at rank 10.
* Re-ranker in-fold validation NDCG@10 = 0.88 — this is *not* comparable to the
  held-out retrieval numbers above and should not be quoted as the model's recall.
* Trend forecast: MAPE **8.9%**, MAE 166.

## SHAP global importance (re-ranker)
| Feature | mean \|SHAP\| |
|---|---|
| Trending this week | 0.042 |
| Product type | 0.023 |
| Colour group | 0.017 |
| High CF rank | 0.012 |
| Garment type | 0.010 |
| Matches price range | 0.007 |
| Popular item | 0.004 |
| Visual style match | 0.004 |

## Honest limitations
* **Absolute recall is low (~0.012).** Next-basket prediction on a sparse
  4.5-month matrix is hard; a longer window did *not* help ALS (recency beats
  volume for fast fashion), and this box's 4 GB free RAM caps how much data the
  training run can hold.
* The gains are **bounded by retrieval** (0.036 ceiling). The productive next
  step is a better candidate generator (two-tower / content-hybrid), not more
  re-ranker features.
* Images cover ~55k of 105k articles (only those in the downloaded zips).
* CLIP visual embeddings (`src/vision/clip_embeddings.py`) are a documented next
  step, not a claimed result — the shipped visual feature is 64-d metadata SVD.
* The Supabase RLS policies are `USING(true)` for `public`; tighten before any
  real deployment.
