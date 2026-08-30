#!/bin/bash
# FashionMind — Clean GitHub Push (semantic commits per phase)
set -e
git init
git remote add origin https://github.com/chapranishika/fashionmind.git

git add .gitignore .env.example requirements.txt requirements-dev.txt Dockerfile docker-compose.yml README.md pytest.ini
git commit -m "chore: project scaffold — gitignore, docker, requirements"

git add src/ingestion/ scripts/
git commit -m "feat(etl): chunked pandas ETL — 105k articles, 1.37M customers, 4 Parquet feature files"

git add src/recsys/
git commit -m "feat(recsys): ALS + BPR collaborative filtering — 310k x 42k sparse matrix, segmented cold-start, ALS candidates"

git add src/vision/
git commit -m "feat(vision): visual FAISS index (SVD 64-dim) + CLIP script for EC2 + outfit pairs"

git add src/trends/
git commit -m "feat(trends): TF-IDF to LSA NLP embeddings + LightGBM weekly demand forecast + Google Trends fusion"

git add src/ranker/
git commit -m "feat(ranker): LightGBM LambdaRank — 13 signals, cached SHAP explanations, paired held-out comparison"

git add src/genai/
git commit -m "feat(genai): Gemini 2.0 stylist — ~180-chunk RAG + 4 ML tools + SSE streaming + tool-calling"

git add api/ frontend/ deploy/ tests/
git commit -m "feat(api): FastAPI (25 endpoints) + vanilla-JS SPA + Docker + EC2 deploy script + pytest suite"

git add README.md
git commit -m "docs: README — metrics table, architecture, Myntra stack mapping"

git push -u origin main
echo "Pushed to GitHub"
