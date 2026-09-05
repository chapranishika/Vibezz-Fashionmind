#!/usr/bin/env python3
"""
Push this repo to its Hugging Face Space (Docker SDK) — code + the trained
artifacts that are too big for GitHub, plus the Space secrets.

Why this exists: models/ and data/features/ are git-ignored (~1 GB), so a plain
GitHub -> Space sync produces a container that boots with zero models. This
script uploads the code and the artifacts together, via HF's native LFS.

Usage (PowerShell / cmd):

    set HF_TOKEN=hf_xxx                        # a *write* token for the Space owner
    python deploy/hf_space_sync.py deploy      # code -> restart -> wait healthy -> smoke  (the usual one)
    python deploy/hf_space_sync.py all         # + secrets + the ~650MB LFS artifacts
    python deploy/hf_space_sync.py code        # one stage: code | secrets | models | features | restart | verify

Env overrides:
    HF_SPACE_REPO   default "Nishika1202/vibezz-fashionmind-api"
    HF_MODELS_DIR   default "<repo>/models"        (may be a junction/symlink)
    HF_FEATURES_DIR default "<repo>/data/features"
    SITE_URL        default "https://<subdomain>.hf.space"  (for OPENROUTER_SITE_URL)

Secrets come from the repo's .env. `customers_clean.parquet` (~160 MB, not read
at runtime) is skipped.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

REPO_DIR = Path(__file__).resolve().parent.parent
SPACE = os.environ.get("HF_SPACE_REPO", "Nishika1202/vibezz-fashionmind-api")
MODELS_DIR = Path(os.environ.get("HF_MODELS_DIR", REPO_DIR / "models"))
FEATURES_DIR = Path(os.environ.get("HF_FEATURES_DIR", REPO_DIR / "data" / "features"))
# HF Space subdomains are "{owner}-{space}.hf.space" -- the whole slug, not just
# the space name (a "{space}.hf.space" bug cost 15 min elsewhere this project).
SITE_URL = os.environ.get("SITE_URL", f"https://{SPACE.replace('/', '-').lower()}.hf.space")

TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not TOKEN:
    sys.exit("set HF_TOKEN to a write token for the Space owner")

api = HfApi(token=TOKEN)
stage = sys.argv[1] if len(sys.argv) > 1 else "all"
_STAGES = {
    "deploy": {"code", "restart", "verify"},
    "all":    {"code", "secrets", "models", "features", "restart", "verify"},
}
WANT = _STAGES.get(stage, {stage})

# secret names to push from .env (plus a couple of computed ones)
SECRET_KEYS = ["OPENROUTER_API_KEY", "OPENROUTER_MODEL", "GEMINI_API_KEY",
               "SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY",
               "JWT_SECRET", "SERPAPI_KEY", "DEMO_EMAIL", "DEMO_PASSWORD",
               "DEMO_CUSTOMER_ID"]


def _retry(fn, tries=5, wait=8):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            print(f"    try {i}/{tries}: {type(e).__name__}: {str(e)[:150]}")
            if i == tries:
                raise
            time.sleep(wait)


def _read_env():
    env = {}
    p = REPO_DIR / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


if "code" in WANT:
    print(">> code + Docker config")
    _retry(lambda: api.upload_folder(
        folder_path=str(REPO_DIR), repo_id=SPACE, repo_type="space",
        allow_patterns=["api/**", "src/**", "db/**", "Dockerfile",
                        "requirements-api.txt", "README.md", ".gitattributes"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        commit_message="sync: backend code + Docker config"))

if "secrets" in WANT:
    print(">> secrets")
    env = _read_env()
    env.setdefault("OPENROUTER_SITE_URL", SITE_URL)
    env["ENVIRONMENT"] = "production"
    # a name cannot be both a variable and a secret — clear stale variables first
    try:
        for k in list(api.get_space_variables(SPACE)):
            api.delete_space_variable(repo_id=SPACE, key=k)
            print(f"   cleared stale variable {k}")
    except Exception as e:
        print(f"   (variable cleanup skipped: {type(e).__name__})")
    for k in SECRET_KEYS + ["OPENROUTER_SITE_URL", "ENVIRONMENT"]:
        v = env.get(k, "")
        if not v:
            print(f"   skip {k} (empty)")
            continue
        _retry(lambda k=k, v=v: api.add_space_secret(repo_id=SPACE, key=k, value=v))
        print(f"   set {k}")

if "models" in WANT:
    print(f">> models  <-  {MODELS_DIR}")
    _retry(lambda: api.upload_folder(
        folder_path=str(MODELS_DIR), path_in_repo="models",
        repo_id=SPACE, repo_type="space",
        commit_message="sync: model artifacts (LFS)"), tries=6, wait=12)

if "features" in WANT:
    print(f">> data/features  <-  {FEATURES_DIR}")
    _retry(lambda: api.upload_folder(
        folder_path=str(FEATURES_DIR), path_in_repo="data/features",
        repo_id=SPACE, repo_type="space",
        ignore_patterns=["customers_clean.parquet", "*.bak", "*.bak_*"],
        commit_message="sync: feature parquets (LFS)"), tries=6, wait=12)

if "restart" in WANT:
    print(">> restart")
    _retry(lambda: api.restart_space(repo_id=SPACE))

if "verify" in WANT:
    print(f">> waiting for {SITE_URL} to come back healthy")
    healthy = False
    for i in range(1, 41):                       # up to 10 min
        try:
            with urllib.request.urlopen(f"{SITE_URL}/health", timeout=15) as r:
                if json.loads(r.read()).get("status") == "ok":
                    print(f"   healthy after ~{i * 15}s")
                    healthy = True
                    break
        except Exception as e:
            if i == 1:
                print(f"   not up yet: {type(e).__name__} — will retry")
        time.sleep(15)
    if not healthy:
        sys.exit(f"Space did not report healthy at {SITE_URL}/health — check the Space logs")
    print(">> smoke test")
    env = {**os.environ, "PROD_API_URL": SITE_URL}
    rc = subprocess.call([sys.executable, str(REPO_DIR / "scripts" / "smoke_prod.py")], env=env)
    if rc != 0:
        sys.exit("smoke test failed against the freshly deployed Space")

print("done:", stage)
