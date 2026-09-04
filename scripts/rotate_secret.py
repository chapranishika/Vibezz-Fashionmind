#!/usr/bin/env python3
"""
Rotate one API key end-to-end: validate -> .env -> HF Space secret -> restart
-> wait healthy -> smoke test. One command instead of the five-step manual
dance (test, edit .env, push the Space secret, restart, re-verify) done by
hand for OPENROUTER_API_KEY, SERPAPI_KEY and SUPABASE_SERVICE_KEY in the same
session -- this is the tool that should have existed the first time.

Usage:
    set HF_TOKEN=hf_xxx
    python scripts/rotate_secret.py OPENROUTER_API_KEY sk-or-v1-...
    python scripts/rotate_secret.py SERPAPI_KEY 8bcc...
    python scripts/rotate_secret.py SUPABASE_SERVICE_KEY sb_secret_...
    python scripts/rotate_secret.py SOME_OTHER_KEY value --skip-validate

Refuses to touch .env or the Space if validation fails (unless the key has no
validator, or --skip-validate is passed). Prints exactly what it did and did
not do -- no silent partial rotation.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_DIR / ".env"
SPACE = os.environ.get("HF_SPACE_REPO", "Nishika1202/vibezz-fashionmind-api")
SPACE_URL = os.environ.get("PROD_API_URL", f"https://{SPACE.split('/')[-1].lower()}.hf.space")


def _post_json(url, body, headers=None, timeout=30):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def _get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode()


# ---- per-secret validators: prove the NEW value works before touching anything

def _validate_openrouter(value):
    st, d = _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {"model": "google/gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
        headers={"Authorization": f"Bearer {value}"})
    ok = st == 200 and "choices" in d
    return ok, ("live" if ok else f"HTTP {st}: {d}")


def _validate_serpapi(value):
    st, txt = _get(f"https://serpapi.com/search.json?engine=google_shopping&q=test&api_key={value}")
    d = json.loads(txt)
    ok = st == 200 and "error" not in d
    return ok, ("live" if ok else f"HTTP {st}: {d.get('error', txt[:120])}")


def _validate_supabase_service_key(value):
    url = os.environ.get("SUPABASE_URL", "https://cxfsiotzfshrjdrctmge.supabase.co")
    st, txt = _get(f"{url}/rest/v1/user_auth?select=id&limit=1",
                   headers={"apikey": value, "Authorization": f"Bearer {value}"})
    ok = st == 200
    return ok, ("bypasses RLS, reads fine" if ok else f"HTTP {st}: {txt[:150]}")


VALIDATORS = {
    "OPENROUTER_API_KEY": _validate_openrouter,
    "SERPAPI_KEY": _validate_serpapi,
    "SUPABASE_SERVICE_KEY": _validate_supabase_service_key,
}


def update_env(key, value):
    text = ENV_PATH.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    if not pattern.search(text):
        sys.exit(f".env has no {key}= line -- add it manually first, this tool only rotates existing keys")
    ENV_PATH.write_text(pattern.sub(f"{key}={value}", text), encoding="utf-8")


def update_space_secret(key, value):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("  (HF_TOKEN not set -- skipping Space secret update; set it and re-run,"
             " or update the secret by hand in the Space settings)")
        return False
    api = HfApi(token=token)
    for i in range(1, 5):
        try:
            api.add_space_secret(repo_id=SPACE, key=key, value=value)
            api.restart_space(repo_id=SPACE)
            return True
        except Exception as e:
            print(f"  attempt {i}/4: {type(e).__name__}")
            if i == 4:
                raise
            time.sleep(6)


def wait_healthy(tries=30, wait=15):
    # A Space restart isn't always a quick container bounce -- it can take a
    # full rebuild cycle. First cut of this tool used tries=12 (3 min) and
    # timed out on a rotation that was actually fine 30s later; 30x15s=7.5min
    # matches what's actually been observed for this Space this session.
    for i in range(1, tries + 1):
        try:
            st, txt = _get(f"{SPACE_URL}/health", timeout=20)
            d = json.loads(txt)
            if d.get("status") == "ok":
                print(f"  healthy: {d}")
                return True
        except Exception:
            pass
        time.sleep(wait)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key", help="env var name, e.g. OPENROUTER_API_KEY")
    ap.add_argument("value", help="the new secret value")
    ap.add_argument("--skip-validate", action="store_true", help="skip the live-test step")
    ap.add_argument("--no-space", action="store_true", help="only update .env, don't touch the HF Space")
    ap.add_argument("--full-smoke", action="store_true", help="run scripts/smoke_prod.py with SMOKE_FULL=1 at the end")
    args = ap.parse_args()

    print(f">> validating {args.key}")
    validator = VALIDATORS.get(args.key)
    if args.skip_validate or not validator:
        print("  (no live test for this key)" if not validator else "  (--skip-validate)")
    else:
        ok, detail = validator(args.value)
        print(f"  {'OK' if ok else 'FAILED'}: {detail}")
        if not ok:
            sys.exit("validation failed -- .env and the Space were NOT touched")

    print(f">> updating .env")
    update_env(args.key, args.value)

    if not args.no_space:
        print(f">> updating HF Space secret + restarting")
        if update_space_secret(args.key, args.value):
            print(">> waiting for the Space to come back healthy")
            if not wait_healthy():
                sys.exit("Space did not report healthy in time -- check the Space logs")

    print(">> running smoke test")
    os.environ["PROD_API_URL"] = SPACE_URL if not args.no_space else "http://127.0.0.1:8000"
    if args.full_smoke:
        os.environ["SMOKE_FULL"] = "1"
    rc = os.system(f'{sys.executable} "{REPO_DIR / "scripts" / "smoke_prod.py"}"')
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
