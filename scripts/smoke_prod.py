#!/usr/bin/env python3
"""
End-to-end smoke test against a running API (local or production).

    python scripts/smoke_prod.py              # cheap checks only: no DB writes,
                                               # no LLM spend. Safe to run anytime.
    SMOKE_FULL=1 python scripts/smoke_prod.py  # + /recommend and /chat: writes a
                                               # row to `recommendations` /
                                               # `chat_messages` and spends a real
                                               # OpenRouter call. Reserve this for
                                               # an actual deploy, not a daily cron.

    PROD_API_URL=http://127.0.0.1:8000 python scripts/smoke_prod.py

Exits non-zero if any hard check fails. Soft checks (marked WARN) don't fail the
run — they cover things that legitimately degrade, like the SerpApi free-tier
quota being exhausted.

Any row this writes uses SMOKE_CUSTOMER_ID (default "smoke-test-ci"), never
"guest" — that id is real shared traffic and its rows shouldn't be mixed with
synthetic test data. A daily pg_cron job on the Supabase side purges these rows
automatically (db/migrations/003, job "purge-smoke-test-ci") — this isn't a
"remember to clean up" note, it actually runs without anyone remembering.

This is the check that would have caught every user-facing bug we shipped:
the Stylist being unreachable, prices rendered as "£0.03", "dresses" returning
trousers, and the API running with zero models loaded.
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("PROD_API_URL", "https://nishika1202-vibezz-fashionmind-api.hf.space").rstrip("/")
CUSTOMER_ID = os.environ.get("SMOKE_CUSTOMER_ID", "smoke-test-ci")
FULL = os.environ.get("SMOKE_FULL", "") not in ("", "0", "false", "False")
TIMEOUT = 60

_fail = 0
_warn = 0


def _req(path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status, r.read().decode()


def check(name, fn, soft=False):
    global _fail, _warn
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    tag = "WARN" if (not ok and soft) else ("PASS" if ok else "FAIL")
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        if soft:
            _warn += 1
        else:
            _fail += 1


# ---- cheap checks: read-only, no DB writes, no LLM spend --------------------

def t_health():
    st, txt = _req("/health")
    d = json.loads(txt)
    n = d.get("models_loaded", 0)
    return (st == 200 and d.get("status") == "ok" and n > 0), f"{n} models loaded"


def t_ready():
    """/ready = models loaded AND DB reachable (503 otherwise)."""
    try:
        st, txt = _req("/ready")
    except urllib.error.HTTPError as e:
        st, txt = e.code, e.read().decode()
    d = json.loads(txt)
    if d.get("ready"):
        return True, f"{d.get('models')} models, db_ok"
    return False, f"not ready: {json.dumps({k: v for k, v in d.items() if 'error' in k or not v})}"


def t_products():
    st, txt = _req("/products?page_size=5")
    return bool(json.loads(txt).get("products")), "catalogue reachable"


def t_photos():
    st, txt = _req("/catalog/photos?q=black+dress+women&n=3")
    return bool(json.loads(txt).get("photos")), "image search returning results"


def t_db_posture():
    """The watchdog-on-the-watchdogs: /health/db reports whether the RLS
    lockdown, the auto-revoke event trigger, and the pg_cron jobs
    (db/migrations/002-004) are all still there. 503 if any are gone."""
    try:
        st, txt = _req("/health/db")
    except urllib.error.HTTPError as e:
        st, txt = e.code, e.read().decode()
    d = json.loads(txt)
    if d.get("ok"):
        return True, "RLS locked, event trigger + 3 cron jobs present"
    return False, f"posture broken: {json.dumps({k: v for k, v in d.items() if k != 'checked_at'})}"


# ---- full checks: write a row to Supabase + spend an OpenRouter call --------

def t_recommend():
    st, txt = _req("/recommend", "POST", {"customer_id": CUSTOMER_ID, "n": 5})
    recs = json.loads(txt).get("recommendations", [])
    if not recs:
        return False, "empty recommendations"
    missing = [r.get("product_name") for r in recs if not r.get("price_inr")]
    if missing:
        return False, f"{len(missing)} recs missing price_inr"
    return True, f"{len(recs)} recs, all priced in INR"


def t_chat_currency():
    st, txt = _req("/chat", "POST", {"message": "2 work tops please",
                                     "customer_id": CUSTOMER_ID, "history": []})
    resp = json.loads(txt).get("response", "")
    if "Demo mode" in resp:
        return False, "assistant in demo mode (no LLM key)"
    if "£" in resp or "$" in resp:
        return False, f"non-INR currency in reply: {resp[:80]}"
    if "₹" not in resp:
        return False, f"no ₹ price in reply: {resp[:80]}"
    return True, "reply priced in ₹, LLM live"


def t_chat_category():
    st, txt = _req("/chat/stream", "POST", {"message": "show me some dresses",
                                            "customer_id": CUSTOMER_ID, "history": []})
    prods = []
    for line in txt.splitlines():
        if line.startswith("data:"):
            try:
                j = json.loads(line[5:])
            except Exception:
                continue
            if j.get("products"):
                prods = j["products"]
    if not prods:
        return False, "no product cards for 'dresses'"
    types = {p.get("product_type_name", "") for p in prods}
    if types and not any("dress" in t.lower() for t in types):
        return False, f"asked for dresses, got {types}"
    return True, f"{len(prods)} dress cards"


print(f"smoke test -> {BASE}  (mode: {'full' if FULL else 'cheap'}, customer_id={CUSTOMER_ID!r})")
check("health / models loaded", t_health)
check("ready (models + db)", t_ready)
check("db security posture (RLS + triggers + cron)", t_db_posture)
check("/products catalogue", t_products)
check("/catalog/photos (SerpApi)", t_photos, soft=True)
if FULL:
    check("POST /recommend priced in INR", t_recommend)
    check("chat reply in ₹, not demo mode", t_chat_currency)
    check("chat 'dresses' returns dresses", t_chat_category)
else:
    print("  (skipping /recommend + /chat — set SMOKE_FULL=1 to include them;"
         " they write to Supabase and spend an OpenRouter call)")

print(f"\n{_fail} failed, {_warn} warnings")
sys.exit(1 if _fail else 0)
