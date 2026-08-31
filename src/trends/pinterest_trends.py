"""
FashionMind — Pinterest trend provider
======================================
Returns currently-rising fashion search terms with a 0-100 interest score.

Resolution order (each step degrades gracefully to the next):
  1. Official Pinterest API v5  — GET /v5/trends/keywords/{region}/top/{type}
     Needs env PINTEREST_ACCESS_TOKEN (OAuth token with trends scope).
  2. Unofficial trends.pinterest.com JSON — no auth, undocumented, geo-gated,
     rate-limited, against Pinterest ToS. Best-effort only.
  3. Cached seed — data/features/pinterest_trends_cache.json (shipped), so the
     rest of the pipeline always has a signal to fuse.

Public API:
  get_pinterest_trends(category=None, region="IN", limit=25) -> list[dict]
      [{ "keyword": str, "score": int(0..100), "pct_change": str|None,
         "source": "api"|"unofficial"|"cache" }]
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except Exception:  # pragma: no cover - requests is a hard dep of pytrends anyway
    requests = None

_BASE = Path(__file__).resolve().parent.parent.parent
_CACHE_PATH = _BASE / "data" / "features" / "pinterest_trends_cache.json"
_RUNTIME_CACHE_TTL = 60 * 60 * 6          # 6h in-process memo
_runtime_cache: dict = {}

PINTEREST_API = "https://api.pinterest.com/v5"
PINTEREST_UNOFFICIAL = "https://trends.pinterest.com/api/v1"

# Shipped so the pipeline works with zero configuration and no network.
# Refresh by running:  python -m src.trends.pinterest_trends --refresh
_SEED = {
    "region": "IN",
    "fetched_at": "2026-08-01T00:00:00Z",
    "source": "cache",
    "keywords": [
        {"keyword": "barrel jeans", "score": 100, "pct_change": "+180%"},
        {"keyword": "boho outfit", "score": 96, "pct_change": "+140%"},
        {"keyword": "ballet flats", "score": 93, "pct_change": "+90%"},
        {"keyword": "co-ord set", "score": 92, "pct_change": "+85%"},
        {"keyword": "oversized blazer", "score": 90, "pct_change": "+60%"},
        {"keyword": "cargo pants", "score": 88, "pct_change": "+55%"},
        {"keyword": "sheer top", "score": 86, "pct_change": "+70%"},
        {"keyword": "maxi skirt", "score": 85, "pct_change": "+50%"},
        {"keyword": "waistcoat", "score": 84, "pct_change": "+65%"},
        {"keyword": "linen set", "score": 83, "pct_change": "+40%"},
        {"keyword": "corset top", "score": 82, "pct_change": "+45%"},
        {"keyword": "butter yellow", "score": 81, "pct_change": "+120%"},
        {"keyword": "chocolate brown outfit", "score": 80, "pct_change": "+75%"},
        {"keyword": "crochet dress", "score": 79, "pct_change": "+35%"},
        {"keyword": "parachute pants", "score": 78, "pct_change": "+95%"},
        {"keyword": "quiet luxury", "score": 77, "pct_change": "+30%"},
        {"keyword": "polka dot dress", "score": 76, "pct_change": "+40%"},
        {"keyword": "chunky loafers", "score": 75, "pct_change": "+35%"},
        {"keyword": "denim maxi skirt", "score": 74, "pct_change": "+80%"},
        {"keyword": "bomber jacket", "score": 73, "pct_change": "+25%"},
        {"keyword": "metallic top", "score": 72, "pct_change": "+50%"},
        {"keyword": "capri pants", "score": 71, "pct_change": "+60%"},
        {"keyword": "animal print", "score": 70, "pct_change": "+30%"},
        {"keyword": "drop waist dress", "score": 69, "pct_change": "+55%"},
        {"keyword": "statement collar", "score": 68, "pct_change": "+20%"},
        {"keyword": "kurta set", "score": 67, "pct_change": "+28%"},
        {"keyword": "organza saree", "score": 66, "pct_change": "+33%"},
        {"keyword": "mesh flats", "score": 65, "pct_change": "+40%"},
    ],
}


# ── helpers ──────────────────────────────────────────────────────────────────
def _normalise(items: list[dict], source: str) -> list[dict]:
    out = []
    for it in items:
        kw = (it.get("keyword") or it.get("term") or it.get("query") or "").strip().lower()
        if not kw:
            continue
        score = it.get("score")
        if score is None:
            score = it.get("interest") or it.get("value") or 50
        out.append({
            "keyword": kw,
            "score": int(max(0, min(100, round(float(score))))),
            "pct_change": it.get("pct_change") or it.get("pct_growth") or it.get("growth"),
            "source": source,
        })
    return out


def _from_cache() -> list[dict]:
    for p in (_CACHE_PATH,):
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                return _normalise(data.get("keywords", []), "cache")
        except Exception:
            pass
    return _normalise(_SEED["keywords"], "cache")


def _from_official(region: str, trend_type: str) -> list[dict]:
    token = os.getenv("PINTEREST_ACCESS_TOKEN", "")
    if not token or requests is None:
        return []
    url = f"{PINTEREST_API}/trends/keywords/{region}/top/{trend_type}"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                         params={"limit": 50}, timeout=8)
        if r.status_code != 200:
            return []
        body = r.json()
        rows = body.get("trends") or body.get("data") or body.get("items") or []
        # v5 shape: [{"keyword": "...", "pct_growth_wow": 12, "time_series": {...}}]
        items = []
        for row in rows:
            ts = row.get("time_series") or {}
            last = list(ts.values())[-1] if isinstance(ts, dict) and ts else row.get("score", 50)
            items.append({"keyword": row.get("keyword"), "score": last,
                          "pct_change": row.get("pct_growth_wow") or row.get("pct_growth_yoy")})
        return _normalise(items, "api")
    except Exception:
        return []


def _from_unofficial(region: str) -> list[dict]:
    if requests is None:
        return []
    # Undocumented; endpoint/param names drift. Kept behind an opt-in flag so a
    # ToS-sensitive deployment never touches it.
    if os.getenv("PINTEREST_ALLOW_UNOFFICIAL", "").lower() not in ("1", "true", "yes"):
        return []
    try:
        r = requests.get(f"{PINTEREST_UNOFFICIAL}/trends",
                         params={"region": region, "type": "fashion"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if r.status_code != 200:
            return []
        rows = r.json().get("trends", [])
        return _normalise(rows, "unofficial")
    except Exception:
        return []


# ── public ──────────────────────────────────────────────────────────────────
def get_pinterest_trends(category: Optional[str] = None, region: str = "IN",
                         trend_type: str = "growing", limit: int = 25) -> list[dict]:
    key = (region, trend_type)
    memo = _runtime_cache.get(key)
    if memo and time.time() - memo[0] < _RUNTIME_CACHE_TTL:
        rows = memo[1]
    else:
        rows = (_from_official(region, trend_type)
                or _from_unofficial(region)
                or _from_cache())
        _runtime_cache[key] = (time.time(), rows)

    if category:
        c = category.strip().lower()
        rows = [r for r in rows if c in r["keyword"]] or rows
    return sorted(rows, key=lambda r: -r["score"])[:limit]


def refresh_cache(region: str = "IN") -> int:
    """Fetch live trends and overwrite the on-disk cache. Returns row count."""
    rows = _from_official(region, "growing") or _from_unofficial(region)
    if not rows:
        print("No live source available (set PINTEREST_ACCESS_TOKEN or "
              "PINTEREST_ALLOW_UNOFFICIAL=1). Cache unchanged.")
        return 0
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(
        {"region": region, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "source": rows[0]["source"], "keywords": rows}, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} keywords to {_CACHE_PATH}")
    return len(rows)


if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv(_BASE / ".env", override=False)
    except Exception:
        pass
    if "--refresh" in sys.argv:
        refresh_cache()
    else:
        for row in get_pinterest_trends(limit=15):
            print(f"  {row['score']:3d}  {row['keyword']:<28} "
                  f"{row['pct_change'] or '':>6}  [{row['source']}]")
