"""
FashionMind — product source abstraction
========================================
One interface, several backends, so the aggregator does not care whether a
product came from the trained H&M catalogue, a curated seed file, or a live
shopping API.

Canonical product shape (`Product` = dict):
    id, source, title, brand, retailer, buy_url, product_type, colour, gender,
    price_min, price_max, currency, price_is_estimate, price_note,
    image_query, image, look, tags

Sources, in priority order:
  · HMLocalSource        — the trained H&M catalogue (105k items), no live price
  · CuratedTrendSource   — data/catalog/trend_products.json (India trend seed)
  · SerpApiShoppingSource— live Google Shopping (gl=in); needs SERPAPI_KEY
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

try:
    import requests
except Exception:
    requests = None

Product = dict
_BASE = Path(__file__).resolve().parent.parent.parent
_CATALOG_JSON = _BASE / "data" / "catalog" / "trend_products.json"

_HM_SEARCH = "https://www2.hm.com/en_in/search-results.html?q={q}"


def _img(query: str) -> str:
    # Redirect-based Unsplash source; the frontend may override with its own pool.
    return f"https://source.unsplash.com/featured/480x600/?{urllib.parse.quote(query + ',fashion')}"


def _match(hay: str, needles: list[str]) -> bool:
    h = (hay or "").lower()
    return any(n.lower() in h for n in needles)


# ── interface ───────────────────────────────────────────────────────────────
class ProductSource(ABC):
    name: str = "base"
    enabled: bool = True

    @abstractmethod
    def search(self, query: Optional[str] = None, product_types: Optional[list[str]] = None,
               tags: Optional[list[str]] = None, colours: Optional[list[str]] = None,
               limit: int = 24) -> list[Product]:
        ...

    def get(self, product_id: str) -> Optional[Product]:  # optional per-source
        return None


# ── curated seed ────────────────────────────────────────────────────────────
class CuratedTrendSource(ProductSource):
    name = "curated"

    def __init__(self, path: Path = _CATALOG_JSON):
        self._path = path
        self._items: list[Product] = []
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._items = data.get("products", [])
        except Exception:
            self._items = []

    def search(self, query=None, product_types=None, tags=None, colours=None, limit=24):
        rows = list(self._items)
        norm = lambda s: (s or "").lower().replace("-", " ")
        if product_types:
            rows = [r for r in rows if r.get("product_type") in product_types]
        if tags:
            tl = {norm(t) for t in tags}
            rows = [r for r in rows
                    if tl & {norm(t) for t in r.get("tags", [])} or norm(r.get("look")) in tl]
        if colours:
            rows = [r for r in rows if r.get("colour") in colours]
        if query:
            q = norm(query)
            def _blob(r):
                return norm(" ".join([r.get("title", ""), r.get("search_query", ""),
                                      " ".join(r.get("tags", [])), r.get("look", "")]))
            strict = [r for r in rows if q in _blob(r)]
            # `query` hard-filters only when it is the sole criterion; otherwise it
            # just re-ranks the already-narrowed set so a hyphen/space mismatch
            # (e.g. "quiet luxury" vs "quiet-luxury") does not zero the result.
            if strict and (product_types or tags):
                rest = [r for r in rows if r not in strict]
                rows = strict + rest
            elif not (product_types or tags):
                rows = strict
        out = []
        for r in rows[:limit]:
            r = dict(r)
            r.setdefault("image", _img(r.get("image_query", r["title"])))
            out.append(r)
        return out

    def get(self, product_id):
        for r in self._items:
            if r["id"] == product_id:
                r = dict(r)
                r.setdefault("image", _img(r.get("image_query", r["title"])))
                return r
        return None


# ── trained H&M catalogue ───────────────────────────────────────────────────
class HMLocalSource(ProductSource):
    name = "hm-demo"

    def _M(self):
        from src.genai.stylist_chatbot import _M, _load
        _load()
        return _M

    def _row_to_product(self, aid: str, r) -> Product:
        M = self._M()
        pt = str(r.get("product_type_name", "")) if r is not None else ""
        title = str(r.get("prod_name", pt)) if r is not None else aid
        colour = str(r.get("colour_group_name", "")) if r is not None else ""
        q = f"{title} {pt}".strip()
        return {
            "id": f"hm_{aid}", "source": "hm-demo", "title": title, "brand": "H&M",
            "retailer": "H&M (trained catalogue)",
            "buy_url": _HM_SEARCH.format(q=urllib.parse.quote_plus(q or title)),
            "product_type": pt, "colour": colour, "gender": "unisex",
            "price_min": None, "price_max": None, "currency": "INR",
            "price_is_estimate": True,
            "price_note": "Trained-catalogue item — no live price.",
            "image_query": q, "image": _img(q or title),
            "look": "", "tags": [t for t in [pt.lower(), colour.lower()] if t],
            "hm_article_id": aid,
        }

    def search(self, query=None, product_types=None, tags=None, colours=None, limit=24):
        M = self._M()
        art = M.get("art")
        if art is None:
            return []
        sub = art
        if product_types:
            sub = sub[sub["product_type_name"].isin(product_types)]
        if colours:
            sub = sub[sub["colour_group_name"].isin(colours)]
        if query:
            q = query.lower()
            sub = sub[sub["prod_name"].str.lower().str.contains(q, na=False)
                      | sub["product_type_name"].str.lower().str.contains(q, na=False)]
        # rank by popularity so the demo catalogue leads with strong items
        pop = M.get("pop_s", {})
        sub = sub.assign(_p=sub["article_id"].map(lambda a: pop.get(a, 0.0))) \
                 .sort_values("_p", ascending=False)
        out = []
        lu = M["art_lu"]
        for aid in sub["article_id"].head(limit):
            r = lu.loc[aid] if aid in lu.index else None
            out.append(self._row_to_product(aid, r))
        return out

    def get(self, product_id):
        if not product_id.startswith("hm_"):
            return None
        aid = product_id[3:]
        M = self._M()
        lu = M["art_lu"]
        return self._row_to_product(aid, lu.loc[aid] if aid in lu.index else None)


# ── live Google Shopping via SerpApi ────────────────────────────────────────
class SerpApiShoppingSource(ProductSource):
    name = "serpapi"
    _warned = False
    # Free plan is 100 searches/month, so every distinct query is cached on disk
    # for a day. Override the window with SERPAPI_CACHE_TTL (seconds).
    _CACHE_DIR = _BASE / "data" / "catalog" / ".serpapi_cache"
    _TTL = int(os.getenv("SERPAPI_CACHE_TTL", str(24 * 3600)))

    def __init__(self, gl: str = "in", hl: str = "en"):
        self.gl, self.hl = gl, hl
        self.enabled = bool(os.getenv("SERPAPI_KEY"))

    def _cache_path(self, q: str):
        import hashlib
        h = hashlib.md5(f"{self.gl}|{self.hl}|{q}".encode()).hexdigest()[:16]
        return self._CACHE_DIR / f"{h}.json"

    def search(self, query=None, product_types=None, tags=None, colours=None, limit=24):
        key = os.getenv("SERPAPI_KEY")
        if not key or requests is None:
            if not SerpApiShoppingSource._warned:
                print("[catalog] SerpApiShoppingSource inactive — set SERPAPI_KEY to enable live products.")
                SerpApiShoppingSource._warned = True
            return []
        q = query or " ".join(tags or []) or " ".join(product_types or []) or "trending fashion"

        cp = self._cache_path(q)
        try:
            if cp.exists() and time.time() - cp.stat().st_mtime < self._TTL:
                results = json.loads(cp.read_text(encoding="utf-8"))
            else:
                r = requests.get("https://serpapi.com/search", timeout=12, params={
                    "engine": "google_shopping", "q": q, "gl": self.gl, "hl": self.hl,
                    "google_domain": "google.co.in" if self.gl == "in" else "google.com",
                    "num": min(limit, 40), "api_key": key,
                })
                if r.status_code != 200:
                    return []
                results = r.json().get("shopping_results", [])
                try:
                    self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cp.write_text(json.dumps(results), encoding="utf-8")
                except Exception:
                    pass
        except Exception:
            return []
        out = []
        for it in results[:limit]:
            price = it.get("extracted_price")
            direct = it.get("link") or ""
            buy = direct if direct and "google." not in direct else (
                it.get("product_link") or direct or it.get("link", ""))
            out.append({
                "id": f"serp_{it.get('product_id') or abs(hash(it.get('link','')))}",
                "source": "serpapi", "title": it.get("title", ""),
                "brand": it.get("source", ""), "retailer": it.get("source", ""),
                "buy_url": buy,
                "product_type": (product_types or [""])[0],
                "colour": "", "gender": "unisex",
                "price_min": price, "price_max": price,
                "currency": "INR" if self.gl == "in" else "USD",
                "price_is_estimate": False,
                "price_note": "Live price from Google Shopping via SerpApi.",
                "image_query": it.get("title", ""), "image": it.get("thumbnail", ""),
                "look": "", "tags": tags or [],
                "rating": it.get("rating"), "reviews": it.get("reviews"),
            })
        return out


# ── registry + aggregate ────────────────────────────────────────────────────
_SOURCES: list[ProductSource] = []


def active_sources() -> list[ProductSource]:
    global _SOURCES
    if not _SOURCES:
        _SOURCES = [HMLocalSource(), CuratedTrendSource(), SerpApiShoppingSource()]
    return [s for s in _SOURCES if getattr(s, "enabled", True)]


def aggregate_search(query=None, product_types=None, tags=None, colours=None,
                     limit=24, sources: Optional[list[str]] = None) -> list[Product]:
    """Merge results across sources, curated + live first, dedupe by (title, retailer)."""
    seen, merged = set(), []
    order = {"curated": 0, "serpapi": 1, "hm-demo": 2}
    srcs = sorted(active_sources(), key=lambda s: order.get(s.name, 9))
    if sources:
        srcs = [s for s in srcs if s.name in sources]
    per = max(4, limit // max(1, len(srcs)))
    for s in srcs:
        try:
            rows = s.search(query=query, product_types=product_types, tags=tags,
                            colours=colours, limit=per + 6)
        except Exception:
            rows = []
        for r in rows:
            k = (r.get("title", "").lower().strip(), r.get("retailer", "").lower())
            if k in seen:
                continue
            seen.add(k)
            merged.append(r)
            if len(merged) >= limit:
                return merged
    return merged


def get_product(product_id: str) -> Optional[Product]:
    for s in active_sources():
        try:
            p = s.get(product_id)
        except Exception:
            p = None
        if p:
            return p
    return None


if __name__ == "__main__":
    for p in aggregate_search(tags=["balletcore"], limit=6):
        print(f"  [{p['source']:8}] {p['title'][:40]:40} {p['retailer']:22} "
              f"{p.get('price_min')}  {p['buy_url'][:60]}")
