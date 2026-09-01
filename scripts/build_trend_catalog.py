"""
Generate data/catalog/trend_products.json — a curated trend-product seed for the
FashionMind aggregator (India market).

Links are retailer SEARCH URLs (always valid, no fabricated SKUs). Prices are
category estimates, not live SKU prices; a real product API replaces this via
src/catalog/sources.py.

Run:  python scripts/build_trend_catalog.py
"""
import hashlib
import json
import pathlib
import sys
import urllib.parse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

RETAILERS = {
    "Myntra":        "https://www.myntra.com/{slug}",
    "Ajio":          "https://www.ajio.com/search/?text={q}",
    "Nykaa Fashion": "https://www.nykaafashion.com/search?q={q}",
    "H&M India":     "https://www2.hm.com/en_in/search-results.html?q={q}",
    "Zara India":    "https://www.zara.com/in/en/search?searchTerm={q}",
    "Urbanic":       "https://in.urbanic.com/search?keyword={q}",
    "Snitch":        "https://www.snitch.co.in/search?q={q}",
    "FabIndia":      "https://www.fabindia.com/search?q={q}",
    "Savana":        "https://savana.in/search?q={q}",
    "Newme":         "https://newme.asia/search?q={q}",
    "Bonkers Corner":"https://bonkerscorner.com/search?q={q}",
    "Lifestyle":     "https://www.lifestylestores.com/in/en/search/?q={q}",
    "JAM":           "https://www.google.com/search?tbm=shop&q={q}%20JAM%20clothing",
    "Bershka":       "https://www.bershka.com/in/search?q={q}",
    "Littlebox":     "https://littleboxindia.com/search?q={q}",
    "Lulu & Sky":    "https://www.luluandsky.com/catalogsearch/result/?q={q}",
    "Virgio":        "https://www.virgio.com/search?q={q}",
}


def buy_url(retailer: str, query: str) -> str:
    tpl = RETAILERS[retailer]
    return tpl.format(slug=query.strip().lower().replace(" ", "-"),
                      q=urllib.parse.quote_plus(query))


# title, brand, retailer, query, product_type, colour, price_min, price_max, gender, look, tags
ROWS = [
    ("Loose barrel-leg jeans", "Levi's", "Myntra", "barrel jeans", "Trousers", "Dark Blue", 2299, 3999, "women", "denim-dressed-up", ["denim", "barrel-jeans", "wide-leg"]),
    ("High-rise barrel jeans", "H&M", "H&M India", "barrel jeans women", "Trousers", "Blue", 1799, 2999, "women", "denim-dressed-up", ["denim", "barrel-jeans"]),
    ("Balloon-leg jeans", "AJIO", "Ajio", "balloon jeans", "Trousers", "Light Blue", 1499, 2499, "women", "weekend-casual", ["denim", "balloon", "wide-leg"]),
    ("Parachute cargo trousers", "Urbanic", "Urbanic", "parachute pants", "Trousers", "Greenish Khaki", 1690, 2490, "women", "utility", ["parachute", "utility", "cargo"]),
    ("Wide-leg cargo pants", "Snitch", "Snitch", "cargo pants men", "Trousers", "Greenish Khaki", 1299, 1999, "men", "utility", ["cargo", "utility"]),
    ("Pleated wide-leg trousers", "Zara", "Zara India", "wide leg trousers", "Trousers", "Black", 2590, 3590, "women", "office-siren", ["wide-leg", "tailoring", "office-siren"]),
    ("Linen-blend capri trousers", "FabIndia", "FabIndia", "capri trousers", "Trousers", "Off White", 1799, 2799, "women", "summer-linen", ["capri", "linen", "summer"]),
    ("Denim maxi skirt", "Nykaa Fashion", "Nykaa Fashion", "denim maxi skirt", "Skirt", "Blue", 1799, 2999, "women", "denim-dressed-up", ["denim", "maxi", "skirt"]),
    ("Bias-cut satin midi skirt", "AJIO", "Ajio", "satin midi skirt", "Skirt", "Dark Green", 1299, 2299, "women", "going-out", ["midi", "satin", "skirt"]),
    ("Crochet knit midi skirt", "Urbanic", "Urbanic", "crochet skirt", "Skirt", "Beige", 1490, 2290, "women", "boho-summer", ["crochet", "boho", "skirt"]),
    ("Mesh ballet flats", "H&M", "H&M India", "mesh ballet flats", "Ballerinas", "Black", 1299, 1999, "women", "balletcore", ["mesh", "ballet-flats", "balletcore", "shoes"]),
    ("Bow ballet flats", "Nykaa Fashion", "Nykaa Fashion", "ballet flats", "Ballerinas", "Light Pink", 1499, 2499, "women", "balletcore", ["ballet-flats", "balletcore", "romantic", "shoes"]),
    ("Chunky penny loafers", "Myntra", "Myntra", "chunky loafers", "Other shoe", "Black", 2199, 3499, "women", "old-money", ["loafers", "chunky", "old-money", "shoes"]),
    ("Suede boat shoes", "Snitch", "Snitch", "boat shoes men", "Other shoe", "Dark Beige", 1799, 2799, "men", "preppy", ["boat-shoes", "preppy", "shoes"]),
    ("Retro low-top sneakers", "Adidas", "Myntra", "samba sneakers", "Sneakers", "Off White", 6999, 9999, "unisex", "blokecore", ["sneakers", "blokecore", "shoes"]),
    ("Linen co-ord shirt + shorts set", "H&M", "H&M India", "linen co-ord set", "Garment Set", "Beige", 2499, 3799, "men", "summer-linen", ["linen", "co-ord", "summer", "matching-set"]),
    ("Textured knit co-ord set", "Urbanic", "Urbanic", "knit co ord set", "Garment Set", "Light Beige", 2190, 3290, "women", "co-ord", ["co-ord", "knit", "matching-set"]),
    ("Printed kurta co-ord set", "FabIndia", "FabIndia", "kurta co-ord set", "Garment Set", "Dark Blue", 2799, 4499, "women", "festive", ["kurta", "ethnic", "co-ord", "festive"]),
    ("Organza embroidered saree", "Nykaa Fashion", "Nykaa Fashion", "organza saree", "Dress", "Light Pink", 3999, 7999, "women", "festive", ["saree", "organza", "ethnic", "occasion", "festive"]),
    ("Oversized single-breasted blazer", "Zara", "Zara India", "oversized blazer", "Blazer", "Black", 4590, 5990, "women", "office-siren", ["blazer", "oversized", "tailoring", "office-siren"]),
    ("Relaxed linen blazer", "H&M", "H&M India", "linen blazer", "Blazer", "Beige", 3499, 4999, "men", "summer-linen", ["blazer", "linen", "tailoring"]),
    ("Longline tailored waistcoat", "AJIO", "Ajio", "waistcoat women", "Blazer", "Dark Grey", 1699, 2699, "women", "office-siren", ["waistcoat", "vest", "tailoring", "office-siren"]),
    ("Nylon bomber jacket", "Snitch", "Snitch", "bomber jacket men", "Jacket", "Black", 2299, 3299, "men", "street", ["bomber", "jacket", "street"]),
    ("Cropped corset-detail top", "Urbanic", "Urbanic", "corset top", "Top", "Black", 990, 1690, "women", "going-out", ["corset", "going-out"]),
    ("Sheer mesh long-sleeve top", "H&M", "H&M India", "sheer mesh top", "Blouse", "Black", 999, 1599, "women", "going-out", ["sheer", "mesh", "going-out"]),
    ("Ruffled statement-collar blouse", "Nykaa Fashion", "Nykaa Fashion", "statement collar blouse", "Blouse", "Off White", 1299, 2199, "women", "romantic", ["collar", "romantic", "blouse"]),
    ("Crochet halter dress", "AJIO", "Ajio", "crochet dress", "Dress", "Beige", 1999, 3299, "women", "boho-summer", ["crochet", "boho", "summer"]),
    ("Polka-dot midi tea dress", "Zara", "Zara India", "polka dot midi dress", "Dress", "Black", 3290, 4290, "women", "retro", ["polka-dot", "retro", "midi"]),
    ("Drop-waist mini dress", "Urbanic", "Urbanic", "drop waist dress", "Dress", "Dark Red", 1690, 2690, "women", "retro", ["drop-waist", "retro"]),
    ("Satin slip midi dress", "H&M", "H&M India", "slip dress", "Dress", "Dark Green", 1799, 2799, "women", "going-out", ["slip-dress", "minimal", "going-out"]),
    ("Tiered boho maxi dress", "FabIndia", "FabIndia", "boho maxi dress", "Dress", "Dark Yellow", 2999, 4999, "women", "boho-summer", ["boho", "maxi", "summer"]),
    ("Fine-gauge crew knit", "Uniqlo", "Myntra", "fine merino sweater", "Sweater", "Beige", 2490, 3490, "unisex", "quiet-luxury", ["quiet-luxury", "minimal", "neutral", "knit"]),
    ("Straight-leg wool-blend trousers", "Zara", "Zara India", "wool blend trousers", "Trousers", "Dark Grey", 3590, 4590, "unisex", "quiet-luxury", ["quiet-luxury", "tailoring", "neutral"]),
    ("Boxy poplin oxford shirt", "H&M", "H&M India", "oxford shirt", "Shirt", "Light Blue", 1499, 2299, "men", "old-money", ["old-money", "preppy", "shirt"]),
    ("Pleated tennis skirt", "Nykaa Fashion", "Nykaa Fashion", "tennis skirt", "Skirt", "Off White", 1199, 1999, "women", "tenniscore", ["tenniscore", "preppy", "sporty", "skirt"]),
    ("Metallic lame cami top", "Urbanic", "Urbanic", "metallic top", "Top", "Gold", 1190, 1890, "women", "party", ["metallic", "going-out", "party"]),
    ("Sequin mini skirt", "AJIO", "Ajio", "sequin skirt", "Skirt", "Grey", 1799, 2999, "women", "party", ["sequin", "party"]),
    ("Butter-yellow poplin shirt", "H&M", "H&M India", "butter yellow shirt", "Shirt", "Light Yellow", 1299, 1999, "unisex", "pastel", ["butter-yellow", "pastel", "shirt"]),
    ("Chocolate-brown faux-leather tote", "AJIO", "Ajio", "brown tote bag", "Bag", "Dark Beige", 1999, 3499, "women", "neutral", ["chocolate-brown", "tote", "bag", "neutral"]),
    ("Leopard-print silky scarf", "Myntra", "Myntra", "leopard scarf", "Scarf", "Dark Beige", 699, 1299, "women", "animal-print", ["leopard", "animal-print", "accessory"]),
    ("Cropped parachute skirt", "Urbanic", "Urbanic", "parachute skirt", "Skirt", "Greenish Khaki", 1390, 2190, "women", "utility", ["parachute", "utility", "skirt"]),
    ("Oversized barrel-fit jeans", "Snitch", "Snitch", "barrel jeans men", "Trousers", "Dark Blue", 1799, 2799, "men", "denim-dressed-up", ["denim", "barrel-jeans", "men"]),
    ("Ribbed tank + wide trouser co-ord", "AJIO", "Ajio", "tank trouser co ord", "Garment Set", "Black", 1899, 2999, "women", "co-ord", ["co-ord", "matching-set"]),
    ("Cotton poplin shorts co-ord set", "H&M", "H&M India", "poplin shorts set", "Garment Set", "Light Blue", 1799, 2699, "women", "summer-linen", ["co-ord", "summer", "shorts"]),
    ("Sculptural gold hoop earrings", "Nykaa Fashion", "Nykaa Fashion", "chunky gold hoops", "Earring", "Gold", 499, 1299, "women", "going-out", ["gold", "statement", "accessory"]),
    ("Wide woven leather belt", "Zara", "Zara India", "woven leather belt", "Belt", "Dark Beige", 1290, 1990, "unisex", "old-money", ["belt", "old-money", "accessory"]),
    ("Suede baseball cap", "Snitch", "Snitch", "suede cap", "Cap/peaked", "Dark Beige", 899, 1499, "unisex", "street", ["cap", "street", "accessory"]),
    ("Sheer-panel maxi dress", "Urbanic", "Urbanic", "sheer maxi dress", "Dress", "Black", 2190, 3490, "women", "going-out", ["sheer", "maxi", "going-out"]),
    ("Cropped cardigan", "H&M", "H&M India", "cropped cardigan", "Cardigan", "Light Pink", 1299, 1999, "women", "balletcore", ["balletcore", "cardigan", "romantic", "knit"]),
    ("Wide-brim raffia hat", "FabIndia", "FabIndia", "raffia hat", "Hat/brim", "Beige", 1299, 2199, "women", "boho-summer", ["boho", "summer", "hat", "accessory"]),

    # ── Festive / ethnic edit ───────────────────────────────────────────────
    ("Organza embroidered saree", "Nykaa Fashion", "Nykaa Fashion", "organza embroidered saree", "Dress", "Light Pink", 3999, 8999, "women", "festive", ["festive", "saree", "organza", "ethnic", "occasion"]),
    ("Bandhani cotton kurta set", "FabIndia", "FabIndia", "bandhani kurta set", "Garment Set", "Dark Red", 2799, 4999, "women", "festive", ["festive", "kurta", "bandhani", "ethnic", "co-ord"]),
    ("Sequin lehenga choli", "Savana", "Savana", "sequin lehenga", "Dress", "Dark Green", 4999, 11999, "women", "festive", ["festive", "lehenga", "sequin", "ethnic", "wedding"]),
    ("Anarkali floor-length kurta", "Ajio", "Ajio", "anarkali kurta", "Dress", "Dark Blue", 2499, 5999, "women", "festive", ["festive", "anarkali", "kurta", "ethnic"]),
    ("Chikankari straight kurta", "Lifestyle", "Lifestyle", "chikankari kurta", "Blouse", "Off White", 1799, 3499, "women", "festive", ["festive", "chikankari", "kurta", "ethnic"]),
    ("Indo-western drape dress", "Newme", "Newme", "indo western drape dress", "Dress", "Gold", 2299, 4299, "women", "festive", ["festive", "indo-western", "occasion", "party"]),
    ("Silk-blend sherwani", "Myntra", "Myntra", "sherwani men", "Blazer", "Dark Beige", 5999, 14999, "men", "festive", ["festive", "sherwani", "ethnic", "wedding", "men"]),
    ("Bandhgala bandi jacket", "Bonkers Corner", "Bonkers Corner", "bandhgala jacket", "Blazer", "Dark Blue", 2299, 4999, "men", "festive", ["festive", "bandhgala", "ethnic", "men"]),
    ("Nehru jacket + kurta pyjama set", "Lifestyle", "Lifestyle", "nehru jacket kurta set", "Garment Set", "Beige", 2999, 5999, "men", "festive", ["festive", "nehru-jacket", "kurta", "ethnic", "men"]),
    ("Banarasi silk dupatta", "FabIndia", "FabIndia", "banarasi dupatta", "Scarf", "Dark Red", 1499, 3499, "women", "festive", ["festive", "banarasi", "dupatta", "ethnic", "accessory"]),
    ("Embroidered potli bag", "Nykaa Fashion", "Nykaa Fashion", "potli bag", "Bag", "Gold", 799, 1999, "women", "festive", ["festive", "potli", "bag", "ethnic", "accessory"]),
    ("Mirror-work embroidered juttis", "Ajio", "Ajio", "juttis women", "Other shoe", "Dark Red", 999, 2499, "women", "festive", ["festive", "juttis", "ethnic", "shoes"]),
    ("Kundan choker necklace set", "Nykaa Fashion", "Nykaa Fashion", "kundan choker set", "Necklace", "Gold", 1299, 3999, "women", "festive", ["festive", "kundan", "jewellery", "ethnic", "accessory"]),
    ("Zari-border cotton saree", "Myntra", "Myntra", "zari border cotton saree", "Dress", "Off White", 1999, 4499, "women", "festive", ["festive", "saree", "cotton", "ethnic"]),
    ("Velvet embroidered ethnic co-ord", "JAM", "JAM", "velvet ethnic co-ord", "Garment Set", "Dark Purple", 2799, 4999, "women", "festive", ["festive", "velvet", "co-ord", "ethnic", "party"]),

    # ── More fast-fashion retailers (India online) ─────────────────────────
    ("Baggy carpenter jeans", "Bershka", "Bershka", "baggy jeans women", "Trousers", "Light Blue", 2290, 3290, "women", "weekend-casual", ["denim", "baggy", "wide-leg"]),
    ("Cropped rib tank top", "Bershka", "Bershka", "rib tank top", "Vest top", "White", 590, 990, "women", "everyday", ["tank", "rib", "basic"]),
    ("Faux-leather biker jacket", "Bershka", "Bershka", "faux leather jacket", "Jacket", "Black", 3290, 4590, "women", "going-out", ["leather", "jacket", "biker"]),
    ("Floral puff-sleeve mini dress", "Littlebox", "Littlebox", "floral mini dress", "Dress", "Light Pink", 999, 1799, "women", "brunch", ["floral", "mini", "romantic"]),
    ("Co-ord blazer + shorts set", "Littlebox", "Littlebox", "blazer shorts co ord set", "Garment Set", "Beige", 1499, 2599, "women", "office-siren", ["co-ord", "tailoring", "matching-set"]),
    ("Satin corset-detail top", "Lulu & Sky", "Lulu & Sky", "satin corset top", "Top", "Dark Green", 1290, 2190, "women", "going-out", ["corset", "satin", "going-out"]),
    ("Ruched bodycon midi dress", "Lulu & Sky", "Lulu & Sky", "ruched bodycon dress", "Dress", "Black", 1690, 2890, "women", "party", ["bodycon", "ruched", "party"]),
    ("Wide-leg pleated trousers", "Virgio", "Virgio", "pleated wide leg trousers", "Trousers", "Off White", 1990, 2990, "women", "office-siren", ["wide-leg", "pleated", "tailoring"]),
    ("Oversized poplin shirt", "Virgio", "Virgio", "oversized poplin shirt", "Shirt", "Light Blue", 1490, 2290, "women", "minimal", ["poplin", "oversized", "minimal"]),
    ("Linen-blend slip dress", "Virgio", "Virgio", "linen slip dress", "Dress", "Beige", 1990, 3190, "women", "summer-linen", ["linen", "slip-dress", "minimal"]),
    ("Waistcoat + trouser co-ord", "Urbanic", "Urbanic", "waistcoat trouser co ord", "Garment Set", "Dark Grey", 2190, 3390, "women", "office-siren", ["waistcoat", "tailoring", "co-ord"]),
    ("Denim maxi shirt dress", "Zara", "Zara India", "denim maxi shirt dress", "Dress", "Blue", 3590, 4590, "women", "denim-dressed-up", ["denim", "maxi", "shirt-dress"]),
]


# ── programmatic fan-out: every trend keyword × every retailer × a few cuts ──
# Gives each trend a deep shelf of shoppable links across all India sites.
_BANDS = {
    "Trousers": (1299, 3499), "Skirt": (1199, 2999), "Dress": (1499, 4999),
    "Top": (699, 1999), "Blouse": (999, 2499), "Shirt": (1199, 2799),
    "T-shirt": (599, 1699), "Vest top": (499, 1299), "Bodysuit": (799, 1999),
    "Sweater": (1499, 3499), "Cardigan": (1299, 2999), "Hoodie": (999, 2499),
    "Jacket": (2499, 5999), "Blazer": (2999, 6999), "Coat": (3999, 8999),
    "Garment Set": (1999, 4999), "Jumpsuit/Playsuit": (1699, 3999),
    "Ballerinas": (1299, 2999), "Sneakers": (2499, 6999), "Other shoe": (1499, 3999),
    "Boots": (2999, 6999), "Bag": (899, 3499), "Scarf": (499, 1999),
    "Sunglasses": (799, 2499), "Earring": (299, 1499), "Necklace": (699, 2999),
    "Leggings/Tights": (699, 1699), "Polo shirt": (899, 2199),
}
_DEFAULT_BAND = (999, 2999)
_CUTS = ["", "Relaxed-fit ", "High-rise ", "Cropped ", "Oversized ", "Classic ",
         "Everyday ", "Premium ", "Structured ", "Soft ", "Tailored ", "Statement ",
         "Elevated ", "Essential ", "Modern ", "Signature ", "Fluid "]
_PALETTE = ["Black", "White", "Off White", "Blue", "Dark Blue", "Light Blue",
            "Beige", "Dark Beige", "Grey", "Dark Grey", "Green", "Dark Green",
            "Pink", "Light Pink", "Dark Red", "Gold", "Greenish Khaki"]
_VARIANTS = 2   # cuts per (keyword, retailer)


def _keywords():
    try:
        from src.trends.pinterest_trends import _SEED
        kws = [k["keyword"] for k in _SEED["keywords"]]
    except Exception:
        kws = []
    try:
        from src.trends.festivals import _FESTIVALS
        for f in _FESTIVALS:
            kws += list(f[3])
    except Exception:
        pass
    seen, out = set(), []
    for k in kws:
        k = k.strip().lower()
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out


def _fanout_rows():
    from src.trends.trend_map import map_trend
    rows = []
    for kw in _keywords():
        m = map_trend(kw)
        pts = m["product_types"] or ["Top"]
        tags = m["tags"] or [kw.replace(" ", "-")]
        colours = m["colours"] or []
        gender = "men" if any(w in kw for w in (" men", "sherwani", "bandhgala", "bandhi", "nehru jacket", "dhoti")) else "women"
        look = tags[0] if tags else "trend"
        base = cap = kw.title()
        for ri, ret in enumerate(RETAILERS):
            pt = pts[ri % len(pts)]
            band = _BANDS.get(pt, _DEFAULT_BAND)
            for v in range(_VARIANTS):
                cut = _CUTS[(ri + v * 5) % len(_CUTS)]
                title = f"{cut}{base}".strip()
                col = (colours[(ri + v) % len(colours)] if colours
                       else _PALETTE[(hash((kw, ret, v)) // 7) % len(_PALETTE)])
                lo = band[0] + ((ri * 137 + v * 311) % max(1, (band[1] - band[0]) // 3))
                hi = min(band[1], lo + (band[1] - band[0]) // 2 + 200)
                rows.append((title, ret, ret, kw, pt, col, lo, hi, gender, look,
                             sorted(set(tags + [kw.replace(" ", "-")]))))
    return rows


def build() -> dict:
    items, seen_ids = [], set()

    def _add(title, brand, ret, q, pt, col, pmin, pmax, gender, look, tags, curated):
        pid = ("cur_" if curated else "agg_") + hashlib.md5(f"{title}|{ret}|{q}".encode()).hexdigest()[:10]
        if pid in seen_ids:
            return
        seen_ids.add(pid)
        items.append({
            "id": pid, "source": "curated", "title": title, "brand": brand,
            "retailer": ret, "buy_url": buy_url(ret, q), "search_query": q,
            "product_type": pt, "colour": col, "gender": gender,
            "price_min": pmin, "price_max": pmax, "currency": "INR",
            "price_note": "Category estimate - confirm on retailer site.",
            "price_is_estimate": True,
            "image_query": q, "look": look, "tags": tags,
        })

    for row in ROWS:
        _add(*row, curated=True)
    for row in _fanout_rows():
        _add(*row, curated=False)
    return {
        "_meta": {
            "note": ("Curated trend-product seed for the FashionMind aggregator (India). "
                     "buy_url values are retailer SEARCH URLs (always valid, no fabricated "
                     "SKUs). Prices are category estimates, not live SKU prices. Swap in a "
                     "real product API via src/catalog/sources.py."),
            "market": "IN", "count": len(items), "generated": "2026-08-31",
        },
        "products": items,
    }


if __name__ == "__main__":
    out = build()
    p = pathlib.Path(__file__).resolve().parent.parent / "data" / "catalog" / "trend_products.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {p}  ({out['_meta']['count']} items, {p.stat().st_size} bytes)")
