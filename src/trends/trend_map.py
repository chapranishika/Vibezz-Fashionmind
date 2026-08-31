"""
FashionMind — trend keyword -> catalogue mapping
================================================
Turns a free-text Pinterest / Google trend keyword into structured catalogue
filters so the signal can actually join the H&M taxonomy and the curated
aggregator catalogue.

    map_trend(keyword) -> {
        "product_types": [<product_type_name>, ...],   # H&M taxonomy
        "colours":       [<colour_group_name>, ...],
        "tags":          [<style tag>, ...],           # for trend_products.json
    }

`product_types` values are exact H&M `product_type_name` strings so a filter
like  art[art.product_type_name.isin(pt)]  works directly.
"""
from __future__ import annotations

import re

# keyword (lowercase, matched as a substring) -> mapping
_RULES: list[tuple[str, dict]] = [
    ("barrel jeans",        {"product_types": ["Trousers"], "tags": ["denim", "barrel-jeans", "wide-leg"]}),
    ("balloon jeans",       {"product_types": ["Trousers"], "tags": ["denim", "balloon", "wide-leg"]}),
    ("parachute pants",     {"product_types": ["Trousers"], "tags": ["parachute", "utility", "wide-leg"]}),
    ("cargo pants",         {"product_types": ["Trousers"], "tags": ["cargo", "utility"]}),
    ("capri pants",         {"product_types": ["Trousers", "Leggings/Tights"], "tags": ["capri", "cropped"]}),
    ("wide leg",            {"product_types": ["Trousers"], "tags": ["wide-leg"]}),
    ("denim maxi skirt",    {"product_types": ["Skirt"], "tags": ["denim", "maxi", "skirt"]}),
    ("maxi skirt",          {"product_types": ["Skirt"], "tags": ["maxi", "skirt"]}),
    ("midi skirt",          {"product_types": ["Skirt"], "tags": ["midi", "skirt"]}),
    ("ballet flats",        {"product_types": ["Ballerinas", "Other shoe"], "tags": ["ballet-flats", "balletcore", "shoes"]}),
    ("mesh flats",          {"product_types": ["Ballerinas", "Other shoe"], "tags": ["mesh", "flats", "shoes"]}),
    ("chunky loafers",      {"product_types": ["Other shoe", "Boots"], "tags": ["loafers", "chunky", "shoes"]}),
    ("boat shoes",          {"product_types": ["Other shoe"], "tags": ["boat-shoes", "preppy", "shoes"]}),
    ("sneakers",            {"product_types": ["Sneakers"], "tags": ["sneakers", "shoes"]}),
    ("co-ord set",          {"product_types": ["Garment Set", "Jumpsuit/Playsuit"], "tags": ["co-ord", "matching-set"]}),
    ("coord set",           {"product_types": ["Garment Set"], "tags": ["co-ord", "matching-set"]}),
    ("linen set",           {"product_types": ["Garment Set", "Shirt", "Trousers"], "tags": ["linen", "co-ord", "summer"]}),
    ("kurta set",           {"product_types": ["Garment Set", "Dress"], "tags": ["kurta", "ethnic", "co-ord"]}),
    ("organza saree",       {"product_types": ["Dress"], "tags": ["saree", "organza", "ethnic", "occasion"]}),
    ("oversized blazer",    {"product_types": ["Blazer", "Jacket"], "tags": ["blazer", "oversized", "tailoring"]}),
    ("waistcoat",           {"product_types": ["Blazer", "Vest top"], "tags": ["waistcoat", "vest", "tailoring"]}),
    ("bomber jacket",       {"product_types": ["Jacket"], "tags": ["bomber", "jacket"]}),
    ("trench coat",         {"product_types": ["Coat"], "tags": ["trench", "coat"]}),
    ("corset top",          {"product_types": ["Top", "Bodysuit", "Blouse"], "tags": ["corset", "going-out"]}),
    ("sheer top",           {"product_types": ["Blouse", "Top"], "tags": ["sheer", "mesh", "going-out"]}),
    ("mesh top",            {"product_types": ["Top"], "tags": ["mesh", "sheer"]}),
    ("statement collar",    {"product_types": ["Blouse", "Shirt"], "tags": ["collar", "romantic"]}),
    ("crochet dress",       {"product_types": ["Dress"], "tags": ["crochet", "boho", "summer"]}),
    ("crochet",             {"product_types": ["Top", "Dress", "Cardigan"], "tags": ["crochet", "boho"]}),
    ("polka dot dress",     {"product_types": ["Dress"], "tags": ["polka-dot", "retro"]}),
    ("drop waist dress",    {"product_types": ["Dress"], "tags": ["drop-waist", "retro"]}),
    ("slip dress",          {"product_types": ["Dress"], "tags": ["slip-dress", "minimal"]}),
    ("maxi dress",          {"product_types": ["Dress"], "tags": ["maxi", "boho"]}),
    ("boho",                {"product_types": ["Dress", "Blouse", "Skirt"], "tags": ["boho", "bohemian"]}),
    ("bohemian",            {"product_types": ["Dress", "Blouse", "Skirt"], "tags": ["boho", "bohemian"]}),
    ("quiet luxury",        {"product_types": ["Sweater", "Trousers", "Blazer", "Coat"],
                             "colours": ["Beige", "Dark Beige", "Black", "Off White", "Grey"],
                             "tags": ["quiet-luxury", "minimal", "neutral"]}),
    ("old money",           {"product_types": ["Blazer", "Shirt", "Trousers"], "tags": ["old-money", "preppy"]}),
    ("office siren",        {"product_types": ["Blouse", "Blazer", "Skirt"], "tags": ["office-siren", "tailoring"]}),
    ("tenniscore",          {"product_types": ["Polo shirt", "Skirt", "Sneakers"], "tags": ["tenniscore", "preppy", "sporty"]}),
    ("blokecore",           {"product_types": ["T-shirt", "Sneakers"], "tags": ["blokecore", "sporty"]}),
    ("balletcore",          {"product_types": ["Ballerinas", "Cardigan", "Skirt"], "tags": ["balletcore", "romantic"]}),
    ("metallic",            {"product_types": ["Top", "Skirt", "Dress"],
                             "colours": ["Gold", "Bronze/Copper", "Grey"], "tags": ["metallic", "going-out"]}),
    ("sequin",              {"product_types": ["Top", "Dress"], "tags": ["sequin", "party"]}),
    ("butter yellow",       {"product_types": ["Top", "Shirt", "Blouse", "Dress", "Sweater", "Cardigan", "Trousers"],
                             "colours": ["Light Yellow", "Yellow"], "tags": ["butter-yellow", "pastel"]}),
    ("chocolate brown",     {"product_types": ["Trousers", "Sweater", "Coat", "Bag"],
                             "colours": ["Dark Beige", "Beige"], "tags": ["chocolate-brown", "neutral"]}),
    ("red",                 {"colours": ["Dark Red", "Light Red", "Other Red"], "tags": ["red", "statement-colour"]}),
    ("animal print",        {"product_types": ["Top", "Skirt", "Dress", "Scarf"], "tags": ["animal-print", "leopard"]}),
    ("leopard",             {"product_types": ["Top", "Skirt", "Bag"], "tags": ["leopard", "animal-print"]}),
    ("polka dot",           {"product_types": ["Dress", "Blouse", "Top"], "tags": ["polka-dot", "retro"]}),
    ("cardigan",            {"product_types": ["Cardigan"], "tags": ["cardigan", "knit"]}),
    ("vest top",            {"product_types": ["Vest top"], "tags": ["vest", "layering"]}),
    ("hoodie",              {"product_types": ["Hoodie"], "tags": ["hoodie", "athleisure"]}),
    ("jumpsuit",            {"product_types": ["Jumpsuit/Playsuit"], "tags": ["jumpsuit"]}),
    ("scarf",               {"product_types": ["Scarf"], "tags": ["scarf", "accessory"]}),
    ("tote bag",            {"product_types": ["Bag"], "tags": ["tote", "bag"]}),
    ("sunglasses",          {"product_types": ["Sunglasses"], "tags": ["sunglasses", "accessory"]}),
]

def map_trend(keyword: str) -> dict:
    kw = (keyword or "").strip().lower()
    hit = {"product_types": [], "colours": [], "tags": []}   # fresh lists per call
    matched = False
    for needle, mapping in _RULES:
        if needle in kw:
            for k in ("product_types", "colours", "tags"):
                for v in mapping.get(k, []):
                    if v not in hit[k]:
                        hit[k].append(v)
            matched = True
    if not matched:
        # last resort: keep the bare tokens as tags so text search can still use them
        hit["tags"] = [t for t in re.split(r"[^a-z0-9]+", kw) if len(t) > 2]
    return hit


def product_types_for_trends(keywords: list[str]) -> list[str]:
    out: list[str] = []
    for kw in keywords:
        for pt in map_trend(kw)["product_types"]:
            if pt not in out:
                out.append(pt)
    return out


if __name__ == "__main__":
    for kw in ["barrel jeans", "quiet luxury", "balletcore", "butter yellow", "random new thing"]:
        print(f"{kw:20} -> {map_trend(kw)}")
