"""
Extract a small subset of the H&M product photos — just the article_ids the app
actually surfaces (cold-start pop + candidates) — from the raw zips into
data/raw/images/<prefix>/<id>.jpg so the API can serve them at /images/.

Run: python scripts/extract_hm_images.py
"""
import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("HM_RAW_DIR", r"D:\fashionmind_data\raw"))
OUT = ROOT / "data" / "raw" / "images"
ZIPS = ["010 (2).zip", "021.zip", "031.zip", "041.zip", "046.zip", "051.zip", "061.zip"]


def wanted_ids() -> set[str]:
    ids: set[str] = set()
    for f in ("cold_start_popular", "cold_start_popular_global", "final_recommendations"):
        p = ROOT / "data" / "features" / f"{f}.parquet"
        if p.exists():
            ids |= set(pd.read_parquet(p)["article_id"].astype(str))
    cf = ROOT / "data" / "features" / "cf_candidates.parquet"
    if cf.exists():
        ids |= set(pd.read_parquet(cf)["article_id"].astype(str).head(40000))
    return {i.zfill(10) for i in ids if i.isdigit()}


def main() -> None:
    ids = wanted_ids()
    want_names = {f"{i[:3]}/{i}.jpg" for i in ids}
    print(f"{len(ids):,} article_ids wanted")
    OUT.mkdir(parents=True, exist_ok=True)
    got = 0
    for zn in ZIPS:
        zp = RAW / zn
        if not zp.exists():
            print(f"  skip {zn} (not found)")
            continue
        with zipfile.ZipFile(zp) as zf:
            members = [n for n in zf.namelist() if n in want_names]
            if not members:
                continue
            print(f"  {zn}: extracting {len(members):,}...")
            for m in members:
                dst = OUT / m
                if dst.exists():
                    got += 1
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(dst, "wb") as out:
                    out.write(src.read())
                got += 1
    size_mb = sum(f.stat().st_size for f in OUT.rglob("*.jpg")) / 1e6
    print(f"done — {got:,} images in {OUT}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
