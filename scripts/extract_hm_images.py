"""
Extract every H&M product photo present in the raw zips into
data/raw/images/<prefix>/<id>.jpg, resized to 512px on the long edge so the
whole set is a couple of GB, not 15. The API serves them at /images/.

Run: python scripts/extract_hm_images.py
"""
import io
import os
import time
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("HM_RAW_DIR", r"D:\fashionmind_data\raw"))
OUT = ROOT / "data" / "raw" / "images"
ZIPS = ["010 (2).zip", "021.zip", "031.zip", "041.zip", "046.zip", "051.zip", "061.zip"]
EDGE, Q = 512, 82


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    done = {p.name for p in OUT.rglob("*.jpg")}
    print(f"{len(done):,} already extracted")
    t0, n, skipped, err = time.time(), 0, 0, 0
    for zn in ZIPS:
        zp = RAW / zn
        if not zp.exists():
            print(f"  skip {zn} (missing)")
            continue
        with zipfile.ZipFile(zp) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".jpg")]
            print(f"  {zn}: {len(members):,} images")
            for m in members:
                name = m.split("/")[-1]
                if name in done:
                    skipped += 1
                    continue
                dst = OUT / m
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    im = Image.open(io.BytesIO(zf.read(m))).convert("RGB")
                    w, h = im.size
                    if max(w, h) > EDGE:
                        s = EDGE / max(w, h)
                        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
                    im.save(dst, "JPEG", quality=Q, optimize=True)
                    n += 1
                    if n % 2000 == 0:
                        print(f"    {n:,} written · {time.time()-t0:.0f}s")
                except Exception:
                    err += 1
    size_gb = sum(f.stat().st_size for f in OUT.rglob("*.jpg")) / 1e9
    print(f"done — {n:,} new, {skipped:,} skipped, {err} errors · "
          f"{len(list(OUT.rglob('*.jpg'))):,} images total, {size_gb:.2f} GB · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
