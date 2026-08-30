"""
FashionMind — Phase 3B: CLIP Image Embeddings (AWS EC2)
========================================================
Run on EC2 after uploading images to data/raw/images/.
Replaces metadata SVD embeddings with real 512-dim CLIP vectors.
Gracefully falls back to padded metadata for articles with no image.

Install: pip install torch transformers pillow
Run    : python3 src/vision/clip_embeddings.py
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sklearn.preprocessing import normalize
warnings.filterwarnings('ignore')


def encode_images_clip(image_paths, batch_size=16):
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image

    print("  Loading CLIP ViT-B/32...")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    embeddings = {}
    t0 = time.time()
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        images, aids = [], []
        for p in batch:
            try:
                images.append(Image.open(p).convert("RGB"))
                aids.append(Path(p).stem)
            except: pass
        if not images: continue
        with torch.no_grad():
            inp  = processor(images=images, return_tensors="pt", padding=True)
            feat = model.get_image_features(**inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        for aid, emb in zip(aids, feat.cpu().numpy()):
            embeddings[aid] = emb
        if (i // batch_size) % 20 == 0:
            print(f"  {i+len(batch):>6,}/{len(image_paths)} | "
                  f"{len(embeddings):,} encoded | {time.time()-t0:.0f}s")
    return embeddings


def run():
    print("="*60)
    print("  FashionMind — Phase 3B: CLIP Visual Embeddings")
    print("="*60)

    # Check deps
    try:
        import torch, transformers
        print(f"\n  torch {torch.__version__} ✓ | transformers {transformers.__version__} ✓")
    except ImportError:
        print("\n  Missing torch/transformers. Install:")
        print("  pip install torch transformers pillow")
        return

    # Find images
    img_dir   = Path("data/raw/images")
    img_paths = sorted(img_dir.rglob("*.jpg"))
    if not img_paths:
        print(f"\n  No images in {img_dir}. Upload image folders first.")
        return
    print(f"\n  {len(img_paths):,} images found across "
          f"{len(set(p.parent.name for p in img_paths))} folders")

    # Encode
    print("\n[1/4] Encoding with CLIP ViT-B/32...")
    clip_embs = encode_images_clip([str(p) for p in img_paths])

    # Load articles + metadata fallback
    print("\n[2/4] Building hybrid embedding matrix...")
    art = pd.read_parquet("data/features/article_content_features.parquet")
    art['article_id'] = art['article_id'].astype(str)

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.decomposition import TruncatedSVD
    import pickle

    cat_features = ['product_type_name','colour_group_name','perceived_colour_master_name',
                    'graphical_appearance_name','garment_group_name',
                    'index_group_name','product_group_name']
    enc  = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_oh = enc.fit_transform(art[cat_features].fillna('Unknown'))
    svd  = TruncatedSVD(n_components=64, random_state=42)
    X_64 = normalize(svd.fit_transform(X_oh), norm='l2').astype(np.float32)
    # Pad metadata to 512 (zeros mark as "no image" for downstream models)
    DIM     = 512
    X_meta  = np.hstack([X_64, np.zeros((len(X_64), DIM-64), dtype=np.float32)])
    aids    = art['article_id'].values
    final   = np.zeros((len(aids), DIM), dtype=np.float32)
    has_clip= []
    for i, aid in enumerate(aids):
        if aid in clip_embs:
            final[i] = clip_embs[aid]; has_clip.append(aid)
        else:
            final[i] = X_meta[i]
    final = normalize(final, norm='l2')
    coverage = len(has_clip)/len(aids)*100
    print(f"  CLIP coverage: {len(has_clip):,}/{len(aids):,} ({coverage:.1f}%)")

    # Save
    np.save("models/visual_features.npy", final.astype(np.float32))
    np.save("models/visual_article_ids.npy", aids)
    print("  Saved visual_features.npy (512-dim) ✓")

    # Rebuild FAISS
    print("\n[3/4] Rebuilding FAISS index (512-dim)...")
    nlist = int(np.sqrt(len(final)))
    q     = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFFlat(q, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(final); index.add(final); index.nprobe = 20
    faiss.write_index(index, "models/visual_faiss.index")
    print(f"  FAISS index rebuilt: {index.ntotal:,} vectors ✓")

    # Rebuild outfit pairs
    print("\n[4/4] Rebuilding outfit pairs with CLIP similarity...")
    index.nprobe = 50
    a2i       = {a:i for i,a in enumerate(aids)}
    lower_set = set(art[art['product_group_name']=='Garment Lower body']['article_id'].values)
    records   = []
    for uid in art[art['product_group_name']=='Garment Upper body']['article_id'].values:
        if uid not in a2i: continue
        D, I = index.search(final[a2i[uid]:a2i[uid]+1], 200)
        for sim, i in zip(D[0], I[0]):
            if i<0 or i>=len(aids): continue
            cand = aids[i]
            if cand in lower_set and cand != uid:
                records.append({'upper_article_id':uid,'lower_article_id':cand,
                                'compatibility_score':float(sim)})
                break
    pd.DataFrame(records).to_parquet('data/features/outfit_pairs.parquet', index=False)
    json.dump({'clip_coverage_pct':round(coverage,1),'n_clip':len(has_clip),
               'n_total':len(aids),'dim':DIM},
              open('models/clip_coverage.json','w'), indent=2)
    print(f"  {len(records):,} outfit pairs rebuilt ✓")

    print(f"\nPhase 3B complete ✓  CLIP coverage: {coverage:.1f}%")
    print("Re-run Phase 5 (reranker.py) to update visual_sim feature scores.")


if __name__ == "__main__":
    run()