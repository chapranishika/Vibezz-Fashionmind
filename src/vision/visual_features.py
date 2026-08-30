"""
FashionMind — Phase 3A: Visual Features + FAISS (metadata-based)
=================================================================
Runs without images. Builds 64-dim SVD embeddings from article
metadata — product type, colour, garment group, appearance.
Covers 100% of catalogue. CLIP replaces this on AWS EC2.

Outputs : models/visual_features.npy
          models/visual_article_ids.npy
          models/visual_faiss.index
          models/visual_encoder.pkl
          data/features/outfit_pairs.parquet
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.decomposition import TruncatedSVD
warnings.filterwarnings('ignore')
os.makedirs('models', exist_ok=True)


def run():
    print("="*60)
    print("  FashionMind — Phase 3A: Visual Features + FAISS")
    print("="*60)

    print("\n[1/5] Loading articles...")
    art = pd.read_parquet('data/features/article_content_features.parquet')
    art['article_id'] = art['article_id'].astype(str)
    print(f"  {len(art):,} articles")

    print("[2/5] Building visual embeddings (SVD on metadata)...")
    cat_features = [
        'product_type_name', 'colour_group_name', 'perceived_colour_master_name',
        'graphical_appearance_name', 'garment_group_name',
        'index_group_name', 'product_group_name',
    ]
    enc   = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_oh  = enc.fit_transform(art[cat_features].fillna('Unknown'))
    svd   = TruncatedSVD(n_components=64, random_state=42)
    X_64  = svd.fit_transform(X_oh)
    X_norm= normalize(X_64, norm='l2').astype(np.float32)
    expl  = svd.explained_variance_ratio_.sum()
    print(f"  Shape: {X_norm.shape} | Variance explained: {expl:.2%}")

    pickle.dump({'enc':enc,'svd':svd}, open('models/visual_encoder.pkl','wb'))
    np.save('models/visual_features.npy', X_norm)
    np.save('models/visual_article_ids.npy', art['article_id'].values)
    print("  Saved visual_features.npy + visual_article_ids.npy")

    print("[3/5] Building FAISS IVFFlat index...")
    d     = 64
    nlist = int(np.sqrt(len(X_norm)))
    q     = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(X_norm); index.add(X_norm); index.nprobe = 20
    faiss.write_index(index, 'models/visual_faiss.index')
    print(f"  IVFFlat index: {index.ntotal:,} vectors | nlist={nlist}")

    print("[4/5] Verifying similarity search...")
    art_lu = art.set_index('article_id')
    aids   = art['article_id'].values
    a2i    = {a:i for i,a in enumerate(aids)}
    for ptype in ['Dress', 'Sweater', 'Trousers']:
        sample = art[art['product_type_name']==ptype]
        if sample.empty: continue
        aid = sample.iloc[0]['article_id']
        idx = a2i[aid]
        D, I = index.search(X_norm[idx:idx+1], 4)
        results = [art_lu.loc[aids[i],'product_type_name']
                   for i in I[0][1:] if i>=0 and aids[i] in art_lu.index]
        print(f"  Query '{ptype}' → {results[:3]}")

    print("[5/5] Building outfit compatibility pairs...")
    lower_set = set(art[art['product_group_name']=='Garment Lower body']['article_id'].values)
    upper_ids = art[art['product_group_name']=='Garment Upper body']['article_id'].values
    index.nprobe = 50
    records = []
    for uid in upper_ids:
        if uid not in a2i: continue
        D, I = index.search(X_norm[a2i[uid]:a2i[uid]+1], 200)
        for sim, i in zip(D[0], I[0]):
            if i<0 or i>=len(aids): continue
            cand = aids[i]
            if cand in lower_set and cand != uid:
                records.append({'upper_article_id':uid,'lower_article_id':cand,
                                'compatibility_score':float(sim)})
                break
    pd.DataFrame(records).to_parquet('data/features/outfit_pairs.parquet', index=False)
    print(f"  {len(records):,} outfit pairs saved ✓")

    print("\nPhase 3A complete ✓  (run clip_embeddings.py on EC2 for CLIP upgrade)")


if __name__ == "__main__":
    run()