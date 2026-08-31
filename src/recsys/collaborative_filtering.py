"""
FashionMind — Phase 2: Collaborative Filtering
===============================================
Models  : ALS (baseline), BPR, Cold-start popularity
Outputs : models/als_model.pkl
          models/bpr_model.pkl
          models/user_item_matrix.npz
          models/user_encoder.pkl
          models/item_encoder.pkl
          data/features/cf_candidates.parquet
          data/features/cold_start_popular.parquet
          data/features/cf_results.csv
          data/features/ground_truth.parquet
          data/features/user_avg_price.parquet
          data/features/art_avg_price.parquet
          data/features/user_history.pkl
"""

import os, gc, pickle, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')   # cap BLAS scratch memory
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import scipy.sparse as sp
warnings.filterwarnings('ignore')
os.makedirs('models', exist_ok=True)
os.makedirs('data/features', exist_ok=True)


def recall_at_k(rec, actual, k=10):
    return len(set(rec[:k]) & set(actual)) / min(len(actual), k) if actual else 0.0

def ndcg_at_k(rec, actual, k=10):
    dcg  = sum(1/np.log2(i+2) for i,r in enumerate(rec[:k]) if r in set(actual))
    idcg = sum(1/np.log2(i+2) for i in range(min(len(actual),k)))
    return dcg/idcg if idcg>0 else 0.0

def map_at_k(rec, actual, k=12):
    act=set(actual); h=s=0.0
    for i,r in enumerate(rec[:k]):
        if r in act: h+=1; s+=h/(i+1)
    return s/min(len(actual),k) if actual else 0.0


def run():
    print("="*60)
    print("  FashionMind — Phase 2: Collaborative Filtering")
    print("="*60)

    # ── Load transactions in chunks (memory-safe) ─────────────────
    # Training window: a recent multi-week slice of the 2-year corpus.
    #   train  = WINDOW_START .. SPLIT_DATE   (interactions used to fit ALS/BPR)
    #   test   = SPLIT_DATE   .. WINDOW_END   (future holdout — never fit on)
    # The split is strictly temporal, so evaluation is a true "predict the
    # next weeks" task rather than a random hold-out.
    # ~4.5-month window — as long as this 4 GB-free box can hold in memory
    # alongside the ALS factor matrices.
    WINDOW_START, SPLIT_DATE, WINDOW_END = '2020-05-01', '2020-09-08', '2020-09-22'
    print(f"\n[1/7] Loading transactions (chunked) — window {WINDOW_START} .. {WINDOW_END}, "
          f"train/test split at {SPLIT_DATE}...")
    CUTOFF  = pd.Timestamp(SPLIT_DATE)

    parts, n_scanned = [], 0
    for chunk in pd.read_csv('data/raw/transactions_train.csv',
                             usecols=['t_dat','customer_id','article_id','price'],
                             dtype={'article_id':str,'customer_id':str,'price':float},
                             chunksize=1_000_000):
        n_scanned += len(chunk)
        chunk['t_dat'] = pd.to_datetime(chunk['t_dat'])
        chunk = chunk[(chunk['t_dat'] >= WINDOW_START) & (chunk['t_dat'] <= WINDOW_END)]
        if chunk.empty:
            print(f"  scanned {n_scanned:,} rows...", end='\r'); continue
        chunk['article_id'] = chunk['article_id'].str.lstrip('0')
        parts.append(chunk.dropna())
        print(f"  scanned {n_scanned:,} rows | kept {sum(len(p) for p in parts):,}...", end='\r')

    tx = pd.concat(parts, ignore_index=True); del parts
    n_total = len(tx)
    print(f"\n  {n_total:,} transactions in window "
          f"({tx.customer_id.nunique():,} customers, {tx.article_id.nunique():,} articles)")

    train_tx = tx[tx.t_dat <= CUTOFF]
    valid_tx = tx[tx.t_dat >  CUTOFF]

    # Vectorised aggregation. Keep the last 50 items per user (an 8-month window
    # means heavy shoppers have far more than 20 purchases — more history is
    # more CF signal).
    HIST_N = 50
    user_price_acc = train_tx.groupby('customer_id')['price'].apply(list).to_dict()
    user_items_acc = (train_tx.sort_values('t_dat')
                              .groupby('customer_id')['article_id']
                              .apply(lambda s: s.tolist()[-HIST_N:]).to_dict())
    art_price_acc  = tx.groupby('article_id')['price'].apply(list).to_dict()
    valid_rows     = [valid_tx[['customer_id', 'article_id']].copy()]
    cs_tx          = train_tx[['customer_id', 'article_id']].copy()   # for cold-start [5/7]

    print(f"  train: {len(train_tx):,} rows / {len(user_items_acc):,} users | "
          f"holdout: {len(valid_tx):,} rows")
    del tx, train_tx, valid_tx; gc.collect()
    try:
        import json as _j
        _j.dump({"training_window": {"train": f"{WINDOW_START} .. {SPLIT_DATE}",
                                     "holdout": f"{SPLIT_DATE} .. {WINDOW_END}",
                                     "months": round((pd.Timestamp(SPLIT_DATE)-pd.Timestamp(WINDOW_START)).days/30.4, 1)},
                 "window_transactions": int(n_total)},
                open('data/features/model_card.json', 'w'), indent=2)
    except Exception:
        pass

    # Save pre-computed summaries (avoids re-loading CSV in later phases)
    user_avg_price = {c: np.mean(p) for c,p in user_price_acc.items()}
    art_avg_price  = {a: np.mean(p) for a,p in art_price_acc.items()}

    pd.DataFrame({'customer_id': list(user_avg_price.keys()),
                  'avg_price':   list(user_avg_price.values())}
                ).to_parquet('data/features/user_avg_price.parquet', index=False)
    pd.DataFrame({'article_id': list(art_avg_price.keys()),
                  'avg_price':  list(art_avg_price.values())}
                ).to_parquet('data/features/art_avg_price.parquet', index=False)
    pickle.dump(user_items_acc, open('data/features/user_history.pkl','wb'))

    valid_df = pd.concat(valid_rows, ignore_index=True)
    ground_truth = {}
    for row in valid_df.itertuples(index=False):
        ground_truth.setdefault(row.customer_id, []).append(row.article_id)
    gt = pd.DataFrame({
        'customer_id': list(ground_truth.keys()),
        'article_id': list(ground_truth.values())
    })
    gt.to_parquet('data/features/ground_truth.parquet', index=False)
    print(f"  Ground truth: {len(ground_truth):,} validation users")

    # ── Build interaction matrix (active users only) ───────────────
    print("\n[2/7] Building user-item interaction matrix...")
    keys, values = [], []
    for cid, items in user_items_acc.items():
        for aid in items:
            keys.append(cid)
            values.append(aid)
    train_df = pd.DataFrame({'customer_id': keys, 'article_id': values})
    del keys, values; gc.collect()
    active = train_df.groupby('customer_id').size()
    train_df = train_df[train_df.customer_id.isin(active[active >= 5].index)]
    # drop long-tail items (bought < 3× in the window) — trims the matrix width
    icnt = train_df.groupby('article_id').size()
    train_df = train_df[train_df.article_id.isin(icnt[icnt >= 3].index)]

    train_df['user_idx'] = train_df['customer_id'].astype('category').cat.codes
    train_df['item_idx'] = train_df['article_id'].astype('category').cat.codes

    user_cats = train_df['customer_id'].astype('category').cat.categories
    item_cats = train_df['article_id'].astype('category').cat.categories
    user_enc  = dict(enumerate(user_cats))
    item_enc  = dict(enumerate(item_cats))
    user_dec  = {v:k for k,v in user_enc.items()}

    grp = train_df.groupby(['user_idx','item_idx']).size().reset_index(name='cnt')
    n_u, n_i = len(user_enc), len(item_enc)
    matrix = sp.csr_matrix(
        (grp['cnt'].values.astype(np.float32),
         (grp['user_idx'].values, grp['item_idx'].values)),
        shape=(n_u, n_i))

    sp.save_npz('models/user_item_matrix.npz', matrix)
    pickle.dump({'enc':user_enc,'dec':user_dec}, open('models/user_encoder.pkl','wb'))
    pickle.dump({'enc':item_enc,'dec':{v:k for k,v in item_enc.items()}},
                open('models/item_encoder.pkl','wb'))
    print(f"  Matrix: {n_u:,} users × {n_i:,} items | density: {matrix.nnz/(n_u*n_i)*100:.4f}%")
    del train_df, grp, active, icnt; gc.collect()

    # ── Train ALS ─────────────────────────────────────────────────
    print("\n[3/7] Training ALS...")
    import implicit
    als = implicit.cpu.als.AlternatingLeastSquares(
        factors=64, regularization=0.01, iterations=150, random_state=42)
    als.fit(matrix)
    pickle.dump(als, open('models/als_model.pkl','wb'))
    print("  ALS done ✓")

    # ── Train BPR ─────────────────────────────────────────────────
    # Reference model only — ALS is what feeds the Phase-5 re-ranker. 20
    # iterations left BPR badly undertrained; 120 is enough to converge on a
    # matrix this size.
    print("[4/7] Training BPR...")
    bpr = implicit.cpu.bpr.BayesianPersonalizedRanking(
        factors=64, learning_rate=0.01, regularization=0.01,
        iterations=120, random_state=42)
    bpr.fit(matrix)
    pickle.dump(bpr, open('models/bpr_model.pkl','wb'))
    print("  BPR done ✓")

    # ── Cold-start: personalised by age-bucket × club-membership ─
    # One popularity list per (age_bucket × club) segment so cold-start users
    # still get age/membership-appropriate picks, plus a global fallback.
    # Fully vectorised (a Python loop over 500k users here would take hours).
    print("[5/7] Building segmented cold-start model...")
    cust_raw = pd.read_csv('data/raw/customers.csv',
                           usecols=['customer_id','age','club_member_status'],
                           dtype={'customer_id':str})
    cust_raw['age_bucket'] = pd.cut(cust_raw['age'], bins=[0,25,35,50,200],
                                    labels=['16-25','26-35','36-50','50+']).astype('object')
    cust_raw['club'] = cust_raw['club_member_status'].fillna('NONE').str.upper()
    cust_raw['seg'] = cust_raw['age_bucket'].astype(str) + '|' + cust_raw['club']

    # map (not merge) the segment onto each transaction row — a 10M-row merge
    # against 1.4M customers OOMs on a small box; a dict .map does not.
    seg_map = cust_raw.set_index('customer_id')['seg'].to_dict()
    seg_tx = cs_tx
    seg_tx['seg'] = seg_tx['customer_id'].map(seg_map)
    seg_tx = seg_tx[seg_tx['seg'].notna() & ~seg_tx['seg'].str.startswith('nan|')]
    seg_pop = seg_tx.groupby(['seg', 'article_id']).size().reset_index(name='count')
    seg_pop[['age_bucket', 'club']] = seg_pop['seg'].str.split('|', n=1, expand=True)
    seg_pop = seg_pop.drop(columns='seg')
    seg_pop['max_c'] = seg_pop.groupby(['age_bucket','club'])['count'].transform('max')
    seg_pop['score'] = seg_pop['count'] / seg_pop['max_c']
    seg_pop['rk'] = seg_pop.groupby(['age_bucket','club'])['count'] \
                           .rank(method='first', ascending=False)
    seg_df = (seg_pop[seg_pop['rk'] <= 500]
              [['age_bucket','club','article_id','score']]
              .reset_index(drop=True))
    seg_df.to_parquet('data/features/cold_start_popular.parquet', index=False)

    glob = cs_tx.groupby('article_id').size().reset_index(name='count') \
                   .sort_values('count', ascending=False)
    glob['score'] = glob['count'] / glob['count'].max()
    pop = glob                      # used by the eval block below (raw string IDs)
    glob.head(500).to_parquet('data/features/cold_start_popular_global.parquet', index=False)
    n_seg = seg_df.groupby(['age_bucket','club']).ngroups
    print(f"  {n_seg} segments · {len(seg_df):,} segment rows saved ✓")

    # ── Evaluate ─────────────────────────────────────────────────
    print("[6/7] Evaluating models on 3,000 users...")
    # FIX: ensure pop article_ids match ground_truth format (raw string IDs, not encoded ints)
    pop_top12 = pop.head(12)['article_id'].tolist()
    # item_enc maps encoded_int → raw_article_id; use reverse map if needed
    if pop_top12 and isinstance(pop_top12[0], (int, np.integer)):
        pop_top12 = [item_enc.get(int(a), str(a)) for a in pop_top12]
    eval_uids = [u for u in list(ground_truth.keys()) if u in user_dec][:3000]

    def evaluate(model, name):
        r10,n10,m12=[],[],[]
        for cid in eval_uids:
            uidx=int(user_dec[cid])
            try:
                ids,_=model.recommend(uidx,matrix[uidx],N=12,filter_already_liked_items=True)
                rec=[item_enc[int(i)] for i in ids]
            except: continue
            act=ground_truth[cid]
            r10.append(recall_at_k(rec,act)); n10.append(ndcg_at_k(rec,act))
            m12.append(map_at_k(rec,act))
        return {'model':name,'recall@10':np.mean(r10),'ndcg@10':np.mean(n10),'map@12':np.mean(m12),'n':len(r10)}

    def eval_pop():
        r10,n10,m12=[],[],[]
        for cid in eval_uids:
            act=ground_truth.get(cid,[])
            r10.append(recall_at_k(pop_top12,act)); n10.append(ndcg_at_k(pop_top12,act))
            m12.append(map_at_k(pop_top12,act))
        return {'model':'Popularity baseline','recall@10':np.mean(r10),'ndcg@10':np.mean(n10),'map@12':np.mean(m12),'n':len(r10)}

    results = [eval_pop(), evaluate(als,'ALS (baseline)'), evaluate(bpr,'BPR')]
    pd.DataFrame(results).to_csv('data/features/cf_results.csv', index=False)

    print("\n  ┌─────────────────────────┬────────────┬──────────┬──────────┐")
    print("  │ Model                   │ Recall@10  │ NDCG@10  │ MAP@12   │")
    print("  ├─────────────────────────┼────────────┼──────────┼──────────┤")
    for r in results:
        print(f"  │ {r['model']:<23s}  │  {r['recall@10']:.4f}    │  {r['ndcg@10']:.4f}  │  {r['map@12']:.4f}  │")
    print("  └─────────────────────────┴────────────┴──────────┴──────────┘")

    # ── Candidate generation (sample dump for inspection) ────────
    # The re-ranker builds its own candidates fresh; this is a small artefact.
    print("\n[7/7] Generating top-200 candidates (3k users)...")
    records = []
    for uidx in range(min(3_000, n_u)):
        cid = user_enc[uidx]
        try:
            ids, scores = als.recommend(uidx, matrix[uidx], N=200,
                                         filter_already_liked_items=True)
            for rank,(iidx,sc) in enumerate(zip(ids, scores)):
                records.append({'customer_id':cid,'article_id':item_enc[int(iidx)],
                                'als_score':float(sc),'rank':rank})
        except: continue
    pd.DataFrame(records).to_parquet('data/features/cf_candidates.parquet', index=False)
    print(f"  {len(records):,} candidate pairs saved ✓")

    print("\nPhase 2 complete ✓")


if __name__ == "__main__":
    run()