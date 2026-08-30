"""
FashionMind — Phase 5: LightGBM Re-ranker + SHAP + A/B Testing
===============================================================
Fuses 13 signals from all previous phases into a single ranked list.
Evaluates all models end-to-end and runs A/B test with hypothesis testing.

Outputs : models/reranker.pkl
          data/features/reranker_features.parquet
          data/features/final_recommendations.parquet
          data/features/shap_explanations.parquet
          data/features/final_metrics.csv
          data/features/ab_test_results.csv
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import faiss, shap
import scipy.sparse as sp
from scipy import stats
warnings.filterwarnings('ignore')
os.makedirs('models', exist_ok=True)

FEAT_COLS = ['als_score','visual_sim','nlp_sim','trend_score','price_affinity',
             'category_match','popularity_score','rank_norm','engagement_score',
             'age_norm','ptype_idx','colour_idx','garment_idx']

FEAT_LABELS = {
    'als_score':       'Personalised match',
    'visual_sim':      'Visual style match',
    'nlp_sim':         'Description similarity',
    'trend_score':     'Trending this week',
    'price_affinity':  'Matches price range',
    'category_match':  'Favourite category',
    'popularity_score':'Popular item',
    'rank_norm':       'High CF rank',
    'engagement_score':'Engagement level',
    'age_norm':        'Age relevance',
    'ptype_idx':       'Product type',
    'colour_idx':      'Colour group',
    'garment_idx':     'Garment type',
}


def recall_at_k(rec, actual, k=10):
    return len(set(rec[:k])&set(actual))/min(len(actual),k) if actual else 0.0
def ndcg_at_k(rec, actual, k=10):
    dcg =sum(1/np.log2(i+2) for i,r in enumerate(rec[:k]) if r in set(actual))
    idcg=sum(1/np.log2(i+2) for i in range(min(len(actual),k)))
    return dcg/idcg if idcg>0 else 0.0
def map_at_k(rec, actual, k=12):
    s=set(actual); h=sc=0.0
    for i,r in enumerate(rec[:k]):
        if r in s: h+=1; sc+=h/(i+1)
    return sc/min(len(actual),k) if actual else 0.0


def run():
    print("="*60)
    print("  FashionMind — Phase 5: Re-ranker + SHAP + A/B Test")
    print("="*60)

    # ── Load all pre-computed resources ───────────────────────────
    print("\n[1/8] Loading models and features...")
    als    = pickle.load(open('models/als_model.pkl','rb'))
    matrix = sp.load_npz('models/user_item_matrix.npz')
    ue     = pickle.load(open('models/user_encoder.pkl','rb'))
    ie     = pickle.load(open('models/item_encoder.pkl','rb'))
    cid2u  = ue['dec']; u2cid=ue['enc']; i2aid=ie['enc']

    vis_feats= np.load('models/visual_features.npy')
    vis_aids = np.load('models/visual_article_ids.npy', allow_pickle=True)
    nlp_feats= np.load('models/nlp_features.npy')
    nlp_aids = np.load('models/nlp_article_ids.npy', allow_pickle=True)
    vis_a2i  = {a:i for i,a in enumerate(vis_aids)}
    nlp_a2i  = {a:i for i,a in enumerate(nlp_aids)}

    art    = pd.read_parquet('data/features/article_content_features.parquet')
    art['article_id'] = art['article_id'].astype(str)
    art_lu = art.set_index('article_id')

    pop    = pd.read_parquet('data/features/cold_start_popular.parquet')
    pop_s  = pop.set_index('article_id')['score'].to_dict()
    pop12  = pop.head(12)['article_id'].tolist()

    trend  = pd.read_parquet('data/features/trend_scores.parquet')
    lw     = trend['week'].max()
    lt_map = trend[trend.week==lw].set_index('product_type_name')['trend_score'].to_dict()

    cust   = pd.read_parquet('data/features/customer_segments.parquet').set_index('customer_id')
    u_price= pd.read_parquet('data/features/user_avg_price.parquet').set_index('customer_id')['avg_price'].to_dict()
    a_price= pd.read_parquet('data/features/art_avg_price.parquet').set_index('article_id')['avg_price'].to_dict()
    u_hist = pickle.load(open('data/features/user_history.pkl','rb'))

    gt_df  = pd.read_parquet('data/features/ground_truth.parquet')
    ground_truth = {r.customer_id: list(r.article_id) for _,r in gt_df.iterrows()}

    u_top_ptype = {}
    for cid, items in u_hist.items():
        pt = [art_lu.loc[a,'product_type_name'] for a in items if a in art_lu.index]
        if pt: u_top_ptype[cid] = max(set(pt), key=pt.count)

    print(f"  Ground truth: {len(ground_truth):,} users | Matrix: {matrix.shape}")

    # ── Build re-ranker feature matrix ────────────────────────────
    # FIX: Split eligible users into non-overlapping train and held-out test pools
    # before building any features. The test pool is NEVER touched during training.
    N_TRAIN, N_TEST, N_CANDS = 3000, 500, 50  # 3k train users for meaningful LambdaRank signal
    print(f"\n[2/8] Building feature matrix "
          f"({N_TRAIN} train users + {N_TEST} held-out test users × {N_CANDS} candidates)...")
    eligible = [c for c in list(ground_truth.keys()) if c in cid2u]
    train_cids = eligible[:N_TRAIN]           # used for LambdaRank training
    test_cids  = eligible[N_TRAIN:N_TRAIN+N_TEST]  # held-out, never seen during training
    eval_cids  = train_cids  # feature matrix built only from train split

    rows = []
    for cid in eval_cids:
        uidx = int(cid2u[cid])
        try: ids, als_scores = als.recommend(uidx, matrix[uidx], N=N_CANDS,
                                              filter_already_liked_items=True)
        except: continue

        up = u_price.get(cid, 0.025)
        ut = u_top_ptype.get(cid, '')
        ue_val = float(cust.loc[cid,'engagement_score']) if cid in cust.index else 0.5
        ua = float(cust.loc[cid,'age_norm']) if cid in cust.index else 0.3

        items = u_hist.get(cid,[])[:20]
        vi=[vis_a2i[a] for a in items if a in vis_a2i]
        ni=[nlp_a2i[a] for a in items if a in nlp_a2i]
        vc=vis_feats[vi].mean(0) if vi else np.zeros(vis_feats.shape[1])
        nc=nlp_feats[ni].mean(0) if ni else np.zeros(64)
        vc/=(np.linalg.norm(vc)+1e-8); nc/=(np.linalg.norm(nc)+1e-8)

        for rank,(iidx,alsc) in enumerate(zip(ids,als_scores)):
            aid=i2aid[int(iidx)]; aids=aid.lstrip('0')
            if aids in art_lu.index:
                r=art_lu.loc[aids]
                ptype=str(r['product_type_name']); pi=int(r['product_type_name_idx'])
                ci=int(r['colour_group_name_idx']); gi=int(r['garment_group_name_idx'])
            else: ptype=''; pi=ci=gi=0
            vv=vis_feats[vis_a2i[aids]] if aids in vis_a2i else np.zeros(vis_feats.shape[1])
            nv=nlp_feats[nlp_a2i[aids]] if aids in nlp_a2i else np.zeros(64)
            ip=a_price.get(aids,0.025)
            label=1 if cid in ground_truth and aids in ground_truth[cid] else 0
            rows.append([cid,aids,label,
                float(alsc), float(np.dot(vv,vc)), float(np.dot(nv,nc)),
                lt_map.get(ptype,0.5),
                float(max(0,1-abs(ip-up)/(up+1e-6))),
                int(ptype==ut), float(pop_s.get(aids,0.0)),
                1-rank/N_CANDS, ue_val, ua, pi, ci, gi])

    df = pd.DataFrame(rows, columns=['customer_id','article_id','label']+FEAT_COLS)
    df.to_parquet('data/features/reranker_features.parquet', index=False)
    print(f"  {len(df):,} rows | {int(df.label.sum())} positives ({df.label.mean()*100:.2f}%)")

    # ── Train LambdaRank ──────────────────────────────────────────
    print("\n[3/8] Training LightGBM LambdaRank...")
    users = df.customer_id.unique()
    sp_   = int(len(users)*0.8)
    tr = df[df.customer_id.isin(users[:sp_])]
    va = df[df.customer_id.isin(users[sp_:])]

    ds_tr = lgb.Dataset(tr[FEAT_COLS].values, label=tr.label.values,
                        group=tr.groupby('customer_id').size().values,
                        feature_name=FEAT_COLS)
    ds_va = lgb.Dataset(va[FEAT_COLS].values, label=va.label.values,
                        group=va.groupby('customer_id').size().values, reference=ds_tr)

    params = {'objective':'lambdarank','metric':'ndcg','ndcg_eval_at':[10,12],
              'learning_rate':0.05,'num_leaves':31,'max_depth':6,
              'min_child_samples':3,'subsample':0.8,'colsample_bytree':0.8,
              'verbose':-1,'random_state':42}
    reranker = lgb.train(params, ds_tr, num_boost_round=300, valid_sets=[ds_va],
                         callbacks=[lgb.early_stopping(30,verbose=False),
                                    lgb.log_evaluation(-1)])
    pickle.dump(reranker, open('models/reranker.pkl','wb'))
    ndcg10 = reranker.best_score['valid_0'].get('ndcg@10',0)
    print(f"  Best iter: {reranker.best_iteration} | NDCG@10: {ndcg10:.4f}")

    # ── Evaluate all models on HELD-OUT test_cids ────────────────
    # test_cids were never seen during LambdaRank training — no leakage.
    # We also keep a per-user hit vector (1 = at least one ground-truth item in
    # the top-12) so the pipeline-vs-ALS comparison below is *paired* on the
    # same users rather than an unpaired split of the training set.
    print(f"\n[4/8] Evaluating all models on {len(test_cids)} held-out test users...")
    results = {}
    hit_pop, hit_als, hit_pipe = {}, {}, {}   # cid -> 0/1

    r10,n10,m12=[],[],[]
    for cid in test_cids:
        act=ground_truth.get(cid,[])
        r10.append(recall_at_k(pop12,act)); n10.append(ndcg_at_k(pop12,act))
        m12.append(map_at_k(pop12,act))
        if act: hit_pop[cid]=1 if set(pop12[:12])&set(act) else 0
    results['Popularity baseline']={'recall@10':np.mean(r10),'ndcg@10':np.mean(n10),'map@12':np.mean(m12)}

    r10,n10,m12=[],[],[]
    for cid in test_cids:
        uidx=int(cid2u[cid]) if cid in cid2u else None
        if uidx is not None:
            try:
                ids,_=als.recommend(uidx,matrix[uidx],N=12,filter_already_liked_items=True)
                rec=[i2aid[int(i)] for i in ids]
            except: rec=pop12
        else: rec=pop12
        act=ground_truth.get(cid,[])
        r10.append(recall_at_k(rec,act)); n10.append(ndcg_at_k(rec,act))
        m12.append(map_at_k(rec,act))
        if act: hit_als[cid]=1 if set(rec[:12])&set(act) else 0
    results['ALS (Phase 2)']={'recall@10':np.mean(r10),'ndcg@10':np.mean(n10),'map@12':np.mean(m12)}

    # Score test users with the trained reranker (no leakage)
    r10,n10,m12=[],[],[]
    for cid in test_cids:
        uidx=int(cid2u[cid]) if cid in cid2u else None
        if uidx is None: continue
        try: ids,als_scores=als.recommend(uidx,matrix[uidx],N=N_CANDS,filter_already_liked_items=True)
        except: continue
        up=u_price.get(cid,0.025); ut=u_top_ptype.get(cid,'')
        ue_val=float(cust.loc[cid,'engagement_score']) if cid in cust.index else 0.5
        ua=float(cust.loc[cid,'age_norm']) if cid in cust.index else 0.3
        items=u_hist.get(cid,[])[:20]
        vi=[vis_a2i[a] for a in items if a in vis_a2i]
        ni=[nlp_a2i[a] for a in items if a in nlp_a2i]
        vc=vis_feats[vi].mean(0) if vi else np.zeros(vis_feats.shape[1])
        nc=nlp_feats[ni].mean(0) if ni else np.zeros(64)
        vc/=(np.linalg.norm(vc)+1e-8); nc/=(np.linalg.norm(nc)+1e-8)
        test_rows=[]
        for rank,(iidx,alsc) in enumerate(zip(ids,als_scores)):
            aid=i2aid[int(iidx)]; aids=aid.lstrip('0')
            if aids in art_lu.index:
                r=art_lu.loc[aids]
                ptype=str(r['product_type_name']); pi=int(r['product_type_name_idx'])
                ci=int(r['colour_group_name_idx']); gi=int(r['garment_group_name_idx'])
            else: ptype=''; pi=ci=gi=0
            vv=vis_feats[vis_a2i[aids]] if aids in vis_a2i else np.zeros(vis_feats.shape[1])
            nv=nlp_feats[nlp_a2i[aids]] if aids in nlp_a2i else np.zeros(64)
            ip=a_price.get(aids,0.025)
            test_rows.append([float(alsc),float(np.dot(vv,vc)),float(np.dot(nv,nc)),
                lt_map.get(ptype,0.5),float(max(0,1-abs(ip-up)/(up+1e-6))),
                int(ptype==ut),float(pop_s.get(aids,0.0)),1-rank/N_CANDS,ue_val,ua,pi,ci,gi])
        if not test_rows: continue
        sc=reranker.predict(np.array(test_rows))
        rec=[i2aid[int(ids[j])] for j in np.argsort(-sc)]
        act=ground_truth.get(cid,[])
        r10.append(recall_at_k(rec,act)); n10.append(ndcg_at_k(rec,act))
        m12.append(map_at_k(rec,act))
        if act: hit_pipe[cid]=1 if set(rec[:12])&set(act) else 0
    results['Full pipeline (Phase 5)']={'recall@10':np.mean(r10) if r10 else 0,
        'ndcg@10':np.mean(n10) if n10 else 0,'map@12':np.mean(m12) if m12 else 0}

    pd.DataFrame([{'model':k,**v} for k,v in results.items()])\
      .to_csv('data/features/final_metrics.csv', index=False)

    base_r = results['Popularity baseline']['recall@10']
    print("\n  ┌──────────────────────────┬────────────┬──────────┬──────────┐")
    print("  │ Model                    │ Recall@10  │ NDCG@10  │ MAP@12   │")
    print("  ├──────────────────────────┼────────────┼──────────┼──────────┤")
    for name,r in results.items():
        lift=f" (+{(r['recall@10']/base_r-1)*100:.0f}%)" if name!='Popularity baseline' and base_r>0 else "       "
        print(f"  │ {name:<24s}  │  {r['recall@10']:.4f}{lift:<8s}│  {r['ndcg@10']:.4f}  │  {r['map@12']:.4f}  │")
    print("  └──────────────────────────┴────────────┴──────────┴──────────┘")

    # ── SHAP ─────────────────────────────────────────────────────
    print("\n[5/8] SHAP explanations...")
    explainer = shap.TreeExplainer(reranker)
    X_sh = df.head(200)[FEAT_COLS].values
    sv   = explainer.shap_values(X_sh)

    shap_rows = []
    for i in range(min(100, len(df))):
        row=df.iloc[i]; s=sv[i]
        top3=np.argsort(np.abs(s))[::-1][:3]
        reasons=[f"{'↑' if s[f]>0 else '↓'} {FEAT_LABELS[FEAT_COLS[f]]}" for f in top3]
        shap_rows.append({'customer_id':row.customer_id,'article_id':row.article_id,
            'reason_1':reasons[0],'reason_2':reasons[1] if len(reasons)>1 else '',
            'reason_3':reasons[2] if len(reasons)>2 else '','shap_sum':float(s.sum())})
    pd.DataFrame(shap_rows).to_parquet('data/features/shap_explanations.parquet', index=False)

    gi = np.abs(sv).mean(0)
    fi = pd.Series(gi, index=FEAT_COLS).sort_values(ascending=False)
    print("\n  SHAP global importance (top 8):")
    for feat,val in fi.head(8).items():
        print(f"    {FEAT_LABELS[feat]:<28s} {val:.4f}  {'█'*int(val*400)}")

    # ── Paired offline comparison: ALS vs full pipeline ──────────
    # This is NOT a live A/B test (no traffic). It is a paired comparison of
    # the two rankers on the SAME held-out users, using per-user hit@12.
    #   · McNemar exact test on the discordant pairs (correct test for paired
    #     binary outcomes)
    #   · 10k-resample paired bootstrap 95% CI on the hit-rate difference
    print("\n[6/8] Paired comparison: ALS vs full pipeline (held-out users)...")
    paired = [c for c in test_cids if c in hit_als and c in hit_pipe]
    a = np.array([hit_als[c]  for c in paired], dtype=int)
    b = np.array([hit_pipe[c] for c in paired], dtype=int)
    n = len(paired)
    ca, cb = float(a.mean()) if n else 0.0, float(b.mean()) if n else 0.0
    diff = cb - ca

    b01 = int(np.sum((a == 0) & (b == 1)))   # pipeline wins
    b10 = int(np.sum((a == 1) & (b == 0)))   # ALS wins
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        p_val = float(mcnemar([[0, b01], [b10, 0]], exact=True).pvalue)
    except Exception:
        # exact binomial fallback: discordants ~ Binom(b01+b10, 0.5)
        k, m_ = min(b01, b10), b01 + b10
        p_val = float(min(1.0, 2 * stats.binom.cdf(k, m_, 0.5))) if m_ else 1.0

    rng = np.random.default_rng(42)
    if n:
        idx = rng.integers(0, n, size=(10_000, n))
        boot = b[idx].mean(1) - a[idx].mean(1)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    else:
        ci_lo = ci_hi = 0.0
    lift_str = f"{diff/ca*100:+.1f}%" if ca >= 0.005 else "N/A (ALS hit-rate too low for a stable ratio)"
    sig = "significant" if p_val < 0.05 else "not significant"

    pd.DataFrame([{'model_a':'ALS','model_b':'Full pipeline',
        'ctr_a':round(ca,4),'ctr_b':round(cb,4),
        'abs_diff':round(diff,4),
        'lift_pct': round(diff/ca*100,2) if ca>=0.005 else None,
        'p_value':round(p_val,4),'significant':bool(p_val<0.05),
        'ci_lo':round(float(ci_lo),4),'ci_hi':round(float(ci_hi),4),
        'n_users':n,'n_pipeline_wins':b01,'n_als_wins':b10,
        'test':'McNemar exact; 95% CI = 10k paired bootstrap'}
    ]).to_csv('data/features/ab_test_results.csv', index=False)
    # ── Also persist to Supabase if configured ────────────────────
    try:
        import os
        if os.getenv('SUPABASE_SERVICE_KEY'):
            import sys; sys.path.insert(0, '.')
            from api.db import save_ab_result
            save_ab_result({
                'model_a': 'ALS', 'model_b': 'Full pipeline',
                'ctr_a': round(ca,4), 'ctr_b': round(cb,4),
                'lift_pct': round(diff/ca*100, 2) if ca >= 0.005 else None,
                'p_value': round(p_val,4), 'significant': bool(p_val<0.05),
                'ci_lo': round(float(ci_lo),4), 'ci_hi': round(float(ci_hi),4),
                'n_users_a': n, 'n_users_b': n,
            })
            print("  Comparison result saved to Supabase ✓")
    except Exception as e:
        print(f"  (Supabase comparison log skipped: {e})")

    print(f"  Paired users: {n}   (pipeline wins {b01}, ALS wins {b10}, ties {n-b01-b10})")
    print(f"  ALS hit@12:      {ca:.4f}")
    print(f"  Pipeline hit@12: {cb:.4f}")
    print(f"  Abs difference:  {diff:+.4f}   relative: {lift_str}")
    print(f"  McNemar exact p={p_val:.4f}  ({sig})   95% CI on diff: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── Save final recommendations ─────────────────────────────────
    print("\n[7/8] Saving final recommendations...")
    final=[]
    for cid,grp in df.groupby('customer_id'):
        sc=reranker.predict(grp[FEAT_COLS].values)
        for rank,aid in enumerate(grp.assign(s=sc).sort_values('s',ascending=False)\
                                     .head(12)['article_id']):
            final.append({'customer_id':cid,'article_id':aid,'rank':rank})
    pd.DataFrame(final).to_parquet('data/features/final_recommendations.parquet', index=False)
    print(f"  {len(final):,} recommendations saved ✓")

    print("\nPhase 5 complete ✓")


if __name__ == "__main__":
    run()