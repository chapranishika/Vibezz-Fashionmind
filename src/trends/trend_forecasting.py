"""
FashionMind — Phase 4: NLP + Trend Forecasting
================================================
Part A — TF-IDF + LSA text embeddings (NLP gap)
Part B — LightGBM weekly demand forecast (Trend & Virality JD bullet)
Part C — Google Trends fusion (works on AWS EC2 with open internet)

Outputs : models/nlp_tfidf.pkl
          models/nlp_svd.pkl
          models/nlp_features.npy
          models/nlp_article_ids.npy
          models/nlp_faiss.index
          models/trend_lgbm.pkl
          data/features/trend_scores.parquet
          data/features/text_similarity_sample.parquet
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
warnings.filterwarnings('ignore')
os.makedirs('models', exist_ok=True)


def run():
    print("="*60)
    print("  FashionMind — Phase 4: NLP + Trend Forecasting")
    print("="*60)

    # ── PART A: NLP text embeddings ───────────────────────────────
    print("\n── Part A: NLP Text Embeddings ──────────────────────────")

    print("\n[A1] Loading articles...")
    art = pd.read_parquet('data/features/article_content_features.parquet')
    art['article_id'] = art['article_id'].astype(str)
    art['text'] = (
        art['product_type_name'].fillna('') + ' ' +
        art['product_group_name'].fillna('') + ' ' +
        art['colour_group_name'].fillna('') + ' ' +
        art['garment_group_name'].fillna('') + ' ' +
        art['graphical_appearance_name'].fillna('') + ' ' +
        art['detail_desc'].fillna('')
    ).str.lower().str.replace(r'[^a-z0-9 ]', ' ', regex=True).str.strip()
    print(f"  {len(art):,} articles | avg {art['text'].str.len().mean():.0f} chars/doc")

    print("[A2] TF-IDF vectorisation...")
    tfidf = TfidfVectorizer(max_features=15_000, ngram_range=(1,2),
                            min_df=3, max_df=0.85, sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(art['text'])
    print(f"  Shape: {X_tfidf.shape} | vocab: {len(tfidf.vocabulary_):,}")
    pickle.dump(tfidf, open('models/nlp_tfidf.pkl','wb'))

    print("[A3] LSA → 64-dim embeddings...")
    svd = TruncatedSVD(n_components=64, n_iter=7, random_state=42)
    X_lsa = normalize(svd.fit_transform(X_tfidf), norm='l2').astype(np.float32)
    print(f"  Variance explained: {svd.explained_variance_ratio_.sum():.2%}")
    pickle.dump(svd, open('models/nlp_svd.pkl','wb'))
    np.save('models/nlp_features.npy', X_lsa)
    np.save('models/nlp_article_ids.npy', art['article_id'].values)

    print("[A4] Building NLP FAISS index...")
    nlist = int(np.sqrt(len(X_lsa)))
    q     = faiss.IndexFlatIP(64)
    index = faiss.IndexIVFFlat(q, 64, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(X_lsa); index.add(X_lsa); index.nprobe = 10
    faiss.write_index(index, 'models/nlp_faiss.index')
    print(f"  NLP FAISS index: {index.ntotal:,} vectors")

    print("[A5] Verifying text similarity...")
    aids   = art['article_id'].values
    art_lu = art.set_index('article_id')
    a2i    = {a:i for i,a in enumerate(aids)}
    for ptype in ['Dress', 'Sweater', 'Trousers']:
        sample = art[art['product_type_name']==ptype]
        if sample.empty: continue
        aid = sample.iloc[0]['article_id']
        if aid not in a2i: continue
        D, I = index.search(X_lsa[a2i[aid]:a2i[aid]+1], 4)
        results = [art_lu.loc[aids[i],'product_type_name']
                   for i in I[0][1:] if i>=0 and aids[i] in art_lu.index]
        print(f"  Query '{ptype}' → {results[:3]}")

    # Save sample similarity pairs
    sample_idx = np.random.choice(len(X_lsa), 1000, replace=False)
    D_b, I_b   = index.search(X_lsa[sample_idx], 3)
    sim_records = []
    for qi,(dists,idxs) in enumerate(zip(D_b, I_b)):
        for sim,i in zip(dists[1:],idxs[1:]):
            if 0<=i<len(aids):
                sim_records.append({'article_id_a':aids[sample_idx[qi]],
                                    'article_id_b':aids[i],'text_similarity':float(sim)})
    pd.DataFrame(sim_records).to_parquet('data/features/text_similarity_sample.parquet', index=False)

    # ── PART B: Demand forecasting ─────────────────────────────────
    print("\n\n── Part B: Demand Forecasting (LightGBM) ────────────────")

    # Use the full transaction history up to the forecast horizon so the
    # LightGBM demand model has enough weeks for 12-week lags and seasonality.
    FORECAST_END = '2020-09-22'
    print(f"\n[B1] Loading transactions (chunked) — history through {FORECAST_END}...")
    art_pt = art[['article_id', 'product_type_name']].drop_duplicates('article_id')
    weekly_parts = []
    for chunk in pd.read_csv('data/raw/transactions_train.csv',
                             usecols=['t_dat','article_id','price'],
                             dtype={'article_id':str,'price':float}, chunksize=1_000_000):
        chunk['t_dat'] = pd.to_datetime(chunk['t_dat'])
        chunk = chunk[chunk['t_dat'] <= FORECAST_END]
        if chunk.empty: continue
        chunk['article_id'] = chunk['article_id'].str.lstrip('0')
        chunk['week'] = chunk['t_dat'].dt.to_period('W').dt.start_time
        chunk = chunk.merge(art_pt, on='article_id', how='left')
        g = (chunk.groupby(['week', 'product_type_name'])
                  .agg(sales=('price', 'size'), revenue=('price', 'sum'))
                  .reset_index())
        weekly_parts.append(g)

    weekly = (pd.concat(weekly_parts, ignore_index=True)
                .groupby(['week', 'product_type_name'], as_index=False)
                .agg(sales=('sales', 'sum'), revenue=('revenue', 'sum'))
                .sort_values(['product_type_name', 'week']).reset_index(drop=True))
    print(f"  {len(weekly):,} weekly rows | {weekly.product_type_name.nunique()} types")
    print(f"  Date range: {weekly.week.min().date()} → {weekly.week.max().date()}")

    print("[B2] Feature engineering...")
    weekly['ptype_idx']  = weekly['product_type_name'].astype('category').cat.codes
    weekly['month']      = weekly['week'].dt.month
    weekly['week_num']   = weekly['week'].dt.isocalendar().week.astype(int)
    weekly['quarter']    = weekly['week'].dt.quarter
    weekly['is_holiday'] = weekly['month'].isin([11,12,1]).astype(int)
    weekly['is_summer']  = weekly['month'].isin([6,7,8]).astype(int)

    for lag in [1,2,4,8,12]:
        weekly[f'lag_{lag}w'] = weekly.groupby('product_type_name')['sales'].shift(lag)
        weekly[f'rev_lag_{lag}w'] = weekly.groupby('product_type_name')['revenue'].shift(lag)
    for w in [4,8]:
        weekly[f'roll_mean_{w}w'] = weekly.groupby('product_type_name')['sales']\
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        weekly[f'roll_std_{w}w'] = weekly.groupby('product_type_name')['sales']\
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
    weekly['wow_growth'] = weekly.groupby('product_type_name')['sales']\
        .pct_change().replace([np.inf,-np.inf],0).fillna(0)

    wm = weekly.dropna(subset=['lag_1w','lag_4w']).copy()

    FEAT = ['ptype_idx','month','week_num','quarter','is_holiday','is_summer',
            'lag_1w','lag_2w','lag_4w','lag_8w','lag_12w',
            'rev_lag_1w','rev_lag_4w',
            'roll_mean_4w','roll_std_4w','roll_mean_8w','roll_std_8w','wow_growth']

    cutoff = wm['week'].max() - pd.Timedelta(weeks=4)
    tr = wm[wm.week <= cutoff]; va = wm[wm.week > cutoff]
    print(f"  Train: {len(tr):,} | Valid: {len(va):,}")

    print("[B3] Training LightGBM forecaster...")
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                               num_leaves=31, min_child_samples=20, subsample=0.8,
                               colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                               random_state=42, verbose=-1)
    model.fit(tr[FEAT].fillna(0), np.log1p(tr['sales']),
              eval_set=[(va[FEAT].fillna(0), np.log1p(va['sales']))],
              callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    pickle.dump(model, open('models/trend_lgbm.pkl','wb'))

    y_pred = np.expm1(model.predict(va[FEAT].fillna(0)))
    mae    = mean_absolute_error(va['sales'], y_pred)
    mape   = mean_absolute_percentage_error(va['sales'], y_pred)*100
    print(f"  Best iter: {model.best_iteration_} | MAE: {mae:.1f} | MAPE: {mape:.1f}%")

    wm = wm.copy()
    wm['predicted']     = np.expm1(model.predict(wm[FEAT].fillna(0)))
    wm['trend_zscore']  = ((wm['predicted'] - wm['roll_mean_4w']) /
                            (wm['roll_std_4w'] + 1e-6))
    wm['trend_score']   = 1/(1+np.exp(-wm['trend_zscore'].clip(-5,5)))

    trend_out = wm[['week','product_type_name','sales','predicted',
                    'trend_score','trend_zscore']].reset_index(drop=True)
    trend_out.to_parquet('data/features/trend_scores.parquet', index=False)
    print(f"  {len(trend_out):,} trend score rows saved ✓")

    last = trend_out[trend_out.week==trend_out.week.max()]\
           .sort_values('trend_score',ascending=False)
    print("\n  Top trending (last week):")
    for _, r in last.head(8).iterrows():
        print(f"    {r['product_type_name']:<28} {r['trend_score']:.3f} "
              f"{'█'*int(r['trend_score']*15)}")

    # ── PART C: Google Trends ──────────────────────────────────────
    print("\n\n── Part C: Google Trends Signal ─────────────────────────")
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl='en-US', tz=0, timeout=(5,10))
        kws = {'Dress':'dress fashion','Trousers':'trousers fashion',
               'Sweater':'sweater fashion','T-shirt':'t-shirt fashion'}
        fused_rows = []
        for ptype, kw in kws.items():
            try:
                pt.build_payload([kw], timeframe='2018-09-01 2019-03-31')
                df_t = pt.interest_over_time()
                if not df_t.empty and kw in df_t.columns:
                    sub = trend_out[trend_out.product_type_name==ptype].copy()
                    for _, row in sub.iterrows():
                        cl = df_t.index[np.argmin(np.abs(
                             pd.to_datetime(df_t.index)-row['week']))]
                        g  = float(df_t.loc[cl,kw])/100
                        fused_rows.append({
                            'week':row['week'],'product_type_name':ptype,
                            'trend_score':row['trend_score'],'google_interest':g,
                            'fused_score':0.6*row['trend_score']+0.4*g})
                    print(f"  {ptype}: ✓")
            except Exception as e:
                print(f"  {ptype}: {type(e).__name__}")
        if fused_rows:
            pd.DataFrame(fused_rows).to_parquet(
                'data/features/fused_trend_scores.parquet', index=False)
            print(f"  Fused scores saved ✓")
        else:
            print("  Google Trends not reachable — will work on AWS EC2")
    except Exception as e:
        print(f"  Skipped ({e}) — code ready, runs on AWS EC2")

    print("\nPhase 4 complete ✓")


if __name__ == "__main__":
    run()