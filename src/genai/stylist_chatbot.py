"""
FashionMind — Phase 6B: Gemini Stylist Chatbot
===============================================
Tools  : get_recommendations | get_trend_report
         get_outfit_suggestion | explain_recommendation
RAG    : TF-IDF + SVD retrieval (37 fashion knowledge chunks)
Stream : SSE word-by-word via stream_chat() generator
API    : imported by api/main.py
"""
import os, json, pickle, re, warnings
import numpy as np
import pandas as pd
import faiss, scipy.sparse as sp
import lightgbm as lgb, shap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
warnings.filterwarnings('ignore')

# ── Feature columns (must match Phase 5) ──────────────────────────
FEAT_COLS = ['als_score','visual_sim','nlp_sim','trend_score','price_affinity',
             'category_match','popularity_score','rank_norm','engagement_score',
             'age_norm','ptype_idx','colour_idx','garment_idx']

FEAT_LABELS = {
    0:'Personalised match', 1:'Visual style match',  2:'Description similarity',
    3:'Trending this week', 4:'Matches price range',  5:'Favourite category',
    6:'Popular item',       7:'High CF rank',          8:'Engagement level',
    9:'Age relevance',     10:'Product type',         11:'Colour group',
   12:'Garment type',
}

# ── Lazy-loaded model registry (populated on first call) ──────────
_M = {}

def _load():
    if _M: return
    print("Loading FashionMind models...", end=' ', flush=True)
    _M['als']      = pickle.load(open('models/als_model.pkl','rb'))
    _M['matrix']   = sp.load_npz('models/user_item_matrix.npz')
    _M['ue']       = pickle.load(open('models/user_encoder.pkl','rb'))
    _M['ie']       = pickle.load(open('models/item_encoder.pkl','rb'))
    _M['reranker'] = pickle.load(open('models/reranker.pkl','rb'))
    _M['vis_feats']= np.load('models/visual_features.npy')
    _M['vis_aids'] = np.load('models/visual_article_ids.npy', allow_pickle=True)
    _M['nlp_feats']= np.load('models/nlp_features.npy')
    _M['nlp_aids'] = np.load('models/nlp_article_ids.npy', allow_pickle=True)
    _M['vis_a2i']  = {a:i for i,a in enumerate(_M['vis_aids'])}
    _M['nlp_a2i']  = {a:i for i,a in enumerate(_M['nlp_aids'])}
    _M['vis_index']= faiss.read_index('models/visual_faiss.index')
    _M['nlp_index']= faiss.read_index('models/nlp_faiss.index')
    _M['art']      = pd.read_parquet('data/features/article_content_features.parquet')
    _M['art']['article_id'] = _M['art']['article_id'].astype(str)
    _M['art_lu']   = _M['art'].set_index('article_id')
    # Segmented cold-start (age_bucket × club) + global popularity fallback.
    _M['pop_seg']  = pd.read_parquet('data/features/cold_start_popular.parquet')
    _M['pop']      = _M['pop_seg']  # back-compat alias
    _glob_path     = 'data/features/cold_start_popular_global.parquet'
    _glob          = (pd.read_parquet(_glob_path) if os.path.exists(_glob_path)
                      else _M['pop_seg'].drop_duplicates('article_id').head(500))
    _M['pop_s']    = _glob.set_index('article_id')['score'].to_dict()
    # Cold-start feed should read like a fashion edit — raw popularity is dominated
    # by socks / underwear / swimwear / basics, so drop those product types here.
    _HIDE_PT = {'socks','underwear','underwear bottom','bra','bikini top','bikini bottom',
                'swimwear bottom','swimwear top','swimwear set','night wear','nightwear',
                'pyjama set','pyjama bottom','pyjama top','pyjama jumpsuit/playsuit',
                'leggings/tights','tights','slippers','dungarees','sleep bag'}
    _pt_lu = {str(k): str(v).lower() for k, v in _M['art_lu']['product_type_name'].to_dict().items()}
    _clean = [a for a in _glob['article_id'].tolist() if _pt_lu.get(str(a), '') not in _HIDE_PT]
    _M['pop12']    = (_clean or _glob['article_id'].tolist())[:40]
    _M['trend']    = pd.read_parquet('data/features/trend_scores.parquet')
    lw = _M['trend']['week'].max()
    _M['lt_map']   = _M['trend'][_M['trend'].week==lw].set_index('product_type_name')['trend_score'].to_dict()
    # Blend the Pinterest culture signal into the per-product-type trend score
    # so trending searches also nudge the re-ranker's `trend_score` feature.
    _pin_path = 'data/features/pinterest_fused_scores.parquet'
    if os.path.exists(_pin_path):
        for _r in pd.read_parquet(_pin_path).itertuples(index=False):
            _M['lt_map'][_r.product_type_name] = float(_r.fused_score)
        _M['pinterest_boosted'] = True
    _M['outfit_df']= pd.read_parquet('data/features/outfit_pairs.parquet')
    _M['shap_df']  = pd.read_parquet('data/features/shap_explanations.parquet')
    _M['u_price']  = pd.read_parquet('data/features/user_avg_price.parquet').set_index('customer_id')['avg_price'].to_dict()
    _M['a_price']  = pd.read_parquet('data/features/art_avg_price.parquet').set_index('article_id')['avg_price'].to_dict()
    _M['u_hist']   = pickle.load(open('data/features/user_history.pkl','rb'))
    _M['cust']     = pd.read_parquet('data/features/customer_segments.parquet').set_index('customer_id')
    _M['cid2u']    = _M['ue']['dec']
    _M['i2aid']    = _M['ie']['enc']
    _M['u_top_pt'] = {}
    art_pt_map     = _M['art_lu']['product_type_name'].to_dict()
    for cid, items in _M['u_hist'].items():
        pt = [art_pt_map[a] for a in items if a in art_pt_map]
        if pt: _M['u_top_pt'][cid] = max(set(pt), key=pt.count)
    _M['kb']       = pickle.load(open('models/rag_kb_chunks.pkl','rb'))
    _M['kb_embs']  = np.load('models/rag_kb_embeddings.npy')

    # Optional GRU4Rec retrieval — used only when RECS_USE_GRU is set. It doubles
    # candidate recall@100 offline (scripts/eval_retrieval.py) but the shipped
    # reranker was fit on ALS candidates only, so this stays opt-in until a
    # re-fit. Missing artifacts = feature simply off, never a load failure.
    _M['gru'] = None
    if os.getenv('RECS_USE_GRU') and os.path.exists('models/gru4rec.pt'):
        try:
            import torch
            from src.recsys.sequence import GRU4Rec
            voc = pickle.load(open('models/gru4rec_vocab.pkl', 'rb'))
            g = GRU4Rec(vocab_size=len(voc['a2i']))
            g._load(torch.load('models/gru4rec.pt'))
            _M['gru'] = {'model': g, 'a2i': voc['a2i'], 'i2a': voc['i2a']}
            print("  + GRU4Rec retrieval enabled")
        except Exception as e:
            print(f"  GRU4Rec load skipped: {e}")
    print("done ✓")


# ══════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════

def retrieve_context(query: str, k: int = 3) -> list:
    _load()
    kb  = _M['kb']; embs = _M['kb_embs']
    qv  = kb['tfidf'].transform([query])
    qe  = normalize(kb['svd'].transform(qv), norm='l2').astype(np.float32)
    sim = cosine_similarity(qe, embs)[0]
    top = sim.argsort()[::-1][:k]
    return [kb['chunks'][i] for i in top if sim[i] > 0.05]


def _expand_meta(aid: str) -> dict:
    aids = str(aid).lstrip('0')
    if aids in _M['art_lu'].index:
        r = _M['art_lu'].loc[aids]
        prod_name = str(r.get('prod_name', 'Cotton Styled Shirt'))
        ptype = str(r.get('product_type_name', 'T-shirt'))
        colour = str(r.get('colour_group_name', 'Classic'))
        pgroup = str(r.get('product_group_name', 'Garment Upper body'))
        ggrp = str(r.get('garment_group_name', 'Essentials'))
        desc = str(r.get('detail_desc', ''))
    else:
        prod_name = 'Cotton Styled Shirt'
        ptype = 'T-shirt'
        colour = 'Classic'
        pgroup = 'Garment Upper body'
        ggrp = 'Essentials'
        desc = ''
    ip = _M['a_price'].get(aids, 0.025)
    inr = int(round(ip * 41500)) if ip < 50 else int(round(ip))   # same scale as the UI
    from urllib.parse import quote_plus
    shop_q = quote_plus(f"{prod_name} {colour} {ptype}".strip())
    return {
        'article_id': aids,
        'product_name': prod_name,
        'product_type_name': ptype,
        'colour_group_name': colour,
        'product_group_name': pgroup,
        'garment_group_name': ggrp,
        'detail_desc': desc,
        'avg_price': float(ip),
        'price': float(ip),
        'price_inr': inr,
        'currency': 'INR',
        'shop_url': f"https://www.google.com/search?tbm=shop&q={shop_q}",
    }


_CAT_ALIASES = {
    "dress": "dress", "dresses": "dress", "gown": "dress", "frock": "dress", "maxi": "dress",
    "jean": "trouser", "jeans": "trouser", "denim": "trouser", "trouser": "trouser",
    "trousers": "trouser", "pant": "trouser", "pants": "trouser", "slacks": "trouser",
    "jogger": "trouser", "joggers": "trouser", "chino": "trouser", "chinos": "trouser",
    "top": "top", "tee": "t-shirt", "tees": "t-shirt", "tshirt": "t-shirt", "t-shirt": "t-shirt",
    "shirt": "shirt", "blouse": "blouse", "tank": "vest top", "cami": "vest top", "camisole": "vest top",
    "skirt": "skirt", "skirts": "skirt", "shorts": "short", "short": "short",
    "jacket": "jacket", "jackets": "jacket", "blazer": "blazer", "coat": "coat", "coats": "coat",
    "cardigan": "cardigan", "sweater": "sweater", "sweaters": "sweater", "jumper": "sweater",
    "knit": "sweater", "knitwear": "sweater", "hoodie": "hoodie", "sweatshirt": "sweater",
    "jumpsuit": "jumpsuit", "playsuit": "jumpsuit", "co-ord": "garment set", "coord": "garment set",
    "shoe": "shoe", "shoes": "shoe", "sneaker": "sneaker", "sneakers": "sneaker",
    "heel": "heel", "heels": "heel", "boot": "boot", "boots": "boot", "sandal": "sandal",
    "sandals": "sandal", "flat": "ballerina", "flats": "ballerina", "loafer": "loafer",
    "bag": "bag", "bags": "bag", "handbag": "bag", "tote": "bag", "scarf": "scarf",
    "sunglasses": "sunglass", "jewellery": "necklace", "jewelry": "necklace",
}
_OCCASION_WORDS = {"work", "casual", "party", "everyday", "office", "brunch", "formal",
                   "date", "weekend", "beach", "gym", "wedding", "festive", "smart", "lounge"}


def _popular_in_category(cat: str, n: int = 12, max_price: float = None) -> list:
    """Most-popular catalogue items of a garment category (for guests / category asks)."""
    art = _M.get('art')
    if art is None or not cat:
        return []
    toks = [w for w in re.split(r"[^a-z]+", cat.lower()) if len(w) > 2]
    keys = {_CAT_ALIASES.get(w, w) for w in toks} or {cat.lower()}
    ptn = art['product_type_name'].astype(str).str.lower()
    grp = (art['product_group_name'].astype(str).str.lower()
           if 'product_group_name' in art.columns else ptn)
    mask = None
    for k in keys:
        m = ptn.str.contains(k, na=False, regex=False) | grp.str.contains(k, na=False, regex=False)
        mask = m if mask is None else (mask | m)
    if mask is None:
        return []
    sub = art[mask]
    if sub.empty:
        return []
    pop_s = _M.get('pop_s', {})
    aids = sub['article_id'].astype(str)
    sub = sub.assign(_pop=[pop_s.get(a.lstrip('0'), pop_s.get(a, 0.0)) for a in aids])
    sub = sub.sort_values('_pop', ascending=False)
    out, per_name = [], {}
    for a in sub['article_id'].astype(str).head(max(n * 8, 120)):
        m = _expand_meta(a)
        if max_price and m['avg_price'] > max_price:
            continue
        nm = re.sub(r"\s*\(\d+\)\s*$", "", str(m['product_name'])).strip().lower()
        if per_name.get(nm, 0) >= 1:          # at most one colourway per style
            continue
        per_name[nm] = per_name.get(nm, 0) + 1
        m['score'] = 0.0
        m['reasons'] = [f"Popular in {cat.strip().lower()}"]
        out.append(m)
        if len(out) >= n:
            break
    return out


def get_recommendations(customer_id: str, occasion: str = None,
                        max_price: float = None, n: int = 12,
                        category: str = None) -> dict:
    _load()
    cid = str(customer_id)

    # A garment-type request ("dresses", "black jackets") is served from the
    # catalogue directly — the popularity list alone can't answer it.
    cat = (category or "").strip()
    if not cat and occasion and occasion.strip().lower() not in _OCCASION_WORDS:
        cat = occasion.strip()
    if cat:
        picked = _popular_in_category(cat, n, max_price)
        if picked:
            return {"customer_id": cid, "category": cat,
                    "is_cold_start": cid not in _M['cid2u'],
                    "recommendations": picked,
                    "explanation": f"Popular {cat.lower()} picks."}

    if cid not in _M['cid2u']:
        # FIX: use segmented cold-start (age-bucket × club) instead of global popularity
        pop_seg = _M.get('pop_seg')
        if pop_seg is None:
            pop_seg = _M.get('pop')
        cust_df = _M.get('cust')
        seg_items = None
        if pop_seg is not None and cust_df is not None and cid in cust_df.index:
            row = cust_df.loc[cid]
            age = float(row.get('age', 30)) if 'age' in cust_df.columns else 30
            bucket = '16-25' if age<26 else ('26-35' if age<36 else ('36-50' if age<51 else '50+'))
            club   = str(row.get('club', row.get('club_member_status', 'NONE'))).upper()
            if 'age_bucket' in pop_seg.columns:
                seg = pop_seg[(pop_seg['age_bucket']==bucket)&(pop_seg['club']==club)]
                if len(seg) >= n:
                    seg_items = seg.nlargest(n,'score')[['article_id','score']].to_dict('records')
        if seg_items is None:
            # global popularity fallback (segment too small or profile unknown)
            seg_items = [{'article_id': a, 'score': float(_M['pop_s'].get(a, 0.0))}
                         for a in _M['pop12'][:n]]
        recs = []
        for item in seg_items:
            meta = _expand_meta(item['article_id'])
            meta['score'] = round(float(item.get('score', 0.0)), 4)
            meta['reasons'] = ['Trending this week', 'Popular item']
            recs.append(meta)
        return {"customer_id":cid,"is_cold_start":True,"recommendations":recs,
                "explanation":f"New user — showing top picks for your profile."}

    uidx = int(_M['cid2u'][cid])
    try: ids, als_scores = _M['als'].recommend(uidx, _M['matrix'][uidx], N=50,
                                                filter_already_liked_items=True)
    except Exception as e: return {"error": str(e)}

    # union GRU4Rec candidates ahead of ALS's (experimental; RECS_USE_GRU).
    # als_score for a GRU-only item is 0 — the reranker treats it as a weak-CF
    # item, which is imperfect until it's re-fit on the mixed candidate set.
    if _M.get('gru'):
        g = _M['gru']
        seq = [g['a2i'][a] for a in _M['u_hist'].get(cid, []) if a in g['a2i']][-20:]
        if seq:
            already = set(_M['u_hist'].get(cid, []))
            g_aids = [g['i2a'][i] for i in g['model'].topk(seq, k=50) if g['i2a'][i] not in already]
            aid2i = {_M['i2aid'][int(x)].lstrip('0'): (int(x), float(s))
                     for x, s in zip(ids, als_scores)}
            merged_ids, merged_sc, seen = [], [], set()
            for a in g_aids[:25] + [_M['i2aid'][int(x)].lstrip('0') for x in ids]:
                if a in seen:
                    continue
                seen.add(a)
                if a in aid2i:
                    merged_ids.append(aid2i[a][0]); merged_sc.append(aid2i[a][1])
                else:
                    # map article_id back to an item index for the feature loop
                    inv = _M.get('_aid2iidx')
                    if inv is None:
                        inv = {v.lstrip('0'): k for k, v in enumerate(_M['i2aid'])}
                        _M['_aid2iidx'] = inv
                    if a in inv:
                        merged_ids.append(inv[a]); merged_sc.append(0.0)
            if merged_ids:
                ids, als_scores = np.array(merged_ids), np.array(merged_sc)

    up = _M['u_price'].get(cid, 0.025)
    ut = _M['u_top_pt'].get(cid, '')
    ue = float(_M['cust'].loc[cid,'engagement_score']) if cid in _M['cust'].index else 0.5
    ua = float(_M['cust'].loc[cid,'age_norm'])         if cid in _M['cust'].index else 0.3

    items = _M['u_hist'].get(cid,[])[:20]
    vi=[_M['vis_a2i'][a] for a in items if a in _M['vis_a2i']]
    ni=[_M['nlp_a2i'][a] for a in items if a in _M['nlp_a2i']]
    vc=_M['vis_feats'][vi].mean(0) if vi else np.zeros(_M['vis_feats'].shape[1])
    nc=_M['nlp_feats'][ni].mean(0) if ni else np.zeros(64)
    vc/=(np.linalg.norm(vc)+1e-8); nc/=(np.linalg.norm(nc)+1e-8)

    rows, meta = [], []
    for rank,(iidx,alsc) in enumerate(zip(ids, als_scores)):
        aid=_M['i2aid'][int(iidx)]; aids=aid.lstrip('0')
        if aids in _M['art_lu'].index:
            r=_M['art_lu'].loc[aids]
            ptype=str(r['product_type_name']); pi=int(r['product_type_name_idx'])
            ci=int(r['colour_group_name_idx']); gi=int(r['garment_group_name_idx'])
            colour=str(r['colour_group_name']); pgroup=str(r['product_group_name'])
            prod_name=str(r.get('prod_name','Cotton Styled Shirt'))
            ggrp=str(r.get('garment_group_name','Essentials'))
            desc=str(r.get('detail_desc',''))
        else: ptype=colour=pgroup=prod_name=ggrp=desc=''; pi=ci=gi=0
        ip=_M['a_price'].get(aids,0.025)
        if max_price and ip>max_price: continue
        vv=_M['vis_feats'][_M['vis_a2i'][aids]] if aids in _M['vis_a2i'] else np.zeros(_M['vis_feats'].shape[1])
        nv=_M['nlp_feats'][_M['nlp_a2i'][aids]] if aids in _M['nlp_a2i'] else np.zeros(64)
        rows.append([float(alsc),float(np.dot(vv,vc)),float(np.dot(nv,nc)),
                     _M['lt_map'].get(ptype,0.5),
                     float(max(0,1-abs(ip-up)/(up+1e-6))),
                     int(ptype==ut),float(_M['pop_s'].get(aids,0.0)),
                     1-rank/50,ue,ua,pi,ci,gi])
        meta.append({
            'article_id': aids,
            'product_name': prod_name,
            'product_type_name': ptype,
            'colour_group_name': colour,
            'product_group_name': pgroup,
            'garment_group_name': ggrp,
            'detail_desc': desc,
            'avg_price': float(ip),
            'price': float(ip)
        })

    if not rows: return {"recommendations":[],"explanation":"No matching items."}

    scores   = _M['reranker'].predict(np.array(rows,dtype=np.float32))
    ranked   = sorted(zip(scores,meta,rows), key=lambda x:-x[0])
    # FIX: use pre-computed SHAP from shap_explanations.parquet (avoids ~3s latency per request)
    shap_df  = _M.get('shap_df', None)
    shap_lkp = {}
    if shap_df is not None and 'customer_id' in shap_df.columns:
        user_shap = shap_df[shap_df['customer_id']==cid]
        shap_lkp  = dict(zip(user_shap['article_id'],
                             zip(user_shap['reason_1'],user_shap['reason_2'],user_shap['reason_3'])))

    recs = []
    for sc, m, fr in ranked[:n]:
        if m['article_id'] in shap_lkp:
            reasons = [r for r in shap_lkp[m['article_id']] if r]
        else:
            # Fast fallback: score-based signal names without SHAP recompute
            feat_names = list(FEAT_LABELS.values())
            top3 = sorted(range(len(fr)), key=lambda i: -abs(fr[i]))[:3]
            reasons = [f"{'↑' if fr[f]>0 else '↓'} {feat_names[f] if f<len(feat_names) else 'signal'}" for f in top3]
        recs.append({**m, "score":round(float(sc),4), "reasons":reasons})

    return {"customer_id":cid,"occasion":occasion or "general",
            "recommendations":recs,"n_candidates":len(rows)}


# Demand-forecast "trending" is only meaningful for wearable categories — the
# raw H&M taxonomy also has Dog Wear, Umbrella, Waterbottle, Sleeping sack, etc.
_GARMENT_TYPES = {
    'Trousers','Jeans','Dress','Sweater','Cardigan','T-shirt','Top','Blouse','Shirt','Polo shirt',
    'Jacket','Blazer','Coat','Hoodie','Vest top','Bodysuit','Skirt','Shorts','Jumpsuit/Playsuit',
    'Garment Set','Leggings/Tights','Sneakers','Boots','Sandals','Ballerinas','Other shoe',
    'Bag','Scarf','Belt','Hat/beanie','Hat/brim','Cap/peaked','Sunglasses','Earring','Necklace',
    'Swimwear bottom','Swimsuit','Bikini top','Sunglasses',
}


def get_trend_report(category: str = None) -> dict:
    _load()
    lw  = _M['trend']['week'].max()
    rec = _M['trend'][_M['trend'].week==lw].sort_values('trend_score',ascending=False)
    rec = rec[rec['product_type_name'].isin(_GARMENT_TYPES)]
    if category:
        rec = rec[rec['product_type_name'].str.contains(category,case=False,na=False)]
    top = rec.head(10)[['product_type_name','trend_score','sales','predicted']].copy()
    top['trend_score'] = top['trend_score'].round(3)
    out = {"week":str(lw.date()),"trending":top.to_dict('records'),
           "note":"Trend score 0-1 based on LightGBM demand forecast + seasonality"}
    # Live-culture signal from Pinterest, layered on top of the demand forecast.
    try:
        from src.trends.pinterest_trends import get_pinterest_trends as _pt
        pins = _pt(category=category, limit=8)
        out["pinterest_rising"] = [
            {"keyword": p["keyword"], "score": p["score"], "pct_change": p["pct_change"]}
            for p in pins
        ]
    except Exception:
        out["pinterest_rising"] = []
    return out


def get_pinterest_trends(category: str = None, region: str = "IN") -> dict:
    """Currently-rising fashion searches on Pinterest (with catalogue mapping)."""
    from src.trends.pinterest_trends import get_pinterest_trends as _pt
    from src.trends.trend_map import map_trend
    pins = _pt(category=category, region=region, limit=15)
    rows = []
    for p in pins:
        m = map_trend(p["keyword"])
        rows.append({"keyword": p["keyword"], "score": p["score"],
                     "pct_change": p["pct_change"], "source": p["source"],
                     "maps_to": {"product_types": m["product_types"], "tags": m["tags"]}})
    return {"region": region, "source": rows[0]["source"] if rows else "cache",
            "rising": rows,
            "note": "Pinterest search interest, 0-100. Use shop_the_trend to turn a keyword into buyable products."}


def shop_the_trend(keyword: str, budget_max: float = None, n: int = 8) -> dict:
    """Turn a trend keyword (e.g. 'barrel jeans', 'balletcore') into products
    from the aggregator: curated India retail links + trained catalogue + any
    live shopping source that is configured."""
    from src.trends.trend_map import map_trend
    from src.catalog import aggregate_search
    m = map_trend(keyword)
    items = aggregate_search(query=keyword, product_types=m["product_types"] or None,
                             tags=m["tags"] or None, colours=m["colours"] or None,
                             limit=max(n * 2, 12))
    if budget_max:
        items = [it for it in items
                 if (it.get("price_min") or 0) <= budget_max or it.get("price_min") is None]
    return {"trend": keyword, "maps_to": m["product_types"],
            "products": items[:n],
            "note": "buy_url opens the item (or a search for it) on the retailer's own site."}


_JUNK_TITLE = re.compile(
    r"\b(pack of|combo|set of \d|multipack|\d ?pcs|nightwear|inner ?wear|camisole|"
    r"slip for|thermal|shapewear|for men.*women|for women.*men|kids?|boys?|girls?|"
    r"toddler|infant)\b", re.I)


def _clean_title(t: str) -> bool:
    return bool(t) and len(t) < 90 and not _JUNK_TITLE.search(t)


# keyword gates so a slot only accepts a title that reads like that garment
_SLOT_KW = {
    "top":       (r"top|tee|t-?shirt|shirt|blouse|tank|cami|bodysuit|corset|"
                  r"blazer|jacket|cardigan|sweater|knit|kurta|kurti", r"jean|trouser|pant|skirt|short|dress|shoe|sneaker|bag|heel|sandal"),
    "bottom":    (r"jean|trouser|pant|denim|skirt|short|legging|culotte|palazzo|cargo|chino", r"top|tee|t-?shirt|blouse|dress|shoe|sneaker|bag|jacket|blazer"),
    "layer":     (r"blazer|jacket|coat|shrug|overshirt|waistcoat|bomber|trench", r"jean|trouser|pant|dress|shoe|bag|skirt"),
    "one-piece": (r"dress|gown|jumpsuit|playsuit|co-?ord|two[- ]?piece|kurta set|"
                  r"saree|lehenga|anarkali|sharara|set", r"shoe|sneaker|bag|heel"),
    "shoes":     (r"shoe|sneaker|trainer|heel|boot|sandal|flat|loafer|mule|slipper|jutti|pump", r""),
    "accessory": (r"bag|tote|clutch|purse|backpack|sling|hobo|scarf|stole|belt|"
                  r"earring|necklace|choker|sunglass|shades|hat|cap", r""),
}


def _cat_ok(slot: str, title: str) -> bool:
    t = (title or "").lower()
    inc, exc = _SLOT_KW.get(slot, (r".", r"(?!x)x"))
    return bool(re.search(inc, t)) and not (exc and re.search(exc, t))


def build_trend_outfit(keyword: str, budget_max: float = None) -> dict:
    """Assemble a coherent full look for a trend — one top (or dress), a bottom,
    shoes and an accessory — with a focused search per slot so the pieces make
    sense together. Curated items win ties (clean names + correct categories)."""
    from src.trends.trend_map import map_trend
    from src.catalog import aggregate_search
    m = map_trend(keyword)
    kl = keyword.lower()
    gender = "men" if any(w in kl for w in
        ("men", "sherwani", "bandhgala", "nehru jacket", "dhoti", "kurta pyjama")) else "women"

    TOP   = {"Top","Blouse","Shirt","T-shirt","Sweater","Vest top","Bodysuit","Cardigan","Polo shirt"}
    LAYER = {"Blazer","Jacket","Coat"}
    DRESS = {"Dress","Jumpsuit/Playsuit","Garment Set"}
    BOTTOM= {"Trousers","Skirt","Shorts","Leggings/Tights"}
    SHOE  = {"Ballerinas","Sneakers","Boots","Sandals","Other shoe","Slippers","Flats"}
    ACC   = {"Bag","Scarf","Belt","Sunglasses","Earring","Necklace","Hat/beanie","Other accessories"}

    trend_toks = [w for w in re.split(r"[^a-z]+", kl) if len(w) > 2]
    _WRONG_GENDER = (re.compile(r"\b(men'?s?|boys?)\b", re.I) if gender == "women"
                     else re.compile(r"\b(women'?s?|girls?|ladies)\b", re.I))

    def _gender_ok(t):
        t = t or ""
        return not (_WRONG_GENDER.search(t) and not re.search(
            r"\bwomen\b" if gender == "women" else r"\bmen\b", t, re.I))

    used: set = set()

    def _filter(rows):
        rows = [r for r in rows if (r.get("gender") or "women") in (gender, "unisex")
                and _gender_ok(r.get("title", ""))]
        if budget_max:
            rows = [r for r in rows
                    if (r.get("price_min") or 0) <= budget_max or r.get("price_min") is None]
        return rows

    def take(slot, query, kinds, strict=True, want_trend=False, neutral=False):
        rows = _filter(aggregate_search(query=query, product_types=list(kinds),
                                        tags=m["tags"] or None, limit=24))

        def _trend_hit(r):
            t = (r.get("title", "") or "").lower()
            return r.get("source") == "curated" or any(w in t for w in trend_toks)

        rows.sort(key=lambda r: (not r.get("image"),
                                 want_trend and not _trend_hit(r),
                                 not _cat_ok(slot, r.get("title", "")),
                                 not _clean_title(r.get("title", "")),
                                 r.get("source") != "curated"))
        passes = []
        if want_trend:
            passes.append(lambda r: r.get("image") and _clean_title(r.get("title", ""))
                          and _cat_ok(slot, r.get("title", "")) and _trend_hit(r))
        passes.append(lambda r: r.get("image") and _clean_title(r.get("title", ""))
                      and _cat_ok(slot, r.get("title", "")))
        passes.append(lambda r: _clean_title(r.get("title", "")) and _cat_ok(slot, r.get("title", "")))
        for ok in passes:
            for r in rows:
                if r.get("id") not in used and ok(r):
                    used.add(r.get("id")); return r
        if not strict:
            for r in rows:
                if r.get("id") not in used and _clean_title(r.get("title", "")):
                    used.add(r.get("id")); return r
        if neutral:
            # last resort: trust the structured product_type, ignore the title
            # gate — keeps the look complete when search returns little (offline,
            # or an obscure trend). Prefers an in-kind item, then anything.
            pool = _filter(aggregate_search(product_types=list(kinds), limit=20)) + rows
            for want_kind in (True, False):
                for r in pool:
                    if r.get("id") in used:
                        continue
                    if not want_kind or r.get("product_type") in kinds:
                        used.add(r.get("id")); return r
        return None

    mapped = set(m["product_types"] or [])
    dress_led = bool({"Dress", "Garment Set", "Jumpsuit/Playsuit"} & mapped)
    outer_led = bool(LAYER & mapped)
    bottom_led = bool(BOTTOM & mapped)
    top_led = bool(TOP & mapped)

    look = []
    if dress_led:
        d = take("one-piece", keyword, DRESS, want_trend=True) \
            or take("one-piece", f"{gender} {keyword}", DRESS, strict=False)
        if d: look.append({"slot": "one-piece", **d})
    else:
        if outer_led:
            l = take("layer", keyword, LAYER, want_trend=True) \
                or take("layer", f"{gender} blazer", LAYER, strict=False)
            if l: look.append({"slot": "layer", **l})
        t = (take("top", keyword, TOP, want_trend=True) if top_led
             else take("top", f"{gender} plain top", TOP, strict=False)) \
            or take("top", f"{gender} basic top", TOP, strict=False, neutral=True)
        if t: look.append({"slot": "top", **t})
        b = (take("bottom", keyword, BOTTOM, want_trend=True) if bottom_led
             else take("bottom", f"{gender} wide leg trousers", BOTTOM)) \
            or take("bottom", f"{gender} trousers", BOTTOM, strict=False, neutral=True)
        if b: look.append({"slot": "bottom", **b})

    s = (take("shoes", f"{keyword} shoes", SHOE)
         or take("shoes", f"{gender} ballet flats", SHOE)
         or take("shoes", f"{gender} white sneakers", SHOE, strict=False, neutral=True))
    if s: look.append({"slot": "shoes", **s})
    a = (take("accessory", f"{gender} shoulder bag", ACC)
         or take("accessory", f"{gender} tote bag", ACC, strict=False, neutral=True))
    if a: look.append({"slot": "accessory", **a})

    total_lo = total_hi = 0
    for it in look:
        total_lo += it.get("price_min") or 0
        total_hi += it.get("price_max") or it.get("price_min") or 0
    return {"trend": keyword, "look": look,
            "estimated_total": {"min": total_lo, "max": total_hi, "currency": "INR"},
            "note": "Prices are category estimates unless a live source is configured; confirm on each retailer."}


def get_outfit_suggestion(upper_article_id: str = None,
                          style: str = None) -> dict:
    _load()
    df = _M['outfit_df']
    if upper_article_id:
        m = df[df['upper_article_id']==upper_article_id]
        if not m.empty:
            best = m.sort_values('compatibility_score',ascending=False).iloc[0]
            info = {}
            if best['lower_article_id'] in _M['art_lu'].index:
                r = _M['art_lu'].loc[best['lower_article_id']]
                info = {'product_type':str(r['product_type_name']),
                        'colour':str(r['colour_group_name'])}
            return {"upper":upper_article_id,"lower":best['lower_article_id'],
                    "compatibility_score":round(float(best['compatibility_score']),3),
                    "lower_details":info}
    top = df.sort_values('compatibility_score',ascending=False).head(5)
    return {"top_outfit_pairs":top.to_dict('records')}


def explain_recommendation(customer_id: str, article_id: str) -> dict:
    _load()
    mask = ((_M['shap_df']['customer_id']==customer_id) &
            (_M['shap_df']['article_id']==article_id))
    if mask.any():
        row = _M['shap_df'][mask].iloc[0]
        return {"customer_id":customer_id,"article_id":article_id,
                "reasons":[row['reason_1'],row['reason_2'],row['reason_3']],
                "source":"pre-computed"}
    return {"customer_id":customer_id,"article_id":article_id,
            "reasons":["↑ Personalised match","↑ Trending this week","↑ Matches price range"],
            "note":"Run get_recommendations for fresh SHAP explanations"}


# ══════════════════════════════════════════════════════════════════
# GEMINI TOOL DECLARATIONS
# ══════════════════════════════════════════════════════════════════

TOOL_DECLARATIONS = [
    {"name":"get_recommendations",
     "description":"Get personalised fashion recommendations for a customer based on purchase history, style, and trends.",
     "parameters":{"type":"object","properties":{
         "customer_id":{"type":"string"},
         "category":   {"type":"string","description":"Garment type the user asked for, e.g. 'dress', 'jacket', 'jeans', 'skirt', 'sneakers'. Set this whenever the request names a kind of item."},
         "occasion":   {"type":"string","description":"casual|work|party|beach|gym"},
         "max_price":  {"type":"number","description":"Max price in INR (₹)"},
         "n":          {"type":"integer","description":"Number of items (default 12)"}
     },"required":["customer_id"]}},
    {"name":"get_trend_report",
     "description":"Get currently trending fashion categories based on demand forecasting.",
     "parameters":{"type":"object","properties":{
         "category":{"type":"string","description":"Optional category filter"}
     }}},
    {"name":"get_outfit_suggestion",
     "description":"Suggest compatible outfit pairings — upper and lower body combinations.",
     "parameters":{"type":"object","properties":{
         "upper_article_id":{"type":"string"},
         "style":           {"type":"string","description":"casual|smart|sporty"}
     }}},
    {"name":"explain_recommendation",
     "description":"Explain why a specific item was recommended using SHAP feature importance.",
     "parameters":{"type":"object","properties":{
         "customer_id":{"type":"string"},
         "article_id": {"type":"string"}
     },"required":["customer_id","article_id"]}},
]

TOOL_MAP = {
    "get_recommendations":   get_recommendations,
    "get_trend_report":      get_trend_report,
    "get_outfit_suggestion": get_outfit_suggestion,
    "explain_recommendation":explain_recommendation,
}

SYSTEM_PROMPT = """You are the Vibezz Stylist — a warm, direct personal shopping assistant.

Tools:
- get_recommendations: personalised picks (each item: product_name, colour, type, price_inr, a short reason, shop_url)
- explain_recommendation: why a specific item was picked
- get_trend_report: which categories are trending right now
- get_outfit_suggestion: matching top / bottom pairings

Rules:
- Ground every answer in tool output. Never invent items, prices or links.
- When the user names a kind of item ("dresses", "a black jacket", "jeans"), call get_recommendations with `category` set to that garment word. Only skip it for vague asks ("something for work").
- Recommend only items whose type actually matches what they asked. If a result set has none, say so plainly and offer the closest thing — don't pass off trousers as a dress.
- ALL prices are Indian Rupees. Write "₹1,371" using the item's price_inr field. NEVER use £, $, or the raw decimal price.
- Make each item name a markdown link to its shop_url, e.g. [Jade Skinny Jeans](https://...).
- The app renders product cards under your message, so be brief: one line of intro, then a tight bullet list — [name](shop_url) · colour · ₹price · one short clause on why it fits. No paragraph-long descriptions.
- No technical jargon. Friendly and to the point.
"""


def run_tool(name: str, args: dict) -> str:
    if name not in TOOL_MAP:
        return json.dumps({"error":f"Unknown tool: {name}"})
    try:
        return json.dumps(TOOL_MAP[name](**args), default=str)
    except Exception as e:
        return json.dumps({"error":str(e)})


def _augment(message: str) -> str:
    chunks = retrieve_context(message, k=3)
    ctx = "\n".join(f"• {c}" for c in chunks)
    return f"[Style context]\n{ctx}\n\n[Message]\n{message}" if ctx else message


def chat(message: str, customer_id: str, history: list, api_key: str = ""):
    """Single-turn chat. Returns (response_text, updated_history).
    Uses OpenRouter when OPENROUTER_API_KEY is set, otherwise Gemini."""
    full_msg = _augment(message)

    if os.getenv("OPENROUTER_API_KEY"):
        from src.genai.llm import run_chat
        return run_chat(SYSTEM_PROMPT, history, full_msg,
                        TOOL_DECLARATIONS, run_tool)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY", ""))

    contents = [types.Content(role=t['role'],
                              parts=[types.Part(text=t['content'])])
                for t in history]
    contents.append(types.Content(role='user', parts=[types.Part(text=full_msg)]))

    tools  = [types.Tool(function_declarations=[
        types.FunctionDeclaration(**td) for td in TOOL_DECLARATIONS])]
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                         tools=tools, temperature=0.7,
                                         max_output_tokens=1024)

    for _ in range(3):
        resp  = client.models.generate_content(model='gemini-2.0-flash',
                                                contents=contents, config=config)
        cand  = resp.candidates[0]
        tcs   = [p for p in cand.content.parts
                 if hasattr(p,'function_call') and p.function_call]
        if not tcs:
            text = ''.join(p.text for p in cand.content.parts
                           if hasattr(p,'text') and p.text)
            history.append({'role':'user',  'content':message})
            history.append({'role':'model', 'content':text})
            return text, history
        contents.append(cand.content)
        tool_results = []
        for p in tcs:
            fc = p.function_call
            tool_results.append(types.Part(function_response=types.FunctionResponse(
                name=fc.name, response={"result":run_tool(fc.name,dict(fc.args))})))
        contents.append(types.Content(role='user', parts=tool_results))

    return "I can help you find the perfect outfit! What occasion are you dressing for?", history


def _to_inr(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return int(round(v * 41500)) if v < 50 else int(round(v))


def _products_from_tools(events):
    """Pull shoppable items out of tool results so the chat UI can show cards."""
    out, seen = [], set()
    for name, res in events:
        try:
            d = json.loads(res)
        except Exception:
            continue
        rows = (d.get("recommendations") or d.get("products")
                or d.get("look") or d.get("outfit") or [])
        if not isinstance(rows, list):
            continue
        for r in rows[:8]:
            if not isinstance(r, dict):
                continue
            aid = str(r.get("article_id") or r.get("id") or "")
            nm = r.get("product_name") or r.get("title") or ""
            key = (aid or nm).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append({
                "article_id": aid,
                "product_name": nm,
                "product_type_name": r.get("product_type_name") or r.get("product_type") or "",
                "colour_group_name": r.get("colour_group_name") or r.get("colour") or "",
                "price_inr": r.get("price_inr") or _to_inr(r.get("price_min") or r.get("avg_price")),
                "shop_url": r.get("shop_url") or r.get("buy_url") or "",
                "reason": (r.get("reasons") or [None])[0] if r.get("reasons") else "",
            })
    return out[:8]


def stream_chat(message: str, customer_id: str, history: list, api_key: str = ""):
    """SSE generator for FastAPI StreamingResponse."""
    full_msg = _augment(message)

    if os.getenv("OPENROUTER_API_KEY"):
        from src.genai.llm import run_chat
        events = []                       # (tool_name, result_json_str)
        text, _ = run_chat(SYSTEM_PROMPT, history, full_msg,
                           TOOL_DECLARATIONS, run_tool,
                           on_tool=lambda n, res="": events.append((n, res)))
        for t, _res in events:
            yield f"data: {json.dumps({'tool_call': t, 'done': False})}\n\n"
        prods = _products_from_tools(events)
        if prods:
            yield f"data: {json.dumps({'products': prods, 'done': False})}\n\n"
        words = text.split(' ')
        for i, w in enumerate(words):
            yield f"data: {json.dumps({'token': w + (' ' if i < len(words)-1 else ''), 'done': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True, 'full': text})}\n\n"
        return

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY", ""))

    contents = [types.Content(role=t['role'],
                              parts=[types.Part(text=t['content'])])
                for t in history]
    contents.append(types.Content(role='user', parts=[types.Part(text=full_msg)]))

    tools  = [types.Tool(function_declarations=[
        types.FunctionDeclaration(**td) for td in TOOL_DECLARATIONS])]
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                         tools=tools, temperature=0.7,
                                         max_output_tokens=1024)

    for _ in range(3):
        resp = client.models.generate_content(model='gemini-2.0-flash',
                                               contents=contents, config=config)
        cand = resp.candidates[0]
        tcs  = [p for p in cand.content.parts
                if hasattr(p,'function_call') and p.function_call]
        if not tcs:
            text = ''.join(p.text for p in cand.content.parts
                           if hasattr(p,'text') and p.text)
            for i, word in enumerate(text.split(' ')):
                chunk = word + (' ' if i < len(text.split(' '))-1 else '')
                yield f"data: {json.dumps({'token':chunk,'done':False})}\n\n"
            yield f"data: {json.dumps({'token':'','done':True,'full':text})}\n\n"
            return
        contents.append(cand.content)
        tool_results = []
        for p in tcs:
            fc = p.function_call
            yield f"data: {json.dumps({'tool_call':fc.name,'done':False})}\n\n"
            tool_results.append(types.Part(function_response=types.FunctionResponse(
                name=fc.name, response={"result":run_tool(fc.name,dict(fc.args))})))
        contents.append(types.Content(role='user', parts=tool_results))

    yield f"data: {json.dumps({'token':'How can I help you today?','done':True})}\n\n"


if __name__ == "__main__":
    print("FashionMind Stylist — tool test (no API key needed)")
    print("\n[1] RAG retrieval:")
    for q in ["beach party outfit","office wear","colour combinations"]:
        c = retrieve_context(q, k=1)
        print(f"  '{q}' → {c[0][:70]}..." if c else f"  '{q}' → no results")

    print("\n[2] Trend report:")
    r = get_trend_report()
    for t in r['trending'][:5]:
        print(f"  {t['product_type_name']:<28} {t['trend_score']:.3f}")

    print("\n[3] Cold-start recommendations:")
    r = get_recommendations("unknown_user_xyz", n=3)
    print(f"  is_cold_start={r.get('is_cold_start')} | {len(r.get('recommendations',[]))} items")

    print("\nAll tools verified ✓")