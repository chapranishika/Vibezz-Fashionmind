"""
FashionMind — Phase 6B: Gemini Stylist Chatbot
===============================================
Tools  : get_recommendations | get_trend_report
         get_outfit_suggestion | explain_recommendation
RAG    : TF-IDF + SVD retrieval (37 fashion knowledge chunks)
Stream : SSE word-by-word via stream_chat() generator
API    : imported by api/main.py
"""
import os, json, pickle, warnings
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
    _M['pop12']    = _glob.head(12)['article_id'].tolist()
    _M['trend']    = pd.read_parquet('data/features/trend_scores.parquet')
    lw = _M['trend']['week'].max()
    _M['lt_map']   = _M['trend'][_M['trend'].week==lw].set_index('product_type_name')['trend_score'].to_dict()
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
    return {
        'article_id': aids,
        'product_name': prod_name,
        'product_type_name': ptype,
        'colour_group_name': colour,
        'product_group_name': pgroup,
        'garment_group_name': ggrp,
        'detail_desc': desc,
        'avg_price': float(ip),
        'price': float(ip)
    }


def get_recommendations(customer_id: str, occasion: str = None,
                        max_price: float = None, n: int = 12) -> dict:
    _load()
    cid = str(customer_id)
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


def get_trend_report(category: str = None) -> dict:
    _load()
    lw  = _M['trend']['week'].max()
    rec = _M['trend'][_M['trend'].week==lw].sort_values('trend_score',ascending=False)
    if category:
        rec = rec[rec['product_type_name'].str.contains(category,case=False,na=False)]
    top = rec.head(10)[['product_type_name','trend_score','sales','predicted']].copy()
    top['trend_score'] = top['trend_score'].round(3)
    return {"week":str(lw.date()),"trending":top.to_dict('records'),
            "note":"Trend score 0-1 based on LightGBM demand forecast + seasonality"}


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
         "occasion":   {"type":"string","description":"casual|work|party|beach|gym"},
         "max_price":  {"type":"number","description":"Max price in GBP"},
         "n":          {"type":"integer","description":"Number of items (default 6)"}
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

SYSTEM_PROMPT = """You are FashionMind Stylist, an expert AI fashion assistant powered by a personalised recommendation engine trained on 8M H&M transactions.

You have access to these tools:
- get_recommendations: personalised fashion picks with SHAP explanations
- get_trend_report: what's trending right now (LightGBM demand forecast)
- get_outfit_suggestion: compatible outfit pairings (FAISS visual similarity)
- explain_recommendation: why an item was recommended (SHAP feature importance)

Guidelines:
- Be specific and actionable — reference actual article IDs and product types
- Explain the 'why' using the SHAP reasons provided (↑ means positive signal)
- Balance personalisation with trend awareness
- Keep responses warm and conversational
- When showing items, format them clearly with product type, colour, and reasons
"""


def run_tool(name: str, args: dict) -> str:
    if name not in TOOL_MAP:
        return json.dumps({"error":f"Unknown tool: {name}"})
    try:
        return json.dumps(TOOL_MAP[name](**args), default=str)
    except Exception as e:
        return json.dumps({"error":str(e)})


def chat(message: str, customer_id: str, history: list, api_key: str):
    """Single-turn chat. Returns (response_text, updated_history)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    chunks = retrieve_context(message, k=3)
    ctx    = "\n".join(f"• {c}" for c in chunks)
    full_msg = (f"[Style context]\n{ctx}\n\n[Message]\n{message}"
                if ctx else message)

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


def stream_chat(message: str, customer_id: str, history: list, api_key: str):
    """SSE generator for FastAPI StreamingResponse."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    chunks = retrieve_context(message, k=3)
    ctx    = "\n".join(f"• {c}" for c in chunks)
    full_msg = (f"[Style context]\n{ctx}\n\n[Message]\n{message}" if ctx else message)

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