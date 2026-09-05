#!/usr/bin/env python3
"""
Bootstrap confidence intervals + sliced metrics on the held-out eval.

`reranker.py` reports point estimates (recall@10 = 0.0123, ...). A point
estimate on ~1000 users is not a result without a ±, and an aggregate hides
whether a model is carried by the head and useless on the tail. This script:

  * rebuilds the SAME held-out test set reranker.py uses (eligible[6000:7000]),
  * scores Popularity / ALS / Full-pipeline per user (reusing the trained
    models/reranker.pkl — it does NOT retrain),
  * reports 95% bootstrap CIs (1000 resamples over users) for every model x
    metric, overall and broken out by user slice:
        history length (cold-start proxy) · age bucket · club tier · price tier

Outputs:
  data/features/eval_slices.csv   (long: model, slice, group, metric, mean, lo, hi, n)
  prints a readable table

Runs against the trained artifacts in models/ and data/features/. Feature-row
construction mirrors reranker.run() — keep the two in sync if the 13 features
change.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ranker.reranker import FEAT_COLS, recall_at_k, ndcg_at_k, map_at_k  # noqa: E402

N_TRAIN, N_TEST, N_CANDS = 6000, 1000, 100
RNG = np.random.default_rng(42)


# ── pure stats helpers (unit-tested in tests/test_eval_slices.py) ────────────

def bootstrap_ci(values, n_boot=1000, alpha=0.05, rng=RNG):
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(v.mean()), float(lo), float(hi))


def history_bucket(n):
    if n == 0:
        return "cold (0)"
    if n < 5:
        return "light (1-4)"
    if n < 20:
        return "medium (5-19)"
    return "heavy (20+)"


def tercile_labels(series):
    """Map a numeric series to 'low'/'mid'/'high' by its own terciles."""
    q1, q2 = series.quantile([1 / 3, 2 / 3])
    return series.apply(lambda x: "low" if x <= q1 else ("mid" if x <= q2 else "high"))


# ── eval ───────────────────────────────────────────────────────────────────

def _load():
    import pickle
    import scipy.sparse as sp
    M = {}
    M["als"] = pickle.load(open("models/als_model.pkl", "rb"))
    M["matrix"] = sp.load_npz("models/user_item_matrix.npz")
    ue = pickle.load(open("models/user_encoder.pkl", "rb"))
    ie = pickle.load(open("models/item_encoder.pkl", "rb"))
    M["cid2u"] = ue["dec"]; M["i2aid"] = ie["enc"]
    M["reranker"] = pickle.load(open("models/reranker.pkl", "rb"))
    M["vis_feats"] = np.load("models/visual_features.npy")
    M["nlp_feats"] = np.load("models/nlp_features.npy")
    va = np.load("models/visual_article_ids.npy", allow_pickle=True)
    na = np.load("models/nlp_article_ids.npy", allow_pickle=True)
    M["vis_a2i"] = {a: i for i, a in enumerate(va)}
    M["nlp_a2i"] = {a: i for i, a in enumerate(na)}
    art = pd.read_parquet("data/features/article_content_features.parquet")
    art["article_id"] = art["article_id"].astype(str)
    M["art_lu"] = art.set_index("article_id")
    pop = pd.read_parquet("data/features/cold_start_popular.parquet")
    M["pop_s"] = pop.set_index("article_id")["score"].to_dict()
    M["pop12"] = pop.head(12)["article_id"].tolist()
    trend = pd.read_parquet("data/features/trend_scores.parquet")
    lw = trend["week"].max()
    M["lt_map"] = trend[trend.week == lw].set_index("product_type_name")["trend_score"].to_dict()
    M["cust"] = pd.read_parquet("data/features/customer_segments.parquet").set_index("customer_id")
    M["u_price"] = pd.read_parquet("data/features/user_avg_price.parquet").set_index("customer_id")["avg_price"].to_dict()
    M["a_price"] = pd.read_parquet("data/features/art_avg_price.parquet").set_index("article_id")["avg_price"].to_dict()
    M["u_hist"] = pickle.load(open("data/features/user_history.pkl", "rb"))
    gt = pd.read_parquet("data/features/ground_truth.parquet")
    M["ground_truth"] = {r.customer_id: list(r.article_id) for _, r in gt.iterrows()}
    M["u_top_ptype"] = {}
    for cid, items in M["u_hist"].items():
        pt = [M["art_lu"].loc[a, "product_type_name"] for a in items if a in M["art_lu"].index]
        if pt:
            M["u_top_ptype"][cid] = max(set(pt), key=pt.count)
    return M


def _feature_rows(M, cid, ids, als_scores):
    up = M["u_price"].get(cid, 0.025)
    ut = M["u_top_ptype"].get(cid, "")
    ue_val = float(M["cust"].loc[cid, "engagement_score"]) if cid in M["cust"].index else 0.5
    ua = float(M["cust"].loc[cid, "age_norm"]) if cid in M["cust"].index else 0.3
    items = M["u_hist"].get(cid, [])[:20]
    vi = [M["vis_a2i"][a] for a in items if a in M["vis_a2i"]]
    ni = [M["nlp_a2i"][a] for a in items if a in M["nlp_a2i"]]
    vc = M["vis_feats"][vi].mean(0) if vi else np.zeros(M["vis_feats"].shape[1])
    nc = M["nlp_feats"][ni].mean(0) if ni else np.zeros(64)
    vc /= (np.linalg.norm(vc) + 1e-8); nc /= (np.linalg.norm(nc) + 1e-8)
    out = []
    for rank, (iidx, alsc) in enumerate(zip(ids, als_scores)):
        aids = M["i2aid"][int(iidx)].lstrip("0")
        if aids in M["art_lu"].index:
            r = M["art_lu"].loc[aids]
            ptype = str(r["product_type_name"]); pi = int(r["product_type_name_idx"])
            ci = int(r["colour_group_name_idx"]); gi = int(r["garment_group_name_idx"])
        else:
            ptype = ""; pi = ci = gi = 0
        vv = M["vis_feats"][M["vis_a2i"][aids]] if aids in M["vis_a2i"] else np.zeros(M["vis_feats"].shape[1])
        nv = M["nlp_feats"][M["nlp_a2i"][aids]] if aids in M["nlp_a2i"] else np.zeros(64)
        ip = M["a_price"].get(aids, 0.025)
        out.append([float(alsc), float(np.dot(vv, vc)), float(np.dot(nv, nc)),
                    M["lt_map"].get(ptype, 0.5), float(max(0, 1 - abs(ip - up) / (up + 1e-6))),
                    int(ptype == ut), float(M["pop_s"].get(aids, 0.0)),
                    1 - rank / N_CANDS, ue_val, ua, pi, ci, gi])
    return out


def per_user_metrics(M, test_cids):
    """One row per (user), with recall/ndcg/map for each of the 3 models."""
    recs = []
    for cid in test_cids:
        act = M["ground_truth"].get(cid, [])
        if not act:
            continue
        uidx = int(M["cid2u"][cid]) if cid in M["cid2u"] else None

        row = {"customer_id": cid}
        # popularity
        row["pop_recall"] = recall_at_k(M["pop12"], act)
        row["pop_ndcg"] = ndcg_at_k(M["pop12"], act)
        row["pop_map"] = map_at_k(M["pop12"], act)

        if uidx is None:
            for m in ("als", "pipe"):
                row[f"{m}_recall"] = row["pop_recall"]
                row[f"{m}_ndcg"] = row["pop_ndcg"]
                row[f"{m}_map"] = row["pop_map"]
            recs.append(row)
            continue

        try:
            ids100, als100 = M["als"].recommend(uidx, M["matrix"][uidx], N=N_CANDS,
                                                filter_already_liked_items=True)
        except Exception:
            continue
        als_rec = [M["i2aid"][int(i)] for i in ids100[:12]]
        row["als_recall"] = recall_at_k(als_rec, act)
        row["als_ndcg"] = ndcg_at_k(als_rec, act)
        row["als_map"] = map_at_k(als_rec, act)

        frows = _feature_rows(M, cid, ids100, als100)
        if frows:
            sc = M["reranker"].predict(np.array(frows))
            pipe_rec = [M["i2aid"][int(ids100[j])] for j in np.argsort(-sc)]
        else:
            pipe_rec = als_rec
        row["pipe_recall"] = recall_at_k(pipe_rec, act)
        row["pipe_ndcg"] = ndcg_at_k(pipe_rec, act)
        row["pipe_map"] = map_at_k(pipe_rec, act)
        recs.append(row)
    return pd.DataFrame(recs)


MODELS = {"Popularity": "pop", "ALS": "als", "Full pipeline": "pipe"}
METRICS = ["recall", "ndcg", "map"]


def summarise(df, slice_name, group_series):
    out = []
    df = df.assign(_grp=group_series.values)
    for grp, sub in df.groupby("_grp"):
        for mname, mp in MODELS.items():
            for met in METRICS:
                mean, lo, hi = bootstrap_ci(sub[f"{mp}_{met}"].values)
                out.append(dict(model=mname, slice=slice_name, group=str(grp),
                                metric=f"{met}@{10 if met != 'map' else 12}",
                                mean=mean, lo=lo, hi=hi, n=len(sub)))
    return out


def main():
    M = _load()
    eligible = [c for c in M["ground_truth"] if c in M["cid2u"]]
    test_cids = eligible[N_TRAIN:N_TRAIN + N_TEST]
    print(f"scoring {len(test_cids)} held-out users x {N_CANDS} candidates x 3 models ...")
    du = per_user_metrics(M, test_cids)
    print(f"  {len(du)} users with >=1 ground-truth item")

    hist_len = du["customer_id"].map(lambda c: len(M["u_hist"].get(c, [])))
    seg = M["cust"].reindex(du["customer_id"])
    price = du["customer_id"].map(lambda c: M["u_price"].get(c, 0.025))

    rows = []
    rows += summarise(du, "overall", pd.Series(["all"] * len(du)))
    rows += summarise(du, "history", hist_len.map(history_bucket))
    rows += summarise(du, "age_bucket", seg["age_bucket"].fillna("unknown").reset_index(drop=True))
    rows += summarise(du, "club", seg["club"].fillna("unknown").reset_index(drop=True))
    rows += summarise(du, "price_tier", tercile_labels(price).reset_index(drop=True))

    res = pd.DataFrame(rows)
    Path("data/features").mkdir(parents=True, exist_ok=True)
    res.to_csv("data/features/eval_slices.csv", index=False)

    pd.set_option("display.width", 140)
    for slc in ["overall", "history", "age_bucket", "club", "price_tier"]:
        print(f"\n── {slc} ──")
        piv = res[(res.slice == slc) & (res.metric == "ndcg@10")].copy()
        piv["ci"] = piv.apply(lambda r: f"{r['mean']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]", axis=1)
        print(piv.pivot(index="group", columns="model", values="ci").to_string())
    print("\nwrote data/features/eval_slices.csv")


if __name__ == "__main__":
    main()
