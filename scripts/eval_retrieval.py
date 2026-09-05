#!/usr/bin/env python3
"""
Candidate recall@100 — the retrieval ceiling — for ALS, the content two-tower,
and their union, on the same held-out users reranker.py / eval_slices.py use.

recall@100 here = fraction of a user's held-out ground-truth items that appear
anywhere in the 100-candidate pool. It's the hard cap on everything the
re-ranker can do (0.036 for ALS alone). The question: does adding a
content-retrieval source raise it?

    python scripts/eval_retrieval.py

Needs models/two_tower_item_emb.npy (scripts/train_two_tower.py).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recsys.two_tower import load_content, user_content, candidates_union  # noqa: E402
from scripts.eval_slices import bootstrap_ci                                   # noqa: E402

N_TRAIN, N_TEST, N_CANDS = 6000, 1000, 100


def _recall(cands, gt):
    return len(set(cands) & set(gt)) / min(len(gt), N_CANDS) if gt else 0.0


def main():
    import scipy.sparse as sp
    als = pickle.load(open("models/als_model.pkl", "rb"))
    matrix = sp.load_npz("models/user_item_matrix.npz")
    ue = pickle.load(open("models/user_encoder.pkl", "rb"))
    ie = pickle.load(open("models/item_encoder.pkl", "rb"))
    cid2u, i2aid = ue["dec"], ie["enc"]

    gt_df = pd.read_parquet("data/features/ground_truth.parquet")
    ground_truth = {r.customer_id: list(r.article_id) for _, r in gt_df.iterrows()}

    content, a2i, aids = load_content()
    tt_emb = np.load("models/two_tower_item_emb.npy")            # (n_items, 64)
    tt_ids = np.array([str(a) for a in np.load("models/two_tower_item_ids.npy", allow_pickle=True)])
    u_hist = pickle.load(open("data/features/user_history.pkl", "rb"))
    seg = pd.read_parquet("data/features/customer_segments.parquet").set_index("customer_id")
    seg_lu = seg[["age_norm", "engagement_score"]].to_dict("index")

    import torch
    from src.recsys.two_tower import TwoTower
    tt = TwoTower(content_dim=content.shape[1])
    st = torch.load("models/two_tower.pt")
    tt.item_net.load_state_dict(st["item_net"]); tt.user_net.load_state_dict(st["user_net"])

    eligible = [c for c in ground_truth if c in cid2u]
    test_cids = eligible[N_TRAIN:N_TRAIN + N_TEST]
    print(f"{len(test_cids)} held-out users")

    rec = {"ALS": [], "two-tower": [], "ALS ∪ two-tower": []}
    for cid in test_cids:
        gt = ground_truth.get(cid, [])
        if not gt:
            continue
        uidx = int(cid2u[cid]) if cid in cid2u else None
        als_ids = []
        if uidx is not None:
            try:
                ids, _ = als.recommend(uidx, matrix[uidx], N=N_CANDS, filter_already_liked_items=True)
                als_ids = [i2aid[int(i)] for i in ids]
            except Exception:
                pass

        items = [a for a in u_hist.get(cid, []) if a in a2i][:20]
        uc = user_content(items, a2i, content)
        s = seg_lu.get(cid, {"age_norm": 0.3, "engagement_score": 0.5})
        u_emb = tt.embed_user(uc, [s["age_norm"], s["engagement_score"]])
        top = np.argpartition(-(tt_emb @ u_emb), N_CANDS)[:N_CANDS]
        tt_cand = list(tt_ids[top[np.argsort(-(tt_emb[top] @ u_emb))]])

        rec["ALS"].append(_recall(als_ids, gt))
        rec["two-tower"].append(_recall(tt_cand, gt))
        rec["ALS ∪ two-tower"].append(_recall(candidates_union(als_ids, tt_cand, N_CANDS), gt))

    rows = []
    print("\ncandidate recall@100 (the ceiling):")
    for k, v in rec.items():
        m, lo, hi = bootstrap_ci(v)
        rows.append(dict(source=k, recall_at_100=m, ci_lo=lo, ci_hi=hi, n=len(v)))
        print(f"  {k:<18s} {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(rows).to_csv("data/features/retrieval_ceiling.csv", index=False)
    print("\nwrote data/features/retrieval_ceiling.csv")


if __name__ == "__main__":
    main()
