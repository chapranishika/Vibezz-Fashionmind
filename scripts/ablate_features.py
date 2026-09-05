#!/usr/bin/env python3
"""
Feature ablation for the LambdaRank re-ranker.

"13 signals" is a count until you show which ones carry the model. For each of
the 13 features this zeroes that column across every candidate row, re-ranks
with the *same* trained reranker.pkl (no retrain), and measures the change in
held-out NDCG@10 vs the full-feature model — on the exact held-out set
reranker.py / eval_slices.py use.

Zeroing (not permuting): a tree ranker that never split on a feature is
unaffected; a big drop means the model leans on it. Reported with a 95%
bootstrap CI on the per-user delta.

Output: data/features/feature_ablation.csv + a printed table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ranker.reranker import FEAT_COLS, ndcg_at_k                      # noqa: E402
from scripts.eval_slices import _load, _feature_rows, bootstrap_ci, N_TRAIN, N_TEST, N_CANDS  # noqa: E402


def _score(reranker, i2aid, ids, frows, act, drop_col=None):
    X = np.array(frows, dtype=float)
    if drop_col is not None:
        X = X.copy()
        X[:, drop_col] = 0.0
    order = np.argsort(-reranker.predict(X))
    rec = [i2aid[int(ids[j])] for j in order]
    return ndcg_at_k(rec, act)


def main():
    M = _load()
    eligible = [c for c in M["ground_truth"] if c in M["cid2u"]]
    test_cids = eligible[N_TRAIN:N_TRAIN + N_TEST]
    print(f"ablating {len(FEAT_COLS)} features over {len(test_cids)} held-out users ...")

    # per-user NDCG for the full model and for each single-feature ablation
    full, ablate = [], {c: [] for c in FEAT_COLS}
    for cid in test_cids:
        act = M["ground_truth"].get(cid, [])
        uidx = int(M["cid2u"][cid]) if cid in M["cid2u"] else None
        if not act or uidx is None:
            continue
        try:
            ids, als = M["als"].recommend(uidx, M["matrix"][uidx], N=N_CANDS,
                                          filter_already_liked_items=True)
        except Exception:
            continue
        frows = _feature_rows(M, cid, ids, als)
        if not frows:
            continue
        full.append(_score(M["reranker"], M["i2aid"], ids, frows, act))
        for j, col in enumerate(FEAT_COLS):
            ablate[col].append(_score(M["reranker"], M["i2aid"], ids, frows, act, drop_col=j))

    full = np.array(full)
    base_mean, base_lo, base_hi = bootstrap_ci(full)
    print(f"\nfull model NDCG@10: {base_mean:.4f} [{base_lo:.4f}, {base_hi:.4f}]  (n={len(full)})")

    rows = []
    for col in FEAT_COLS:
        a = np.array(ablate[col])
        delta = a - full                                  # per-user change when col is zeroed
        d_mean, d_lo, d_hi = bootstrap_ci(delta)
        rows.append(dict(feature=col, ndcg_without=float(a.mean()),
                         delta=d_mean, delta_lo=d_lo, delta_hi=d_hi,
                         matters=bool(d_hi < 0)))          # CI of the drop excludes 0
    res = pd.DataFrame(rows).sort_values("delta")          # most negative delta = most important
    res.to_csv("data/features/feature_ablation.csv", index=False)

    print("\nfeature                 NDCG w/o    Δ (95% CI)                 relies on it?")
    for _, r in res.iterrows():
        rel = (r["delta"] / base_mean * 100) if base_mean else 0
        flag = "yes" if r["matters"] else ""
        print(f"  {r['feature']:<20s}  {r['ndcg_without']:.4f}   "
              f"{r['delta']:+.4f} [{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]  {rel:+5.0f}%  {flag}")
    print("\nwrote data/features/feature_ablation.csv")


if __name__ == "__main__":
    main()
