# Re-fit the re-ranker on ALS ∪ GRU candidates

`RECS_USE_GRU` (in `get_recommendations`) already unions GRU4Rec candidates
ahead of ALS before re-ranking, but the shipped `reranker.pkl` was trained on
ALS-only candidates, so:

- a GRU-only item gets `als_score = 0` → the model reads it as a weak-CF item,
- `rank_norm` for it is its position in the *merged* list, a different
  distribution than training,
- there is no feature carrying the GRU signal itself.

That's why the hook is behind a flag. This is the re-fit that makes it real.

## Change (three sites, one feature)

1. **`FEAT_COLS`** (`src/ranker/reranker.py`): append `gru_score`. 13 → 14.
   `FEAT_LABELS['gru_score'] = 'Sequence match'`.

2. **`reranker.py`, both feature-build loops** (train matrix ~L122, held-out
   eval ~L216):
   - after `als.recommend(..., N=100)`, also compute
     `gru_ids = GRU.topk(user_seq, k=50)` with already-bought filtered
     (load `models/gru4rec.pt` + `gru4rec_vocab.pkl` once at the top of `run()`;
     skip the whole block if the artifacts are absent so the script still runs
     ALS-only).
   - build the candidate pool as `union(als_ids, gru_ids, cap≈120)` keeping,
     per candidate: `als_score` (real if from ALS else 0.0), `gru_score`
     (softmax prob or `1 - gru_rank/50` if from GRU else 0.0), `rank_norm` =
     `1 - pool_pos / len(pool)`.
   - `label` unchanged (ground-truth membership).
   - append `gru_score` to each feature row in `FEAT_COLS` order.

3. **`src/genai/stylist_chatbot.py`, `get_recommendations` serving loop** (~L292):
   the existing `RECS_USE_GRU` block already merges `ids`/`als_scores`; carry a
   parallel `gru_scores` list through the merge and append it as the 14th
   feature value. Keep the order identical to `FEAT_COLS`.

## Retrain + verify

```
python src/ranker/reranker.py            # rebuilds reranker.pkl on the mixed pool
python scripts/eval_slices.py            # bootstrap CIs vs ALS
python scripts/ablate_features.py        # does gru_score carry weight?
```

Expect: `gru_score` shows up as a real feature in the ablation (CI excludes 0);
the pipeline's candidate recall@100 rises toward the GRU ceiling (0.076) because
the pool now contains GRU's hits; NDCG@10 hopefully rises too. If `gru_score`
ablates to ~0, the re-ranker isn't using the sequence signal and the honest
move is to serve GRU candidates directly (skip the second stage) — the
retrieval-ceiling numbers already argue for that.

## Then

`RECS_USE_GRU=1` in the Space secrets, redeploy, and the flag is no longer
experimental. Consider making it the default once an online A/B
(`docs/AB_TEST_PLAN.md`, swap `control` = ALS-only for `control` = current) is
green.
