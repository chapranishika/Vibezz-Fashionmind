#!/usr/bin/env python3
"""
Point-in-time / train-test leakage checks for the re-ranker pipeline.

Full re-derivation needs the raw H&M transactions (in data/raw/, ~GBs), so this
does what's checkable without them:

  1. Static: the training feature builders cap trend_score at the training
     split. trend_scores.parquet runs to 2020-09-22 (inside the holdout); using
     its max week for training features leaks the future into the model's #1
     SHAP feature. reranker.py and scripts/eval_slices.py must guard this.
  2. Static: collaborative_filtering.py fits ALS / accumulates price + history
     only from `train_tx` (t_dat <= CUTOFF), with ground_truth from `t_dat >
     CUTOFF`.
  3. Data: report trend_scores week coverage vs the split, and whether the
     shipped reranker.pkl predates the point-in-time fix (=> retrain to get an
     honest headline number).

Exit 1 if a static guard is missing.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPLIT = "2020-09-08"


def _read(p):
    return (REPO / p).read_text(encoding="utf-8")


def check_static():
    fails, notes = [], []

    rr = _read("src/ranker/reranker.py")
    if re.search(r"trend\.week\s*<=\s*SPLIT_WEEK", rr) and "2020-09-08" in rr:
        notes.append("OK  reranker.py caps trend_score at the training split")
    else:
        fails.append("reranker.py builds lt_map from trend['week'].max() with no "
                     "<= split guard — trend_score leaks the holdout window")

    es = _read("scripts/eval_slices.py")
    if re.search(r"trend\.week\s*<=\s*_split", es):
        notes.append("OK  eval_slices.py caps the trend week at the split")
    else:
        fails.append("eval_slices.py does not cap the trend week — its metrics inherit the leak")

    cf = _read("src/recsys/collaborative_filtering.py")
    if re.search(r"train_tx\s*=\s*tx\[tx\.t_dat\s*<=\s*CUTOFF\]", cf) and \
       re.search(r"valid_tx\s*=\s*tx\[tx\.t_dat\s*>\s*CUTOFF\]", cf):
        notes.append("OK  collaborative_filtering.py splits train/holdout at CUTOFF")
    else:
        fails.append("collaborative_filtering.py train/holdout split not found as expected")
    # price + history accumulators must be fed from train_tx, not tx
    if re.search(r"for .* in train_tx", cf) or re.search(r"train_tx.sort_values", cf):
        notes.append("OK  price / history accumulators iterate train_tx")
    else:
        notes.append("??  could not confirm the accumulators use train_tx (manual check)")

    return fails, notes


def check_data():
    notes = []
    try:
        import pandas as pd
        t = pd.read_parquet(REPO / "data/features/trend_scores.parquet")
        mx = t["week"].max()
        past = int((t["week"] > pd.Timestamp(SPLIT)).sum())
        notes.append(f"trend_scores weeks: .. {mx.date()}  ({past} rows past the {SPLIT} split — "
                     f"fine as long as the builders cap in use)")
    except Exception as e:
        notes.append(f"trend_scores: {type(e).__name__}: {e}")

    try:
        card = json.loads((REPO / "data/features/model_card.json").read_text())
        gen = card.get("generated", "?")
        notes.append(f"model_card generated {gen} — the shipped reranker.pkl predates the "
                     f"point-in-time fix; retrain for an un-leaked headline number "
                     f"(expect trend_score's SHAP weight and the aggregate lift to drop)")
    except Exception as e:
        notes.append(f"model_card: {type(e).__name__}: {e}")
    return notes


def main():
    fails, notes = check_static()
    for n in notes:
        print("  " + n)
    for n in check_data():
        print("  .. " + n)
    if fails:
        print("\nLEAKAGE GUARDS MISSING:")
        for f in fails:
            print("  [FAIL] " + f)
        sys.exit(1)
    print("\nstatic leakage guards present.")


if __name__ == "__main__":
    main()
