#!/usr/bin/env python3
"""
Train the GRU4Rec next-item model on pre-split, time-ordered histories.

    python scripts/train_sequence.py [--epochs 8] [--max-len 20] [--min-count 5]

user_history.pkl is time-ordered (collaborative_filtering.py sorts by t_dat
before accumulating), so it is a legitimate pre-split sequence source. Only
users with >= 2 items contribute.

Output:
    models/gru4rec.pt          emb + gru + proj state
    models/gru4rec_vocab.pkl   {"a2i": ..., "i2a": ...}
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recsys.sequence import build_vocab, make_examples, GRU4Rec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    u_hist = pickle.load(open("data/features/user_history.pkl", "rb"))
    seqs = [h for h in u_hist.values() if len(h) >= 2]
    print(f"{len(seqs):,} users with >=2 history items "
          f"(of {len(u_hist):,}; the rest can't train a sequence model)")

    a2i, i2a = build_vocab(seqs, min_count=args.min_count)
    X, y = make_examples(seqs, a2i, max_len=args.max_len)
    print(f"vocab {len(a2i):,} items | {len(X):,} (prefix -> next) examples")

    m = GRU4Rec(vocab_size=len(a2i))
    best = m.fit(X, y, epochs=args.epochs, lr=args.lr)
    print(f"best val loss {best:.4f}")

    import torch
    torch.save(m._state(), "models/gru4rec.pt")
    pickle.dump({"a2i": a2i, "i2a": i2a}, open("models/gru4rec_vocab.pkl", "wb"))
    print("wrote models/gru4rec.pt  models/gru4rec_vocab.pkl")


if __name__ == "__main__":
    main()
