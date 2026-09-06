#!/usr/bin/env python3
"""
Train the content two-tower on pre-split purchase pairs and dump item
embeddings for retrieval.

    python scripts/train_two_tower.py [--epochs 8] [--max-per-user 20]

Positives come from user_history.pkl (pre-split, same source ALS uses), so this
does not touch the holdout. Output:
    models/two_tower_item_emb.npy      (n_items, 64) L2-normalised
    models/two_tower_item_ids.npy      article-id order for the rows above
    models/two_tower.pt                item_net + user_net state
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recsys.two_tower import TwoTower, load_content, user_content  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-per-user", type=int, default=15)
    ap.add_argument("--max-users", type=int, default=200_000,
                    help="cap #users to keep the per-user matrices small")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    content, a2i, aids = load_content()
    print(f"content: {content.shape}")

    u_hist = pickle.load(open("data/features/user_history.pkl", "rb"))
    seg = pd.read_parquet("data/features/customer_segments.parquet").set_index("customer_id")
    seg_lu = seg[["age_norm", "engagement_score"]].to_dict("index")

    # per-UNIQUE-user content + extra (small), and (user_row, item_row) pairs
    # as indices — never N_pairs × 128 copies.
    uc_rows, ux_rows, user_of_pair, item_of_pair = [], [], [], []
    for cid, items in list(u_hist.items())[: args.max_users]:
        items = [a for a in items if a in a2i][: args.max_per_user]
        if not items:
            continue
        u = len(uc_rows)
        uc_rows.append(user_content(items, a2i, content))
        s = seg_lu.get(cid, {"age_norm": 0.3, "engagement_score": 0.5})
        ux_rows.append(np.array([s["age_norm"], s["engagement_score"]], np.float32))
        for a in items:
            user_of_pair.append(u); item_of_pair.append(a2i[a])

    uc_rows = np.asarray(uc_rows, np.float32)
    ux_rows = np.asarray(ux_rows, np.float32)
    user_of_pair = np.asarray(user_of_pair, np.int64)
    item_of_pair = np.asarray(item_of_pair, np.int64)
    print(f"{len(item_of_pair):,} pairs from {len(uc_rows):,} users "
          f"(uc matrix {uc_rows.nbytes/1e6:.0f} MB)")

    tt = TwoTower(content_dim=content.shape[1])
    best = tt.fit(item_of_pair, content,
                  {"content": uc_rows, "extra": ux_rows},
                  user_idx=user_of_pair, epochs=args.epochs, lr=args.lr)
    print(f"best val loss {best:.4f}")

    emb = tt.all_item_embeddings(content)
    np.save("models/two_tower_item_emb.npy", emb.astype(np.float32))
    np.save("models/two_tower_item_ids.npy", aids)
    import torch
    torch.save({"item_net": tt.item_net.state_dict(),
                "user_net": tt.user_net.state_dict()}, "models/two_tower.pt")
    print("wrote models/two_tower_item_emb.npy  models/two_tower_item_ids.npy  models/two_tower.pt")


if __name__ == "__main__":
    main()
