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
    ap.add_argument("--max-per-user", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    content, a2i, aids = load_content()
    print(f"content: {content.shape}")

    u_hist = pickle.load(open("data/features/user_history.pkl", "rb"))
    seg = pd.read_parquet("data/features/customer_segments.parquet").set_index("customer_id")
    seg_lu = seg[["age_norm", "engagement_score"]].to_dict("index")

    uc_rows, ux_rows, item_rows = [], [], []
    for cid, items in u_hist.items():
        items = [a for a in items if a in a2i][: args.max_per_user]
        if not items:
            continue
        uc = user_content(items, a2i, content)
        s = seg_lu.get(cid, {"age_norm": 0.3, "engagement_score": 0.5})
        ux = np.array([s["age_norm"], s["engagement_score"]], np.float32)
        for a in items:
            uc_rows.append(uc); ux_rows.append(ux); item_rows.append(a2i[a])

    uc_rows = np.asarray(uc_rows, np.float32)
    ux_rows = np.asarray(ux_rows, np.float32)
    item_rows = np.asarray(item_rows, np.int64)
    print(f"{len(item_rows):,} (user, item) pairs from {len(u_hist):,} users")

    tt = TwoTower(content_dim=content.shape[1])
    best = tt.fit(item_rows, content,
                  {"content": uc_rows, "extra": ux_rows},
                  epochs=args.epochs, lr=args.lr)
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
