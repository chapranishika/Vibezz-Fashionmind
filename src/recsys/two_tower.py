"""
Content two-tower retrieval — a second candidate source alongside ALS.

ALS's candidate recall@100 is 0.036: it never surfaces 96% of what users go on
to buy, and no re-ranker can beat that ceiling. A two-tower retrieves by
*content* (visual + text features that already exist) rather than co-purchase,
so its misses are different from ALS's — the union covers more.

Towers
------
ItemTower : MLP over concat(visual_64, nlp_64) -> 64-d, L2-normalised.
UserTower : MLP over concat(mean history-item content_128, [age_norm,
            engagement]) -> 64-d, L2-normalised.

Training  : in-batch sampled softmax. A batch is B (user, bought-item) pairs
            from the pre-split window (user_history.pkl); every other item in
            the batch is a negative. score = <u, i> / temperature.

Serving   : item embeddings -> FAISS inner-product index; a user embedding is
            one ANN query. `candidates_for` unions the two-tower top-K with an
            ALS list.
"""
from __future__ import annotations

import numpy as np


# ── feature assembly (numpy, no torch needed) ──────────────────────────────

def load_content(models_dir="models"):
    vf = np.load(f"{models_dir}/visual_features.npy").astype(np.float32)
    nf = np.load(f"{models_dir}/nlp_features.npy").astype(np.float32)
    aids = np.load(f"{models_dir}/visual_article_ids.npy", allow_pickle=True)
    aids = np.array([str(a) for a in aids])
    content = np.concatenate([vf, nf], axis=1)                 # (n_items, 128)
    return content, {a: i for i, a in enumerate(aids)}, aids


def user_content(hist_item_ids, a2i, content):
    idx = [a2i[a] for a in hist_item_ids if a in a2i]
    if not idx:
        return np.zeros(content.shape[1], dtype=np.float32)
    return content[idx].mean(0)


# ── model (torch) ─────────────────────────────────────────────────────────

def _mlp(torch, d_in, d_out, hidden=128):
    nn = torch.nn
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, d_out))


class TwoTower:
    def __init__(self, content_dim=128, user_extra=2, emb_dim=64, temp=0.05, seed=42):
        import torch
        torch.manual_seed(seed)
        self.torch = torch
        self.item_net = _mlp(torch, content_dim, emb_dim)
        self.user_net = _mlp(torch, content_dim + user_extra, emb_dim)
        self.temp = temp
        self.emb_dim = emb_dim

    def _norm(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def item_emb(self, content_batch):
        return self._norm(self.item_net(content_batch))

    def user_emb(self, user_content_batch, extra_batch):
        x = self.torch.cat([user_content_batch, extra_batch], dim=-1)
        return self._norm(self.user_net(x))

    def parameters(self):
        return list(self.item_net.parameters()) + list(self.user_net.parameters())

    # ---- training ----

    def fit(self, item_idx, content, user_extra, *, user_idx=None,
            epochs=8, batch=1024, lr=1e-3, val_frac=0.1, log=print):
        """item_idx: (N,) item row per training pair.
        user_extra["content"]: (U, 128) per-*unique-user* content (not per pair)
        user_extra["extra"]:   (U, 2)   per-unique-user [age_norm, engagement]
        user_idx: (N,) row into those U — which user each pair belongs to.
        Kept as indices so N ~millions of pairs don't materialise N×128 copies."""
        torch = self.torch
        content_t = torch.tensor(content)
        uc_all = torch.tensor(user_extra["content"])        # (U, 128)
        ux_all = torch.tensor(user_extra["extra"])          # (U, 2)
        item_idx = torch.tensor(item_idx, dtype=torch.long) # (N,)
        user_idx = (torch.tensor(user_idx, dtype=torch.long)
                    if user_idx is not None else torch.arange(len(item_idx)))
        n = len(item_idx)
        perm = torch.randperm(n)
        n_val = int(n * val_frac)
        val, tr = perm[:n_val], perm[n_val:]
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        best, best_state, patience = 1e9, None, 2

        for ep in range(1, epochs + 1):
            self.item_net.train(); self.user_net.train()
            order = tr[torch.randperm(len(tr))]
            tot = 0.0
            for s in range(0, len(order), batch):
                b = order[s:s + batch]
                if len(b) < 2:
                    continue
                u = user_idx[b]
                ue = self.user_emb(uc_all[u], ux_all[u])                # (B, d)
                ie = self.item_emb(content_t[item_idx[b]])              # (B, d)
                logits = ue @ ie.T / self.temp                         # (B, B)
                target = torch.arange(len(b))
                loss = loss_fn(logits, target)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(b)
            vl = self._val_loss(val, user_idx, uc_all, ux_all, content_t, item_idx, loss_fn, batch)
            log(f"  epoch {ep}: train {tot/len(order):.4f}  val {vl:.4f}")
            if vl < best - 1e-4:
                best, best_state, wait = vl, self._state(), 0
            else:
                wait = getattr(self, "_wait", 0) + 1
                self._wait = wait
                if wait >= patience:
                    log("  early stop"); break
        if best_state:
            self._load(best_state)
        return best

    def _val_loss(self, val, user_idx, uc_all, ux_all, content, item_idx, loss_fn, batch):
        torch = self.torch
        self.item_net.eval(); self.user_net.eval()
        with torch.no_grad():
            tot = 0.0
            for s in range(0, len(val), batch):
                b = val[s:s + batch]
                if len(b) < 2:
                    continue
                u = user_idx[b]
                ue = self.user_emb(uc_all[u], ux_all[u])
                ie = self.item_emb(content[item_idx[b]])
                logits = ue @ ie.T / self.temp
                tot += loss_fn(logits, torch.arange(len(b))).item() * len(b)
            return tot / max(len(val), 1)

    def _state(self):
        return {k: v.clone() for k, v in
                {**self.item_net.state_dict(), **{"u." + k: v for k, v in self.user_net.state_dict().items()}}.items()}

    def _load(self, st):
        self.item_net.load_state_dict({k: v for k, v in st.items() if not k.startswith("u.")})
        self.user_net.load_state_dict({k[2:]: v for k, v in st.items() if k.startswith("u.")})

    # ---- serving ----

    def all_item_embeddings(self, content):
        torch = self.torch
        self.item_net.eval()
        with torch.no_grad():
            return self.item_emb(torch.tensor(content)).cpu().numpy()

    def embed_user(self, u_content, u_extra):
        torch = self.torch
        self.user_net.eval()
        with torch.no_grad():
            return self.user_emb(torch.tensor(u_content[None]),
                                 torch.tensor(np.asarray(u_extra, np.float32)[None])).cpu().numpy()[0]


def candidates_union(als_ids, tt_ids, cap=100):
    """ALS list first (co-purchase signal), then two-tower fills to `cap`."""
    seen, out = set(), []
    for a in list(als_ids) + list(tt_ids):
        if a not in seen:
            seen.add(a); out.append(a)
        if len(out) >= cap:
            break
    return out
