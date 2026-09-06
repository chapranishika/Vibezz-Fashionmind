"""
GRU4Rec-style next-item model — a recency-aware candidate source.

ALS and the content two-tower are order-blind: they see a user's history as a
set. Fashion is sequential — season, recency, "bought jeans -> wants a top". A
GRU over the (time-ordered) history predicts the next item; its top-K is a
third candidate source whose hits differ from the other two.

Model
-----
item embedding table (|V| x d) -> GRU(d -> h) -> Linear(h -> d) -> scores =
projected_hidden @ item_emb.T over the whole vocab (full softmax; the vocab of
history items is ~tens of thousands, fine on CPU for a small run).

Training
--------
Autoregressive next-item on pre-split sequences (user_history.pkl is
time-ordered). A length-L history yields L-1 (prefix -> next) examples. Users
with <2 history items contribute nothing — the same sparsity that caps the
rest of this project.
"""
from __future__ import annotations

import numpy as np


def build_vocab(histories, min_count=1):
    """histories: iterable of lists of article_id (str). Returns id<->idx maps.
    idx 0 is reserved for padding."""
    from collections import Counter
    c = Counter(a for h in histories for a in h)
    items = [a for a, n in c.items() if n >= min_count]
    a2i = {a: i + 1 for i, a in enumerate(sorted(items))}
    i2a = {i: a for a, i in a2i.items()}
    return a2i, i2a


def make_examples(histories, a2i, max_len=20):
    """(prefix, next) pairs as padded index arrays. prefix is left-padded to
    max_len."""
    X, y = [], []
    for h in histories:
        seq = [a2i[a] for a in h if a in a2i]
        for t in range(1, len(seq)):
            pref = seq[max(0, t - max_len):t]
            pref = [0] * (max_len - len(pref)) + pref
            X.append(pref)
            y.append(seq[t])
    return np.asarray(X, np.int64), np.asarray(y, np.int64)


class GRU4Rec:
    def __init__(self, vocab_size, emb_dim=64, hidden=64, seed=42):
        import torch
        torch.manual_seed(seed)
        self.torch = torch
        nn = torch.nn
        self.V = vocab_size
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, emb_dim)
        self.emb_dim = emb_dim

    def parameters(self):
        return (list(self.emb.parameters()) + list(self.gru.parameters())
                + list(self.proj.parameters()))

    def _hidden(self, x):
        e = self.emb(x)                          # (B, L, d)
        _, h = self.gru(e)                       # h: (1, B, hidden)
        return self.proj(h[-1])                  # (B, d)

    def scores(self, x):
        """(B, L) index sequences -> (B, V) scores over items 1..V."""
        h = self._hidden(x)                      # (B, d)
        W = self.emb.weight[1:]                  # (V, d)  drop padding row
        return h @ W.T

    def fit(self, X, y, *, epochs=8, batch=512, lr=1e-3, val_frac=0.1, log=print):
        torch = self.torch
        X_t, y_t = torch.tensor(X), torch.tensor(y) - 1     # labels 0..V-1
        n = len(X_t)
        perm = torch.randperm(n)
        nv = int(n * val_frac)
        val, tr = perm[:nv], perm[nv:]
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        lf = torch.nn.CrossEntropyLoss()
        best, best_state, wait = 1e9, None, 0

        for ep in range(1, epochs + 1):
            self.emb.train(); self.gru.train(); self.proj.train()
            order = tr[torch.randperm(len(tr))]
            tot = 0.0
            for s in range(0, len(order), batch):
                b = order[s:s + batch]
                loss = lf(self.scores(X_t[b]), y_t[b])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(b)
            vl = self._val_loss(val, X_t, y_t, lf, batch)
            log(f"  epoch {ep}: train {tot/max(len(order),1):.4f}  val {vl:.4f}")
            if vl < best - 1e-4:
                best, best_state, wait = vl, self._state(), 0
            else:
                wait += 1
                if wait >= 2:
                    log("  early stop"); break
        if best_state:
            self._load(best_state)
        return best

    def _val_loss(self, val, X_t, y_t, lf, batch):
        torch = self.torch
        self.emb.eval(); self.gru.eval(); self.proj.eval()
        with torch.no_grad():
            tot = 0.0
            for s in range(0, len(val), batch):
                b = val[s:s + batch]
                tot += lf(self.scores(X_t[b]), y_t[b]).item() * len(b)
            return tot / max(len(val), 1)

    def _state(self):
        return {"emb": self.emb.state_dict(), "gru": self.gru.state_dict(),
                "proj": self.proj.state_dict()}

    def _load(self, st):
        self.emb.load_state_dict(st["emb"]); self.gru.load_state_dict(st["gru"])
        self.proj.load_state_dict(st["proj"])

    def topk(self, seq_idx, k=100):
        """seq_idx: 1-D list/array of item indices (already mapped). -> item
        indices (1..V) ranked."""
        torch = self.torch
        x = torch.tensor(np.asarray(seq_idx, np.int64)[None])
        with torch.no_grad():
            sc = self.scores(x)[0].cpu().numpy()
        top = np.argpartition(-sc, min(k, len(sc) - 1))[:k]
        return (top[np.argsort(-sc[top])] + 1).tolist()      # back to 1..V
