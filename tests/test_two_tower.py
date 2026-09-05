"""Two-tower: shapes, L2-normalisation, a training step that reduces loss,
and the candidate-union helper."""
import numpy as np
import pytest

from src.recsys.two_tower import TwoTower, candidates_union, user_content


def test_candidates_union_dedups_and_caps():
    als = ["a", "b", "c"]
    tt = ["c", "d", "e", "f"]
    out = candidates_union(als, tt, cap=5)
    assert out == ["a", "b", "c", "d", "e"]        # als first, tt fills, no dupes, cap 5


def test_user_content_mean_or_zero():
    content = np.arange(12, dtype=np.float32).reshape(4, 3)
    a2i = {"w": 0, "x": 1, "y": 2, "z": 3}
    assert np.allclose(user_content(["x", "z"], a2i, content), content[[1, 3]].mean(0))
    assert np.allclose(user_content(["missing"], a2i, content), 0)


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n_items, dim = 200, 16
    content = rng.normal(size=(n_items, dim)).astype(np.float32)
    n_pairs = 2000
    item_idx = rng.integers(0, n_items, n_pairs).astype(np.int64)
    # user content = the bought item's content + noise -> learnable signal
    uc = content[item_idx] + rng.normal(0, 0.3, (n_pairs, dim)).astype(np.float32)
    ux = rng.random((n_pairs, 2)).astype(np.float32)
    return torch, content, item_idx, uc, ux


def test_embeddings_are_unit_norm(tiny):
    torch, content, item_idx, uc, ux = tiny
    tt = TwoTower(content_dim=content.shape[1])
    ie = tt.item_emb(torch.tensor(content[:8]))
    ue = tt.user_emb(torch.tensor(uc[:8]), torch.tensor(ux[:8]))
    assert torch.allclose(ie.norm(dim=-1), torch.ones(8), atol=1e-4)
    assert torch.allclose(ue.norm(dim=-1), torch.ones(8), atol=1e-4)


def test_training_reduces_val_loss(tiny):
    torch, content, item_idx, uc, ux = tiny
    tt = TwoTower(content_dim=content.shape[1])
    before = tt._val_loss(torch.arange(400), torch.tensor(uc), torch.tensor(ux),
                          torch.tensor(content), torch.tensor(item_idx),
                          torch.nn.CrossEntropyLoss(), 256)
    best = tt.fit(item_idx, content, {"content": uc, "extra": ux},
                  epochs=6, batch=256, log=lambda *_: None)
    assert best < before                       # learned something from the signal
    emb = tt.all_item_embeddings(content)
    assert emb.shape == (content.shape[0], tt.emb_dim)
