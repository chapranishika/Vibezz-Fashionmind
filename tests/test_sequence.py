"""GRU4Rec: vocab / example construction, and a training step that learns a
recency pattern a set model can't."""
import numpy as np
import pytest

from src.recsys.sequence import build_vocab, make_examples, GRU4Rec


def test_build_vocab_reserves_zero_for_padding():
    a2i, i2a = build_vocab([["a", "b"], ["b", "c"]])
    assert 0 not in a2i.values()
    assert set(a2i) == {"a", "b", "c"}
    assert i2a[a2i["b"]] == "b"


def test_make_examples_left_pads_and_shifts():
    a2i, _ = build_vocab([["a", "b", "c", "d"]])
    X, y = make_examples([["a", "b", "c", "d"]], a2i, max_len=3)
    assert X.shape == (3, 3) and len(y) == 3
    assert X[0].tolist() == [0, 0, a2i["a"]] and y[0] == a2i["b"]      # predict b from [a]
    assert X[2].tolist() == [a2i["a"], a2i["b"], a2i["c"]] and y[2] == a2i["d"]


def test_users_with_one_item_produce_no_examples():
    a2i, _ = build_vocab([["a"]])
    X, y = make_examples([["a"]], a2i)
    assert len(X) == 0


def test_training_learns_a_sequential_rule():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    V = 20
    # rule: item i is always followed by item (i % V) + 1  -> pure recency signal
    seqs = []
    for _ in range(400):
        start = rng.integers(1, V + 1)
        s = [start]
        for _ in range(5):
            s.append((s[-1] % V) + 1)
        seqs.append([f"i{j}" for j in s])
    a2i, _ = build_vocab(seqs)
    X, y = make_examples(seqs, a2i, max_len=5)

    m = GRU4Rec(vocab_size=len(a2i))
    before = m._val_loss(torch.arange(len(X)), torch.tensor(X), torch.tensor(y) - 1,
                         torch.nn.CrossEntropyLoss(), 256)
    best = m.fit(X, y, epochs=12, batch=128, log=lambda *_: None)
    assert best < before * 0.5                       # the rule is very learnable

    # after "i3", the next item should rank i4 (=(3%20)+1) at the top
    nxt = m.topk([a2i["i3"]], k=3)
    assert nxt[0] == a2i["i4"]
