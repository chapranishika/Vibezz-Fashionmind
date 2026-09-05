"""Pure-function checks for the bootstrap / slicing helpers in scripts/eval_slices.py."""
import numpy as np
import pytest

from scripts.eval_slices import bootstrap_ci, history_bucket, tercile_labels
import pandas as pd


def test_bootstrap_ci_brackets_the_mean_and_is_ordered():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, size=500)
    mean, lo, hi = bootstrap_ci(x, n_boot=500, rng=rng)
    assert lo < mean < hi
    assert abs(mean - x.mean()) < 1e-9          # mean is the point estimate, exact
    assert hi - lo < 0.05                       # 500 samples -> tight-ish CI


def test_bootstrap_ci_widens_with_less_data():
    rng = np.random.default_rng(1)
    big = bootstrap_ci(rng.normal(0, 1, 2000), n_boot=400, rng=rng)
    small = bootstrap_ci(rng.normal(0, 1, 20), n_boot=400, rng=rng)
    assert (small[2] - small[1]) > (big[2] - big[1])


def test_bootstrap_ci_empty_is_nan_not_crash():
    m, lo, hi = bootstrap_ci([])
    assert np.isnan(m) and np.isnan(lo) and np.isnan(hi)


@pytest.mark.parametrize("n,expected", [
    (0, "cold (0)"), (1, "light (1-4)"), (4, "light (1-4)"),
    (5, "medium (5-19)"), (19, "medium (5-19)"), (20, "heavy (20+)"), (200, "heavy (20+)"),
])
def test_history_bucket(n, expected):
    assert history_bucket(n) == expected


def test_tercile_labels_splits_roughly_evenly():
    s = pd.Series(range(300))
    lab = tercile_labels(s)
    counts = lab.value_counts()
    assert set(counts.index) == {"low", "mid", "high"}
    assert counts.min() >= 90 and counts.max() <= 110   # ~100 each
