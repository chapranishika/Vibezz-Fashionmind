"""Unit tests for the ranking metrics used across Phases 2 and 5."""
import math

from src.recsys.collaborative_filtering import recall_at_k, ndcg_at_k, map_at_k


def test_recall_perfect_and_zero():
    assert recall_at_k(["a", "b"], ["a", "b"], k=10) == 1.0
    assert recall_at_k(["x", "y"], ["a", "b"], k=10) == 0.0


def test_recall_partial_is_hits_over_min_len_k():
    # 1 hit, |actual| = 2  ->  1 / min(2, 10) = 0.5
    assert recall_at_k(["a", "x", "y"], ["a", "b"], k=10) == 0.5
    # k caps the denominator: 1 hit, |actual| = 5, k = 2 -> 1 / 2
    assert recall_at_k(["a", "x"], ["a", "b", "c", "d", "e"], k=2) == 0.5


def test_recall_empty_actual_is_zero():
    assert recall_at_k(["a", "b"], [], k=10) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], k=10) == 1.0


def test_ndcg_rewards_earlier_hits():
    early = ndcg_at_k(["a", "x", "y"], ["a"], k=10)
    late = ndcg_at_k(["x", "y", "a"], ["a"], k=10)
    assert early > late
    assert math.isclose(early, 1.0)  # single relevant item ranked first


def test_ndcg_empty_actual_is_zero():
    assert ndcg_at_k(["a"], [], k=10) == 0.0


def test_map_perfect_and_partial():
    assert map_at_k(["a", "b"], ["a", "b"], k=12) == 1.0
    # hit at rank 2 only: precision 1/2 at that hit, averaged over min(1,12)
    assert math.isclose(map_at_k(["x", "a", "y"], ["a"], k=12), 0.5)


def test_map_ordering_matters():
    assert map_at_k(["a", "x", "b"], ["a", "b"], k=12) > \
           map_at_k(["x", "a", "b"], ["a", "b"], k=12)


def test_all_metrics_handle_short_reclists():
    assert recall_at_k([], ["a"], k=10) == 0.0
    assert ndcg_at_k([], ["a"], k=10) == 0.0
    assert map_at_k([], ["a"], k=10) == 0.0
