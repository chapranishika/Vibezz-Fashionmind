"""Data-contract validators: clean frames pass, deliberately broken ones fail."""
import numpy as np
import pandas as pd

from src.data import contracts as C


def _articles(n=100_000):
    return pd.DataFrame({
        "article_id": [f"{i:09d}" for i in range(n)],
        "product_type_name_idx": np.random.randint(0, 130, n),
        "colour_group_name_idx": np.random.randint(0, 50, n),
        "is_clothing": np.random.randint(0, 2, n),
        "desc_length": np.random.randint(0, 200, n),
    })


def _segments(n=1_200_000):
    return pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "age": np.random.randint(16, 90, n).astype(float),
        "age_norm": np.random.rand(n),
        "age_bucket": np.random.choice(["young", "mature"], n),
        "club": np.random.choice(["ACTIVE", "PRE-CREATE"], n),
        "is_active_member": np.random.randint(0, 2, n),
        "receives_news": np.random.randint(0, 2, n),
        "engagement_score": np.random.rand(n),
    })


def test_clean_articles_pass():
    assert C.check_articles(_articles()) == []


def test_duplicate_article_id_is_error():
    df = _articles(1000)
    df.loc[1, "article_id"] = df.loc[0, "article_id"]
    errs = [v for v in C.check_articles(df) if v.severity == "error"]
    assert any("unique" in v.check for v in errs)


def test_negative_idx_is_error():
    df = _articles(1000)
    df.loc[5, "product_type_name_idx"] = -1
    assert any(v.severity == "error" and ">= 0" in v.check for v in C.check_articles(df))


def test_engagement_score_out_of_range_is_error():
    df = _segments(2000)
    df.loc[3, "engagement_score"] = 1.5
    assert any(v.severity == "error" and "engagement_score" in v.check
               for v in C.check_customer_segments(df))


def test_price_negative_is_error():
    df = pd.DataFrame({"article_id": ["a", "b"], "avg_price": [0.1, -0.2]})
    assert any(v.severity == "error" for v in C.check_price(df, "art_avg_price", "article_id"))


def test_referential_orphans_flagged():
    articles = pd.DataFrame({"article_id": ["1", "2", "3"]})
    art_price = pd.DataFrame({"article_id": ["1", "999"], "avg_price": [0.1, 0.2]})
    cold = pd.DataFrame({"article_id": ["2"], "age_bucket": ["16-25"], "club": ["ACTIVE"], "score": [0.5]})
    out = C.check_referential(articles, art_price, cold)
    assert any("not in the catalogue" in v.detail for v in out)


def test_vocab_mismatch_is_warn_not_error():
    seg = pd.DataFrame({"age_bucket": ["young", "mature"]})
    cold = pd.DataFrame({"age_bucket": ["16-25", "50+"]})
    out = C.check_vocab_consistency(seg, cold)
    assert out and out[0].severity == "warn"


def test_outfit_self_pair_is_error():
    df = pd.DataFrame({"upper_article_id": ["a", "b"], "lower_article_id": ["x", "b"],
                       "compatibility_score": [0.5, 0.9]})
    assert any(v.severity == "error" and "self-pair" in v.check for v in C.check_outfit_pairs(df))
