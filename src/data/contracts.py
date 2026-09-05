"""
Data contracts for the feature parquets — explicit assertions, no Pandera.

Each `check_<table>(df, ...)` returns a list of Violation. `check_all(dir)`
loads every parquet and runs the lot, including cross-table referential /
vocabulary checks. `scripts/validate_features.py` is the operator entry point;
run it after the ETL and before a deploy.

Severity:
  error  -> the pipeline output is wrong; block.
  warn   -> suspicious but tolerable (e.g. a few price orphans).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path("data/features")


@dataclass
class Violation:
    table: str
    check: str
    severity: str          # "error" | "warn"
    detail: str

    def __str__(self):
        return f"[{self.severity.upper():5}] {self.table}.{self.check}: {self.detail}"


def _v(table, check, sev, detail):
    return Violation(table, check, sev, detail)


# ── per-column primitives ──────────────────────────────────────────────────

def _unique_nonnull(df, col, table, key=None, sev="error"):
    out = []
    n_null = int(df[col].isna().sum())
    if n_null:
        out.append(_v(table, f"{col} not null", sev, f"{n_null} nulls"))
    dupes = int(df.duplicated(subset=key or col).sum())
    if dupes:
        out.append(_v(table, f"{key or col} unique", sev, f"{dupes} duplicate rows"))
    return out


def _in_range(df, col, lo, hi, table, sev="error", allow_null=True):
    s = df[col]
    if not allow_null and s.isna().any():
        return [_v(table, f"{col} in [{lo},{hi}]", sev, f"{int(s.isna().sum())} nulls")]
    bad = s.dropna()
    bad = bad[(bad < lo) | (bad > hi) | ~np.isfinite(bad)]
    if len(bad):
        return [_v(table, f"{col} in [{lo},{hi}]", sev,
                   f"{len(bad)} out of range (e.g. {bad.iloc[0]!r})")]
    return []


def _in_set(df, col, allowed, table, sev="error"):
    seen = set(df[col].dropna().unique())
    extra = seen - set(allowed)
    if extra:
        return [_v(table, f"{col} in known set", sev, f"unexpected values {sorted(extra)[:8]}")]
    return []


def _rowcount(df, lo, hi, table):
    if not (lo <= len(df) <= hi):
        return [_v(table, "row count", "warn", f"{len(df):,} outside [{lo:,}, {hi:,}]")]
    return []


# ── per-table contracts ───────────────────────────────────────────────────

def check_articles(df):
    t = "article_content_features"
    out = _unique_nonnull(df, "article_id", t)
    out += _rowcount(df, 90_000, 130_000, t)
    for c in [c for c in df.columns if c.endswith("_idx")]:
        if (df[c] < 0).any():
            out.append(_v(t, f"{c} >= 0", "error", f"{int((df[c] < 0).sum())} negative"))
    for c in [c for c in df.columns if c.startswith("is_")]:
        out += _in_set(df, c, {0, 1}, t)
    if "desc_length" in df:
        out += _in_range(df, "desc_length", 0, 100_000, t)
    return out


def check_customer_segments(df):
    t = "customer_segments"
    out = _unique_nonnull(df, "customer_id", t)
    out += _rowcount(df, 1_000_000, 1_600_000, t)
    out += _in_range(df, "age", 14, 105, t, sev="warn")
    out += _in_range(df, "age_norm", 0.0, 1.0, t)
    out += _in_range(df, "engagement_score", 0.0, 1.0, t)
    for c in ("is_active_member", "receives_news"):
        if c in df:
            out += _in_set(df, c, {0, 1}, t)
    if df["club"].nunique() > 12:
        out.append(_v(t, "club cardinality", "warn", f"{df['club'].nunique()} distinct clubs"))
    return out


def check_price(df, table, id_col):
    out = _unique_nonnull(df, id_col, table)
    out += _in_range(df, "avg_price", 0.0, 5.0, table)      # normalized scale, ~0..1
    if (df["avg_price"] == 0).mean() > 0.5:
        out.append(_v(table, "avg_price non-trivial", "warn",
                      f"{(df['avg_price'] == 0).mean():.0%} of prices are exactly 0"))
    return out


def check_trend_scores(df):
    t = "trend_scores"
    out = _unique_nonnull(df, "week", t, key=["week", "product_type_name"])
    out += _in_range(df, "trend_score", 0.0, 1.0, t)
    if df["week"].isna().any():
        out.append(_v(t, "week not null", "error", f"{int(df['week'].isna().sum())} null weeks"))
    return out


def check_cold_start_popular(df):
    t = "cold_start_popular"
    out = _unique_nonnull(df, "article_id", t, key=["age_bucket", "club", "article_id"])
    out += _in_range(df, "score", 0.0, 1.0, t)
    empty = df.groupby(["age_bucket", "club"]).size()
    if (empty == 0).any():
        out.append(_v(t, "every segment populated", "error", "a (age_bucket, club) group is empty"))
    return out


def check_outfit_pairs(df):
    t = "outfit_pairs"
    out = _in_range(df, "compatibility_score", 0.0, 1.0, t)
    self_pairs = int((df["upper_article_id"] == df["lower_article_id"]).sum())
    if self_pairs:
        out.append(_v(t, "no self-pairs", "error", f"{self_pairs} rows pair an item with itself"))
    return out


# ── cross-table ───────────────────────────────────────────────────────────

def check_referential(articles, art_price, cold_start):
    out = []
    valid = set(articles["article_id"].astype(str))
    for name, df in (("art_avg_price", art_price), ("cold_start_popular", cold_start)):
        orphan = ~df["article_id"].astype(str).isin(valid)
        frac = orphan.mean()
        if frac > 0:
            sev = "warn" if frac < 0.02 else "error"
            out.append(_v(name, "article_id -> article_content_features", sev,
                          f"{orphan.sum()} ({frac:.1%}) article_ids not in the catalogue"))
    return out


def check_vocab_consistency(customer_segments, cold_start):
    """The same field name should mean the same thing across tables.

    Known: cold_start_popular.age_bucket is ['16-25', ...] while
    customer_segments.age_bucket is ['young', 'mature', ...]. It's a trap for
    anyone doing a segment join, but not an active bug — get_recommendations
    recomputes the '16-25' bucket from raw age rather than joining on
    customer_segments. Warn so it's visible; fixing it means an ETL change +
    retrain, tracked separately.
    """
    a = set(customer_segments["age_bucket"].dropna().unique())
    b = set(cold_start["age_bucket"].dropna().unique())
    if a and b and a.isdisjoint(b):
        return [_v("cold_start_popular", "age_bucket vocab == customer_segments", "warn",
                   f"disjoint vocabularies: {sorted(a)} vs {sorted(b)} — see docstring")]
    return []


# ── runner ────────────────────────────────────────────────────────────────

_LOADERS = {
    "articles": "article_content_features.parquet",
    "customer_segments": "customer_segments.parquet",
    "art_avg_price": "art_avg_price.parquet",
    "user_avg_price": "user_avg_price.parquet",
    "trend_scores": "trend_scores.parquet",
    "cold_start_popular": "cold_start_popular.parquet",
    "outfit_pairs": "outfit_pairs.parquet",
}


def check_all(features_dir=DEFAULT_DIR):
    d = Path(features_dir)
    frames = {}
    missing = []
    for k, fn in _LOADERS.items():
        p = d / fn
        if p.exists():
            frames[k] = pd.read_parquet(p)
        else:
            missing.append(fn)

    out = [_v("(dir)", "file present", "error", f"missing {m}") for m in missing]
    if "articles" in frames:
        out += check_articles(frames["articles"])
    if "customer_segments" in frames:
        out += check_customer_segments(frames["customer_segments"])
    if "art_avg_price" in frames:
        out += check_price(frames["art_avg_price"], "art_avg_price", "article_id")
    if "user_avg_price" in frames:
        out += check_price(frames["user_avg_price"], "user_avg_price", "customer_id")
    if "trend_scores" in frames:
        out += check_trend_scores(frames["trend_scores"])
    if "cold_start_popular" in frames:
        out += check_cold_start_popular(frames["cold_start_popular"])
    if "outfit_pairs" in frames:
        out += check_outfit_pairs(frames["outfit_pairs"])
    if {"articles", "art_avg_price", "cold_start_popular"} <= frames.keys():
        out += check_referential(frames["articles"], frames["art_avg_price"], frames["cold_start_popular"])
    if {"customer_segments", "cold_start_popular"} <= frames.keys():
        out += check_vocab_consistency(frames["customer_segments"], frames["cold_start_popular"])
    return out


def summarise(violations):
    errors = [v for v in violations if v.severity == "error"]
    warns = [v for v in violations if v.severity == "warn"]
    return errors, warns
