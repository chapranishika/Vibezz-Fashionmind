"""
Off-policy / counterfactual estimators for "what would CTR be under a different
ranker?", using only logged (context, action, reward, propensity) data.

Estimators
----------
ips   — inverse propensity scoring. Unbiased if p_logged > 0 wherever p_target
        > 0 (common support) and propensities are correct. High variance when
        the policies disagree.
snips — self-normalised IPS. Slightly biased, much lower variance; normalises
        by the sum of weights instead of N. Almost always preferred over raw
        IPS in practice.
dr    — doubly robust. IPS on the residual reward plus a direct model of the
        reward. Consistent if *either* the propensities or the reward model is
        right; lower variance than IPS when the reward model is decent.

Everything here takes a tidy DataFrame with columns:
    context_id   grouping key (a request / impression)
    reward       0/1 (clicked) or continuous
    p_logged     P(this action | context) under the policy that generated the log
    p_target     P(this action | context) under the policy being evaluated
    r_hat        (dr only) reward-model prediction for the logged action
    r_hat_target (dr only) E_target[reward] — expected reward of the target
                 policy's action(s), from the same reward model

CRITICAL: p_logged must come from a *stochastic* logging policy. A deterministic
top-N ranker has p_logged ∈ {0, 1}; IPS/SNIPS/DR are then undefined or
degenerate. `assert_stochastic_log` enforces this. Make serving stochastic
(ε-greedy or Plackett–Luce sampling) and log the propensity to unlock real OPE
— see scripts/offpolicy_eval.py and migration
20260910000000_recs_propensity.sql.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class DeterministicLogError(RuntimeError):
    pass


def assert_stochastic_log(df: pd.DataFrame, col: str = "p_logged", tol: float = 1e-6):
    p = df[col].to_numpy(dtype=float)
    if np.any(p <= 0) or np.any(p > 1):
        raise DeterministicLogError(f"{col} must be in (0, 1]; got min={p.min()}, max={p.max()}")
    near_one = float(np.mean(p > 1 - tol))
    if near_one > 0.99:
        raise DeterministicLogError(
            f"{near_one:.0%} of {col} ≈ 1 — the log looks deterministic. "
            "OPE needs a stochastic logging policy; see the module docstring."
        )


def _weights(df):
    return (df["p_target"].to_numpy(float) / df["p_logged"].to_numpy(float))


def ips(df: pd.DataFrame, clip: float | None = 20.0) -> dict:
    assert_stochastic_log(df)
    w = _weights(df)
    if clip is not None:
        w = np.minimum(w, clip)
    r = df["reward"].to_numpy(float)
    vals = w * r
    est = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    return {"estimator": "ips", "value": est, "se": se,
            "ci95": (est - 1.96 * se, est + 1.96 * se),
            "ess": float(w.sum() ** 2 / np.sum(w ** 2))}      # effective sample size


def snips(df: pd.DataFrame, clip: float | None = 20.0) -> dict:
    assert_stochastic_log(df)
    w = _weights(df)
    if clip is not None:
        w = np.minimum(w, clip)
    r = df["reward"].to_numpy(float)
    est = float(np.sum(w * r) / np.sum(w))
    # delta-method SE for the ratio estimator
    n = len(w)
    wr, w_bar = w * r, w.mean()
    var = np.sum((wr - est * w) ** 2) / (n * (n * w_bar ** 2))
    se = float(np.sqrt(var))
    return {"estimator": "snips", "value": est, "se": se,
            "ci95": (est - 1.96 * se, est + 1.96 * se),
            "ess": float(w.sum() ** 2 / np.sum(w ** 2))}


def dr(df: pd.DataFrame, clip: float | None = 20.0) -> dict:
    """Needs r_hat (reward model on the logged action) and r_hat_target
    (expected reward of the target policy under the same model)."""
    assert_stochastic_log(df)
    for c in ("r_hat", "r_hat_target"):
        if c not in df:
            raise ValueError(f"dr needs column {c!r}")
    w = _weights(df)
    if clip is not None:
        w = np.minimum(w, clip)
    r = df["reward"].to_numpy(float)
    r_hat = df["r_hat"].to_numpy(float)
    r_hat_t = df["r_hat_target"].to_numpy(float)
    vals = r_hat_t + w * (r - r_hat)
    est = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    return {"estimator": "dr", "value": est, "se": se,
            "ci95": (est - 1.96 * se, est + 1.96 * se)}


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Run whichever estimators the columns support; return a tidy table."""
    rows = [ips(df), snips(df)]
    if {"r_hat", "r_hat_target"} <= set(df.columns):
        rows.append(dr(df))
    out = pd.DataFrame(rows)
    out["ci_lo"] = out["ci95"].map(lambda t: t[0])
    out["ci_hi"] = out["ci95"].map(lambda t: t[1])
    return out.drop(columns=["ci95"])
