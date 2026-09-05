"""
OPE estimators validated on a synthetic contextual bandit with a KNOWN
target-policy value. This is how you check an OPE implementation: if it can't
recover a value you computed in closed form, it's wrong.
"""
import numpy as np
import pandas as pd
import pytest

from src.eval.offpolicy import ips, snips, dr, evaluate, assert_stochastic_log, DeterministicLogError

RNG = np.random.default_rng(7)
N_ACTIONS = 5


def _bandit(n, logging_temp=1.0, target_temp=0.3, seed=0):
    """n impressions. Each context has action reward-probabilities; the logging
    policy samples softmax(scores/logging_temp), the target policy is
    softmax(scores/target_temp). Returns (log_df, true_target_value)."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n, N_ACTIONS))
    reward_p = 1 / (1 + np.exp(-(scores - 0.5)))            # true P(click | context, action)

    def softmax(x, t):
        z = np.exp((x - x.max(axis=1, keepdims=True)) / t)
        return z / z.sum(axis=1, keepdims=True)

    p_log_all = softmax(scores, logging_temp)
    p_tgt_all = softmax(scores, target_temp)

    a = np.array([rng.choice(N_ACTIONS, p=p_log_all[i]) for i in range(n)])
    r = rng.binomial(1, reward_p[np.arange(n), a])

    true_value = float((p_tgt_all * reward_p).sum(axis=1).mean())   # E_context E_{a~target}[reward]

    df = pd.DataFrame({
        "context_id": np.arange(n),
        "reward": r.astype(float),
        "p_logged": p_log_all[np.arange(n), a],
        "p_target": p_tgt_all[np.arange(n), a],
        # a deliberately-imperfect reward model for DR
        "r_hat": reward_p[np.arange(n), a] + rng.normal(0, 0.1, n),
        "r_hat_target": (p_tgt_all * reward_p).sum(axis=1) + rng.normal(0, 0.1, n),
    })
    return df, true_value


def test_ips_is_unbiased_over_many_logs():
    ests, truth = [], None
    for s in range(60):
        df, tv = _bandit(4000, seed=s)
        truth = tv
        ests.append(ips(df, clip=None)["value"])
    # mean of the estimator across logs should sit on the true value
    assert abs(np.mean(ests) - truth) < 0.01


def test_snips_lower_variance_than_ips():
    ips_v, snips_v = [], []
    for s in range(60):
        df, _ = _bandit(3000, seed=s)
        ips_v.append(ips(df, clip=None)["value"])
        snips_v.append(snips(df, clip=None)["value"])
    assert np.var(snips_v) < np.var(ips_v)


def test_dr_recovers_value_with_a_wrong_reward_model():
    df, tv = _bandit(20000, seed=1)
    df["r_hat"] = 0.5                       # deliberately useless reward model
    df["r_hat_target"] = 0.5
    out = dr(df)
    assert out["value"] == pytest.approx(tv, abs=0.02)   # propensities carry it


def test_dr_recovers_value_with_wrong_propensities_but_good_model():
    df, tv = _bandit(20000, seed=2)
    df["p_logged"] = 1.0 / N_ACTIONS       # wrong: pretend uniform logging
    df["p_target"] = 1.0 / N_ACTIONS
    # r_hat_target is the good model here -> DR should still be close
    out = dr(df)
    assert out["value"] == pytest.approx(tv, abs=0.03)


def test_evaluate_returns_all_supported_estimators():
    df, _ = _bandit(2000, seed=3)
    tbl = evaluate(df)
    assert set(tbl["estimator"]) == {"ips", "snips", "dr"}
    assert (tbl["ci_lo"] < tbl["value"]).all() and (tbl["value"] < tbl["ci_hi"]).all()


def test_deterministic_log_is_rejected():
    df = pd.DataFrame({"reward": [1, 0, 1], "p_logged": [1.0, 1.0, 1.0], "p_target": [0.4, 0.1, 0.5]})
    with pytest.raises(DeterministicLogError):
        assert_stochastic_log(df)
    with pytest.raises(DeterministicLogError):
        ips(df)
