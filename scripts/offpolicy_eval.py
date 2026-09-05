#!/usr/bin/env python3
"""
Off-policy evaluation on the logged `recommendations` table.

Question answered: *if we served the deterministic score-ordered top item
instead of what the ε-greedy logging policy actually served, what would the
rank-1 click-through rate be?* — estimated from logs alone via IPS / SNIPS.

    python scripts/offpolicy_eval.py               # pull from Supabase
    python scripts/offpolicy_eval.py --csv recs.csv

Needs impressions logged with exploration (`explore_eps > 0`, `p_logged`
populated) — set `RECS_EXPLORE_EPS=0.1` on the API and let it collect. Until
then this prints the synthetic validation instead (the estimators recovering a
known target value) and the instructions to start collecting.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.offpolicy import evaluate, DeterministicLogError, assert_stochastic_log  # noqa: E402


def _load_supabase():
    import urllib.request
    url = os.environ.get("SUPABASE_URL", "https://cxfsiotzfshrjdrctmge.supabase.co")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not key:
        sys.exit("SUPABASE_SERVICE_KEY not set (or pass --csv)")
    q = (f"{url}/rest/v1/recommendations"
         "?select=customer_id,article_id,score,rank,clicked,p_logged,explore_eps,generated_at"
         "&order=generated_at.desc&limit=50000")
    req = urllib.request.Request(q, headers={"apikey": key, "Authorization": f"Bearer {key}"})
    import json
    return pd.DataFrame(json.loads(urllib.request.urlopen(req, timeout=30).read()))


def _synthetic_demo(n_draws=40):
    from tests.test_offpolicy import _bandit
    from src.eval.offpolicy import ips, snips, dr
    print("\nNo explored impressions yet — validating the estimators on a synthetic\n"
          f"bandit with a known target value ({n_draws} independent logs):\n")
    truth, acc = None, {"ips": [], "snips": [], "dr": []}
    for s in range(n_draws):
        df, truth = _bandit(4000, seed=s)
        acc["ips"].append(ips(df, clip=None)["value"])
        acc["snips"].append(snips(df, clip=None)["value"])
        acc["dr"].append(dr(df)["value"])
    print(f"  true target value : {truth:.4f}")
    for k, v in acc.items():
        v = np.array(v)
        print(f"  {k:<6s} mean {v.mean():.4f}  (bias {v.mean()-truth:+.4f}, sd across logs {v.std():.4f})")
    print("\nIPS ~unbiased, SNIPS lower variance, DR robust — the machinery works.\n"
          "To run on real data: set RECS_EXPLORE_EPS=0.1 on the API, wait for traffic,\n"
          "then re-run. Migration 20260910000000_recs_propensity adds p_logged / explore_eps.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv) if args.csv else _load_supabase()
    print(f"loaded {len(df):,} recommendation rows")

    explored = df[df.get("explore_eps", pd.Series(0, index=df.index)).fillna(0) > 0].copy()
    if explored.empty:
        _synthetic_demo()
        return

    # one row per impression = the item shown at rank 1
    imp = explored.sort_values("rank").groupby(["customer_id", "generated_at"], as_index=False).first()
    # deterministic target policy: rank-1 item = the highest score in that slate
    best_score = explored.groupby(["customer_id", "generated_at"])["score"].transform("max")
    explored["_is_best"] = explored["score"] >= best_score - 1e-9
    top = explored[explored["rank"] == 1].set_index(["customer_id", "generated_at"])
    imp = imp.set_index(["customer_id", "generated_at"])
    imp["p_target"] = top["_is_best"].astype(float)             # 1 if logged rank-1 == argmax score
    frame = pd.DataFrame({
        "context_id": np.arange(len(imp)),
        "reward": imp["clicked"].fillna(False).astype(float).values,
        "p_logged": imp["p_logged"].values,
        "p_target": imp["p_target"].values,
    })

    try:
        assert_stochastic_log(frame)
    except DeterministicLogError as e:
        print(f"\n{e}\nCollect more explored traffic (higher RECS_EXPLORE_EPS or longer window).")
        return

    print(f"\n{len(frame):,} explored impressions | logged rank-1 CTR = "
          f"{frame['reward'].mean():.4f}")
    tbl = evaluate(frame)
    print("\nestimated rank-1 CTR under the deterministic score-ordered policy:")
    for _, r in tbl.iterrows():
        print(f"  {r['estimator']:<6s} {r['value']:.4f}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
              + (f"  ESS={r['ess']:.0f}" if "ess" in r and pd.notna(r['ess']) else ""))


if __name__ == "__main__":
    main()
