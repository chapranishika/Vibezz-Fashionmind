# A/B test plan — ship the LambdaRank re-ranker

This is the online experiment that would justify (or kill) the second ranking
stage. Offline, the re-ranker's aggregate lift over ALS is inside the noise on
1,000 users (see README → Results: the marginal NDCG CIs overlap); the paired
test clears zero but a paired offline test on logged data is not proof of an
online effect. This plan is written to be runnable, not run — there is no live
traffic yet.

## Hypothesis

Re-ranking the top-100 ALS candidates with the 13-signal LambdaRank model
increases the rate at which users click a recommended item, versus serving the
ALS top-N directly, with no material regression in latency or catalogue
diversity.

## Unit, arms, allocation

- **Randomisation unit:** `customer_id` (hash-bucketed, sticky). Guests
  (`customer_id = "guest"`) are excluded — they can't be consistently bucketed
  and their behaviour is not representative.
- **Arms:** `control` = ALS top-N; `treatment` = ALS→LambdaRank. 50/50.
- **Assignment** at the `/recommend` boundary; logged to `recommendations`
  alongside the existing rows (add an `arm` column).

## Metrics

| | metric | definition | direction |
|---|---|---|---|
| **Primary** | rec CTR | clicked recs / served recs, per user, over the window | ↑ |
| Secondary | add-to-cart rate from a rec | cart rows with a `rec_id` / served recs | ↑ |
| Secondary | first-click rank | mean rank of the first clicked rec | ↓ |
| **Guardrail** | p95 `/recommend` latency | from `http_request_duration_seconds` | no worse than +20 ms |
| **Guardrail** | catalogue coverage | distinct article_ids served / catalogue, per day | no worse than −10% rel |
| **Guardrail** | 5xx rate on `/recommend` | from `http_requests_total{status=~"5.."}` | no increase |

## Sizing

Baseline rec CTR assumed **3.3%** (the ALS paired-hit@12 figure is the best
proxy available pre-launch). Minimum detectable effect **+15% relative**
(3.3% → 3.8%), two-sided, α = 0.05, power = 0.80.

- n per arm ≈ `16 * p(1-p) / (p*mde)^2` ≈ **~26,000 users per arm** (~52k total)
  at the user level. Recompute from the real baseline once traffic exists —
  this number moves fast with `p`.
- At an assumed 2,000 bucketed users/day, that's **~26 days** to power. Round
  to a **4-week** run so it spans whole weekly cycles (fashion demand is
  weekly-seasonal).

## Analysis

- Primary: two-proportion z-test on per-user CTR, plus a CUPED-adjusted
  estimate using each user's pre-period CTR as the covariate (variance
  reduction; fashion CTR is heavy-tailed).
- Report the effect with a 95% CI, not just a p-value. Slice the effect by the
  same cuts as the offline eval (history length, age bucket, price tier) —
  the offline slices show the re-ranker helps mature/higher-spend users and
  *hurts* mid-age/budget users, so a flat average could hide a bad trade.
- Guardrails checked daily; any breach → pause.

## Stopping rules

- **No peeking for significance.** Fixed horizon (4 weeks) unless a guardrail
  breaches. If sequential looks are needed, use an always-valid method
  (e.g. mSPRT / group-sequential O'Brien–Fleming), not repeated z-tests.
- **Ship** if primary CI excludes 0 on the positive side AND no guardrail
  regressed AND no user slice regressed by >10% relative.
- **Roll back** immediately on: p95 latency +50 ms, any 5xx increase, coverage
  −20% relative, or primary CI excluding 0 on the *negative* side.
- **Inconclusive** (CI spans 0 at horizon): keep control. A re-ranker that
  can't beat retrieval online is not worth the second stage's latency and
  operational surface.
