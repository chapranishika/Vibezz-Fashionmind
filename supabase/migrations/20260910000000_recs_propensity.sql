-- 20260910000000_recs_propensity.sql
--
-- Make the recommendations log usable for off-policy evaluation. The serving
-- policy is deterministic (top-N by score), so p_logged is always 1 and
-- IPS/SNIPS/DR are degenerate. Set RECS_EXPLORE_EPS > 0 and /recommend will
-- ε-greedy-shuffle the slate and record the propensity of the realised order.
--
--   p_logged  P(this item was shown at this rank) under the logging policy
--             = (1 - eps) * 1[deterministic rank == this rank] + eps / N
--   explore_eps  the eps in force for this impression (0 = deterministic)

alter table public.recommendations
  add column if not exists p_logged   numeric,
  add column if not exists explore_eps numeric not null default 0;
