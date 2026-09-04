# Security notes

## If a secret leaks (chat transcript, screenshot, commit, anywhere)

Rotate it with `scripts/rotate_secret.py` — validates the new value against the
live service, updates `.env`, updates the HF Space secret, restarts, waits for
healthy, runs the smoke test. Refuses to touch anything if validation fails.

```
set HF_TOKEN=hf_xxx
python scripts/rotate_secret.py OPENROUTER_API_KEY sk-or-v1-...
python scripts/rotate_secret.py SERPAPI_KEY ...
python scripts/rotate_secret.py SUPABASE_SERVICE_KEY sb_secret_...
```

Then kill the old value at the source (revoke/regenerate on the provider's
side) — rotating our copy doesn't do that by itself. Verify it's actually dead:

```python
# OpenRouter — expect 401
curl https://openrouter.ai/api/v1/chat/completions -H "Authorization: Bearer <old>" ...
# SerpApi — expect 401 "Invalid API key"
curl "https://serpapi.com/search.json?api_key=<old>&q=test"
# Supabase — expect 401 (after disabling legacy keys, see below)
curl "$SUPABASE_URL/rest/v1/user_auth?select=id&limit=1" -H "apikey: <old>" -H "Authorization: Bearer <old>"
```

HF write tokens have no revoke-via-API — the token can't cover its own tracks.
Kill it at https://huggingface.co/settings/tokens by hand; there's no
`rotate_secret.py` path for this one.

## Disabling the leaked Supabase legacy `service_role` / `anon` keys

Verified working path (two wrong turns got taken before landing here — this
is the one that actually works):

1. supabase.com/dashboard → the project → **Project Settings → API Keys**
   (NOT "JWT Keys" — that panel controls Supabase Auth session signing, which
   this app doesn't use; nothing there affects the `service_role`/`anon` keys).
2. Click the **"Legacy anon, service_role API keys"** tab.
3. Scroll to **"Disable legacy API keys"** → **"Disable JWT-based API keys"**.
4. Confirm.

This immediately 401s every legacy `anon`/`service_role` JWT ever issued for
the project — no need to hunt for a "regenerate JWT secret" button, no need
to touch JWT Signing Keys. Do this only *after* the replacement `sb_secret_...`
key is confirmed working everywhere (`rotate_secret.py` handles that ordering).

## Row Level Security

Every `public` table should have RLS enabled with **zero** grants to
`anon`/`authenticated` — the FastAPI backend (using the `service_role`/
`sb_secret_...` key, which bypasses RLS) is the only DB client. The frontend
never talks to Supabase directly (verified: no supabase-js, no anon key in
`frontend/index.html`).

This is enforced two ways (`db/migrations/002`, `003`):
- An event trigger (`auto_revoke_new_tables`) revokes anon/authenticated on
  every `CREATE TABLE` in `public`, immediately, regardless of which role
  created it — closes the gap that `ALTER DEFAULT PRIVILEGES` couldn't (we
  don't have permission to alter `supabase_admin`'s own default ACL).
- A weekly `pg_cron` job (`reassert-anon-revoke`) re-runs the revoke as a
  backstop, in case something re-grants access some other way.

Spot-check it's actually working:

```sql
select grantee, count(*) from information_schema.role_table_grants
where table_schema='public' and grantee in ('anon','authenticated')
group by grantee;
-- must return zero rows
```

## Prod smoke test data hygiene

`scripts/smoke_prod.py`'s full mode writes to `recommendations` /
`chat_messages` under `customer_id = 'smoke-test-ci'` — never `'guest'`, which
is real anonymous-visitor traffic. A daily `pg_cron` job (`purge-smoke-test-ci`,
`db/migrations/003`) deletes those rows automatically; nothing needs to
remember to do it by hand.

The CI workflow (`.github/workflows/ci.yml`) only runs full mode (LLM spend +
DB writes) on a push to `main`; the daily cron check runs cheap-only
(health/products/photos — no cost, no writes).
