# Migrations moved

The canonical migrations now live in [`../../supabase/migrations/`](../../supabase/migrations/)
so the Supabase CLI can apply them:

```
supabase link --project-ref cxfsiotzfshrjdrctmge
supabase db push          # apply new migrations to the linked project
supabase db reset         # rebuild the whole schema from zero (local stack)
```

`.github/workflows/db.yml` runs `supabase db reset` on every change under
`supabase/` and asserts `security_posture()` comes out green — that's the
"reproducible from zero" check.

The files here are kept for history. `20260901000000_baseline_schema.sql` is a
reverse-engineered snapshot of the tables that were originally created ad hoc
via the dashboard and never captured; `002`–`005` are the RLS lockdown,
auto-revoke trigger, pg_cron maintenance, and posture watchdog, unchanged.
