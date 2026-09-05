-- 004_security_posture_watchdog.sql
-- Project: fashionmind (cxfsiotzfshrjdrctmge)
--
-- "Who watches the watchdogs" — 002/003 lock RLS and add an event trigger + two
-- pg_cron jobs, but nothing checked they were still there tomorrow. If a future
-- migration or a plan change silently drops one, the hole reopens quietly.
--
-- This adds:
--   1. public.security_posture() — one function that reports every invariant
--      (zero anon/authenticated grants, RLS on every table, the event trigger
--      present and enabled, all three cron jobs active) as a jsonb {ok, ...}.
--      The API exposes it at GET /health/db (503 when ok=false), and
--      scripts/smoke_prod.py asserts it on every run — so a failed daily CI
--      run, which GitHub emails the repo owner about, is the alert channel.
--   2. security-selfheal — an hourly cron job that re-asserts the revoke and
--      re-creates the event trigger if it has vanished. Auto-remediation, so
--      drift is fixed within the hour regardless of whether anyone reads the
--      email.
--
-- No unwatched link: function missing -> /health/db 500 -> smoke fail -> email.
-- trigger/cron/grants drift -> ok=false -> smoke fail -> email, and selfheal
-- fixes it on the next hour.

create or replace function public.security_posture() returns jsonb
language plpgsql security definer set search_path = public, pg_catalog as $fn$
declare
  anon_grants int;
  trg_present bool;
  rls_all bool;
  cron_purge bool;
  cron_reassert bool;
  cron_selfheal bool;
begin
  select count(*) into anon_grants
    from information_schema.role_table_grants
    where table_schema = 'public' and grantee in ('anon', 'authenticated');

  select exists(
    select 1 from pg_event_trigger
    where evtname = 'auto_revoke_new_tables' and evtenabled <> 'D'
  ) into trg_present;

  select bool_and(c.relrowsecurity) into rls_all
    from pg_class c
    join pg_namespace n on n.oid = c.relnamespace
    where n.nspname = 'public' and c.relkind = 'r' and c.relname not like '\_%';

  select exists(select 1 from cron.job where jobname = 'purge-smoke-test-ci'  and active) into cron_purge;
  select exists(select 1 from cron.job where jobname = 'reassert-anon-revoke' and active) into cron_reassert;
  select exists(select 1 from cron.job where jobname = 'security-selfheal'    and active) into cron_selfheal;

  return jsonb_build_object(
    'ok', (anon_grants = 0 and trg_present and coalesce(rls_all, false)
           and cron_purge and cron_reassert and cron_selfheal),
    'anon_grants', anon_grants,
    'event_trigger', trg_present,
    'rls_all_tables', coalesce(rls_all, false),
    'cron', jsonb_build_object('purge', cron_purge, 'reassert', cron_reassert, 'selfheal', cron_selfheal),
    'checked_at', now()
  );
end
$fn$;

revoke all on function public.security_posture() from anon, authenticated;
grant execute on function public.security_posture() to service_role;

select cron.schedule(
  'security-selfheal',
  '0 * * * *',
  $cron$
    revoke all on all tables    in schema public from anon, authenticated;
    revoke all on all sequences in schema public from anon, authenticated;
    revoke all on all functions in schema public from anon, authenticated;
    do $body$
    begin
      if not exists (select 1 from pg_event_trigger where evtname = 'auto_revoke_new_tables') then
        create event trigger auto_revoke_new_tables on ddl_command_end
          when tag in ('CREATE TABLE', 'CREATE TABLE AS')
          execute function public._auto_revoke_new_tables();
      end if;
    end
    $body$;
  $cron$
);

-- Verify:  select public.security_posture();  -> {"ok": true, ...}
--          select jobid, jobname, schedule, active from cron.job;
