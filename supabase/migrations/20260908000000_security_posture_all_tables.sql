-- 005_security_posture_all_tables.sql
-- Project: fashionmind (cxfsiotzfshrjdrctmge)
--
-- 004's security_posture() had `and c.relname not like '\_%'` in the RLS check
-- -- added to skip a throwaway probe table, but the effect is that any table
-- named _<anything> is exempt from the rls_all_tables check. A filter that
-- weakens the check it's inside. Removed: check RLS on every base table in
-- public, and report the names of any that lack it (`rls_gaps`).

create or replace function public.security_posture() returns jsonb
language plpgsql security definer set search_path = public, pg_catalog as $fn$
declare
  anon_grants int;
  trg_present bool;
  rls_all bool;
  rls_gaps text[];
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

  select coalesce(bool_and(c.relrowsecurity), true),
         coalesce(array_agg(c.relname) filter (where not c.relrowsecurity), '{}')
    into rls_all, rls_gaps
    from pg_class c
    join pg_namespace n on n.oid = c.relnamespace
    where n.nspname = 'public' and c.relkind = 'r';

  select exists(select 1 from cron.job where jobname = 'purge-smoke-test-ci'  and active) into cron_purge;
  select exists(select 1 from cron.job where jobname = 'reassert-anon-revoke' and active) into cron_reassert;
  select exists(select 1 from cron.job where jobname = 'security-selfheal'    and active) into cron_selfheal;

  return jsonb_build_object(
    'ok', (anon_grants = 0 and trg_present and rls_all
           and cron_purge and cron_reassert and cron_selfheal),
    'anon_grants', anon_grants,
    'event_trigger', trg_present,
    'rls_all_tables', rls_all,
    'rls_gaps', to_jsonb(rls_gaps),
    'cron', jsonb_build_object('purge', cron_purge, 'reassert', cron_reassert, 'selfheal', cron_selfheal),
    'checked_at', now()
  );
end
$fn$;

revoke all on function public.security_posture() from anon, authenticated;
grant execute on function public.security_posture() to service_role;
