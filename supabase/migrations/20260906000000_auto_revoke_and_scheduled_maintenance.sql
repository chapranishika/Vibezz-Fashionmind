-- 003_auto_revoke_and_scheduled_maintenance.sql
-- Project: fashionmind (cxfsiotzfshrjdrctmge)
--
-- Closes the gap left by 002: ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin
-- returned 42501 (permission denied) -- only supabase_admin can alter its own
-- default ACL, and we don't have that role. Any table created by Supabase's
-- own tooling (dashboard table editor, its migration runner) would have come
-- back with anon/authenticated granted by default, reopening 002's hole.
--
-- Fix: an event trigger that fires on every CREATE TABLE in `public`,
-- regardless of which role created it, and revokes anon/authenticated
-- immediately. Verified live: created a probe table as the migration runner's
-- own role, checked its grants immediately after -- zero rows for
-- anon/authenticated. No window, no dependency on which role owns the table.
--
-- Also schedules two pg_cron jobs (belt-and-suspenders + a bug I introduced
-- myself): a weekly re-assertion of the 002 revoke (catches a manual GRANT or
-- a future migration that reopens it, before it sits open for long), and a
-- daily purge of the synthetic rows scripts/smoke_prod.py writes under
-- customer_id 'smoke-test-ci' -- previously cleaned up by hand after every run.

begin;

create or replace function public._auto_revoke_new_tables() returns event_trigger
language plpgsql security definer as $$
declare r record;
begin
  for r in select objid::regclass::text as tbl
           from pg_event_trigger_ddl_commands()
           where command_tag = 'CREATE TABLE' and schema_name = 'public'
  loop
    execute format('revoke all on %s from anon, authenticated', r.tbl);
  end loop;
end;
$$;

drop event trigger if exists auto_revoke_new_tables;
create event trigger auto_revoke_new_tables
  on ddl_command_end
  when tag in ('CREATE TABLE', 'CREATE TABLE AS')
  execute function public._auto_revoke_new_tables();

commit;

create extension if not exists pg_cron;

select cron.schedule(
  'purge-smoke-test-ci',
  '17 3 * * *',
  $$
    delete from chat_messages where session_id in
      (select id from chat_sessions where customer_id = 'smoke-test-ci');
    delete from chat_sessions   where customer_id = 'smoke-test-ci';
    delete from recommendations where customer_id = 'smoke-test-ci';
  $$
);

select cron.schedule(
  'reassert-anon-revoke',
  '0 4 * * 0',
  $$
    revoke all on all tables    in schema public from anon, authenticated;
    revoke all on all sequences in schema public from anon, authenticated;
    revoke all on all functions in schema public from anon, authenticated;
  $$
);

-- Verify:
--   select jobid, jobname, schedule, active from cron.job;
--   create table public._probe(id int); -- then check role_table_grants for it, drop it
