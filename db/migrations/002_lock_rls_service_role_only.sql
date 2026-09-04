-- 002_lock_rls_service_role_only.sql
-- Project: fashionmind (cxfsiotzfshrjdrctmge)
--
-- Closes the RLS hole: every app table had a PERMISSIVE `ALL ... USING (true)`
-- policy granted to role `public` (i.e. anon + authenticated), and anon/
-- authenticated held full table GRANTs. The anon key ships in the frontend
-- bundle, so anyone could read every bcrypt hash + email in user_auth, dump
-- customers/orders, or TRUNCATE any table — straight past the API.
--
-- This app's design: the FastAPI backend is the ONLY database client, using the
-- service_role key. The frontend never touches Supabase directly (verified —
-- no supabase-js, no anon key in frontend/index.html). So the fix is simply to
-- remove anon/authenticated access entirely. service_role has BYPASSRLS and
-- keeps all its GRANTs, so the API is unaffected.
--
-- PRECONDITION: the backend must be running with the real service_role key
-- (SUPABASE_SERVICE_KEY), not the anon key. Swap that first, verify /health,
-- then apply this — zero downtime.

begin;

-- 1. Drop the "USING (true) TO public" free-for-all policies.
do $$
declare r record;
begin
  for r in
    select schemaname, tablename, policyname
    from pg_policies
    where schemaname = 'public'
      and policyname = 'service_role_all'
  loop
    execute format('drop policy %I on %I.%I', r.policyname, r.schemaname, r.tablename);
  end loop;
end $$;

-- 2. Revoke every anon / authenticated privilege on the public schema.
--    (RLS stays ENABLED on every table -> deny-by-default for anyone who is
--    not service_role. service_role bypasses RLS and its grants are untouched.)
revoke all on all tables    in schema public from anon, authenticated;
revoke all on all sequences in schema public from anon, authenticated;
revoke all on all functions in schema public from anon, authenticated;
revoke usage on schema public from anon, authenticated;

-- 3. Make sure future tables created in this schema don't re-open the gap.
alter default privileges in schema public revoke all on tables    from anon, authenticated;
alter default privileges in schema public revoke all on sequences from anon, authenticated;
alter default privileges in schema public revoke all on functions from anon, authenticated;

commit;

-- Verify afterwards:
--   set role anon;  select * from user_auth limit 1;   -- expect: permission denied
--   reset role;

-- KNOWN GAP (could not close from here — 42501 permission denied to change
-- default privileges as `postgres`; only `supabase_admin` can alter its own):
-- pg_default_acl still has an entry for owner_role=supabase_admin granting
-- anon+authenticated full rights on anything IT creates. Tables created via
-- the Supabase dashboard table editor (or any tooling that runs as
-- supabase_admin) will come back with anon/authenticated wide open by
-- default, same class of hole as this migration fixes.
--
-- Mitigation until someone with supabase_admin runs the ALTER DEFAULT
-- PRIVILEGES themselves: after creating ANY new table, immediately run
--   revoke all on <table> from anon, authenticated;
-- and periodically re-run the verification query below — it must return zero
-- rows:
--   select grantee, count(*) from information_schema.role_table_grants
--   where table_schema='public' and grantee in ('anon','authenticated')
--   group by grantee;
