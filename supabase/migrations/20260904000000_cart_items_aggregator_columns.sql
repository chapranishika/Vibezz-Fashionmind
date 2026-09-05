-- 001_cart_items_aggregator_columns.sql
-- Applied to the `fashionmind` Supabase project (cxfsiotzfshrjdrctmge) on 2026-08-31.
--
-- Adds the trend-aggregator context to a cart line:
--   mode = 'demo'    -> trained H&M catalogue item, goes through fake checkout
--   mode = 'compare' -> aggregator item from a real retailer; "buy" deep-links
--                       out, the cart is a cross-site price/compare list
-- The (customer_id, article_id, size) unique key the app upserts on already
-- existed as cart_items_customer_id_article_id_size_key, so no index is added
-- here.

alter table public.cart_items
  add column if not exists mode     text not null default 'demo',
  add column if not exists source   text,   -- curated | hm-demo | serpapi
  add column if not exists retailer text,
  add column if not exists buy_url  text,
  add column if not exists title    text,
  add column if not exists image    text,
  add column if not exists look     text,   -- trend the item was saved under
  add column if not exists currency text default 'INR';
