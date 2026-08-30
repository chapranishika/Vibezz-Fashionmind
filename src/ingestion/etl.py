"""
FashionMind — Phase 1: Pandas ETL (No-PySpark Windows Fallback)
===============================================================
Inputs  : data/raw/articles.csv, data/raw/customers.csv
Outputs : data/features/articles_clean.parquet
          data/features/customers_clean.parquet
          data/features/article_content_features.parquet
          data/features/customer_segments.parquet
"""

import os
import pandas as pd
import numpy as np

def run():
    print("=" * 60)
    print("  FashionMind — Phase 1: Pandas ETL (Windows Fallback)")
    print("=" * 60)
    os.makedirs("data/features", exist_ok=True)

    print("\n[1/4] Loading articles...")
    art = pd.read_csv("data/raw/articles.csv", dtype={"article_id": str})
    art["article_id"] = art["article_id"].str.lstrip("0")
    print(f"      {len(art):,} articles")

    print("[2/4] Loading customers...")
    cust = pd.read_csv("data/raw/customers.csv", dtype={"customer_id": str})
    print(f"      {len(cust):,} customers")

    print("[3/4] Cleaning & feature engineering...")
    
    # Clean articles
    art["detail_desc"] = art["detail_desc"].fillna("No description available")
    art["article_id_str"] = art["article_id"].astype(str).str.zfill(9)
    art["image_folder"] = art["article_id_str"].str[:3]
    art["image_path"] = "data/raw/images/" + art["image_folder"] + "/" + art["article_id_str"] + ".jpg"
    
    cat_cols = ["product_group_name", "product_type_name",
                "colour_group_name", "garment_group_name",
                "index_group_name", "graphical_appearance_name",
                "perceived_colour_master_name"]
    for col in cat_cols:
        art[f"{col}_idx"] = art[col].fillna("Unknown").astype("category").cat.codes
        
    clothing_groups = ["Garment Upper body", "Garment Lower body",
                       "Garment Full body", "Underwear", "Nightwear",
                       "Swimwear", "Socks & Tights"]
    art["is_clothing"] = art["product_group_name"].isin(clothing_groups).astype(int)
    art["is_ladieswear"] = (art["index_group_name"] == "Ladieswear").astype(int)
    art["is_menswear"] = (art["index_group_name"] == "Menswear").astype(int)
    art["is_kids"] = (art["index_group_name"] == "Baby/Children").astype(int)
    art["is_solid_colour"] = (art["graphical_appearance_name"] == "Solid").astype(int)
    art["desc_length"] = art["detail_desc"].str.len()
    
    # Clean customers
    cust["age"] = cust["age"].fillna(36.0).astype(float)
    # `club` uses the same normalisation as the cold-start segment builder in
    # collaborative_filtering.py (fillna NONE + uppercase) so segment keys line up.
    cust["club"] = cust["club_member_status"].fillna("NONE").str.upper()
    cust["club_member_status"] = cust["club_member_status"].fillna("UNKNOWN")
    cust["fashion_news_frequency"] = cust["fashion_news_frequency"].fillna("NONE")
    cust["FN"] = cust["FN"].fillna(0.0)
    cust["Active"] = cust["Active"].fillna(0.0)
    
    # age buckets
    conditions = [
        (cust["age"] < 23),
        (cust["age"] < 31),
        (cust["age"] < 46),
        (cust["age"] < 65)
    ]
    choices = ["teen", "young", "mid", "mature"]
    cust["age_bucket"] = np.select(conditions, choices, default="senior")
    
    bucket_map = {"teen": 0, "young": 1, "mid": 2, "mature": 3, "senior": 4}
    cust["age_bucket_idx"] = cust["age_bucket"].map(bucket_map)
    cust["age_norm"] = (cust["age"] - 16.0) / (99.0 - 16.0)
    cust["is_active_member"] = (cust["club_member_status"] == "ACTIVE").astype(int)
    cust["receives_news"] = (cust["fashion_news_frequency"] != "NONE").astype(int)
    cust["engagement_score"] = (
        cust["is_active_member"] * 0.4 +
        cust["receives_news"]    * 0.3 +
        cust["FN"]               * 0.2 +
        cust["Active"]           * 0.1
    )
    
    print("[4/4] Saving feature store...")
    art.to_parquet("data/features/articles_clean.parquet", index=False)
    cust.to_parquet("data/features/customers_clean.parquet", index=False)
    
    art_cols = [
        "article_id", "article_id_str", "image_path", "prod_name",
        "product_group_name", "product_type_name", "colour_group_name",
        "garment_group_name", "index_group_name", "graphical_appearance_name",
        "perceived_colour_master_name", "detail_desc",
        "product_group_name_idx", "product_type_name_idx",
        "colour_group_name_idx", "garment_group_name_idx", "index_group_name_idx",
        "graphical_appearance_name_idx",
        "is_clothing", "is_ladieswear", "is_menswear", "is_kids",
        "is_solid_colour", "desc_length"
    ]
    art[art_cols].to_parquet("data/features/article_content_features.parquet", index=False)
    
    cust_cols = [
        "customer_id", "age", "age_norm", "age_bucket", "age_bucket_idx",
        "club", "is_active_member", "receives_news", "engagement_score"
    ]
    cust[cust_cols].to_parquet("data/features/customer_segments.parquet", index=False)
    
    print("\n  ✓ articles_clean.parquet")
    print("  ✓ customers_clean.parquet")
    print("  ✓ article_content_features.parquet")
    print("  ✓ customer_segments.parquet")
    print("\nPhase 1 complete ✓")

if __name__ == "__main__":
    run()