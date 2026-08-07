"""Fashion Dataset Adapter: Converts Fashion Product Images (Small) to TIGeR parquet format.

This script parses styles.csv from the Kaggle dataset 'paramaggarwal/fashion-product-images-small',
filters for our core categories (shirts, shoes, bags, hats), maps the attributes to our strict 
`configs/schema.yaml`, and outputs the `products.parquet` file required by the Sieve.
"""

import json
import logging
import random
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from tiger.schema import Schema
from tiger import text_views

# Mapping Kaggle articleType to our 4 strict categories
CATEGORY_MAP = {
    "Tshirts": "shirts",
    "Shirts": "shirts",
    "Tops": "shirts",
    "Sweatshirts": "shirts",
    "Sweaters": "shirts",
    "Jackets": "shirts",
    
    "Sports Shoes": "shoes",
    "Casual Shoes": "shoes",
    "Formal Shoes": "shoes",
    "Heels": "shoes",
    "Flats": "shoes",
    "Sandals": "shoes",
    
    "Handbags": "bags",
    "Backpacks": "bags",
    "Clutches": "bags",
    "Wallets": "bags",
    "Laptop Bag": "bags",
    "Duffel Bag": "bags",
    "Messenger Bag": "bags",
    
    "Caps": "hats",
    "Hats": "hats",
}

def import_fashion(source_dir: Path, out_dir: Path, schema: Schema, max_items: int = 3000, seed: int = 7):
    """Read styles.csv, filter, standardize, and output products.parquet."""
    rng = random.Random(seed)
    
    csv_path = source_dir / "styles.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find styles.csv in {source_dir}. Did you mount the dataset?")
        
    logging.info(f"Loading {csv_path}...")
    
    # Read CSV, dropping rows with malformed lines (oneline error in the Kaggle dataset)
    df_raw = pd.read_csv(csv_path, on_bad_lines='skip')
    
    rows = []
    
    for _, row in df_raw.iterrows():
        if len(rows) >= max_items:
            break
            
        # 1. Map Category
        article_type = str(row.get("articleType", ""))
        if article_type not in CATEGORY_MAP:
            continue
        category = CATEGORY_MAP[article_type]
        
        # 2. Extract Title
        title = str(row.get("productDisplayName", ""))
        if pd.isna(title) or not title.strip():
            continue
            
        # 3. Extract Image Path
        product_id = str(row.get("id", ""))
        if not product_id:
            continue
            
        img_path = (source_dir / "images" / f"{product_id}.jpg").resolve()
        if not img_path.exists():
            continue # Only include products where the image is actually present
            
        # 4. Extract Attributes mapping to our schema
        raw_color = str(row.get("baseColour", "")).lower()
        
        # Normalize via our schema so we don't get junk data
        attrs = {}
        if raw_color and schema.in_domain("color", raw_color):
            attrs["color"] = schema.normalize("color", raw_color)
            
        # Ensure mandatory fields are present
        if "color" not in attrs:
            continue
            
        rows.append({
            "row_id": product_id,
            "product_id": product_id,
            "title": title,
            "category": category,
            "attributes": json.dumps(attrs, ensure_ascii=False),
            "canonical_text": text_views.canonical_text(title, category, attrs),
            "image_path": str(img_path),
            "is_image_missing": False,
            "is_text_missing": False,
        })
                
    if not rows:
        raise ValueError(f"No valid products matching our 4 categories were found in {source_dir}.")
        
    df = pd.DataFrame(rows)
    
    # Create calibration / report splits
    products = df["product_id"].tolist()
    rng.shuffle(products)
    calibration_fraction = 0.5
    n_cal = int(len(products) * calibration_fraction)
    cal_set = set(products[:n_cal])
    df["split"] = df["product_id"].map(lambda p: "calibration" if p in cal_set else "report")
    
    # Save the output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "products.parquet"
    df.to_parquet(out_file, index=False)
    logging.info(f"Successfully imported {len(df)} Fashion products into {out_file}")
    
    # Save meta
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_dir),
        "seed": seed,
        "n_products": len(df),
        "by_category": df["category"].value_counts().to_dict(),
        "by_split": df["split"].value_counts().to_dict()
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    return df
