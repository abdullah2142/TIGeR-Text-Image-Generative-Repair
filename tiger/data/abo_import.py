"""ABO Dataset Adapter: Converts Amazon Berkeley Objects to TIGeR parquet format.

ABO contains massive JSON metadata files and 398k images. This script parses
the raw ABO metadata, filters for our core categories (shirts, shoes, bags, hats),
maps the Amazon attributes to our strict `configs/schema.yaml`, and outputs
the `products.parquet` file required by the Sieve.
"""

import json
import logging
import random
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from tiger.schema import Schema
from tiger import text_views

# Mapping Amazon ABO 'product_type' (which has hundreds of weird strings) 
# to our 4 strict categories.
CATEGORY_MAP = {
    "SHIRT": "shirts",
    "T_SHIRT": "shirts",
    "SWEATER": "shirts",
    "SHOES": "shoes",
    "BOOT": "shoes",
    "SANDAL": "shoes",
    "HANDBAG": "bags",
    "BACKPACK": "bags",
    "LUGGAGE": "bags",
    "HAT": "hats",
}

def _parse_abo_value(val_list):
    """ABO attributes are often lists of dicts (e.g., [{'value': 'Blue', 'language_tag': 'en_US'}])"""
    if not isinstance(val_list, list):
        return ""
    for item in val_list:
        if isinstance(item, dict) and item.get("language_tag", "").startswith("en"):
            return str(item.get("value", "")).lower()
        if isinstance(item, dict) and not item.get("language_tag"):
            return str(item.get("value", "")).lower()
    return ""

def import_abo(source_dir: Path, out_dir: Path, schema: Schema, max_items: int = 1000, seed: int = 7):
    """Read ABO metadata, filter, standardize, and output products.parquet."""
    rng = random.Random(seed)
    
    # In ABO, the main metadata file is usually something like listings/metadata/listings_0.json.gz
    # For this adapter, we will look for any JSON or JSONL files in the source directory.
    metadata_files = list(source_dir.glob("**/*.json")) + list(source_dir.glob("**/*.json.gz"))
    if not metadata_files:
        raise FileNotFoundError(f"No JSON metadata files found in {source_dir}")
        
    logging.info(f"Found {len(metadata_files)} metadata files. Processing...")
    
    rows = []
    
    for meta_file in metadata_files:
        if len(rows) >= max_items:
            break
            
        # We handle both JSONL (one object per line) and standard JSON list
        # ABO is typically JSONL.
        import gzip
        open_fn = gzip.open if meta_file.suffix == '.gz' else open
        
        with open_fn(meta_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if len(rows) >= max_items:
                    break
                    
                line = line.strip()
                if not line or line.startswith('['): continue
                
                try:
                    # Some files are list of dicts, some are jsonl. Just try parsing the line.
                    item = json.loads(line.rstrip(','))
                except json.JSONDecodeError:
                    continue
                    
                # 1. Map Category
                abo_type = _parse_abo_value(item.get("product_type", [])).upper()
                if abo_type not in CATEGORY_MAP:
                    continue
                category = CATEGORY_MAP[abo_type]
                
                # 2. Extract Title
                title = _parse_abo_value(item.get("item_name", []))
                if not title:
                    continue
                    
                # 3. Extract Image Path
                main_image = item.get("main_image_id")
                if not main_image:
                    continue
                # ABO images are stored as e.g., images/metadata/XY/XYZ123.jpg
                # The Kaggle user will need to mount the image directory.
                # We assume images are at source_dir/images/<main_image_id>.jpg
                img_path = (source_dir / "images" / f"{main_image}.jpg").resolve()
                
                # 4. Extract Attributes mapping to our schema
                raw_color = _parse_abo_value(item.get("color", []))
                raw_material = _parse_abo_value(item.get("material", []))
                raw_pattern = _parse_abo_value(item.get("pattern", []))
                raw_brand = _parse_abo_value(item.get("brand", []))
                
                # Normalize via our schema so we don't get junk data
                attrs = {}
                if raw_color and schema.in_domain("color", raw_color):
                    attrs["color"] = schema.normalize("color", raw_color)
                if raw_material and schema.in_domain("material", raw_material):
                    attrs["material"] = schema.normalize("material", raw_material)
                if raw_pattern and schema.in_domain("pattern", raw_pattern):
                    attrs["pattern"] = schema.normalize("pattern", raw_pattern)
                if raw_brand:
                    attrs["brand"] = raw_brand[:60] # max len from schema
                    
                # Ensure mandatory fields are present
                if "color" not in attrs:
                    continue
                    
                product_id = item.get("item_id", f"abo_{len(rows)}")
                
                rows.append({
                    "row_id": str(product_id),
                    "product_id": str(product_id),
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
    logging.info(f"Successfully imported {len(df)} ABO products into {out_file}")
    
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
