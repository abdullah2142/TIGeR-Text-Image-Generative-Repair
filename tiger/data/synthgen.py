"""Seeded synthetic sample catalogue: products + rendered studio images.

Replaces the lost toy dataset (original lived on another machine) and doubles
as the bundled sample dataset roadmap item 4.2 asks for.  Everything is
deterministic given the seed.

Each product gets:
  - structured attributes (color, material, pattern, size, brand) drawn from
    the schema domains in configs/schema.yaml
  - a natural title (sometimes containing the colour word; a few brands contain
    colour words on purpose, e.g. "Red Wing Supply", to exercise safe patching)
  - a rendered 224x224 studio image: near-white background, category-specific
    silhouette filled with the product colour, striped/dotted pattern rendering,
    seeded jitter in position/scale/shade

Output: data/sample/products.parquet + data/sample/images/*.jpg + meta json.
Rows carry product_id (the duplicate-group / resampling unit, roadmap 1.8) and
a product-level calibration/report split (roadmap 1.3 / 5.5).
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from tiger import text_views
from tiger.schema import Schema

COLOR_RGB = {
    "red": (200, 35, 35),
    "blue": (45, 75, 200),
    "green": (40, 150, 70),
    "black": (28, 28, 30),
    "white": (240, 240, 238),
    "yellow": (230, 205, 50),
    "pink": (235, 105, 170),
    "purple": (130, 60, 180),
    "brown": (125, 82, 48),
    "orange": (235, 130, 40),
    "gray": (128, 130, 132),
}

MATERIALS = {
    "shirts": ["cotton", "polyester", "denim", "wool"],
    "shoes": ["leather", "canvas", "suede"],
    "bags": ["leather", "canvas", "denim"],
    "hats": ["wool", "cotton", "canvas"],
}

SIZES = {
    "shirts": ["XS", "S", "M", "L", "XL"],
    "shoes": ["6", "7", "8", "9", "10", "11"],
    "bags": ["one-size"],
    "hats": ["one-size"],
}

BRANDS = [
    "Northline", "Aster & Co", "Red Wing Supply", "Bluebird Atelier", "Kestrel",
    "Marrow Lane", "Cinder Works", "Golden Fern", "Harbor Twelve", "Quill",
]

TITLE_NOUNS = {
    "shirts": ["Crewneck Shirt", "Oxford Shirt", "Camp Shirt", "Henley Shirt"],
    "shoes": ["Trail Shoe", "Court Sneaker", "Derby Shoe", "Slip-On Shoe"],
    "bags": ["Tote Bag", "Duffel Bag", "Messenger Bag", "Day Bag"],
    "hats": ["Bucket Hat", "Field Cap", "Beanie Hat", "Wide-Brim Hat"],
}

PATTERN_WEIGHTS = [("solid", 0.70), ("striped", 0.20), ("dotted", 0.10)]


def _weighted(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    r = rng.random()
    acc = 0.0
    for name, w in pairs:
        acc += w
        if r <= acc:
            return name
    return pairs[-1][0]


def _jitter_rgb(rng: random.Random, rgb: tuple[int, int, int], amt: int = 18) -> tuple[int, int, int]:
    return tuple(int(np.clip(c + rng.randint(-amt, amt), 0, 255)) for c in rgb)


def _shape_points(category: str, cx: float, cy: float, s: float) -> list[tuple[float, float]]:
    """Category-specific silhouette polygon centred at (cx, cy), scale s."""
    if category == "shirts":  # T-shape: sleeves + body
        return [
            (cx - 0.90 * s, cy - 0.55 * s), (cx - 0.35 * s, cy - 0.75 * s),
            (cx + 0.35 * s, cy - 0.75 * s), (cx + 0.90 * s, cy - 0.55 * s),
            (cx + 0.72 * s, cy - 0.15 * s), (cx + 0.45 * s, cy - 0.25 * s),
            (cx + 0.45 * s, cy + 0.80 * s), (cx - 0.45 * s, cy + 0.80 * s),
            (cx - 0.45 * s, cy - 0.25 * s), (cx - 0.72 * s, cy - 0.15 * s),
        ]
    if category == "bags":  # trapezoid body (handle drawn separately)
        return [
            (cx - 0.70 * s, cy - 0.35 * s), (cx + 0.70 * s, cy - 0.35 * s),
            (cx + 0.85 * s, cy + 0.70 * s), (cx - 0.85 * s, cy + 0.70 * s),
        ]
    if category == "hats":  # crown triangle-ish dome (brim drawn separately)
        return [
            (cx - 0.55 * s, cy + 0.15 * s), (cx - 0.35 * s, cy - 0.55 * s),
            (cx + 0.35 * s, cy - 0.55 * s), (cx + 0.55 * s, cy + 0.15 * s),
        ]
    # shoes: side profile
    return [
        (cx - 0.90 * s, cy + 0.35 * s), (cx - 0.80 * s, cy - 0.15 * s),
        (cx - 0.35 * s, cy - 0.30 * s), (cx + 0.10 * s, cy - 0.60 * s),
        (cx + 0.55 * s, cy - 0.30 * s), (cx + 0.90 * s, cy + 0.05 * s),
        (cx + 0.90 * s, cy + 0.45 * s), (cx - 0.90 * s, cy + 0.45 * s),
    ]


def render_product_image(
    out_path: Path,
    category: str,
    color: str,
    pattern: str,
    rng: random.Random,
    size: int = 224,
) -> None:
    bg_shade = rng.randint(238, 250)
    im = Image.new("RGB", (size, size), (bg_shade, bg_shade, bg_shade))
    d = ImageDraw.Draw(im)

    fill = _jitter_rgb(rng, COLOR_RGB[color])
    cx = size / 2 + rng.uniform(-0.04, 0.04) * size
    cy = size / 2 + rng.uniform(-0.04, 0.04) * size
    s = size * rng.uniform(0.36, 0.44)

    pts = _shape_points(category, cx, cy, s)
    d.polygon(pts, fill=fill, outline=(60, 60, 60))

    if category == "bags":  # handle
        d.arc([cx - 0.45 * s, cy - 0.95 * s, cx + 0.45 * s, cy - 0.05 * s], 180, 360,
              fill=(60, 60, 60), width=max(2, size // 45))
    if category == "hats":  # brim
        d.ellipse([cx - 0.85 * s, cy + 0.02 * s, cx + 0.85 * s, cy + 0.32 * s],
                  fill=fill, outline=(60, 60, 60))
    if category == "shoes":  # sole
        d.rectangle([cx - 0.90 * s, cy + 0.45 * s, cx + 0.90 * s, cy + 0.58 * s],
                    fill=(50, 45, 42))

    # pattern rendering, clipped to a bitmap mask of the silhouette
    if pattern in ("striped", "dotted"):
        mask = Image.new("L", (size, size), 0)
        dm = ImageDraw.Draw(mask)
        dm.polygon(pts, fill=255)
        if category == "hats":
            dm.ellipse([cx - 0.85 * s, cy + 0.02 * s, cx + 0.85 * s, cy + 0.32 * s], fill=255)
        overlay = Image.new("RGB", (size, size), fill)
        od = ImageDraw.Draw(overlay)
        second = (250, 250, 250) if color not in ("white", "gray", "yellow") else (40, 40, 40)
        if pattern == "striped":
            step = max(6, size // 16)
            for x in range(0, size, step * 2):
                od.rectangle([x, 0, x + step, size], fill=second)
        else:
            step = max(10, size // 12)
            rad = max(2, size // 40)
            for y in range(step // 2, size, step):
                for x in range(step // 2, size, step):
                    od.ellipse([x - rad, y - rad, x + rad, y + rad], fill=second)
        im = Image.composite(overlay, im, mask)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="JPEG", quality=92)


def make_title(rng: random.Random, category: str, color: str, material: str, brand: str) -> str:
    noun = rng.choice(TITLE_NOUNS[category])
    style = rng.random()
    if style < 0.6:  # colour word in title (common case; exercises title checks)
        return f"{brand} {color.capitalize()} {material.capitalize()} {noun}"
    if style < 0.8:
        return f"{brand} {material.capitalize()} {noun}"
    return f"{brand} {noun}"


def generate(
    root: Path,
    schema: Schema,
    seed: int = 20260717,
    products_per_category: int = 30,
    image_size: int = 224,
    calibration_fraction: float = 0.5,
    out_dir: str = "data/sample",
) -> pd.DataFrame:
    rng = random.Random(seed)
    out = (root / out_dir).resolve()
    images_dir = out / "images"

    color_domain = [c for c in schema.domain("color") if c != "multicolour"]
    categories = list(schema.categories)

    rows = []
    for category in categories:
        for i in range(products_per_category):
            product_id = f"{category}_{i:03d}"
            color = rng.choice(color_domain)
            material = rng.choice(MATERIALS[category])
            pattern = _weighted(rng, PATTERN_WEIGHTS)
            size_v = rng.choice(SIZES[category])
            brand = rng.choice(BRANDS)

            attrs = {"color": color, "material": material, "pattern": pattern,
                     "size": size_v, "brand": brand}
            title = make_title(rng, category, color, material, brand)

            img_rel = f"{out_dir}/images/{product_id}.jpg"
            render_product_image(images_dir / f"{product_id}.jpg", category, color,
                                 pattern, rng, size=image_size)

            rows.append({
                "row_id": product_id,
                "product_id": product_id,
                "title": title,
                "category": category,
                "attributes": json.dumps(attrs, ensure_ascii=False),
                "canonical_text": text_views.canonical_text(title, category, attrs),
                "image_path": img_rel,
                "is_image_missing": False,
                "is_text_missing": False,
            })
            
    # Force a completely unique product to guarantee a Generative Fallback path
    unique_id = "forced_gen_000"
    unique_img = f"{out_dir}/images/{unique_id}.jpg"
    render_product_image(images_dir / f"{unique_id}.jpg", "shirts", "magenta", "solid", rng, size=image_size)
    
    unique_attrs = {"color": "magenta", "material": "velvet", "pattern": "solid", "size": "L", "brand": "GenerativeLabs"}
    unique_title = "Magenta Velvet Vintage spacesuit"
    rows.append({
        "row_id": unique_id,
        "product_id": unique_id,
        "title": unique_title,
        "category": "shirts",
        "attributes": json.dumps(unique_attrs, ensure_ascii=False),
        "canonical_text": text_views.canonical_text(unique_title, "shirts", unique_attrs),
        "image_path": unique_img,
        "is_image_missing": False,
        "is_text_missing": False,
    })

    df = pd.DataFrame(rows)

    # product-level calibration/report split (roadmap 1.3, 5.5)
    products = df["product_id"].tolist()
    rng.shuffle(products)
    n_cal = int(len(products) * calibration_fraction)
    cal_set = set(products[:n_cal])
    df["split"] = df["product_id"].map(lambda p: "calibration" if p in cal_set else "report")

    df.to_parquet(out / "products.parquet", index=False)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "products_per_category": products_per_category,
        "image_size": image_size,
        "calibration_fraction": calibration_fraction,
        "n_products": len(df),
        "categories": categories,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df
