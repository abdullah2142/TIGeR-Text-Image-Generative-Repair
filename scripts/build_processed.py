from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
import yaml
from PIL import Image


@dataclass
class BuildConfig:
    raw_csv: Path
    processed_dir: Path


def load_config(config_path: Path) -> BuildConfig:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})

    raw_csv = Path(data_cfg.get("raw_csv", "data/raw/rows.csv"))
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))

    return BuildConfig(raw_csv=raw_csv, processed_dir=processed_dir)


def safe_json_load(s: str) -> dict:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return {}
    s = str(s).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# -----------------------------
# Upgrade 2B: Stable, sensitive canonical text (color-first)
# -----------------------------
def _norm_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def build_canonical_text(title: str, category: str, attrs: dict) -> str:
    """
    Deterministic canonical text to improve sensitivity to attribute mutations (esp. color).
    Template:
      "{title}. Color: {color}. Category: {category}. Attributes: k=v, k=v."
    - Color is placed early to increase impact on text embedding.
    - Attributes are sorted for deterministic ordering.
    """
    title = _norm_str(title)
    category = _norm_str(category)
    attrs = dict(attrs) if isinstance(attrs, dict) else {}

    # normalize color if present
    color = attrs.get("color", None)
    if isinstance(color, str):
        color = color.strip().lower()
        if color:
            attrs["color"] = color
        else:
            attrs.pop("color", None)
            color = None
    elif color is None:
        pass
    else:
        # if it's non-string, coerce to string
        color = str(color).strip().lower()
        if color:
            attrs["color"] = color
        else:
            attrs.pop("color", None)
            color = None

    # stable ordering
    items = sorted((str(k), str(v)) for k, v in attrs.items())
    attrs_for_text = ", ".join([f"{k}={v}" for k, v in items])

    color_prefix = f"Color: {color}. " if color else ""
    return f"{title}. {color_prefix}Category: {category}. Attributes: {attrs_for_text}."


def download_image(url: str, out_path: Path, timeout: int = 30) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def normalize_image(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.save(dst, format="JPEG", quality=92)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()

    cfg = load_config(config_path)

    raw_csv = (project_root / cfg.raw_csv).resolve()
    processed_dir = (project_root / cfg.processed_dir).resolve()
    processed_images_dir = processed_dir / "images"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")

    df = pd.read_csv(raw_csv)

    required_cols = ["row_id", "title", "category", "attributes_json", "image_path", "image_url"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    out_rows = []
    missing_images = 0
    downloaded = 0
    normalized = 0

    for _, row in df.iterrows():
        row_id = _norm_str(row.get("row_id", ""))
        title = _norm_str(row.get("title", ""))
        category = _norm_str(row.get("category", ""))
        attrs = safe_json_load(row.get("attributes_json", ""))

        image_path = _norm_str(row.get("image_path", ""))
        image_url = _norm_str(row.get("image_url", ""))

        # Determine source image
        src_local = None
        src_url_downloaded = None

        if image_path:
            p = Path(image_path)
            src_local = (project_root / p).resolve() if not p.is_absolute() else p
        elif image_url:
            tmp_path = processed_dir / "tmp_downloads" / f"{row_id}.img"
            ok = download_image(image_url, tmp_path)
            if ok:
                downloaded += 1
                src_url_downloaded = tmp_path
            else:
                src_url_downloaded = None

        # Decide final processed image path
        processed_image_path = ""
        is_image_missing = False

        if src_local and src_local.exists():
            dst = processed_images_dir / f"{row_id}.jpg"
            ok = normalize_image(src_local, dst)
            if ok:
                normalized += 1
                processed_image_path = str(dst.relative_to(project_root))
            else:
                is_image_missing = True
                missing_images += 1
        elif src_url_downloaded and src_url_downloaded.exists():
            dst = processed_images_dir / f"{row_id}.jpg"
            ok = normalize_image(src_url_downloaded, dst)
            if ok:
                normalized += 1
                processed_image_path = str(dst.relative_to(project_root))
            else:
                is_image_missing = True
                missing_images += 1
        else:
            is_image_missing = True
            missing_images += 1

        is_text_missing = (title == "" and (not attrs or len(attrs) == 0))

        # Canonical text we’ll use later for CLIP text embedding (Upgrade 2B)
        canonical_text = build_canonical_text(title, category, attrs)

        out_rows.append(
            {
                "row_id": row_id,
                "title": title,
                "category": category,
                "attributes": json.dumps(attrs, ensure_ascii=False),
                "canonical_text": canonical_text,
                "image_path": processed_image_path,
                "is_image_missing": bool(is_image_missing),
                "is_text_missing": bool(is_text_missing),
                "source_image_path": image_path,
                "source_image_url": image_url,
            }
        )

    out_df = pd.DataFrame(out_rows)

    out_parquet = processed_dir / "processed_rows.parquet"
    out_df.to_parquet(out_parquet, index=False)

    # Clean temp downloads folder (optional)
    tmp_dir = processed_dir / "tmp_downloads"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("✅ Built:", out_parquet)
    print("Rows:", len(out_df))
    print("Downloaded images:", downloaded)
    print("Normalized images:", normalized)
    print("Missing images:", missing_images)
    print("Missing text rows:", int(out_df["is_text_missing"].sum()))
    print("Example row:\n", out_df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
