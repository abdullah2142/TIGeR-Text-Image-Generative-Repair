from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_attributes(val) -> dict:
    """Attributes may be dict, JSON string, NaN, etc."""
    if isinstance(val, dict):
        return val
    if val is None:
        return {}
    if isinstance(val, float) and np.isnan(val):
        return {}
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def build_attrs_text(row: dict) -> str:
    """Make attribute-only text view: 'Color: purple. Type: duffel.'"""
    attrs = parse_attributes(row.get("attributes", None))
    if not attrs:
        return ""

    parts = []
    preferred = ["color", "material", "type", "brand", "pattern", "size"]
    used = set()

    for k in preferred:
        if k in attrs and attrs[k] not in (None, "", []):
            parts.append(f"{k.replace('_', ' ').title()}: {attrs[k]}")
            used.add(k)

    for k in sorted(attrs.keys()):
        if k in used:
            continue
        v = attrs.get(k)
        if v in (None, "", []):
            continue
        parts.append(f"{k.replace('_', ' ').title()}: {v}")

    if not parts:
        return ""
    return ". ".join(parts) + "."


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def compute_thresholds_iqr_or_quantile(
    df_valid: pd.DataFrame,
    col: str,
    method: str,
    iqr_k: float,
    q: float,
    min_cat_samples: int,
    global_method: str,
    global_q: float,
) -> tuple[dict, dict, float]:
    """
    Per-category thresholds with global fallback.

    Returns:
      thresholds: dict(category -> threshold)
      thr_source: dict(category -> "category"|"global")
      global_thr: float
    """
    vals_all = df_valid[col].dropna().astype(float).values
    if len(vals_all) == 0:
        return {}, {}, np.nan

    # --- GLOBAL threshold ---
    if global_method == "iqr":
        q1g = np.quantile(vals_all, 0.25)
        q3g = np.quantile(vals_all, 0.75)
        iqrg = q3g - q1g
        global_thr = float(q1g - iqr_k * iqrg)
    else:
        global_thr = float(np.quantile(vals_all, global_q))

    thresholds: dict[str, float] = {}
    thr_source: dict[str, str] = {}

    for cat, g in df_valid.groupby("category"):
        vals = g[col].dropna().astype(float).values
        if len(vals) < min_cat_samples:
            thresholds[str(cat)] = global_thr
            thr_source[str(cat)] = "global"
            continue

        if method == "quantile":
            thresholds[str(cat)] = float(np.quantile(vals, q))
        else:
            q1 = np.quantile(vals, 0.25)
            q3 = np.quantile(vals, 0.75)
            iqr = q3 - q1
            thresholds[str(cat)] = float(q1 - iqr_k * iqr)

        thr_source[str(cat)] = "category"

    return thresholds, thr_source, global_thr


def compute_upper_quantile_thresholds(
    df_valid: pd.DataFrame,
    col: str,
    q_hi: float,
    min_cat_samples: int,
    global_q_hi: float,
) -> tuple[dict, dict, float]:
    """
    For "high outliers" like GAP: threshold = upper quantile (e.g., 0.95).
    Outlier rule becomes: value > threshold (and optionally > 0).

    Returns:
      thresholds: dict(category -> threshold)
      thr_source: dict(category -> "category"|"global")
      global_thr: float
    """
    vals_all = df_valid[col].dropna().astype(float).values
    if len(vals_all) == 0:
        return {}, {}, np.nan

    global_thr = float(np.quantile(vals_all, global_q_hi))

    thresholds: dict[str, float] = {}
    thr_source: dict[str, str] = {}

    for cat, g in df_valid.groupby("category"):
        vals = g[col].dropna().astype(float).values
        if len(vals) < min_cat_samples:
            thresholds[str(cat)] = global_thr
            thr_source[str(cat)] = "global"
        else:
            thresholds[str(cat)] = float(np.quantile(vals, q_hi))
            thr_source[str(cat)] = "category"

    return thresholds, thr_source, global_thr


def load_thresholds_json(path: Path) -> tuple[dict[str, float], float | None, dict]:
    """
    Expected JSON formats supported:
      1) {"thresholds": {"bags": 0.23, ...}, "global_threshold": 0.20, ...}
      2) {"thresholds": {"bags": 0.23, ...}}  (global optional)
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    thresholds = obj.get("thresholds", {}) or {}
    thresholds = {str(k): float(v) for k, v in thresholds.items()}

    global_thr = obj.get("global_threshold", None)
    if global_thr is not None:
        global_thr = float(global_thr)

    return thresholds, global_thr, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input_parquet", default="", help="Override input parquet path")

    ap.add_argument(
        "--thresholds_json",
        default="",
        help="If provided, use these per-category thresholds instead of computing from the input set.",
    )
    ap.add_argument(
        "--write_thresholds_json",
        default="",
        help="If provided, writes the computed thresholds JSON to this path (use on CLEAN set to create baseline).",
    )

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(root / args.config)

    # Input parquet
    default_parquet = root / cfg["data"]["processed_dir"] / "processed_rows.parquet"
    processed_parquet = Path(args.input_parquet) if args.input_parquet else default_parquet
    processed_parquet = processed_parquet if processed_parquet.is_absolute() else (root / processed_parquet)
    processed_parquet = processed_parquet.resolve()

    if not processed_parquet.exists():
        raise FileNotFoundError(f"Missing input parquet: {processed_parquet}")

    # Output dirs
    outputs_dir = (root / cfg["data"]["outputs_dir"]).resolve()
    cache_dir = (root / cfg["data"]["cache_dir"]).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_parquet)

    # Config
    model_name = cfg["models"]["clip_model_name"]
    device = cfg.get("sieve", {}).get("device", "cpu")
    batch_size = int(cfg.get("sieve", {}).get("batch_size", 16))

    threshold_method = cfg.get("sieve", {}).get("threshold_method", "iqr")
    iqr_k = float(cfg.get("sieve", {}).get("iqr_k", 1.5))
    quantile_q = float(cfg.get("sieve", {}).get("quantile_q", 0.05))

    min_cat = int(cfg.get("sieve", {}).get("min_category_samples", 8))
    global_method = cfg.get("sieve", {}).get("global_threshold_method", "quantile")
    global_q = float(cfg.get("sieve", {}).get("global_quantile_q", 0.10))

    # ✅ Upgrade 3A knobs (multi-text + attrs outlier)
    enable_multi_text = bool(cfg.get("sieve", {}).get("enable_multi_text", True))
    attrs_outlier_q = float(cfg.get("sieve", {}).get("attrs_outlier_quantile", 0.05))
    attrs_min_cat = int(cfg.get("sieve", {}).get("attrs_min_category_samples", min_cat))
    attrs_global_q = float(cfg.get("sieve", {}).get("attrs_global_quantile", global_q))

    # ✅ Gap probe knobs
    enable_gap_probe = bool(cfg.get("sieve", {}).get("enable_gap_probe", True))
    gap_outlier_q = float(cfg.get("sieve", {}).get("gap_outlier_quantile", 0.95))
    gap_min_cat = int(cfg.get("sieve", {}).get("gap_min_category_samples", min_cat))
    gap_global_q = float(cfg.get("sieve", {}).get("gap_global_quantile", gap_outlier_q))
    gap_positive_only = bool(cfg.get("sieve", {}).get("gap_positive_only", True))

    # Ensure required columns exist
    required_cols = [
        "row_id",
        "category",
        "canonical_text",
        "image_path",
        "is_image_missing",
        "is_text_missing",
        "title",
        "attributes",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = False if col in ["is_image_missing", "is_text_missing"] else ""

    # Make sure missing flags are boolean
    df["is_image_missing"] = df["is_image_missing"].fillna(False).astype(bool)
    df["is_text_missing"] = df["is_text_missing"].fillna(False).astype(bool)

    # Runtime "missing" corrections (in case parquet isn't perfect)
    runtime_is_image_missing = df["is_image_missing"].to_numpy().copy()
    runtime_is_text_missing = df["is_text_missing"].to_numpy().copy()

    print("Loading CLIP:", model_name)
    model = CLIPModel.from_pretrained(model_name)
    try:
        processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    except TypeError:
        processor = CLIPProcessor.from_pretrained(model_name)

    model.eval()
    model.to(device)

    emb_dim = int(getattr(model.config, "projection_dim", 512))

    rows = df.to_dict(orient="records")

    # Similarities
    sim_full = [np.nan] * len(rows)   # df["sim_score"]
    sim_title = [np.nan] * len(rows)
    sim_attrs = [np.nan] * len(rows)

    # Embeddings
    text_emb_full = [None] * len(rows)
    text_emb_title = [None] * len(rows)
    text_emb_attrs = [None] * len(rows)
    image_embs = [None] * len(rows)

    # Batch holders
    batch_indices: list[int] = []
    batch_full: list[str] = []
    batch_title: list[str] = []
    batch_attrs: list[str] = []
    batch_has_title: list[bool] = []
    batch_has_attrs: list[bool] = []
    batch_images: list[Image.Image] = []

    def flush_batch():
        nonlocal batch_indices, batch_full, batch_title, batch_attrs, batch_images, batch_has_title, batch_has_attrs
        if not batch_indices:
            return

        img_inputs = processor(images=batch_images, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}

        txt_full_inputs = processor(text=batch_full, return_tensors="pt", padding=True)
        txt_full_inputs = {k: v.to(device) for k, v in txt_full_inputs.items()}

        if enable_multi_text:
            txt_title_inputs = processor(text=batch_title, return_tensors="pt", padding=True)
            txt_title_inputs = {k: v.to(device) for k, v in txt_title_inputs.items()}

            txt_attrs_inputs = processor(text=batch_attrs, return_tensors="pt", padding=True)
            txt_attrs_inputs = {k: v.to(device) for k, v in txt_attrs_inputs.items()}
        else:
            txt_title_inputs = None
            txt_attrs_inputs = None

        with torch.inference_mode():
            v = model.get_image_features(**img_inputs)
            t_full = model.get_text_features(**txt_full_inputs)

            v = _l2_normalize(v)
            t_full = _l2_normalize(t_full)
            s_full = (t_full * v).sum(dim=-1).detach().cpu().numpy()

            if enable_multi_text:
                t_title = model.get_text_features(**txt_title_inputs)
                t_attrs = model.get_text_features(**txt_attrs_inputs)

                t_title = _l2_normalize(t_title)
                t_attrs = _l2_normalize(t_attrs)

                s_title = (t_title * v).sum(dim=-1).detach().cpu().numpy()
                s_attrs = (t_attrs * v).sum(dim=-1).detach().cpu().numpy()
            else:
                t_title = None
                t_attrs = None
                s_title = None
                s_attrs = None

        v_np = v.detach().cpu().numpy().astype(np.float32)
        t_full_np = t_full.detach().cpu().numpy().astype(np.float32)

        if enable_multi_text:
            t_title_np = t_title.detach().cpu().numpy().astype(np.float32)
            t_attrs_np = t_attrs.detach().cpu().numpy().astype(np.float32)

        for i, df_idx in enumerate(batch_indices):
            sim_full[df_idx] = float(s_full[i])
            image_embs[df_idx] = v_np[i]
            text_emb_full[df_idx] = t_full_np[i]

            if enable_multi_text:
                if batch_has_title[i]:
                    sim_title[df_idx] = float(s_title[i])
                    text_emb_title[df_idx] = t_title_np[i]
                else:
                    sim_title[df_idx] = np.nan
                    text_emb_title[df_idx] = None

                if batch_has_attrs[i]:
                    sim_attrs[df_idx] = float(s_attrs[i])
                    text_emb_attrs[df_idx] = t_attrs_np[i]
                else:
                    sim_attrs[df_idx] = np.nan
                    text_emb_attrs[df_idx] = None

        # Clear batch
        batch_indices, batch_full, batch_title, batch_attrs, batch_images = [], [], [], [], []
        batch_has_title, batch_has_attrs = [], []

    # Encode
    for idx, r in enumerate(rows):
        # Text missing (runtime)
        full_text_raw = str(r.get("canonical_text", "")).strip()
        if not full_text_raw:
            runtime_is_text_missing[idx] = True

        # Image path checks (runtime)
        img_rel = str(r.get("image_path", "")).strip()
        if not img_rel:
            runtime_is_image_missing[idx] = True
            continue

        img_path = (root / Path(img_rel)).resolve()
        if not img_path.exists():
            runtime_is_image_missing[idx] = True
            continue

        try:
            with Image.open(img_path) as im:
                img = im.convert("RGB")
        except Exception:
            runtime_is_image_missing[idx] = True
            continue

        # If marked missing, skip
        if runtime_is_image_missing[idx]:
            continue

        # Build multi-text
        full_text = full_text_raw if full_text_raw else " "
        title_text = str(r.get("title", "")).strip()
        has_title = bool(title_text)
        attrs_text = build_attrs_text(r)
        has_attrs = bool(attrs_text)

        # Avoid empty strings for tokenizer
        if not title_text:
            title_text = " "
        if not attrs_text:
            attrs_text = " "

        batch_indices.append(idx)
        batch_full.append(full_text)
        batch_title.append(title_text)
        batch_attrs.append(attrs_text)
        batch_has_title.append(has_title)
        batch_has_attrs.append(has_attrs)
        batch_images.append(img)

        if len(batch_indices) >= batch_size:
            flush_batch()

    flush_batch()

    # Attach runtime missing flags back
    df["is_image_missing"] = runtime_is_image_missing.astype(bool)
    df["is_text_missing"] = runtime_is_text_missing.astype(bool)

    # Similarity scores
    df["sim_score"] = sim_full
    df["sim_title"] = sim_title
    df["sim_attrs"] = sim_attrs

    # ==========================
    # Thresholds for sim_score (baseline support)
    # ==========================
    df_valid = df[~df["sim_score"].isna()].copy()

    thresholds: dict[str, float] = {}
    thr_source: dict[str, str] = {}
    global_thr: float = float("nan")
    threshold_meta: dict = {}

    if args.thresholds_json:
        thr_path = Path(args.thresholds_json)
        thr_path = thr_path if thr_path.is_absolute() else (root / thr_path)
        thr_path = thr_path.resolve()
        if not thr_path.exists():
            raise FileNotFoundError(f"Missing thresholds_json: {thr_path}")

        thresholds, json_global_thr, threshold_meta = load_thresholds_json(thr_path)

        if json_global_thr is None:
            # fallback compute global from current df_valid
            sims_all = df_valid["sim_score"].dropna().astype(float).values
            if len(sims_all) > 0:
                if global_method == "iqr":
                    q1g = np.quantile(sims_all, 0.25)
                    q3g = np.quantile(sims_all, 0.75)
                    iqrg = q3g - q1g
                    global_thr = float(q1g - iqr_k * iqrg)
                else:
                    global_thr = float(np.quantile(sims_all, global_q))
            else:
                global_thr = float("nan")
        else:
            global_thr = float(json_global_thr)

        all_cats = df["category"].astype(str).unique().tolist()
        thr_source = {cat: "baseline_json" for cat in all_cats}

    else:
        thresholds, thr_source, global_thr = compute_thresholds_iqr_or_quantile(
            df_valid=df_valid,
            col="sim_score",
            method=threshold_method,
            iqr_k=iqr_k,
            q=quantile_q,
            min_cat_samples=min_cat,
            global_method=global_method,
            global_q=global_q,
        )

    df["sieve_threshold"] = df["category"].astype(str).map(thresholds)
    df.loc[df["sieve_threshold"].isna(), "sieve_threshold"] = global_thr
    df["threshold_source"] = df["category"].astype(str).map(thr_source).fillna("global_fallback")
    df["global_threshold"] = float(global_thr) if global_thr == global_thr else np.nan

    if args.write_thresholds_json:
        out_thr = Path(args.write_thresholds_json)
        out_thr = out_thr if out_thr.is_absolute() else (root / out_thr)
        out_thr = out_thr.resolve()
        out_thr.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "created_from": str(processed_parquet),
            "threshold_method": threshold_method,
            "iqr_k": iqr_k,
            "quantile_q": quantile_q,
            "min_category_samples": min_cat,
            "global_threshold_method": global_method,
            "global_quantile_q": global_q,
            "global_threshold": float(global_thr) if global_thr == global_thr else None,
            "thresholds": {k: float(v) for k, v in thresholds.items()},
        }
        out_thr.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"✅ Wrote thresholds JSON: {out_thr}")

    # ==========================
    # Upgrade 3A: attrs outlier thresholding (LOW tail)
    # ==========================
    df["attrs_threshold"] = np.nan
    df["attrs_threshold_source"] = ""
    df["attrs_outlier"] = False

    df_valid_attrs = df[~df["sim_attrs"].isna()].copy()
    sims_attrs_all = df_valid_attrs["sim_attrs"].dropna().astype(float).values

    global_attrs_thr = float(np.quantile(sims_attrs_all, attrs_global_q)) if len(sims_attrs_all) > 0 else np.nan

    attrs_thresholds: dict[str, float] = {}
    attrs_thr_source: dict[str, str] = {}

    for cat, g in df_valid_attrs.groupby("category"):
        sims = g["sim_attrs"].dropna().astype(float).values
        if len(sims) < attrs_min_cat or len(sims) == 0 or np.isnan(global_attrs_thr):
            attrs_thresholds[str(cat)] = global_attrs_thr
            attrs_thr_source[str(cat)] = "global"
        else:
            attrs_thresholds[str(cat)] = float(np.quantile(sims, attrs_outlier_q))
            attrs_thr_source[str(cat)] = "category"

    df["attrs_threshold"] = df["category"].astype(str).map(attrs_thresholds)
    df.loc[df["attrs_threshold"].isna(), "attrs_threshold"] = global_attrs_thr
    df["attrs_threshold_source"] = df["category"].astype(str).map(attrs_thr_source).fillna("global_fallback")

    mask_attrs_outlier = (
        (~df["sim_attrs"].isna())
        & (~df["attrs_threshold"].isna())
        & (df["sim_attrs"] < df["attrs_threshold"])
    )
    df.loc[mask_attrs_outlier, "attrs_outlier"] = True

    # ==========================
    # Gap probe: gap = sim_attrs - sim_score (HIGH outliers)
    # ==========================
    df["gap"] = np.nan
    df["gap_threshold"] = np.nan
    df["gap_threshold_source"] = ""
    df["gap_outlier"] = False

    if enable_gap_probe:
        mask_gap_valid = (~df["sim_attrs"].isna()) & (~df["sim_score"].isna())
        df.loc[mask_gap_valid, "gap"] = df.loc[mask_gap_valid, "sim_attrs"] - df.loc[mask_gap_valid, "sim_score"]

        df_valid_gap = df[~df["gap"].isna()].copy()

        gap_thresholds, gap_thr_source, global_gap_thr = compute_upper_quantile_thresholds(
            df_valid=df_valid_gap,
            col="gap",
            q_hi=gap_outlier_q,
            min_cat_samples=gap_min_cat,
            global_q_hi=gap_global_q,
        )

        df["gap_threshold"] = df["category"].astype(str).map(gap_thresholds)
        df.loc[df["gap_threshold"].isna(), "gap_threshold"] = global_gap_thr
        df["gap_threshold_source"] = df["category"].astype(str).map(gap_thr_source).fillna("global_fallback")

        mask_gap_outlier = (
            (~df["gap"].isna())
            & (~df["gap_threshold"].isna())
            & (df["gap"] > df["gap_threshold"])
        )
        if gap_positive_only:
            mask_gap_outlier = mask_gap_outlier & (df["gap"] > 0)

        df.loc[mask_gap_outlier, "gap_outlier"] = True

    # ==========================
    # Flagging logic
    # ==========================
    df["flagged"] = False

    df.loc[df["is_image_missing"] == True, "flagged"] = True
    df.loc[df["is_text_missing"] == True, "flagged"] = True

    # primary: sim_full threshold
    mask_sim = (
        (~df["sim_score"].isna())
        & (~df["sieve_threshold"].isna())
        & (df["sim_score"] < df["sieve_threshold"])
    )
    df.loc[mask_sim, "flagged"] = True

    # secondary: attrs outlier
    df.loc[df["attrs_outlier"] == True, "flagged"] = True

    # tertiary: gap outlier
    if enable_gap_probe:
        df.loc[df["gap_outlier"] == True, "flagged"] = True

    # Reason (priority order)
    reasons = []
    for _, r in df.iterrows():
        if bool(r.get("is_image_missing", False)):
            reasons.append("missing_image")
        elif bool(r.get("is_text_missing", False)):
            reasons.append("missing_text")
        elif pd.notna(r.get("sim_score")) and pd.notna(r.get("sieve_threshold")) and float(r["sim_score"]) < float(r["sieve_threshold"]):
            reasons.append("low_similarity")
        elif bool(r.get("attrs_outlier", False)):
            reasons.append("attrs_outlier")
        elif enable_gap_probe and bool(r.get("gap_outlier", False)):
            reasons.append("gap_outlier")
        else:
            reasons.append("ok")
    df["flag_reason"] = reasons

    # Output files
    stem = processed_parquet.stem
    out_csv = outputs_dir / f"sieve_results_{stem}.csv"
    out_flagged = outputs_dir / f"flagged_rows_{stem}.csv"
    df.to_csv(out_csv, index=False)
    df[df["flagged"] == True].to_csv(out_flagged, index=False)

    # Embeddings cache
    emb_npz = cache_dir / f"clip_embeddings_{stem}.npz"

    def as_vec(e):
        if e is None:
            return np.zeros((emb_dim,), dtype=np.float32)
        e = np.asarray(e, dtype=np.float32)
        if e.shape[0] != emb_dim:
            out = np.zeros((emb_dim,), dtype=np.float32)
            n = min(emb_dim, e.shape[0])
            out[:n] = e[:n]
            return out
        return e

    np.savez_compressed(
        emb_npz,
        row_id=np.asarray(df["row_id"].astype(str).values, dtype="U"),
        text_emb=np.stack([as_vec(e) for e in text_emb_full]),
        image_emb=np.stack([as_vec(e) for e in image_embs]),
        text_emb_title=np.stack([as_vec(e) for e in text_emb_title]),
        text_emb_attrs=np.stack([as_vec(e) for e in text_emb_attrs]),
        sim_score=np.array(df["sim_score"].fillna(-1.0).values, dtype=np.float32),
        sim_title=np.array(df["sim_title"].fillna(-1.0).values, dtype=np.float32),
        sim_attrs=np.array(df["sim_attrs"].fillna(-1.0).values, dtype=np.float32),
        gap=np.array(df["gap"].fillna(-1.0).values, dtype=np.float32),
        sieve_threshold=np.array(df["sieve_threshold"].fillna(-1.0).values, dtype=np.float32),
        attrs_threshold=np.array(df["attrs_threshold"].fillna(-1.0).values, dtype=np.float32),
        gap_threshold=np.array(df["gap_threshold"].fillna(-1.0).values, dtype=np.float32),
        flagged=np.array(df["flagged"].astype(int).values, dtype=np.int32),
    )

    print("✅ Sieve complete (Upgrade 3A + gap probe)")
    print("Input:", processed_parquet)
    print("Results:", out_csv)
    print("Flagged:", out_flagged)
    print("Embedding cache:", emb_npz)

    print("\nSummary:")
    cols = [
        "row_id",
        "category",
        "sim_score",
        "sim_title",
        "sim_attrs",
        "gap",
        "sieve_threshold",
        "attrs_threshold",
        "gap_threshold",
        "threshold_source",
        "attrs_threshold_source",
        "gap_threshold_source",
        "flagged",
        "flag_reason",
    ]
    keep = [c for c in cols if c in df.columns]
    print(df[keep].to_string(index=False))


if __name__ == "__main__":
    main()
