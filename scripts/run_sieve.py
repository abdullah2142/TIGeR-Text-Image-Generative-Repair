from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def compute_thresholds(
    df_valid: pd.DataFrame,
    method: str,
    iqr_k: float,
    q: float,
    min_cat_samples: int,
    global_method: str,
    global_q: float,
) -> tuple[dict, dict, float]:
    """
    Returns:
      thresholds: dict(category -> threshold)
      thr_source: dict(category -> "category"|"global")
      global_thr: float
    """

    sims_all = df_valid["sim_score"].dropna().astype(float).values
    if len(sims_all) == 0:
        return {}, {}, -1.0

    # --- compute GLOBAL threshold ---
    if global_method == "iqr":
        q1g = np.quantile(sims_all, 0.25)
        q3g = np.quantile(sims_all, 0.75)
        iqrg = q3g - q1g
        global_thr = float(q1g - iqr_k * iqrg)
    else:
        # default: global quantile
        global_thr = float(np.quantile(sims_all, global_q))

    thresholds = {}
    thr_source = {}

    # --- compute per-category thresholds, with fallback ---
    for cat, g in df_valid.groupby("category"):
        sims = g["sim_score"].dropna().astype(float).values

        if len(sims) < min_cat_samples:
            thresholds[cat] = global_thr
            thr_source[cat] = "global"
            continue

        if method == "quantile":
            thresholds[cat] = float(np.quantile(sims, q))
        else:
            # default: IQR lower fence
            q1 = np.quantile(sims, 0.25)
            q3 = np.quantile(sims, 0.75)
            iqr = q3 - q1
            thresholds[cat] = float(q1 - iqr_k * iqr)

        thr_source[cat] = "category"

    return thresholds, thr_source, global_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input_parquet", default="", help="Override input parquet path")
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

    # Ensure required columns exist
    for col in ["row_id", "category", "canonical_text", "image_path", "is_image_missing", "is_text_missing"]:
        if col not in df.columns:
            if col in ["is_image_missing", "is_text_missing"]:
                df[col] = False
            else:
                df[col] = ""

    print("Loading CLIP:", model_name)
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    model.to(device)

    emb_dim = int(getattr(model.config, "projection_dim", 512))

    # Compute similarity for rows that have image present
    rows = df.to_dict(orient="records")
    sim_scores = [np.nan] * len(rows)
    text_embs = [None] * len(rows)
    image_embs = [None] * len(rows)

    batch_indices: list[int] = []
    batch_texts: list[str] = []
    batch_images: list[Image.Image] = []

    def flush_batch():
        nonlocal batch_indices, batch_texts, batch_images
        if not batch_indices:
            return

        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
            t = out.text_embeds.detach().cpu().numpy()   # (B, D)
            v = out.image_embeds.detach().cpu().numpy()  # (B, D)

        for i, df_idx in enumerate(batch_indices):
            t_i = t[i]
            v_i = v[i]
            s = cosine_sim(t_i, v_i)

            sim_scores[df_idx] = s
            text_embs[df_idx] = t_i.astype(np.float32)
            image_embs[df_idx] = v_i.astype(np.float32)

        batch_indices, batch_texts, batch_images = [], [], []

    for idx, r in enumerate(rows):
        if bool(r.get("is_image_missing", False)):
            continue

        img_rel = r.get("image_path", "")
        if not img_rel:
            continue

        img_path = (root / Path(str(img_rel))).resolve()
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        batch_indices.append(idx)
        batch_texts.append(str(r.get("canonical_text", "")))
        batch_images.append(img)

        if len(batch_indices) >= batch_size:
            flush_batch()

    flush_batch()

    # Attach sim scores
    df["sim_score"] = sim_scores

    # Compute thresholds from rows that have sim_score
    df_valid = df[~df["sim_score"].isna()].copy()

    thresholds, thr_source, global_thr = compute_thresholds(
        df_valid=df_valid,
        method=threshold_method,
        iqr_k=iqr_k,
        q=quantile_q,
        min_cat_samples=min_cat,
        global_method=global_method,
        global_q=global_q,
    )

    df["sieve_threshold"] = df["category"].map(thresholds)
    df["threshold_source"] = df["category"].map(thr_source).fillna("unknown")
    df["global_threshold"] = global_thr  # for reference

    # Flagging logic
    df["flagged"] = False
    df.loc[df["is_image_missing"] == True, "flagged"] = True
    df.loc[df["is_text_missing"] == True, "flagged"] = True

    mask_sim = (
        (~df["sim_score"].isna())
        & (~df["sieve_threshold"].isna())
        & (df["sim_score"] < df["sieve_threshold"])
    )
    df.loc[mask_sim, "flagged"] = True

    # Reason
    reasons = []
    for _, r in df.iterrows():
        if bool(r.get("is_image_missing", False)):
            reasons.append("missing_image")
        elif bool(r.get("is_text_missing", False)):
            reasons.append("missing_text")
        elif pd.notna(r.get("sim_score")) and pd.notna(r.get("sieve_threshold")) and float(r["sim_score"]) < float(r["sieve_threshold"]):
            reasons.append("low_similarity")
        else:
            reasons.append("ok")
    df["flag_reason"] = reasons

    # Output file naming (avoid overwrites)
    stem = processed_parquet.stem  # e.g., processed_rows or toy_noisy_rows
    out_csv = outputs_dir / f"sieve_results_{stem}.csv"
    out_flagged = outputs_dir / f"flagged_rows_{stem}.csv"
    df.to_csv(out_csv, index=False)
    df[df["flagged"] == True].to_csv(out_flagged, index=False)

    # Embeddings cache (aligned to df rows)
    emb_npz = cache_dir / f"clip_embeddings_{stem}.npz"

    def as_vec(e):
        if e is None:
            return np.zeros((emb_dim,), dtype=np.float32)
        e = np.asarray(e, dtype=np.float32)
        if e.shape[0] != emb_dim:
            # safety: pad/trim if dimension mismatch
            out = np.zeros((emb_dim,), dtype=np.float32)
            n = min(emb_dim, e.shape[0])
            out[:n] = e[:n]
            return out
        return e

    np.savez_compressed(
        emb_npz,
        row_id=np.asarray(df["row_id"].astype(str).values, dtype="U"),
        text_emb=np.stack([as_vec(e) for e in text_embs]),
        image_emb=np.stack([as_vec(e) for e in image_embs]),
        sim_score=np.array(df["sim_score"].fillna(-1.0).values, dtype=np.float32),
        sieve_threshold=np.array(df["sieve_threshold"].fillna(-1.0).values, dtype=np.float32),
        flagged=np.array(df["flagged"].astype(int).values, dtype=np.int32),
    )

    print("✅ Sieve complete")
    print("Input:", processed_parquet)
    print("Results:", out_csv)
    print("Flagged:", out_flagged)
    print("Embedding cache:", emb_npz)

    print("\nSummary:")
    cols = ["row_id", "category", "sim_score", "sieve_threshold", "threshold_source", "flagged", "flag_reason"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
