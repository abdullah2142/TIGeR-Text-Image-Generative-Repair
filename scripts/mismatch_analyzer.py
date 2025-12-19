from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image


BASIC_COLORS = [
    "black", "white", "gray",
    "red", "green", "blue",
    "yellow", "orange", "pink", "purple", "brown",
]

# RGB prototypes (rough, but works for toy/MVP)
COLOR_PROTOTYPES = {
    "black":  (20, 20, 20),
    "white":  (235, 235, 235),
    "gray":   (128, 128, 128),
    "red":    (210, 50, 50),
    "green":  (50, 170, 70),
    "blue":   (60, 90, 210),
    "yellow": (220, 210, 60),
    "orange": (230, 140, 50),
    "pink":   (220, 90, 170),
    "purple": (140, 80, 180),
    "brown":  (130, 90, 60),
}


def safe_json_load(s: Any) -> Dict[str, Any]:
    if s is None:
        return {}
    if isinstance(s, dict):
        return s
    if isinstance(s, float) and pd.isna(s):
        return {}
    s = str(s).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def extract_color_from_text(text: str) -> str:
    t = (text or "").lower()
    for c in BASIC_COLORS:
        if re.search(rf"\b{re.escape(c)}\b", t):
            return c
    return ""


def dominant_color_name(image_path: Path) -> Tuple[str, Tuple[int, int, int]]:
    """
    Simple dominant color:
    - downscale to 64x64
    - ignore very dark/very light pixels lightly
    - average RGB
    - map to nearest prototype
    """
    with Image.open(image_path).convert("RGB") as im:
        im = im.resize((64, 64))
        arr = np.array(im, dtype=np.float32).reshape(-1, 3)

    # Filter out near-white and near-black extremes a bit (helps)
    brightness = arr.mean(axis=1)
    keep = (brightness > 15) & (brightness < 245)
    if keep.sum() > 100:
        arr = arr[keep]

    avg = arr.mean(axis=0)  # (3,)
    rgb = (int(avg[0]), int(avg[1]), int(avg[2]))

    # nearest prototype
    best = None
    best_d = 1e18
    for name, p in COLOR_PROTOTYPES.items():
        d = (rgb[0] - p[0]) ** 2 + (rgb[1] - p[1]) ** 2 + (rgb[2] - p[2]) ** 2
        if d < best_d:
            best_d = d
            best = name

    return best or "", rgb


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def analyze_row(
    row: pd.Series,
    root: Path,
    rowid_to_idx: Dict[str, int],
    row_ids: np.ndarray,
    text_emb: np.ndarray,
    image_emb: np.ndarray,
) -> Dict[str, Any]:
    row_id = str(row["row_id"])
    own_idx = rowid_to_idx.get(row_id, None)

    # Basic flags
    if bool(row.get("is_image_missing", False)):
        return {
            "row_id": row_id,
            "error_type": "Type III",
            "mismatch_aspects": ["missing_image"],
            "suggested_direction": "HUMAN",
            "confidence": {"image_trust": 0.0, "text_trust": 0.5},
            "proposed_fix": {},
            "notes": "Image is missing.",
        }

    if bool(row.get("is_text_missing", False)):
        return {
            "row_id": row_id,
            "error_type": "Type III",
            "mismatch_aspects": ["missing_text"],
            "suggested_direction": "V2T",
            "confidence": {"image_trust": 0.7, "text_trust": 0.0},
            "proposed_fix": {"text_patch": {"title": "NEEDS_CAPTION"}},
            "notes": "Text is missing; likely generate/fill text from image.",
        }

    img_rel = str(row.get("image_path", "")).strip()
    img_path = (root / img_rel).resolve()
    if not img_path.exists():
        return {
            "row_id": row_id,
            "error_type": "Type III",
            "mismatch_aspects": ["missing_image_file"],
            "suggested_direction": "HUMAN",
            "confidence": {"image_trust": 0.0, "text_trust": 0.5},
            "proposed_fix": {},
            "notes": f"Image file not found at: {img_rel}",
        }

    # --- Color check ---
    attrs = safe_json_load(row.get("attributes", "{}"))
    text_color = str(attrs.get("color", "")).lower().strip()
    if not text_color:
        # fallback: try title
        text_color = extract_color_from_text(str(row.get("title", "")))

    img_color, img_rgb = dominant_color_name(img_path)
    color_mismatch = bool(text_color) and bool(img_color) and (text_color != img_color)

    # --- CLIP retrieval check (swap detection) ---
    # If we don't have embedding index, we can still do color-based classification.
    swap_like = False
    top_match_row = ""
    top_match_sim = None
    own_sim = None
    margin = None

    if own_idx is not None:
        # cosine similarity: use normalized matrices
        v = image_emb[own_idx : own_idx + 1]  # (1, D)
        sims = (text_emb @ v.T).reshape(-1)   # (N,)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        own_sim = float(sims[own_idx]) if 0 <= own_idx < len(sims) else None

        # best "other" (excluding itself)
        sims2 = sims.copy()
        if 0 <= own_idx < len(sims2):
            sims2[own_idx] = -1e9
        best_other_idx = int(np.argmax(sims2))
        best_other_sim = float(sims2[best_other_idx])
        best_other_row = str(row_ids[best_other_idx])

        top_match_row = best_other_row
        top_match_sim = best_other_sim
        margin = (best_other_sim - (own_sim if own_sim is not None else best_sim))

        # Heuristic: looks swapped if another row’s text matches image notably better
        # Tune margin if needed.
        if own_sim is not None and (best_other_sim > own_sim + 0.01):
            swap_like = True

    # --- Decide error type + direction ---
    # For this MVP:
    # - swap_like => Type I (irrelevant / swapped image) => trust text more => T2V (replace/fix image)
    # - else color mismatch => Type II-ish (attribute mismatch) => trust image more => V2T (fix text/attrs)
    # - else => low similarity but unclear => HUMAN
    if swap_like:
        return {
            "row_id": row_id,
            "error_type": "Type I",
            "mismatch_aspects": ["image_relevance"],
            "suggested_direction": "T2V",
            "confidence": {"image_trust": 0.3, "text_trust": 0.8},
            "proposed_fix": {
                "image_replacement_candidate_row_id": top_match_row,
                "notes": "Image seems to match another row's text better; likely swapped.",
            },
            "debug": {
                "own_sim": own_sim,
                "best_other_row": top_match_row,
                "best_other_sim": top_match_sim,
                "margin": margin,
            },
        }

    if color_mismatch:
        # propose patch to attributes.color (and optionally title)
        patch = {"attributes.color": img_color}
        return {
            "row_id": row_id,
            "error_type": "Type II",
            "mismatch_aspects": ["color"],
            "suggested_direction": "V2T",
            "confidence": {"image_trust": 0.75, "text_trust": 0.4},
            "proposed_fix": {
                "text_patch": patch,
                "observed_image_color": img_color,
                "observed_image_rgb": img_rgb,
                "text_color": text_color,
            },
            "debug": {
                "own_sim": own_sim,
                "best_other_row": top_match_row,
                "best_other_sim": top_match_sim,
                "margin": margin,
            },
        }

    return {
        "row_id": row_id,
        "error_type": "Unknown",
        "mismatch_aspects": ["unknown"],
        "suggested_direction": "HUMAN",
        "confidence": {"image_trust": 0.5, "text_trust": 0.5},
        "proposed_fix": {},
        "debug": {
            "own_sim": own_sim,
            "best_other_row": top_match_row,
            "best_other_sim": top_match_sim,
            "margin": margin,
            "img_color": img_color,
            "text_color": text_color,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sieve_csv", required=True, help="Path to sieve_results_*.csv")
    ap.add_argument("--emb_npz", required=True, help="Path to clip_embeddings_*.npz")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sieve_csv = Path(args.sieve_csv)
    emb_npz = Path(args.emb_npz)

    sieve_csv = sieve_csv if sieve_csv.is_absolute() else (root / sieve_csv)
    emb_npz = emb_npz if emb_npz.is_absolute() else (root / emb_npz)

    sieve_csv = sieve_csv.resolve()
    emb_npz = emb_npz.resolve()

    if not sieve_csv.exists():
        raise FileNotFoundError(f"Missing: {sieve_csv}")
    if not emb_npz.exists():
        raise FileNotFoundError(f"Missing: {emb_npz}")

    df = pd.read_csv(sieve_csv)

    # Load embeddings
    z = np.load(emb_npz, allow_pickle=True)
    row_ids = z["row_id"].astype(str)
    text_emb = z["text_emb"].astype(np.float32)
    image_emb = z["image_emb"].astype(np.float32)

    # Normalize embeddings for fast cosine
    text_emb = l2_normalize(text_emb)
    image_emb = l2_normalize(image_emb)

    rowid_to_idx = {rid: i for i, rid in enumerate(row_ids)}

    # Analyze only flagged
    flagged_df = df[df["flagged"] == True].copy()

    reports = []
    for _, r in flagged_df.iterrows():
        rep = analyze_row(
            row=r,
            root=root,
            rowid_to_idx=rowid_to_idx,
            row_ids=row_ids,
            text_emb=text_emb,
            image_emb=image_emb,
        )
        reports.append(rep)

    outputs_dir = (root / "data/outputs").resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    stem = sieve_csv.stem.replace("sieve_results_", "")
    out_jsonl = outputs_dir / f"mismatch_reports_{stem}.jsonl"
    out_csv = outputs_dir / f"arbiter_queue_{stem}.csv"

    # Save JSONL
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rep in reports:
            f.write(json.dumps(rep, ensure_ascii=False) + "\n")

    # Save CSV queue for Arbiter (flatten key columns)
    flat_rows = []
    for rep in reports:
        flat_rows.append({
            "row_id": rep.get("row_id", ""),
            "error_type": rep.get("error_type", ""),
            "suggested_direction": rep.get("suggested_direction", ""),
            "mismatch_aspects": ",".join(rep.get("mismatch_aspects", [])),
            "image_trust": rep.get("confidence", {}).get("image_trust", None),
            "text_trust": rep.get("confidence", {}).get("text_trust", None),
            "proposed_fix": json.dumps(rep.get("proposed_fix", {}), ensure_ascii=False),
        })
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    print("✅ Mismatch Analyzer complete")
    print("Flagged rows analyzed:", len(flagged_df))
    print("JSONL:", out_jsonl)
    print("Queue CSV:", out_csv)

    if len(flat_rows) > 0:
        print("\nQueue preview:")
        print(pd.DataFrame(flat_rows).head(10).to_string(index=False))
    else:
        print("\nNo flagged rows to analyze (queue empty).")


if __name__ == "__main__":
    main()
