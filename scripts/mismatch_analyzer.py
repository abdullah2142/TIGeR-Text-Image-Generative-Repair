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

    brightness = arr.mean(axis=1)
    keep = (brightness > 15) & (brightness < 245)
    if keep.sum() > 100:
        arr = arr[keep]

    avg = arr.mean(axis=0)
    rgb = (int(avg[0]), int(avg[1]), int(avg[2]))

    best = ""
    best_d = 1e18
    for name, p in COLOR_PROTOTYPES.items():
        d = (rgb[0] - p[0]) ** 2 + (rgb[1] - p[1]) ** 2 + (rgb[2] - p[2]) ** 2
        if d < best_d:
            best_d = d
            best = name

    return best, rgb


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
    cat_by_idx: np.ndarray,
    valid_image_idx: np.ndarray,
    same_category_only: bool,
    swap_margin: float,
    candidate_margin: float,
) -> Dict[str, Any]:
    row_id = str(row["row_id"])
    own_idx = rowid_to_idx.get(row_id, None)
    category = str(row.get("category", "")).strip()

    # ---------- Missing checks ----------
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
    if not img_rel or not img_path.exists():
        return {
            "row_id": row_id,
            "error_type": "Type III",
            "mismatch_aspects": ["missing_image_file"],
            "suggested_direction": "HUMAN",
            "confidence": {"image_trust": 0.0, "text_trust": 0.5},
            "proposed_fix": {},
            "notes": f"Image file not found at: {img_rel}",
        }

    # ---------- Color check ----------
    attrs = safe_json_load(row.get("attributes", "{}"))
    text_color = str(attrs.get("color", "")).lower().strip()
    if not text_color:
        text_color = extract_color_from_text(str(row.get("title", "")))

    img_color, img_rgb = dominant_color_name(img_path)
    color_mismatch = bool(text_color) and bool(img_color) and (text_color != img_color)

    # ---------- CLIP retrieval checks ----------
    swap_like = False

    # debug vars
    own_sim_v2t = None
    best_text_row = ""
    best_text_sim = None
    v2t_margin = None

    own_sim_t2v = None
    best_img_row = ""
    best_img_sim = None
    t2v_margin = None

    if own_idx is not None and 0 <= own_idx < len(row_ids):
        # ---- (A) V2T: image -> text ----
        v = image_emb[own_idx: own_idx + 1]          # (1, D)
        sims_text = (text_emb @ v.T).reshape(-1)     # (N,)
        own_sim_v2t = float(sims_text[own_idx])

        sims_text2 = sims_text.copy()
        sims_text2[own_idx] = -1e9
        best_text_idx = int(np.argmax(sims_text2))
        best_text_sim = float(sims_text2[best_text_idx])
        best_text_row = str(row_ids[best_text_idx])
        v2t_margin = best_text_sim - own_sim_v2t

        if best_text_sim > own_sim_v2t + swap_margin:
            swap_like = True

        # ---- (B) T2V: text -> image (replacement candidate) ----
        t = text_emb[own_idx: own_idx + 1]           # (1, D)
        sims_img = (image_emb @ t.T).reshape(-1)     # (N,)
        own_sim_t2v = float(sims_img[own_idx])

        candidate_mask = valid_image_idx.copy()
        candidate_mask[own_idx] = False

        if same_category_only and category:
            candidate_mask &= (cat_by_idx == category)

        if candidate_mask.any():
            sims_img_masked = sims_img.copy()
            sims_img_masked[~candidate_mask] = -1e9
            best_img_idx = int(np.argmax(sims_img_masked))
            best_img_sim = float(sims_img_masked[best_img_idx])
            best_img_row = str(row_ids[best_img_idx])
            t2v_margin = best_img_sim - own_sim_t2v

    # ---------- Decide type ----------
    # Priority: if swap_like, DO NOT trust color patch (color from wrong image would be wrong).
    if swap_like:
        proposed: Dict[str, Any] = {}
        if best_img_row:
            proposed["image_replacement_candidate_row_id"] = best_img_row
            proposed["notes"] = (
                "Swap-like: current image matches another row's text better. "
                "Proposed replacement is the image that best matches THIS row's text "
                f"{'(same-category)' if same_category_only else '(any-category)'}."
            )
        else:
            proposed["notes"] = "Swap-like detected but no valid replacement candidate found."

        # include extra debug: where this image seems to belong
        proposed["image_seems_to_belong_to_row_id"] = best_text_row

        return {
            "row_id": row_id,
            "error_type": "Type I",
            "mismatch_aspects": ["image_relevance"],
            "suggested_direction": "T2V",
            "confidence": {"image_trust": 0.3, "text_trust": 0.8},
            "proposed_fix": proposed,
            "debug": {
                "category": category,
                "v2t": {
                    "own_sim": own_sim_v2t,
                    "best_text_row": best_text_row,
                    "best_text_sim": best_text_sim,
                    "margin": v2t_margin,
                    "swap_margin": swap_margin,
                },
                "t2v": {
                    "own_sim": own_sim_t2v,
                    "best_img_row": best_img_row,
                    "best_img_sim": best_img_sim,
                    "margin": t2v_margin,
                    "candidate_margin": candidate_margin,
                    "same_category_only": same_category_only,
                },
            },
        }

    # If not swap_like but still flagged, we can still propose a candidate replacement
    if best_img_row and own_sim_t2v is not None and best_img_sim is not None:
        if best_img_sim > own_sim_t2v + candidate_margin:
            return {
                "row_id": row_id,
                "error_type": "Type I",
                "mismatch_aspects": ["image_relevance"],
                "suggested_direction": "T2V",
                "confidence": {"image_trust": 0.35, "text_trust": 0.75},
                "proposed_fix": {
                    "image_replacement_candidate_row_id": best_img_row,
                    "notes": (
                        "Low-similarity: proposing a better-matching image for this row's text "
                        f"{'(same-category)' if same_category_only else '(any-category)'}."
                    ),
                },
                "debug": {
                    "category": category,
                    "t2v": {
                        "own_sim": own_sim_t2v,
                        "best_img_row": best_img_row,
                        "best_img_sim": best_img_sim,
                        "margin": t2v_margin,
                        "candidate_margin": candidate_margin,
                    },
                },
            }

    # ---------- (4A) Color mismatch -> V2T patch payload (consistent) ----------
    if color_mismatch:
        # pred_color comes from vision (dominant color), used as patch value
        pred_color = img_color

        return {
        "row_id": row_id,
        "error_type": "Type II",
        "flag_reason": "color_mismatch",              # <-- NEW (helps arbiter/routing)
        "mismatch_aspects": ["color"],
        "suggested_direction": "V2T",
        "confidence": {"image_trust": 0.9, "text_trust": 0.4},   # <-- per spec
        "proposed_fix": {
            "text_patch": {
                "attributes.color": img_color,        # pred_color (from image)
                "title_color_replace": True
            }
        },
        "debug": {
            "category": category,
            "observed_image_color": img_color,
            "observed_image_rgb": list(img_rgb) if isinstance(img_rgb, tuple) else img_rgb,
            "text_color": text_color,
            "v2t": {
                "own_sim": own_sim_v2t,
                "best_text_row": best_text_row,
                "best_text_sim": best_text_sim,
                "margin": v2t_margin,
            },
            "t2v": {
                "own_sim": own_sim_t2v,
                "best_img_row": best_img_row,
                "best_img_sim": best_img_sim,
                "margin": t2v_margin,
            },
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
            "category": category,
            "v2t": {
                "own_sim": own_sim_v2t,
                "best_text_row": best_text_row,
                "best_text_sim": best_text_sim,
                "margin": v2t_margin,
            },
            "t2v": {
                "own_sim": own_sim_t2v,
                "best_img_row": best_img_row,
                "best_img_sim": best_img_sim,
                "margin": t2v_margin,
            },
            "img_color": img_color,
            "text_color": text_color,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sieve_csv", required=True, help="Path to sieve_results_*.csv")
    ap.add_argument("--emb_npz", required=True, help="Path to clip_embeddings_*.npz")

    # ✅ new knobs (won't break your current command)
    ap.add_argument("--same_category_only", type=int, default=1, help="1=restrict replacement candidates to same category (default). 0=allow cross-category.")
    ap.add_argument("--swap_margin", type=float, default=0.01, help="How much better another row's TEXT must match this image to call it swap-like.")
    ap.add_argument("--candidate_margin", type=float, default=0.02, help="How much better a candidate IMAGE must match this row's TEXT to propose replacement.")
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
    df["row_id"] = df["row_id"].astype(str)

    # Load embeddings
    z = np.load(emb_npz, allow_pickle=True)
    row_ids = z["row_id"].astype(str)
    text_emb = z["text_emb"].astype(np.float32)
    image_emb = z["image_emb"].astype(np.float32)

    # Normalize embeddings for fast cosine
    text_emb = l2_normalize(text_emb)
    image_emb = l2_normalize(image_emb)

    rowid_to_idx = {rid: i for i, rid in enumerate(row_ids)}

    # Build per-idx category + valid image mask from df (aligned via row_id mapping)
    df_by_rowid = df.set_index("row_id", drop=False)

    cat_by_idx = np.array(
        [str(df_by_rowid.loc[rid, "category"]) if rid in df_by_rowid.index else "" for rid in row_ids],
        dtype=object,
    )

    valid_image_idx = np.zeros((len(row_ids),), dtype=bool)
    for i, rid in enumerate(row_ids):
        if rid not in df_by_rowid.index:
            continue
        r = df_by_rowid.loc[rid]
        if bool(r.get("is_image_missing", False)):
            continue
        img_rel = str(r.get("image_path", "")).strip()
        if not img_rel:
            continue
        img_path = (root / img_rel).resolve()
        if not img_path.exists():
            continue
        valid_image_idx[i] = True

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
            cat_by_idx=cat_by_idx,
            valid_image_idx=valid_image_idx,
            same_category_only=bool(args.same_category_only),
            swap_margin=float(args.swap_margin),
            candidate_margin=float(args.candidate_margin),
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

    # Save CSV queue for Arbiter
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
